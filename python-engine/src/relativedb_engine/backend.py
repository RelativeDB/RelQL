"""The in-process model backend: :class:`relativedb.scoring.SequenceBackend`
over a :class:`~relativedb_engine.scorer.NativeScorer`, plus the pieces that
are native-only by design — frozen task-head fitting and full fine-tuning
run through librt_c's GPU kernels and never through a remote scorer.
"""
from __future__ import annotations

import ctypes
import json
import os
from typing import Any, Optional, Sequence

import numpy as np

from relativedb.model import ModelConfig, NormalizationMode
from relativedb.relql.ast import ParsedQuery, TaskType
from relativedb.retrieve import RetrieverWiring
from relativedb.schema import Schema
from relativedb.scoring import (D_MODEL, _FT_TASK_OF, FT_BINARY,
                                FT_MULTICLASS, FT_RANKING, FT_REGRESSION,
                                ColumnStats, SequenceBackend, TokenBatch)
from relativedb.task import TaskSpecFactory

from .native import (RT_DEVICE_CUDA, RT_DEVICE_MPS, RtLib, RtModel,
                     RtNativeError, load_lib, resolve_model_path)
from .scorer import NativeScorer, NativeTextEncoder

__all__ = ["RtNativeBackend", "FineTunedHead"]


class FineTunedHead:
    """A trained task head over the frozen backbone's target-cell features.

    The transformer is never updated; this is the small adapter that replaces
    the released checkpoint's zero-shot head. Produced by
    :meth:`~relativedb.engine.Engine.fit_head`, persisted with :meth:`save`,
    and served by passing ``head=`` to :class:`RtNativeBackend` (any
    :class:`~relativedb.scoring.SequenceBackend` accepts it).
    """

    def __init__(self, lib: "RtLib", handle: int, *, task: int,
                 initial_loss: Optional[float] = None,
                 final_loss: Optional[float] = None,
                 seconds: Optional[float] = None,
                 n_examples: Optional[int] = None,
                 classes: Sequence[Any] = (),
                 feat_mu: Optional[np.ndarray] = None,
                 feat_sd: Optional[np.ndarray] = None,
                 column_stats: Optional["ColumnStats"] = None,
                 normalization_mode: NormalizationMode | str = NormalizationMode.ZERO_SHOT):
        self._native = lib
        self._handle = handle
        self.task = task
        self.initial_loss = initial_loss
        self.final_loss = final_loss
        self.seconds = seconds
        self.n_examples = n_examples
        self.classes = tuple(classes)
        # Standardization statistics of the fitted features. Kept on the head
        # so predict() applies exactly the transform fit() saw.
        self.feat_mu = None if feat_mu is None else np.asarray(feat_mu, np.float32)
        self.feat_sd = None if feat_sd is None else np.asarray(feat_sd, np.float32)
        # The normalization this head was fitted under. Serving under a
        # different one silently changes what every number means, so it
        # travels with the weights.
        self.column_stats = column_stats
        self.normalization_mode = NormalizationMode.coerce(normalization_mode)

    def __del__(self):
        try:
            if getattr(self, "_handle", None):
                self._native._lib.rt_finetune_head_free(self._handle)
                self._handle = None
        except Exception:
            pass

    @property
    def n_outputs(self) -> int:
        return int(self._native._lib.rt_finetune_head_outputs(self._handle))

    @property
    def task_name(self) -> str:
        return {FT_BINARY: "binary", FT_REGRESSION: "regression",
                FT_MULTICLASS: "multiclass",
                FT_RANKING: "ranking"}.get(self.task, "unknown")

    def _sidecar(path: str) -> str:                      # noqa: N805
        return str(path) + ".preproc.json"

    def save(self, path: str) -> str:
        """Persist the head, plus the preprocessing it was fitted under.

        The C ABI writes the weights; the feature standardization and the
        column statistics go beside them. All three are one artifact — a head
        served without them predicts on differently-scaled inputs and is
        wrong in a way nothing reports.
        """
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_finetune_head_save(
            self._handle, str(path).encode("utf-8"), err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"saving the fine-tuned head to {path!r} failed: "
                f"{err.value.decode('utf-8', 'replace')}")
        side = {
            "feat_mu": None if self.feat_mu is None else self.feat_mu.tolist(),
            "feat_sd": None if self.feat_sd is None else self.feat_sd.tolist(),
            "column_stats": (None if self.column_stats is None
                             else self.column_stats.to_dict()),
            "normalization_mode": self.normalization_mode.value,
            "classes": [str(c) for c in self.classes],
        }
        with open(FineTunedHead._sidecar(path), "w") as fh:
            json.dump(side, fh)
        return str(path)

    @staticmethod
    def load(path: str) -> "FineTunedHead":
        lib = load_lib()
        err = ctypes.create_string_buffer(512)
        handle = lib._lib.rt_finetune_head_load(
            str(path).encode("utf-8"), err, len(err))
        if not handle:
            raise RtNativeError(
                f"loading a fine-tuned head from {path!r} failed: "
                f"{err.value.decode('utf-8', 'replace')}")
        side_path = FineTunedHead._sidecar(path)
        if not os.path.exists(side_path):
            raise RtNativeError(
                f"{side_path!r} is missing: this head was saved without its "
                f"preprocessing, and serving it would apply the wrong scale "
                f"to every numeric cell. Refit rather than loading it.")
        with open(side_path) as fh:
            side = json.load(fh)
        task = int(lib._lib.rt_finetune_head_task(handle))
        cs = side.get("column_stats")
        return FineTunedHead(
            lib, handle, task=task,
            classes=tuple(side.get("classes") or ()),
            feat_mu=(None if side.get("feat_mu") is None
                     else np.asarray(side["feat_mu"], np.float32)),
            feat_sd=(None if side.get("feat_sd") is None
                     else np.asarray(side["feat_sd"], np.float32)),
            column_stats=None if cs is None else ColumnStats.from_dict(cs),
            normalization_mode=side.get("normalization_mode", "reference"))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Score frozen features ``[N, 512]`` -> logits ``[N, n_outputs]``."""
        f = np.asarray(features, np.float32)
        if self.feat_mu is not None:
            f = (f - self.feat_mu) / self.feat_sd
        f = np.ascontiguousarray(f, np.float32)
        if f.ndim != 2 or f.shape[1] != D_MODEL:
            raise RtNativeError(
                f"features must be [N, {D_MODEL}], got {f.shape}")
        n_out = self.n_outputs
        out = np.zeros(f.shape[0] * n_out, np.float32)
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_finetune_head_predict(
            self._handle, f.shape[0], f.reshape(-1), out, err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"rt_finetune_head_predict failed: "
                f"{err.value.decode('utf-8', 'replace')}")
        return out.reshape(f.shape[0], n_out)

    def __repr__(self) -> str:
        loss = ""
        if self.initial_loss is not None and self.final_loss is not None:
            loss = f" loss {self.initial_loss:.4f}->{self.final_loss:.4f}"
        n = f" on {self.n_examples} examples" if self.n_examples else ""
        return f"<FineTunedHead {self.task_name}{n}{loss}>"


class RtNativeBackend(SequenceBackend):
    """The in-process :class:`~relativedb.engine.ModelBackend`: sequence
    assembly from the base package, MiniLM text embedding and the RT-J
    forward in native code (librt_c).

    Beyond scoring it carries the adaptation surface that is native-only by
    design: :meth:`fit_head` (frozen-backbone task heads, trained on the GPU)
    and the full fine-tuning plumbing used by
    :func:`relativedb.training.finetune`.
    """

    def __init__(self, *, schema: Optional[Schema] = None,
                 wiring: Optional[RetrieverWiring] = None,
                 lib_path: Optional[str] = None,
                 embedder: Optional[NativeTextEncoder] = None,
                 n_threads: int = 0,
                 num_history_windows: int = 3,
                 max_seq_len: int = 2048,      # reference eval uses 8192
                 column_stats: Optional["ColumnStats"] = None,
                 normalization_mode: Optional[NormalizationMode | str] = None,
                 task_spec_factory: Optional[TaskSpecFactory] = None,
                 device: Optional[int] = None,
                 head: Optional[Any] = None,
                 batch_size: Optional[int] = None):
        self._lib_path = lib_path
        if isinstance(head, (str, os.PathLike)):
            head = FineTunedHead.load(str(head))
        scorer = NativeScorer(lib_path=lib_path, embedder=embedder,
                              n_threads=n_threads, device=device)
        super().__init__(scorer, schema=schema, wiring=wiring,
                         n_threads=n_threads,
                         num_history_windows=num_history_windows,
                         max_seq_len=max_seq_len, column_stats=column_stats,
                         normalization_mode=normalization_mode,
                         task_spec_factory=task_spec_factory, head=head,
                         batch_size=batch_size)

    @property
    def embedder(self) -> NativeTextEncoder:
        return self.scorer.embedder

    @property
    def device(self) -> Optional[int]:
        return self.scorer.device

    @device.setter
    def device(self, value: Optional[int]) -> None:
        self.scorer.device = value

    # -- native-only handles (training paths) -------------------------------
    def _model_for(self, model_uri: str) -> RtModel:
        return self.scorer._model_for(model_uri)

    def _collate_native(self, seqs) -> dict:
        """Collate + materialize the full rt_c.h array set (training needs the
        raw arrays, not a :class:`TokenBatch`)."""
        batch: TokenBatch = self._collate(seqs)
        return self.scorer._materialize(batch)

    # -- frozen task-head fitting ------------------------------------------
    def fit_head(self, model: RtModel, task_type: TaskType,
                 features: np.ndarray, labels: np.ndarray,
                 group_offsets: np.ndarray, n_groups: int, *,
                 epochs: int = 100, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 classes: Sequence[Any] = (),
                 normalization_mode: NormalizationMode | str = NormalizationMode.ZERO_SHOT,
                 ) -> "FineTunedHead":
        """Fit a task head on frozen features ``[N, 512]``.

        Training runs on the GPU (Metal or CUDA); inference on the resulting
        head is plain CPU (``rt_finetune_head_predict``), so a head trained
        here serves anywhere.
        """
        lib = load_lib(self._lib_path)
        ft_task = _FT_TASK_OF[task_type]
        if lib.device_available(RT_DEVICE_MPS):
            fit_device = RT_DEVICE_MPS
        elif lib.device_available(RT_DEVICE_CUDA):
            fit_device = RT_DEVICE_CUDA
        else:
            raise RtNativeError(
                "task-head fitting requires a GPU device (Metal or CUDA); "
                "this build or machine has none. Scoring is unaffected.")
        n_outputs = len(classes) if ft_task == FT_MULTICLASS else 1

        class_emb = None
        emb_ptr = None
        if ft_task == FT_MULTICLASS and classes:
            # Seed the head in the checkpoint's own class-embedding basis so it
            # starts from the zero-shot ordering rather than from nothing.
            class_emb = np.ascontiguousarray(
                self.scorer.embed([str(c) for c in classes], normalize=True),
                np.float32)
            emb_ptr = class_emb.ctypes.data_as(ctypes.c_void_p)

        err = ctypes.create_string_buffer(512)
        handle = lib._lib.rt_finetune_head_create(
            model._handle, ft_task, n_outputs, emb_ptr, err, len(err))
        if not handle:
            raise RtNativeError(
                f"rt_finetune_head_create failed: "
                f"{err.value.decode('utf-8', 'replace')}")

        # The backbone's target-cell features sit in a very narrow cone —
        # measured mean pairwise cosine 0.9976 on a 240-issue sample. The
        # shared constant direction then dominates the gradient and the linear
        # head fits only its bias, converging to the label prior and predicting
        # one class for every row. Standardizing per dimension puts the
        # variation on a comparable scale to the mean; on that sample it moved
        # a 4-class probe from 0.450 to 0.817 at identical lr and epochs.
        feats = np.asarray(features, np.float32).reshape(len(labels), -1)
        feat_mu = feats.mean(0)
        feat_sd = feats.std(0) + 1e-6
        feat_mu = np.ascontiguousarray(feat_mu, np.float32)
        feat_sd = np.ascontiguousarray(feat_sd, np.float32)
        rc = lib._lib.rt_finetune_head_reparameterize_standardized(
            handle, feat_mu, feat_sd, err, len(err))
        if rc != 0:
            lib._lib.rt_finetune_head_free(handle)
            raise RtNativeError(
                "zero-shot head reparameterization failed: "
                f"{err.value.decode('utf-8', 'replace')}")
        feats = (feats - feat_mu) / feat_sd

        f = np.ascontiguousarray(feats, np.float32).reshape(-1)
        y = np.ascontiguousarray(labels, np.float32)
        go = np.ascontiguousarray(group_offsets, np.int32)
        i_loss = ctypes.c_float(0.0)
        f_loss = ctypes.c_float(0.0)
        secs = ctypes.c_double(0.0)
        if hasattr(lib._lib, "rt_finetune_head_fit_device"):
            rc = lib._lib.rt_finetune_head_fit_device(
                handle, int(y.shape[0]), f, y, go, int(n_groups), int(epochs),
                float(learning_rate), float(weight_decay), int(fit_device),
                ctypes.byref(i_loss), ctypes.byref(f_loss), ctypes.byref(secs),
                err, len(err))
        else:
            rc = lib._lib.rt_finetune_head_fit_metal(
                handle, int(y.shape[0]), f, y, go, int(n_groups), int(epochs),
                float(learning_rate), float(weight_decay),
                ctypes.byref(i_loss), ctypes.byref(f_loss), ctypes.byref(secs),
                err, len(err))
        if rc != 0:
            lib._lib.rt_finetune_head_free(handle)
            raise RtNativeError(
                f"fine-tune head fit failed: "
                f"{err.value.decode('utf-8', 'replace')}")
        mode = NormalizationMode.coerce(normalization_mode)
        return FineTunedHead(lib, handle, task=ft_task,
                             initial_loss=float(i_loss.value),
                             final_loss=float(f_loss.value),
                             seconds=float(secs.value),
                             n_examples=int(y.shape[0]), classes=classes,
                             feat_mu=feat_mu, feat_sd=feat_sd,
                             column_stats=(self.column_stats
                                           if mode is NormalizationMode.REFERENCE
                                           else None),
                             normalization_mode=mode)
