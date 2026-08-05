"""The in-process model backend: :class:`relativedb.scoring.SequenceBackend`
over a :class:`~relativedb.rt.scorer.RelationalScorer`, plus frozen
task-head fitting on torch. Sequence assembly stays in the base package; the
model runtime is the shared ``relational-transformers`` package.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional, Sequence

import numpy as np

from relativedb.model import NormalizationMode
from relativedb.relql.ast import TaskType
from relativedb.retrieve import RetrieverWiring
from relativedb.schema import Schema
from relativedb.scoring import (D_MODEL, _FT_TASK_OF, FT_BINARY,
                                FT_MULTICLASS, FT_RANKING, FT_REGRESSION,
                                ColumnStats, SequenceBackend, TokenBatch)
from relativedb.task import TaskSpecFactory

from .models import EngineError
from .scorer import RelationalScorer, TextEncoder

__all__ = ["RtBackend", "RtNativeBackend", "FineTunedHead"]

_HEAD_SUFFIX = ".safetensors"


class FineTunedHead:
    """A trained task head over the frozen backbone's target-cell features.

    The transformer is never updated; this is the small linear adapter that
    replaces the released checkpoint's zero-shot head. Produced by
    :meth:`~relativedb.engine.Engine.fit_head`, persisted with :meth:`save`,
    and served by passing ``head=`` to :class:`RtBackend` (any
    :class:`~relativedb.scoring.SequenceBackend` accepts it).
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray, *, task: int,
                 initial_loss: Optional[float] = None,
                 final_loss: Optional[float] = None,
                 seconds: Optional[float] = None,
                 n_examples: Optional[int] = None,
                 classes: Sequence[Any] = (),
                 feat_mu: Optional[np.ndarray] = None,
                 feat_sd: Optional[np.ndarray] = None,
                 column_stats: Optional["ColumnStats"] = None,
                 normalization_mode: NormalizationMode | str = NormalizationMode.ZERO_SHOT):
        self.weight = np.asarray(weight, np.float32)
        self.bias = np.asarray(bias, np.float32)
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

    @property
    def n_outputs(self) -> int:
        return int(self.weight.shape[0])

    @property
    def task_name(self) -> str:
        return {FT_BINARY: "binary", FT_REGRESSION: "regression",
                FT_MULTICLASS: "multiclass",
                FT_RANKING: "ranking"}.get(self.task, "unknown")

    def _sidecar(path: str) -> str:                      # noqa: N805
        return str(path) + ".preproc.json"

    def save(self, path: str) -> str:
        """Persist the head, plus the preprocessing it was fitted under.

        The weights, the feature standardization, and the column statistics
        are one artifact — a head served without them predicts on
        differently-scaled inputs and is wrong in a way nothing reports.
        """
        from safetensors.numpy import save_file

        save_file({"weight": self.weight, "bias": self.bias,
                   "task": np.asarray([self.task], np.int64)}, str(path))
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
        from safetensors.numpy import load_file

        try:
            tensors = load_file(str(path))
            weight, bias = tensors["weight"], tensors["bias"]
            task = int(tensors["task"][0])
        except Exception as e:
            raise EngineError(
                f"loading a fine-tuned head from {path!r} failed: {e}. Heads "
                f"saved by the retired native engine use a different format; "
                f"refit to migrate them.") from e
        side_path = FineTunedHead._sidecar(path)
        if not os.path.exists(side_path):
            raise EngineError(
                f"{side_path!r} is missing: this head was saved without its "
                f"preprocessing, and serving it would apply the wrong scale "
                f"to every numeric cell. Refit rather than loading it.")
        with open(side_path) as fh:
            side = json.load(fh)
        cs = side.get("column_stats")
        return FineTunedHead(
            weight, bias, task=task,
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
            raise EngineError(
                f"features must be [N, {D_MODEL}], got {f.shape}")
        return f @ self.weight.T + self.bias

    def __repr__(self) -> str:
        loss = ""
        if self.initial_loss is not None and self.final_loss is not None:
            loss = f" loss {self.initial_loss:.4f}->{self.final_loss:.4f}"
        n = f" on {self.n_examples} examples" if self.n_examples else ""
        return f"<FineTunedHead {self.task_name}{n}{loss}>"


def _head_loss(torch, logits, labels, task: int, group_offsets, n_groups):
    F = torch.nn.functional
    if task == FT_MULTICLASS:
        return F.cross_entropy(logits, labels.long())
    if task == FT_BINARY:
        return F.binary_cross_entropy_with_logits(logits.reshape(-1), labels)
    if task == FT_REGRESSION:
        return F.huber_loss(logits.reshape(-1), labels)
    # FT_RANKING: listwise cross entropy per candidate group. Relevance is
    # normalized into a target distribution; every group carries at least one
    # positive (enforced upstream in relativedb.training.fit_head).
    scores = logits.reshape(-1)
    total = logits.new_zeros(())
    for g in range(n_groups):
        s, e = int(group_offsets[g]), int(group_offsets[g + 1])
        log_p = F.log_softmax(scores[s:e], dim=0)
        target = labels[s:e] / labels[s:e].sum()
        total = total - (target * log_p).sum()
    return total / max(n_groups, 1)


class RtNativeBackend(SequenceBackend):
    """The in-process :class:`~relativedb.engine.ModelBackend`: sequence
    assembly from the base package, MiniLM text embedding in torch, and the
    RT-J forward through relational-transformers (Triton on CUDA for target
    scores, torch elsewhere, ONNX when selected).

    Beyond scoring it carries the adaptation surface: :meth:`fit_head`
    (frozen-backbone task heads) and the collation plumbing used by
    :func:`relativedb.training.finetune`.
    """

    def __init__(self, *, schema: Optional[Schema] = None,
                 wiring: Optional[RetrieverWiring] = None,
                 lib_path: Optional[str] = None,
                 embedder: Optional[TextEncoder] = None,
                 n_threads: int = 0,
                 num_history_windows: int = 3,
                 max_seq_len: int = 2048,      # reference eval uses 8192
                 column_stats: Optional["ColumnStats"] = None,
                 normalization_mode: Optional[NormalizationMode | str] = None,
                 task_spec_factory: Optional[TaskSpecFactory] = None,
                 device: Optional[int] = None,
                 cuda_backend: str = "triton",
                 inference_backend: str = "auto",
                 onnx_model_path: Optional[str] = None,
                 head: Optional[Any] = None,
                 batch_size: Optional[int] = None):
        self._lib_path = lib_path
        if isinstance(head, (str, os.PathLike)):
            head = FineTunedHead.load(str(head))
        scorer = RelationalScorer(embedder=embedder,
                                  n_threads=n_threads, device=device,
                                  cuda_backend=cuda_backend,
                                  inference_backend=inference_backend,
                                  onnx_model_path=onnx_model_path)
        super().__init__(scorer, schema=schema, wiring=wiring,
                         n_threads=n_threads,
                         num_history_windows=num_history_windows,
                         max_seq_len=max_seq_len, column_stats=column_stats,
                         normalization_mode=normalization_mode,
                         task_spec_factory=task_spec_factory, head=head,
                         batch_size=batch_size)

    @property
    def embedder(self) -> TextEncoder:
        return self.scorer.embedder

    @property
    def device(self) -> Optional[int]:
        return self.scorer.device

    @device.setter
    def device(self, value: Optional[int]) -> None:
        self.scorer.device = value

    # -- model handles (training paths) --------------------------------------
    def _model_for(self, model_uri: str):
        """The torch :class:`relational_transformers.RelationalTransformer`."""
        return self.scorer.torch_model(model_uri)

    def _collate_native(self, seqs) -> dict:
        """Collate + materialize the full array set (training needs the raw
        arrays, not a :class:`TokenBatch`)."""
        batch: TokenBatch = self._collate(seqs)
        return self.scorer._materialize(batch)

    # -- frozen task-head fitting ------------------------------------------
    def fit_head(self, model, task_type: TaskType,
                 features: np.ndarray, labels: np.ndarray,
                 group_offsets: np.ndarray, n_groups: int, *,
                 epochs: int = 100, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 classes: Sequence[Any] = (),
                 normalization_mode: NormalizationMode | str = NormalizationMode.ZERO_SHOT,
                 ) -> "FineTunedHead":
        """Fit a task head on frozen features ``[N, 512]`` with torch AdamW."""
        import torch

        ft_task = _FT_TASK_OF[task_type]
        n_outputs = len(classes) if ft_task == FT_MULTICLASS else 1

        # The backbone's target-cell features sit in a very narrow cone —
        # measured mean pairwise cosine 0.9976 on a 240-issue sample. The
        # shared constant direction then dominates the gradient and the linear
        # head fits only its bias, converging to the label prior and predicting
        # one class for every row. Standardizing per dimension puts the
        # variation on a comparable scale to the mean; on that sample it moved
        # a 4-class probe from 0.450 to 0.817 at identical lr and epochs.
        feats = np.asarray(features, np.float32).reshape(len(labels), -1)
        feat_mu = np.ascontiguousarray(feats.mean(0), np.float32)
        feat_sd = np.ascontiguousarray(feats.std(0) + 1e-6, np.float32)
        feats = (feats - feat_mu) / feat_sd

        head = torch.nn.Linear(D_MODEL, n_outputs)
        if ft_task == FT_MULTICLASS and classes:
            # Seed the head in the checkpoint's own class-embedding basis so it
            # starts from the zero-shot ordering rather than from nothing:
            # logits = <class_emb, text_decoder(feature)>, reparameterized for
            # the standardized features above.
            class_emb = torch.as_tensor(np.ascontiguousarray(
                self.scorer.embed([str(c) for c in classes], normalize=True),
                np.float32))
            decoder = model.model.dec_dict["text"].to("cpu").float()
            weight_raw = class_emb @ decoder.weight            # [C, 512]
            bias_raw = class_emb @ decoder.bias                # [C]
            mu = torch.as_tensor(feat_mu)
            sd = torch.as_tensor(feat_sd)
            with torch.no_grad():
                head.weight.copy_(weight_raw * sd)
                head.bias.copy_(bias_raw + weight_raw @ mu)

        x = torch.as_tensor(feats)
        y = torch.as_tensor(np.ascontiguousarray(labels, np.float32))
        go = np.ascontiguousarray(group_offsets, np.int64)
        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)
        started = time.perf_counter()
        initial_loss = None
        loss_value = None
        head.train()
        for _ in range(max(int(epochs), 1)):
            optimizer.zero_grad(set_to_none=True)
            loss = _head_loss(torch, head(x), y, ft_task, go, n_groups)
            if initial_loss is None:
                initial_loss = float(loss.detach())
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
        head.eval()
        seconds = time.perf_counter() - started

        mode = NormalizationMode.coerce(normalization_mode)
        return FineTunedHead(
            head.weight.detach().numpy(), head.bias.detach().numpy(),
            task=ft_task,
            initial_loss=initial_loss,
            final_loss=loss_value,
            seconds=seconds,
            n_examples=int(y.shape[0]), classes=classes,
            feat_mu=feat_mu, feat_sd=feat_sd,
            column_stats=(self.column_stats
                          if mode is NormalizationMode.REFERENCE
                          else None),
            normalization_mode=mode)


# Primary public name. The historical name remains source-compatible.
RtBackend = RtNativeBackend
