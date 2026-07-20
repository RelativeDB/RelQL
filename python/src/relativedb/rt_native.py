"""Native RT model backend — scores contexts with the golden-verified C++
RT-J engine (``librt_c``). This is the engine's only real scoring backend;
there is no model-free scorer.

Three layers:

1. :class:`RtLib` / :func:`load_lib` — a ctypes binding to the C ABI in
   ``cpp/src/rt_c.h`` (``rt_model_load`` / ``rt_forward`` / ...). The library
   is lazy-loaded from ``RELATIVEDB_RT_LIB`` or the sibling
   ``cpp/build/librt_c.{dylib,so,dll}``; a clear
   :class:`RtNativeUnavailableError` is raised when missing.
2. :class:`TextEmbedder` — a caching wrapper over sentence-transformers
   (``all-MiniLM-L12-v2``, 384-d) for text cells and the frozen
   ``"<column> of <table>"`` schema phrases (F13/F14).
3. :class:`RtNativeBackend` — implements the engine's ``ModelBackend``
   protocol: converts each assembled :class:`~relativedb.engine.EntityContext`
   into the RAW PRE-SORT token arrays the engine consumes (one token per
   feature cell; node graph + FK-parent links; ``bool_as_num`` routing), runs
   one forward pass per batch, and maps the number-head target score back to
   the task's output (sigmoid -> probability for classification, in-context
   denormalization for regression/forecasting).

Checkpoint routing follows :class:`~relativedb.model.ModelConfig`: the engine
passes the already-routed ``model_uri`` (classification vs regression);
``hf://org/repo/subdir`` URIs resolve through huggingface_hub (cache-first)
and plain filesystem paths are accepted directly.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np

from .engine import EntityContext, EntityPrediction, ModelBackend
from .evaluate import eval_bool, eval_value
from .model import ModelConfig
from .relql.ast import (Aggregation, Arith, Case, ColumnRef, Condition, Func,
                        LogicalOp, Not, ParsedQuery, TaskType)
from .retrieve import RetrieverWiring, TemporalBound
from .schema import Schema, ValueType

__all__ = ["RtNativeUnavailableError", "RtNativeError",
           "ContextConnectivityWarning", "ContextTruncationWarning",
           "ColumnStats",
           "RtLib", "RtModel",
           "TextEmbedder", "RtNativeBackend", "FineTunedHead",
           "load_lib", "resolve_model_path"]

D_TEXT = 384
D_MODEL = 512                   # frozen-backbone feature width (rt_c.h)
MAX_F2P = 5

# Compute devices (rt_c.h)
RT_DEVICE_CPU, RT_DEVICE_MPS, RT_DEVICE_CUDA = 0, 1, 2

# Fine-tune task codes (rt_c.h). These are the wire values the C ABI expects,
# not the engine's TaskType.
FT_BINARY, FT_REGRESSION, FT_MULTICLASS, FT_RANKING = 0, 1, 2, 3
# SemType enum from cpp/src/rt.hpp
SEM_NUMBER, SEM_TEXT, SEM_DATETIME, SEM_BOOLEAN = 0, 1, 2, 3

# Shared contract constants — must be byte-for-byte identical across the
# Python / Rust / Java bindings (CONTRACT.md §5).
MAX_MULTICLASS_CLASSES = 1000   # cap on the multiclass label domain
MAX_RANK_CANDIDATES = 1000      # cap on the ranking parent-id candidate set
T_SOFTMAX = 0.1                 # multiclass class_probs softmax temperature

_FT_TASK_OF = {
    TaskType.BINARY_CLASSIFICATION: FT_BINARY,
    TaskType.REGRESSION: FT_REGRESSION,
    TaskType.FORECASTING: FT_REGRESSION,
    TaskType.MULTICLASS_CLASSIFICATION: FT_MULTICLASS,
    TaskType.MULTILABEL_RANKING: FT_RANKING,
}

_SEM_OF_VALUE_TYPE = {
    ValueType.NUMBER: SEM_NUMBER,
    ValueType.TEXT: SEM_TEXT,
    ValueType.DATETIME: SEM_DATETIME,
    ValueType.BOOLEAN: SEM_BOOLEAN,
}


class RtNativeUnavailableError(RuntimeError):
    """librt_c (or a runtime dependency) could not be located/loaded."""


class ContextConnectivityWarning(UserWarning):
    """A context row that other rows hang off emits no tokens, so nothing
    below it can reach the prediction. Declare a feature column on that table
    — or declare its primary key as a column when the key carries meaning."""


class ColumnStats:
    """Per-column ``(mean, std)`` for numeric cells, plus one global normalizer
    for datetimes. Fitted from the data, exactly as the reference does it.

    The reference computes these once at preprocessing (``rustler/src/pre.rs``):
    each numeric column gets the mean and sample standard deviation of its
    whole column with nulls and NaNs excluded, a zero standard deviation is
    replaced by 1.0, and every value is stored as ``(v - mean) / std``. Every
    datetime cell in the dataset shares a single mean/std accumulated across
    all tables.

    This engine previously normalized against the values present in the
    context window instead. That is undefined when a column appears once —
    which is every column on the entity's own row when entities are scored one
    at a time — and it made a value depend on which other rows shared the call.

    Fit under the training temporal bound: statistics drawn from rows after the
    anchor leak the future into every scaled value.
    """

    __slots__ = ("stats", "dt", "bound")

    def __init__(self, stats: dict[tuple[str, str], tuple[float, float]],
                 dt: tuple[float, float] = (0.0, 1.0),
                 bound: str = "unbounded"):
        self.stats = dict(stats)
        self.dt = dt
        self.bound = bound

    @classmethod
    def fit(cls, schema: Schema, wiring: RetrieverWiring,
            bound: Optional[TemporalBound] = None) -> "ColumnStats":
        bound = TemporalBound.unbounded() if bound is None else bound
        out: dict[tuple[str, str], tuple[float, float]] = {}
        dt_vals: list[float] = []
        for table in schema.tables:
            wanted = {c.name: c.type for c in table.columns
                      if c.type in (ValueType.NUMBER, ValueType.BOOLEAN,
                                    ValueType.DATETIME)}
            if not wanted:
                continue
            acc: dict[str, list[float]] = {c: [] for c in wanted}
            scanner = wiring.scanner(table.name)
            for r in scanner(table.name, bound):
                for c, vt in wanted.items():
                    v = r.cells.get(c)
                    if v is None:
                        continue
                    if vt is ValueType.DATETIME:
                        dt_vals.append(_days(v))
                        continue
                    if isinstance(v, bool):
                        v = 1.0 if v else 0.0
                    if isinstance(v, (int, float)) and not math.isnan(float(v)):
                        acc[c].append(float(v))
            for c, vals in acc.items():
                if not vals:
                    continue
                a = np.asarray(vals, float)
                # ddof=1 to match polars' std(1) in the reference
                sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
                out[(table.name, c)] = (float(a.mean()),
                                        sd if sd != 0.0 else 1.0)
        if len(dt_vals) > 1:
            a = np.asarray(dt_vals, float)
            sd = float(a.std(ddof=1))
            dt = (float(a.mean()), sd if sd != 0.0 else 1.0)
        else:
            dt = (0.0, 1.0)
        return cls(out, dt=dt, bound=repr(bound))

    def has(self, table: str, column: str) -> bool:
        return (table, column) in self.stats

    def transform(self, table: str, column: str, x: float) -> float:
        mu, sd = self.stats[(table, column)]
        return (x - mu) / sd

    def transform_datetime(self, x: float) -> float:
        mu, sd = self.dt
        return (x - mu) / sd

    def to_dict(self) -> dict:
        return {"bound": self.bound, "datetime": list(self.dt),
                "stats": {f"{t}.{c}": list(v) for (t, c), v in self.stats.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnStats":
        stats = {}
        for k, v in (d.get("stats") or {}).items():
            t, _, c = k.partition(".")
            stats[(t, c)] = (float(v[0]), float(v[1]))
        dt = tuple(d.get("datetime") or (0.0, 1.0))
        return cls(stats, dt=(float(dt[0]), float(dt[1])),
                   bound=d.get("bound", "unbounded"))

    def __len__(self) -> int:
        return len(self.stats)

    def __repr__(self) -> str:
        return (f"<ColumnStats {len(self.stats)} columns "
                f"datetime={self.dt[0]:.1f}/{self.dt[1]:.1f} bound={self.bound}>")


class ContextTruncationWarning(UserWarning):
    """A context exceeded ``max_seq_len`` and lost cells. Raised because the
    cap bites hardest on the busiest entities, so silent truncation skews a
    comparison rather than merely shrinking it."""


class RtNativeError(RuntimeError):
    """An error reported by the native RT engine."""


# ---------------------------------------------------------------------------
# ctypes binding
# ---------------------------------------------------------------------------

def _lib_filename() -> str:
    if sys.platform == "darwin":
        return "librt_c.dylib"
    if sys.platform.startswith("win"):
        return "rt_c.dll"
    return "librt_c.so"


def _candidate_paths() -> list[str]:
    cands = []
    env = os.environ.get("RELATIVEDB_RT_LIB")
    if env:
        cands.append(env)
    fname = _lib_filename()
    here = os.path.dirname(os.path.abspath(__file__))
    # in-package drop-in, then the sibling C++ build tree of the monorepo
    cands.append(os.path.join(here, fname))
    cands.append(os.path.abspath(os.path.join(
        here, "..", "..", "..", "cpp", "build", fname)))
    return cands


class RtLib:
    """A loaded librt_c with bound signatures (see ``cpp/src/rt_c.h``)."""

    def __init__(self, cdll: ctypes.CDLL, path: str):
        self._lib = cdll
        self.path = path
        lib = self._lib
        lib.rt_model_load.restype = ctypes.c_void_p
        lib.rt_model_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.c_size_t]
        lib.rt_model_free.restype = None
        lib.rt_model_free.argtypes = [ctypes.c_void_p]
        lib.rt_model_num_params.restype = ctypes.c_int64
        lib.rt_model_num_params.argtypes = [ctypes.c_void_p]
        f64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
        u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
        f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        lib.rt_forward.restype = ctypes.c_int
        lib.rt_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p,          # node, f2p, col, table
            u8p, f64p, u8p,                  # is_padding, sem_types, is_target
            f32p, f32p, f32p, f32p, f32p,    # number, datetime, boolean, text, col_name
            ctypes.c_int32, f32p,            # n_threads, out_target_scores
            ctypes.c_char_p, ctypes.c_size_t]
        # rt_forward_ex: rt_forward + a trailing B*384 out_target_text buffer
        # (the dec_dict.text head output at each row's target cell). We always
        # pass a real buffer (multiclass); the NULL case keeps calling rt_forward.
        lib.rt_forward_ex.restype = ctypes.c_int
        lib.rt_forward_ex.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p,          # node, f2p, col, table
            u8p, f64p, u8p,                  # is_padding, sem_types, is_target
            f32p, f32p, f32p, f32p, f32p,    # number, datetime, boolean, text, col_name
            ctypes.c_int32, f32p, f32p,      # n_threads, out_scores, out_target_text
            ctypes.c_char_p, ctypes.c_size_t]

        # rt_forward on an explicit device. rt_forward is the CPU entry point;
        # without this binding every score and every encode ran on CPU while
        # Metal sat idle, which is most of the per-row inference cost.
        lib.rt_forward_device.restype = ctypes.c_int
        lib.rt_forward_device.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p,          # node, f2p, col, table
            u8p, f64p, u8p,                  # is_padding, sem_types, is_target
            f32p, f32p, f32p, f32p, f32p,    # number, datetime, boolean, text, col_name
            ctypes.c_int32, ctypes.c_int32,  # n_threads, device
            f32p,                            # out_target_scores
            ctypes.c_char_p, ctypes.c_size_t]

        # ---- frozen-backbone fine-tuning (see rt_c.h) ----------------------
        # The transformer stays frozen; only a small task head is trained, on
        # its final target-cell states [N, 512].
        lib.rt_encode_targets_device.restype = ctypes.c_int
        lib.rt_encode_targets_device.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p,          # node, f2p, col, table
            u8p, f64p, u8p,                  # is_padding, sem_types, is_target
            f32p, f32p, f32p, f32p, f32p,    # number, datetime, boolean, text, col_name
            ctypes.c_int32, ctypes.c_int32,  # n_threads, device
            f32p,                            # out_target_features [B, 512]
            ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_create.restype = ctypes.c_void_p
        lib.rt_finetune_head_create.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_void_p,                 # class_embeddings or NULL
            ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_load.restype = ctypes.c_void_p
        lib.rt_finetune_head_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                              ctypes.c_size_t]
        lib.rt_finetune_head_free.restype = None
        lib.rt_finetune_head_free.argtypes = [ctypes.c_void_p]
        lib.rt_finetune_head_save.restype = ctypes.c_int
        lib.rt_finetune_head_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                              ctypes.c_char_p, ctypes.c_size_t]
        i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        lib.rt_finetune_head_fit_metal.restype = ctypes.c_int
        lib.rt_finetune_head_fit_metal.argtypes = [
            ctypes.c_void_p, ctypes.c_int32,
            f32p, f32p,                      # features [N,512], labels [N]
            i32p, ctypes.c_int32,            # group_offsets, n_groups
            ctypes.c_int32, ctypes.c_float, ctypes.c_float,   # epochs, lr, wd
            ctypes.POINTER(ctypes.c_float),  # out_initial_loss
            ctypes.POINTER(ctypes.c_float),  # out_final_loss
            ctypes.POINTER(ctypes.c_double), # out_seconds
            ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_predict.restype = ctypes.c_int
        lib.rt_finetune_head_predict.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, f32p, f32p,
            ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_outputs.restype = ctypes.c_int32
        lib.rt_finetune_head_outputs.argtypes = [ctypes.c_void_p]
        lib.rt_finetune_head_task.restype = ctypes.c_int32
        lib.rt_finetune_head_task.argtypes = [ctypes.c_void_p]
        lib.rt_device_available.restype = ctypes.c_int
        lib.rt_device_available.argtypes = [ctypes.c_int32]

    def device_available(self, device: int) -> bool:
        return bool(self._lib.rt_device_available(device))

    def load_model(self, safetensors_path: str) -> "RtModel":
        err = ctypes.create_string_buffer(512)
        handle = self._lib.rt_model_load(
            safetensors_path.encode("utf-8"), err, len(err))
        if not handle:
            raise RtNativeError(
                f"rt_model_load({safetensors_path!r}) failed: "
                f"{err.value.decode('utf-8', 'replace')}")
        return RtModel(self, handle, safetensors_path)


class FineTunedHead:
    """A trained task head over the frozen backbone's target-cell features.

    The transformer is never updated; this is the small adapter that replaces
    the released checkpoint's zero-shot head. Produced by
    :meth:`~relativedb.engine.Engine.finetune`, persisted with :meth:`save`,
    and served by passing ``head=`` to :class:`RtNativeBackend`.
    """

    def __init__(self, lib: "RtLib", handle: int, *, task: int,
                 initial_loss: Optional[float] = None,
                 final_loss: Optional[float] = None,
                 seconds: Optional[float] = None,
                 n_examples: Optional[int] = None,
                 classes: Sequence[Any] = (),
                 feat_mu: Optional[np.ndarray] = None,
                 feat_sd: Optional[np.ndarray] = None,
                 column_stats: Optional["ColumnStats"] = None):
        self._native = lib
        self._handle = handle
        self.task = task
        self.initial_loss = initial_loss
        self.final_loss = final_loss
        self.seconds = seconds
        self.n_examples = n_examples
        self.classes = tuple(classes)
        # Standardization statistics of the fitted features (§ below). Kept on
        # the head so predict() applies exactly the transform fit() saw.
        self.feat_mu = None if feat_mu is None else np.asarray(feat_mu, np.float32)
        self.feat_sd = None if feat_sd is None else np.asarray(feat_sd, np.float32)
        # The normalization this head was fitted under. Serving under a
        # different one silently changes what every number means, so it
        # travels with the weights.
        self.column_stats = column_stats

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
            column_stats=None if cs is None else ColumnStats.from_dict(cs))

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


class RtModel:
    """A loaded RT-J checkpoint living in the native engine."""

    def __init__(self, lib: RtLib, handle: int, path: str):
        self._native = lib
        self._handle = handle
        self.path = path

    @property
    def num_params(self) -> int:
        return int(self._native._lib.rt_model_num_params(self._handle))

    @staticmethod
    def _prep(node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
              is_target, number_v, datetime_v, boolean_v, text_v, col_name_v):
        node_idxs = np.ascontiguousarray(node_idxs, np.int64)
        B, S = node_idxs.shape
        return (B, S, node_idxs,
                np.ascontiguousarray(f2p, np.int64).reshape(B, S, MAX_F2P),
                np.ascontiguousarray(col_idxs, np.int64).reshape(B, S),
                np.ascontiguousarray(table_idxs, np.int64).reshape(B, S),
                np.ascontiguousarray(is_padding, np.uint8).reshape(B, S),
                np.ascontiguousarray(sem_types, np.int64).reshape(B, S),
                np.ascontiguousarray(is_target, np.uint8).reshape(B, S),
                np.ascontiguousarray(number_v, np.float32).reshape(B, S),
                np.ascontiguousarray(datetime_v, np.float32).reshape(B, S),
                np.ascontiguousarray(boolean_v, np.float32).reshape(B, S),
                np.ascontiguousarray(text_v, np.float32).reshape(B, S, D_TEXT),
                np.ascontiguousarray(col_name_v, np.float32).reshape(B, S, D_TEXT))

    def forward(self, *, node_idxs, f2p, col_idxs, table_idxs, is_padding,
                sem_types, is_target, number_v, datetime_v, boolean_v,
                text_v, col_name_v, n_threads: int = 0,
                device: int = RT_DEVICE_CPU) -> np.ndarray:
        """Raw PRE-sort arrays in (see rt_c.h) -> per-row target score [B]."""
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        out = np.zeros(B, np.float32)
        err = ctypes.create_string_buffer(512)
        if device == RT_DEVICE_CPU:
            rc = self._native._lib.rt_forward(
                self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
                is_padding, sem_types, is_target, number_v, datetime_v,
                boolean_v, text_v, col_name_v, int(n_threads), out,
                err, len(err))
        else:
            rc = self._native._lib.rt_forward_device(
                self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
                is_padding, sem_types, is_target, number_v, datetime_v,
                boolean_v, text_v, col_name_v, int(n_threads), int(device),
                out, err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"rt_forward failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return out

    def encode_targets(self, *, node_idxs, f2p, col_idxs, table_idxs,
                       is_padding, sem_types, is_target, number_v, datetime_v,
                       boolean_v, text_v, col_name_v, n_threads: int = 0,
                       device: int = RT_DEVICE_CPU) -> np.ndarray:
        """Frozen-backbone features: the final target-cell state ``[B, 512]``.

        This is what fine-tuning trains on — the transformer is not updated, so
        every example need only be encoded once."""
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        out = np.zeros(B * D_MODEL, np.float32)
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_encode_targets_device(
            self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
            is_padding, sem_types, is_target, number_v, datetime_v,
            boolean_v, text_v, col_name_v, int(n_threads), int(device),
            out, err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"rt_encode_targets_device failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return out.reshape(B, D_MODEL)

    def forward_ex(self, *, node_idxs, f2p, col_idxs, table_idxs, is_padding,
                   sem_types, is_target, number_v, datetime_v, boolean_v,
                   text_v, col_name_v, n_threads: int = 0
                   ) -> tuple[np.ndarray, np.ndarray]:
        """As :meth:`forward`, but also returns the TEXT decoder-head output at
        each row's target cell: ``(scores[B], target_text[B, 384])`` (rt_c.h
        ``rt_forward_ex``). ``target_text`` is NOT L2-normalized."""
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        out = np.zeros(B, np.float32)
        out_text = np.zeros((B, D_TEXT), np.float32)
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_forward_ex(
            self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
            is_padding, sem_types, is_target, number_v, datetime_v,
            boolean_v, text_v, col_name_v, int(n_threads), out, out_text,
            err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"rt_forward_ex failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return out, out_text

    def close(self) -> None:
        if self._handle:
            self._native._lib.rt_model_free(self._handle)
            self._handle = 0

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


_cached_lib: Optional[RtLib] = None


def load_lib(path: Optional[str] = None) -> RtLib:
    """Lazy-load librt_c; raises :class:`RtNativeUnavailableError` listing the
    searched paths when it cannot be found."""
    global _cached_lib
    if _cached_lib is not None and path is None:
        return _cached_lib
    candidates = [path] if path else _candidate_paths()
    tried = []
    for cand in candidates:
        if not cand:
            continue
        tried.append(cand)
        if not os.path.exists(cand):
            continue
        try:
            lib = RtLib(ctypes.CDLL(cand), cand)
        except (OSError, AttributeError) as e:
            raise RtNativeUnavailableError(
                f"found {cand} but could not bind the rt_c ABI: {e}") from e
        if path is None:
            _cached_lib = lib
        return lib
    raise RtNativeUnavailableError(
        "librt_c was not found (build cpp/ with cmake, or set RELATIVEDB_RT_LIB "
        "to the built library). Searched: " + ", ".join(tried))


# ---------------------------------------------------------------------------
# checkpoint URI resolution
# ---------------------------------------------------------------------------

def _quantized_variant() -> str | None:
    """RELATIVEDB_RT_QUANTIZED: 1/true/q8 -> q8, q4 -> q4, f16 -> f16."""
    v = os.environ.get("RELATIVEDB_RT_QUANTIZED", "").lower()
    if v in ("1", "true", "q8"):
        return "q8"
    if v in ("q4", "f16"):
        return v
    return None


def _pick_model(dirpath: str) -> str:
    """dir -> model.<variant>.safetensors (from cpp/rt_quantize) when
    RELATIVEDB_RT_QUANTIZED selects a variant and it is present, else
    model.safetensors."""
    v = _quantized_variant()
    if v:
        q = os.path.join(dirpath, f"model.{v}.safetensors")
        if os.path.isfile(q):
            return q
    return os.path.join(dirpath, "model.safetensors")


def resolve_model_path(uri: str) -> str:
    """Resolve a checkpoint URI to a local ``model.safetensors`` path.

    Accepts a filesystem path (file or directory containing
    ``model.safetensors``) or ``hf://org/repo/subdir`` (resolved through
    huggingface_hub, cache-first). With env ``RELATIVEDB_RT_QUANTIZED=1``,
    an int8 ``model.q8.safetensors`` sibling is preferred when present
    (explicit file paths are always used as given)."""
    if os.path.isfile(uri):
        return uri
    if os.path.isdir(uri):
        p = _pick_model(uri)
        if os.path.isfile(p):
            return p
        raise RtNativeUnavailableError(
            f"directory {uri!r} has no model.safetensors")
    if uri.startswith("hf://"):
        rest = uri[len("hf://"):].strip("/")
        parts = rest.split("/")
        if len(parts) < 2:
            raise RtNativeUnavailableError(f"malformed hf:// URI: {uri!r}")
        repo_id = "/".join(parts[:2])
        sub = "/".join(parts[2:])
        filename = (sub + "/" if sub else "") + "model.safetensors"
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RtNativeUnavailableError(
                f"resolving {uri!r} requires huggingface_hub: "
                f"pip install huggingface_hub") from e
        try:  # cache-first: never hit the network when already downloaded
            path = hf_hub_download(repo_id, filename, local_files_only=True)
        except Exception:
            path = hf_hub_download(repo_id, filename)
        # a quantized sibling lives beside the snapshot file, not in the repo
        picked = _pick_model(os.path.dirname(path))
        return picked if os.path.isfile(picked) else path
    raise RtNativeUnavailableError(
        f"cannot resolve model uri {uri!r} (not a path, not hf://)")


# ---------------------------------------------------------------------------
# text embeddings (MiniLM), cached
# ---------------------------------------------------------------------------

class TextEmbedder:
    """Caching wrapper over sentence-transformers for the pinned MiniLM
    encoder. Lazy: the model loads on first :meth:`encode`."""

    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_norm: dict[str, np.ndarray] = {}

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RtNativeUnavailableError(
                    "RtNativeBackend needs sentence-transformers for the "
                    f"pinned {self.model_name} text encoder: "
                    "pip install sentence-transformers") from e
            self._model = SentenceTransformer(
                f"sentence-transformers/{self.model_name}")
            # Left at the encoder's shipped 128 word-piece window ON PURPOSE.
            # RT-J is frozen and was trained on vectors this encoder produced
            # at that setting; raising it changes the embedding function out
            # from under the backbone. Measured on 60 long issue bodies, 256
            # vs 128 gives mean cosine 0.883 (min 0.520) — a different input
            # distribution, not more information. Text past ~128 word pieces
            # cannot be represented by this encoder as trained; the way to
            # carry more is more ROWS (each gets its own window), not a
            # longer window.
        return self._model

    def encode(self, texts: Sequence[str], *,
               normalize: bool = False) -> list[np.ndarray]:
        """Mean-pooled MiniLM embeddings, cached. ``normalize=True`` returns
        L2-normalized (unit) vectors via SBERT ``normalize_embeddings=True``
        (a separate cache) — used for multiclass class-label embeddings; the
        default (un-normalized) matches training for text CELL values."""
        cache = self._cache_norm if normalize else self._cache
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            embs = self._load().encode(missing, normalize_embeddings=normalize,
                                       show_progress_bar=False)
            for t, e in zip(missing, embs):
                cache[t] = np.asarray(e, np.float32)
        return [cache[t] for t in texts]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# context -> RT token batch conversion
# ---------------------------------------------------------------------------

_TASK_TABLE = "task"
_TASK_TIME_COL = "timestamp"
_TASK_LABEL_COL = "label"


def _sem_of_python_value(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return SEM_BOOLEAN
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return SEM_NUMBER
    if isinstance(v, datetime):
        return SEM_DATETIME
    if isinstance(v, str):
        return SEM_TEXT
    return None  # lists/None/unsupported -> no token


def _days(t: datetime) -> float:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp() / 86400.0


class _Seq:
    """Token accumulator for one entity's context window (pre-sort order)."""

    def __init__(self):
        self.node: list[int] = []
        self.f2p: list[list[int]] = []
        self.col: list[tuple[str, str]] = []      # (column, table) key
        self.tab: list[str] = []
        self.sem: list[int] = []
        self.is_tgt: list[bool] = []
        self.value: list[Any] = []                # raw, normalized at collate

    def add(self, node: int, parents: Sequence[int], col: str, table: str,
            sem: int, value: Any, *, target: bool = False) -> None:
        self.node.append(node)
        self.f2p.append((list(parents) + [-1] * MAX_F2P)[:MAX_F2P])
        self.col.append((col, table))
        self.tab.append(table)
        self.sem.append(sem)
        self.is_tgt.append(target)
        self.value.append(value)

    def __len__(self) -> int:
        return len(self.node)

    def clone(self) -> "_Seq":
        """A deep-enough copy: independent lists so per-candidate ranking rows
        can diverge (target-cell f2p) and normalize their own values."""
        s = _Seq()
        s.node = list(self.node)
        s.f2p = [list(x) for x in self.f2p]
        s.col = list(self.col)
        s.tab = list(self.tab)
        s.sem = list(self.sem)
        s.is_tgt = list(self.is_tgt)
        s.value = list(self.value)
        return s


class RtNativeBackend:
    """A real :class:`~relativedb.engine.ModelBackend` over the C++ RT engine.

    Token mapping (mirrors rt/data.py — the arrays are RAW PRE-SORT; the
    native engine sorts and builds its own attention masks):

    * one token per feature cell ``(value, column, table)`` (F10); FKs become
      the node graph rather than tokens, as does a primary key the schema has
      not also declared as a column;
    * every context row is a graph node: tokens of one row share its
      ``node_idx``; ``f2p[token] = node_idxs`` of the row's FK-parent rows
      that are present in the context (up to 5, -1-padded);
    * the prediction is a synthetic ``task`` row (child of the entity node)
      with a ``timestamp`` cell at the anchor and a masked ``label`` cell
      (``is_target``); past task outcomes evaluated from the entity's own
      history are added as unmasked sibling task rows (self labels, F65);
    * numbers/booleans are z-scored per column over the batch's in-context
      values, datetimes with one global stat (F11/F12); booleans then route
      through the number channel (``bool_as_num``, F52);
    * text cells and ``"<column> of <table>"`` schema phrases embed with the
      pinned MiniLM encoder (F13/F14).

    Classification scores are logits -> sigmoid -> probability; regression
    scores are normalized -> denormalized with the in-context label stats.

    Multiclass classification masks the target cell as TEXT, reads the text
    decoder head via ``rt_forward_ex``, and nearest-neighbor-decodes it against
    L2-normalized MiniLM embeddings of the distinct target-column values
    (CONTRACT.md §2). Ranking scores each candidate parent id's existence
    context through the number head and takes the top-k (§3). Both need a
    ``wiring`` with a ``TableScanner`` to enumerate the class/candidate domain.
    """

    def __init__(self, *, schema: Optional[Schema] = None,
                 wiring: Optional[RetrieverWiring] = None,
                 lib_path: Optional[str] = None,
                 embedder: Optional[TextEmbedder] = None,
                 n_threads: int = 0,
                 num_history_windows: int = 3,
                 max_seq_len: int = 8192,      # reference eval ctx_size
                 column_stats: Optional["ColumnStats"] = None,
                 device: Optional[int] = None,
                 head: Optional[Any] = None):
        self.schema = schema
        self.wiring = wiring
        self._lib_path = lib_path
        self.embedder = embedder or TextEmbedder()
        self.n_threads = n_threads
        self.num_history_windows = max(1, num_history_windows)
        self.max_seq_len = max_seq_len
        # A head fitted under fitted-statistics normalization must be served
        # the same way, so the head's own stats win over anything passed here.
        # Metal when the build and machine provide it, CPU otherwise. Scoring
        # ran on CPU regardless before rt_forward_device was bound, which
        # dominated per-row latency on large contexts.
        self.device = device
        self.column_stats = column_stats
        if head is not None and getattr(head, "column_stats", None) is not None:
            self.column_stats = head.column_stats
        self._models: dict[str, RtModel] = {}
        # A fine-tuned head replaces the checkpoint's zero-shot head for the
        # task it was trained on; every other task still scores zero-shot.
        if isinstance(head, (str, os.PathLike)):
            head = FineTunedHead.load(str(head))
        self.head: Optional[FineTunedHead] = head

    def _head_for(self, task_type: TaskType) -> Optional["FineTunedHead"]:
        """The fine-tuned head, when it was trained for this task type."""
        if self.head is None:
            return None
        return self.head if _FT_TASK_OF.get(task_type) == self.head.task else None

    def _resolve_device(self) -> int:
        if self.device is None:
            lib = load_lib(self._lib_path)
            self.device = (RT_DEVICE_MPS if lib.device_available(RT_DEVICE_MPS)
                           else RT_DEVICE_CPU)
        return self.device

    def _encode(self, model: RtModel, seqs: list["_Seq"]) -> np.ndarray:
        """Frozen-backbone features ``[len(seqs), 512]`` for these sequences."""
        return self._forward(model, seqs, encode=True)

    # -- model handles ------------------------------------------------------
    def _model_for(self, model_uri: str) -> RtModel:
        path = resolve_model_path(model_uri)
        if path not in self._models:
            self._models[path] = load_lib(self._lib_path).load_model(path)
        return self._models[path]

    # -- ModelBackend -------------------------------------------------------
    def score(self, query: ParsedQuery, task_type: TaskType,
              contexts: list[EntityContext], model_uri: str,
              config: ModelConfig) -> list[EntityPrediction]:
        ret = query.ret
        ret_kind = ret.kind if ret is not None else None
        if not contexts:
            return []
        if task_type is TaskType.MULTICLASS_CLASSIFICATION:
            return self._score_multiclass(query, contexts, model_uri)
        if task_type is TaskType.MULTILABEL_RANKING:
            return self._score_ranking(query, contexts, model_uri)
        model = self._model_for(model_uri)
        seqs, label_mu, label_sd = self._build_sequences(query, task_type,
                                                         contexts)
        head = self._head_for(task_type)
        if head is not None:
            # trained head over the frozen backbone's target-cell features
            scores = head.predict(self._encode(model, seqs))[:, 0]
        else:
            scores = self._forward(model, seqs)
        preds: list[EntityPrediction] = []
        for ctx, s in zip(contexts, scores):
            s = float(s)
            if task_type is TaskType.BINARY_CLASSIFICATION:
                p = 1.0 / (1.0 + math.exp(-s))
                preds.append(self._shape_binary(ctx.entity_id, ret_kind, p))
            else:
                # The released head emits a normalized score, so it is scaled
                # back with the in-context label statistics. A fine-tuned head
                # was fitted on raw target values and already predicts in the
                # label's own units — scaling it again applies the transform
                # twice and inflates the error by orders of magnitude.
                v = s if head is not None else s * label_sd + label_mu
                if task_type is TaskType.FORECASTING:
                    n = query.num_forecasts or 1
                    preds.append(EntityPrediction(ctx.entity_id, value=v,
                                                  forecast=tuple([v] * n)))
                else:
                    preds.append(EntityPrediction(ctx.entity_id, value=v))
        return preds

    @staticmethod
    def _shape_binary(entity_id: Any, ret_kind: Optional[str],
                      p: float) -> EntityPrediction:
        """Shape the model's binary probability per the RETURN clause (moved
        here from the deleted history baseline; operates on the model output,
        not on any history-window heuristic)."""
        if ret_kind == "CLASS":
            # Hard decision at threshold 0.5, not the score.
            return EntityPrediction(
                entity_id, predicted_class="true" if p >= 0.5 else "false")
        if ret_kind == "DISTRIBUTION":
            return EntityPrediction(
                entity_id, class_probs={"true": p, "false": 1.0 - p})
        if ret_kind == "EXPECTED_VALUE":
            # Expected value of the 0/1 indicator is p.
            return EntityPrediction(entity_id, value=p)
        # PROBABILITY (explicit) or default.
        return EntityPrediction(entity_id, probability=p)

    # -- batch building -----------------------------------------------------
    def _sem_for_cell(self, table: str, col: str, value: Any) -> Optional[int]:
        if self.schema is not None:
            tdef = self.schema.table(table)
            cdef = tdef.column(col) if tdef else None
            if cdef is not None:
                return _SEM_OF_VALUE_TYPE[cdef.type]
        return _sem_of_python_value(value)

    def _self_labels(self, query: ParsedQuery, task_type: TaskType,
                     ctx: EntityContext) -> list[tuple[datetime, float]]:
        """(timestamp, outcome) pairs from trailing history windows (F65)."""
        aggs = query.target_aggregations
        window = next((a.window for a in aggs if a.window is not None), None)
        span = window.span() if window is not None else None
        if ctx.anchor is None or span is None:
            return []
        rows_by_table = ctx.rows_by_table()
        cells = ctx.entity_cells(query.entity_key.table)
        out = []
        for k in range(1, self.num_history_windows + 1):
            pa = ctx.anchor - span * k
            if task_type is TaskType.BINARY_CLASSIFICATION:
                v = 1.0 if eval_bool(query.target, rows_by_table, cells, pa) \
                    else 0.0
            else:
                ev = eval_value(query.target, rows_by_table, cells, pa)
                if isinstance(ev, bool):
                    ev = 1.0 if ev else 0.0
                if not isinstance(ev, (int, float)):
                    continue
                v = float(ev)
            out.append((pa, v))
        return out

    @staticmethod
    def _target_columns(expr: Any) -> set[tuple[str, str]]:
        """Every ``(table, column)`` the target expression reads."""
        out: set[tuple[str, str]] = set()
        stack = [expr]
        while stack:
            e = stack.pop()
            if isinstance(e, ColumnRef):
                out.add((e.table, e.column))
            elif isinstance(e, Aggregation):
                out.add((e.column.table, e.column.column))
                stack.append(e.filter)
            elif isinstance(e, Condition):
                stack += [e.left, e.right_expr]
            elif isinstance(e, LogicalOp):
                stack += [e.left, e.right]
            elif isinstance(e, Not):
                stack.append(e.expr)
            elif isinstance(e, Arith):
                stack += [e.left, e.right]
            elif isinstance(e, Func):
                stack += list(e.args)
            elif isinstance(e, Case):
                for c, t in e.whens:
                    stack += [c, t]
                stack.append(e.else_)
        return out

    def _fk_to_parent(self) -> dict[str, dict[str, str]]:
        if self.schema is None:
            return {}
        return {t.name: {l.fk_column: l.to_table
                         for l in self.schema.links_from(t.name)}
                for t in self.schema.tables}

    @staticmethod
    def _severed_parents(seq: "_Seq", node_of: dict) -> set:
        """Tables whose rows are referenced as a parent but emit no tokens.

        Attention reaches a row only through its tokens, so a token-less parent
        is a dead end: everything hanging off it — however much context was
        assembled — can never influence the prediction."""
        with_tokens = set(seq.node)
        table_of_node = {n: key[0] for key, n in node_of.items()}
        out = set()
        for parents in seq.f2p:
            for p in parents:
                if p >= 0 and p not in with_tokens:
                    t = table_of_node.get(p)
                    if t is not None and t != _TASK_TABLE:
                        out.add(t)
        return out

    def _build_ctx_seq(self, query: ParsedQuery, task_type: TaskType,
                       ctx: EntityContext, fk_to_parent: dict, all_labels: list,
                       *, target_sem: int = SEM_NUMBER
                       ) -> tuple[_Seq, dict, int, int]:
        """Assemble one entity's context into a token sequence. Returns
        ``(seq, node_of, entity_node, tgt_idx)``. ``target_sem`` overrides the
        masked target cell's sem-type (SEM_TEXT for multiclass, §2.1); ``tgt_idx``
        is that cell's position (used by ranking to rewire its f2p, §3.2)."""
        entity_table = query.entity_key.table
        # Columns the target reads off the entity's own table. The task row
        # carries a masked copy of the answer, but the entity's real row sits
        # in its own context and would otherwise hand the answer straight to
        # the model. Suppressed on that one row only: the same column on every
        # *other* row is legitimate history, and for a static attribute it is
        # the only thing that forms a class domain at all.
        suppressed = {c for t, c in self._target_columns(query.target)
                      if t == entity_table}
        truncated = [False]
        seq = _Seq()
        node_of: dict[tuple[str, Any], int] = {}

        def node(key: tuple[str, Any]) -> int:
            if key not in node_of:
                node_of[key] = len(node_of)
            return node_of[key]

        # rows first claim node ids so f2p links resolve in any order
        for r in ctx.rows:
            node(r.key)
        by_id: dict[Any, list[tuple[str, Any]]] = {}
        for r in ctx.rows:
            by_id.setdefault(r.id, []).append(r.key)

        entity_node = node((entity_table, ctx.entity_id))

        # -- the target task row (masked label) --
        tgt_node = node((_TASK_TABLE, "__target__"))
        if ctx.anchor is not None:
            seq.add(tgt_node, [entity_node], _TASK_TIME_COL, _TASK_TABLE,
                    SEM_DATETIME, ctx.anchor)
        tgt_idx = len(seq)
        seq.add(tgt_node, [entity_node], _TASK_LABEL_COL, _TASK_TABLE,
                target_sem, None, target=True)

        # -- past outcomes of the same task (self labels, F65) --
        for ts, label in self._self_labels(query, task_type, ctx):
            hnode = node((_TASK_TABLE, ts))
            seq.add(hnode, [entity_node], _TASK_LABEL_COL, _TASK_TABLE,
                    SEM_NUMBER, label)
            seq.add(hnode, [entity_node], _TASK_TIME_COL, _TASK_TABLE,
                    SEM_DATETIME, ts)
            all_labels.append(label)

        # -- one token per feature cell of every context row --
        for r in ctx.rows:
            parents: list[int] = []
            for fk, pid in r.parents.items():
                ptable = fk_to_parent.get(r.table, {}).get(fk)
                if ptable is not None:
                    pkey = (ptable, pid)
                    if pkey in node_of:
                        parents.append(node_of[pkey])
                    continue
                # no schema: link by unique id match within the context
                cands = by_id.get(pid, [])
                if len(cands) == 1:
                    parents.append(node_of[cands[0]])
            rnode = node_of[r.key]
            is_entity_row = r.key == (entity_table, ctx.entity_id)
            for col, v in r.cells.items():
                if len(seq) >= self.max_seq_len:
                    # Truncation is never silent: it falls hardest on the
                    # busiest entities, which in an imbalanced task tend to be
                    # the positive class, so a quiet clip biases the result.
                    truncated[0] = True
                    break
                if is_entity_row and col in suppressed:
                    continue
                sem = self._sem_for_cell(r.table, col, v)
                if sem is None:
                    continue
                seq.add(rnode, parents, col, r.table, sem, v)
        if truncated[0]:
            warnings.warn(
                f"context for {entity_table}={ctx.entity_id!r} was truncated at "
                f"max_seq_len={self.max_seq_len}; the tail of its history did "
                f"not reach the model", ContextTruncationWarning, stacklevel=2)
        return seq, node_of, entity_node, tgt_idx

    def _build_sequences(self, query: ParsedQuery, task_type: TaskType,
                         contexts: list[EntityContext], *,
                         target_sem: int = SEM_NUMBER
                         ) -> tuple[list[_Seq], float, float]:
        fk_to_parent = self._fk_to_parent()
        seqs: list[_Seq] = []
        all_labels: list[float] = []
        severed: set = set()
        for ctx in contexts:
            seq, node_of, _, _ = self._build_ctx_seq(
                query, task_type, ctx, fk_to_parent, all_labels,
                target_sem=target_sem)
            severed |= self._severed_parents(seq, node_of)
            seqs.append(seq)
        if severed:
            tables = ", ".join(sorted(repr(t) for t in severed))
            warnings.warn(
                f"context is disconnected: {tables} rows carry no feature "
                f"cells, so nothing linked through them can reach the "
                f"prediction and every entity will score alike. Declare a "
                f"feature column on those tables — or, when the primary key "
                f"itself carries meaning, declare it as a column too.",
                ContextConnectivityWarning, stacklevel=4)

        if all_labels:
            a = np.asarray(all_labels, float)
            label_mu, label_sd = float(a.mean()), float(a.std() + 1e-8)
        else:
            label_mu, label_sd = 0.0, 1.0
        self._normalize(seqs, label_mu, label_sd)
        return seqs, label_mu, label_sd

    def _normalize(self, seqs: list[_Seq], label_mu: float,
                   label_sd: float) -> None:
        """In-place: raw cell values -> normalized floats (F11/F12).

        Numbers and booleans z-score per (column, table) over all in-context
        values of the batch; datetimes share one global stat. The task label
        column uses the self-label stats so history tokens and the model
        output live in the same normalized space."""
        num_vals: dict[tuple[str, str], list[float]] = {}
        dt_vals: list[float] = []
        for seq in seqs:
            for (ck, sem, v, tgt) in zip(seq.col, seq.sem, seq.value,
                                         seq.is_tgt):
                if tgt or v is None:
                    continue
                if sem == SEM_DATETIME:
                    dt_vals.append(_days(v))
                elif sem in (SEM_NUMBER, SEM_BOOLEAN):
                    num_vals.setdefault(ck, []).append(
                        float(v) if not isinstance(v, bool)
                        else (1.0 if v else 0.0))
        # Fitted statistics when this backend has them, in-context otherwise.
        # This is a regime, not a per-column fallback: either the whole batch
        # normalizes against fitted state or none of it does, so a value never
        # depends on which other rows shared the call. See :class:`ColumnStats`
        # for why the released checkpoint uses the in-context regime.
        cs = self.column_stats
        stats: dict[tuple[str, str], tuple[float, float]] = {}
        for ck, vals in num_vals.items():
            if cs is not None and ck[1] != _TASK_TABLE and cs.has(ck[1], ck[0]):
                continue                      # handled by cs.transform below
            # The synthetic task row is not in the schema: its label shares the
            # self-label statistics so history tokens and the model's output
            # live in the same space. Without fitted stats this falls back to
            # the pre-conformance in-context behaviour.
            a = np.asarray(vals, float)
            stats[ck] = (float(a.mean()), float(a.std() + 1e-8))
        stats[(_TASK_LABEL_COL, _TASK_TABLE)] = (label_mu, label_sd)
        if dt_vals:
            a = np.asarray(dt_vals, float)
            dt_mu, dt_sd = float(a.mean()), float(a.std() + 1e-8)
        else:
            dt_mu, dt_sd = 0.0, 1.0
        for seq in seqs:
            for i, (ck, sem, v, tgt) in enumerate(
                    zip(seq.col, seq.sem, seq.value, seq.is_tgt)):
                if tgt or v is None:
                    seq.value[i] = 0.0
                    continue
                if sem == SEM_DATETIME:
                    seq.value[i] = (cs.transform_datetime(_days(v))
                                    if cs is not None
                                    else (_days(v) - dt_mu) / dt_sd)
                elif sem in (SEM_NUMBER, SEM_BOOLEAN):
                    x = float(v) if not isinstance(v, bool) \
                        else (1.0 if v else 0.0)
                    if cs is not None and ck[1] != _TASK_TABLE \
                            and cs.has(ck[1], ck[0]):
                        seq.value[i] = cs.transform(ck[1], ck[0], x)
                    else:
                        mu, sd = stats[ck]
                        seq.value[i] = (x - mu) / sd
                # text values stay as raw strings; embedded at collate

    def _forward(self, model: RtModel, seqs: list[_Seq], *,
                 want_text: bool = False, encode: bool = False):
        B = len(seqs)
        S = max(1, max(len(s) for s in seqs))
        col_vocab: dict[tuple[str, str], int] = {}
        tab_vocab: dict[str, int] = {}
        node_idxs = np.zeros((B, S), np.int64)
        f2p = np.full((B, S, MAX_F2P), -1, np.int64)
        col_idxs = np.zeros((B, S), np.int64)
        table_idxs = np.zeros((B, S), np.int64)
        is_padding = np.ones((B, S), np.uint8)
        sem_types = np.zeros((B, S), np.int64)
        is_target = np.zeros((B, S), np.uint8)
        number_v = np.zeros((B, S), np.float32)
        datetime_v = np.zeros((B, S), np.float32)
        boolean_v = np.zeros((B, S), np.float32)
        text_v = np.zeros((B, S, D_TEXT), np.float32)
        col_name_v = np.zeros((B, S, D_TEXT), np.float32)

        # Schema phrases and text cells embed in one batch, then are looked up
        # by key in the fill loop below. Going back through encode_one() per
        # cell cost a list allocation, a dict.fromkeys and a comprehension for
        # every one of the B*S tokens — no model calls, since the batch has
        # already warmed the cache, but the overhead is paid thousands of times
        # per forward and grew with the context size.
        distinct_cols = {ck for seq in seqs for ck in seq.col}
        texts = list({v for seq in seqs
                      for (sem, v) in zip(seq.sem, seq.value)
                      if sem == SEM_TEXT and isinstance(v, str)})
        phrases = [f"{c} of {t}" for (c, t) in distinct_cols]
        embedded = self.embedder.encode(phrases + texts)
        phrase_emb = dict(zip(distinct_cols, embedded[:len(phrases)]))
        text_emb = dict(zip(texts, embedded[len(phrases):]))

        for b, seq in enumerate(seqs):
            for s in range(len(seq)):
                ck, table, sem = seq.col[s], seq.tab[s], seq.sem[s]
                node_idxs[b, s] = seq.node[s]
                f2p[b, s] = seq.f2p[s]
                col_idxs[b, s] = col_vocab.setdefault(ck, len(col_vocab))
                table_idxs[b, s] = tab_vocab.setdefault(table, len(tab_vocab))
                is_padding[b, s] = 0
                is_target[b, s] = 1 if seq.is_tgt[s] else 0
                col_name_v[b, s] = phrase_emb[ck]
                v = seq.value[s]
                if sem == SEM_TEXT:
                    if isinstance(v, str):
                        text_v[b, s] = text_emb[v]
                    sem_types[b, s] = SEM_TEXT
                elif sem == SEM_DATETIME:
                    datetime_v[b, s] = float(v)
                    sem_types[b, s] = SEM_DATETIME
                else:  # number/boolean -> number channel (bool_as_num, F52)
                    number_v[b, s] = float(v)
                    sem_types[b, s] = SEM_NUMBER
        kw = dict(
            node_idxs=node_idxs, f2p=f2p, col_idxs=col_idxs,
            table_idxs=table_idxs, is_padding=is_padding,
            sem_types=sem_types, is_target=is_target, number_v=number_v,
            datetime_v=datetime_v, boolean_v=boolean_v, text_v=text_v,
            col_name_v=col_name_v, n_threads=self.n_threads)
        if encode:
            return model.encode_targets(device=self._resolve_device(),
                                        **kw)   # frozen features [B, 512]
        if want_text:
            return model.forward_ex(**kw)   # (scores[B], target_text[B, 384])
        return model.forward(device=self._resolve_device(), **kw)

    # -- multiclass / ranking domain enumeration ----------------------------
    def _require_wiring(self, what: str) -> RetrieverWiring:
        if self.wiring is None:
            raise RtNativeError(
                f"{what} requires a wiring with a TableScanner to enumerate "
                f"the domain; construct RtNativeBackend(schema=..., wiring=...)")
        return self.wiring

    @staticmethod
    def _batch_bound(contexts: list[EntityContext]) -> TemporalBound:
        """The query temporal bound reconstructed from the assembled contexts.
        Contexts of one execute share the anchor; take the max (most recent)
        so a shared scan stays 'nothing newer than the anchor' (F24)."""
        anchors = [c.anchor for c in contexts if c.anchor is not None]
        if not anchors:
            return TemporalBound.unbounded()
        return TemporalBound.at_or_before(max(anchors))

    def _target_column(self, query: ParsedQuery) -> ColumnRef:
        t = query.target
        if isinstance(t, ColumnRef):
            return t
        if isinstance(t, Aggregation):
            return t.column
        raise RtNativeError(
            "multiclass target must be a categorical column or "
            "FIRST/LAST(column)")

    def _class_domain(self, table: str, column: str,
                      bound: TemporalBound) -> list[str]:
        """Distinct non-null target-column values via the TableScanner, sorted
        lexicographically (UTF-8 byte order) and capped (§2.5)."""
        scanner = self._require_wiring("multiclass").scanner(table)
        seen: set[str] = set()
        for r in scanner(table, bound):
            v = r.cells.get(column)
            if v is not None:
                seen.add(str(v))
        labels = sorted(seen, key=lambda s: s.encode("utf-8"))
        return labels[:MAX_MULTICLASS_CLASSES]

    def _rank_candidates(self, parent_table: str,
                         bound: TemporalBound) -> list[Row]:
        """Distinct parent-table candidate *rows* via the TableScanner: deduped
        by id, sorted (numeric asc if integral else lexicographic UTF-8 asc),
        capped (§3.1). The full row is kept so its feature cells can be emitted
        into each candidate's context (§3.2) — an id alone gives the model
        nothing to tell candidates apart."""
        scanner = self._require_wiring("ranking").scanner(parent_table)
        rows_by_id: dict[Any, Row] = {}
        for r in scanner(parent_table, bound):
            rows_by_id.setdefault(r.id, r)
        ids = list(rows_by_id)
        if ids and all(isinstance(i, int) and not isinstance(i, bool)
                       for i in ids):
            ids.sort()
        else:
            ids.sort(key=lambda i: str(i).encode("utf-8"))
        return [rows_by_id[i] for i in ids[:MAX_RANK_CANDIDATES]]

    # -- multiclass (CONTRACT.md §2) ----------------------------------------
    def _score_multiclass(self, query: ParsedQuery,
                          contexts: list[EntityContext],
                          model_uri: str) -> list[EntityPrediction]:
        model = self._model_for(model_uri)
        col = self._target_column(query)
        head = self._head_for(TaskType.MULTICLASS_CLASSIFICATION)
        if head is not None:
            return self._score_multiclass_head(query, contexts, model, head)
        bound = self._batch_bound(contexts)
        labels = self._class_domain(col.table, col.column, bound)
        if not labels:
            raise RtNativeError(
                f"multiclass: target column {col} has no observed values "
                f"at or before the anchor to form a class domain")
        # L2-normalized MiniLM embeddings of the raw class strings (E[K, 384]).
        E = np.asarray(self.embedder.encode(labels, normalize=True),
                       np.float32)

        # masked-TEXT target cell -> text decoder head at each entity's target.
        seqs, _, _ = self._build_sequences(
            query, TaskType.MULTICLASS_CLASSIFICATION, contexts,
            target_sem=SEM_TEXT)
        _, pred_text = self._forward(model, seqs, want_text=True)

        preds: list[EntityPrediction] = []
        for ctx, pred in zip(contexts, pred_text):
            pred = np.asarray(pred, np.float32)
            pred = pred / (float(np.linalg.norm(pred)) + 1e-8)  # §2.3
            sims = E @ pred                                     # cosine (§2.6)
            k_best = int(np.argmax(sims))                       # ties: low idx
            logits = sims / T_SOFTMAX
            ex = np.exp(logits - logits.max())                 # log-sum-exp
            probs = ex / ex.sum()
            preds.append(EntityPrediction(
                ctx.entity_id,
                predicted_class=labels[k_best],
                class_probs={labels[i]: float(probs[i])
                             for i in range(len(labels))}))
        return preds

    def _score_multiclass_head(self, query: ParsedQuery,
                               contexts: list[EntityContext], model: RtModel,
                               head: "FineTunedHead") -> list[EntityPrediction]:
        """Multiclass through a trained head: logits over the class list the
        head was fitted on, rather than nearest-neighbour over MiniLM text."""
        labels = list(head.classes)
        if not labels:
            raise RtNativeError(
                "the fine-tuned multiclass head carries no class list; "
                "re-run Engine.finetune to regenerate it")
        seqs, _, _ = self._build_sequences(
            query, TaskType.MULTICLASS_CLASSIFICATION, contexts)
        logits = head.predict(self._encode(model, seqs))
        preds: list[EntityPrediction] = []
        for ctx, row in zip(contexts, logits):
            row = np.asarray(row, np.float32)
            ex = np.exp(row - row.max())
            probs = ex / ex.sum()
            preds.append(EntityPrediction(
                ctx.entity_id,
                predicted_class=labels[int(np.argmax(row))],
                class_probs={labels[i]: float(probs[i])
                             for i in range(len(labels))}))
        return preds

    # -- fine-tuning --------------------------------------------------------
    def fit_head(self, model: RtModel, task_type: TaskType,
                 features: np.ndarray, labels: np.ndarray,
                 group_offsets: np.ndarray, n_groups: int, *,
                 epochs: int = 100, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 classes: Sequence[Any] = ()) -> "FineTunedHead":
        """Fit a task head on frozen features ``[N, 512]``.

        Training runs on Metal; inference on the resulting head is plain CPU
        (``rt_finetune_head_predict``), so a head trained here serves anywhere.
        """
        lib = load_lib(self._lib_path)
        ft_task = _FT_TASK_OF[task_type]
        if not lib.device_available(RT_DEVICE_MPS):
            raise RtNativeError(
                "fine-tuning requires a Metal device (rt_finetune_head_fit_metal); "
                "this build or machine has none. Scoring is unaffected.")
        n_outputs = len(classes) if ft_task == FT_MULTICLASS else 1

        class_emb = None
        emb_ptr = None
        if ft_task == FT_MULTICLASS and classes:
            # Seed the head in the checkpoint's own class-embedding basis so it
            # starts from the zero-shot ordering rather than from nothing.
            class_emb = np.ascontiguousarray(
                self.embedder.encode([str(c) for c in classes], normalize=True),
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
        feats = (feats - feat_mu) / feat_sd

        f = np.ascontiguousarray(feats, np.float32).reshape(-1)
        y = np.ascontiguousarray(labels, np.float32)
        go = np.ascontiguousarray(group_offsets, np.int32)
        i_loss = ctypes.c_float(0.0)
        f_loss = ctypes.c_float(0.0)
        secs = ctypes.c_double(0.0)
        rc = lib._lib.rt_finetune_head_fit_metal(
            handle, int(y.shape[0]), f, y, go, int(n_groups), int(epochs),
            float(learning_rate), float(weight_decay),
            ctypes.byref(i_loss), ctypes.byref(f_loss), ctypes.byref(secs),
            err, len(err))
        if rc != 0:
            lib._lib.rt_finetune_head_free(handle)
            raise RtNativeError(
                f"rt_finetune_head_fit_metal failed: "
                f"{err.value.decode('utf-8', 'replace')}")
        return FineTunedHead(lib, handle, task=ft_task,
                             initial_loss=float(i_loss.value),
                             final_loss=float(f_loss.value),
                             seconds=float(secs.value),
                             n_examples=int(y.shape[0]), classes=classes,
                             feat_mu=feat_mu, feat_sd=feat_sd,
                             column_stats=self.column_stats)

    def ranking_parent_table(self, query: ParsedQuery) -> str:
        """The parent table a ranking query's FK target points at."""
        t = query.target
        if not isinstance(t, Aggregation):
            raise RtNativeError(
                "ranking target must be LIST_DISTINCT(table.fk) or "
                "ARRAY_AGG(table.fk)")
        link = None
        if self.schema is not None:
            link = next((l for l in self.schema.links_from(t.column.table)
                         if l.fk_column == t.column.column), None)
        if link is None:
            raise RtNativeError(
                f"ranking requires LIST_DISTINCT/ARRAY_AGG over a foreign-key "
                f"column: {t.column} is not an FK to a parent table")
        return link.to_table

    def candidate_seqs(self, query: ParsedQuery, ctx: EntityContext,
                       parent_table: str, candidates: list) -> list["_Seq"]:
        """One existence sequence per candidate parent row (§3.2).

        The candidate is attached as the masked target cell's parent; if it is
        not already a context row its feature cells are emitted as a fresh
        node, because an edge to an empty node scores identically for every
        candidate. Shared by scoring and fine-tuning so the head is trained on
        exactly the inputs it will later be served."""
        fk_to_parent = self._fk_to_parent()
        all_labels: list[float] = []
        base, node_of, entity_node, tgt_idx = self._build_ctx_seq(
            query, TaskType.MULTILABEL_RANKING, ctx, fk_to_parent,
            all_labels, target_sem=SEM_NUMBER)
        out: list[_Seq] = []
        for row in candidates:
            s = base.clone()
            cnode = node_of.get((parent_table, row.id))
            if cnode is None:
                cnode = len(node_of)              # fresh node for this candidate
                for col, v in row.cells.items():
                    sem = self._sem_for_cell(parent_table, col, v)
                    if sem is not None:
                        s.add(cnode, [], col, parent_table, sem, v)
            s.f2p[tgt_idx] = ([entity_node, cnode] + [-1] * MAX_F2P)[:MAX_F2P]
            out.append(s)
        self._normalize(out, 0.0, 1.0)
        return out

    # -- ranking (CONTRACT.md §3) -------------------------------------------
    def _score_ranking(self, query: ParsedQuery,
                       contexts: list[EntityContext],
                       model_uri: str) -> list[EntityPrediction]:
        t = query.target
        if not isinstance(t, Aggregation):
            raise RtNativeError(
                "ranking target must be LIST_DISTINCT(table.fk) or "
                "ARRAY_AGG(table.fk)")
        fk_ref = t.column
        link = None
        if self.schema is not None:
            link = next((l for l in self.schema.links_from(fk_ref.table)
                         if l.fk_column == fk_ref.column), None)
        if link is None:
            raise RtNativeError(
                f"ranking requires LIST_DISTINCT/ARRAY_AGG over a foreign-key "
                f"column: "
                f"{fk_ref} is not an FK to a parent table")
        parent_table = link.to_table
        k = query.top_k or 1
        bound = self._batch_bound(contexts)
        candidates = self._rank_candidates(parent_table, bound)
        if not candidates:
            raise RtNativeError(
                f"ranking: parent table {parent_table!r} has no candidate ids "
                f"at or before the anchor")

        model = self._model_for(model_uri)
        rank_head = self._head_for(TaskType.MULTILABEL_RANKING)
        preds: list[EntityPrediction] = []
        for ctx in contexts:
            cand_seqs = self.candidate_seqs(query, ctx, parent_table,
                                            candidates)
            if rank_head is not None:
                logits = rank_head.predict(self._encode(model, cand_seqs))[:, 0]
            else:
                logits = self._forward(model, cand_seqs)
            probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64)))
            order = sorted(range(len(candidates)),
                           key=lambda i: (-probs[i], i))   # ties: low cand idx
            ranked = tuple(str(candidates[i].id) for i in order[:k])
            preds.append(EntityPrediction(ctx.entity_id, ranked=ranked))
        return preds
