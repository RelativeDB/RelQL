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
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np

from .engine import EntityContext, EntityPrediction, ModelBackend
from .evaluate import eval_bool, eval_value
from .model import ModelConfig
from .pql.ast import Aggregation, ColumnRef, ParsedQuery, TaskType
from .retrieve import RetrieverWiring, TemporalBound
from .schema import Schema, ValueType

__all__ = ["RtNativeUnavailableError", "RtNativeError", "RtLib", "RtModel",
           "TextEmbedder", "RtNativeBackend", "load_lib", "resolve_model_path"]

D_TEXT = 384
MAX_F2P = 5
# SemType enum from cpp/src/rt.hpp
SEM_NUMBER, SEM_TEXT, SEM_DATETIME, SEM_BOOLEAN = 0, 1, 2, 3

# Shared contract constants — must be byte-for-byte identical across the
# Python / Rust / Java bindings (CONTRACT.md §5).
MAX_MULTICLASS_CLASSES = 1000   # cap on the multiclass label domain
MAX_RANK_CANDIDATES = 1000      # cap on the ranking parent-id candidate set
T_SOFTMAX = 0.1                 # multiclass class_probs softmax temperature

_SEM_OF_VALUE_TYPE = {
    ValueType.NUMBER: SEM_NUMBER,
    ValueType.TEXT: SEM_TEXT,
    ValueType.DATETIME: SEM_DATETIME,
    ValueType.BOOLEAN: SEM_BOOLEAN,
}


class RtNativeUnavailableError(RuntimeError):
    """librt_c (or a runtime dependency) could not be located/loaded."""


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

    def load_model(self, safetensors_path: str) -> "RtModel":
        err = ctypes.create_string_buffer(512)
        handle = self._lib.rt_model_load(
            safetensors_path.encode("utf-8"), err, len(err))
        if not handle:
            raise RtNativeError(
                f"rt_model_load({safetensors_path!r}) failed: "
                f"{err.value.decode('utf-8', 'replace')}")
        return RtModel(self, handle, safetensors_path)


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
                text_v, col_name_v, n_threads: int = 0) -> np.ndarray:
        """Raw PRE-sort arrays in (see rt_c.h) -> per-row target score [B]."""
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        out = np.zeros(B, np.float32)
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_forward(
            self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
            is_padding, sem_types, is_target, number_v, datetime_v,
            boolean_v, text_v, col_name_v, int(n_threads), out, err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"rt_forward failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return out

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

    * one token per feature cell ``(value, column, table)`` (F10); IDs/FKs are
      never tokens (F17) — they become the node graph instead;
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
    ``RETURN QUANTILES/INTERVAL`` still raise — a single point head exposes no
    empirical distribution.
    """

    def __init__(self, *, schema: Optional[Schema] = None,
                 wiring: Optional[RetrieverWiring] = None,
                 lib_path: Optional[str] = None,
                 embedder: Optional[TextEmbedder] = None,
                 n_threads: int = 0,
                 num_history_windows: int = 3,
                 max_seq_len: int = 1024):
        self.schema = schema
        self.wiring = wiring
        self._lib_path = lib_path
        self.embedder = embedder or TextEmbedder()
        self.n_threads = n_threads
        self.num_history_windows = max(1, num_history_windows)
        self.max_seq_len = max_seq_len
        self._models: dict[str, RtModel] = {}

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
        # A single point head exposes no empirical distribution — these RETURN
        # forms need a quantile/distribution head the checkpoint does not have.
        if ret_kind in ("QUANTILES", "INTERVAL"):
            raise RtNativeError(
                "RETURN QUANTILES/INTERVAL requires a quantile/distribution "
                "head the current checkpoint does not expose")
        if not contexts:
            return []
        if task_type is TaskType.MULTICLASS_CLASSIFICATION:
            return self._score_multiclass(query, contexts, model_uri)
        if task_type is TaskType.MULTILABEL_RANKING:
            return self._score_ranking(query, contexts, model_uri)
        model = self._model_for(model_uri)
        seqs, label_mu, label_sd = self._build_sequences(query, task_type,
                                                         contexts)
        scores = self._forward(model, seqs)
        preds: list[EntityPrediction] = []
        for ctx, s in zip(contexts, scores):
            s = float(s)
            if task_type is TaskType.BINARY_CLASSIFICATION:
                p = 1.0 / (1.0 + math.exp(-s))
                preds.append(self._shape_binary(ctx.entity_id, ret_kind, p))
            else:  # REGRESSION / FORECASTING: denormalize with label stats
                v = s * label_sd + label_mu
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

    def _fk_to_parent(self) -> dict[str, dict[str, str]]:
        if self.schema is None:
            return {}
        return {t.name: {l.fk_column: l.to_table
                         for l in self.schema.links_from(t.name)}
                for t in self.schema.tables}

    def _build_ctx_seq(self, query: ParsedQuery, task_type: TaskType,
                       ctx: EntityContext, fk_to_parent: dict, all_labels: list,
                       *, target_sem: int = SEM_NUMBER
                       ) -> tuple[_Seq, dict, int, int]:
        """Assemble one entity's context into a token sequence. Returns
        ``(seq, node_of, entity_node, tgt_idx)``. ``target_sem`` overrides the
        masked target cell's sem-type (SEM_TEXT for multiclass, §2.1); ``tgt_idx``
        is that cell's position (used by ranking to rewire its f2p, §3.2)."""
        entity_table = query.entity_key.table
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
            for col, v in r.cells.items():
                if len(seq) >= self.max_seq_len:
                    break
                sem = self._sem_for_cell(r.table, col, v)
                if sem is None:
                    continue
                seq.add(rnode, parents, col, r.table, sem, v)
        return seq, node_of, entity_node, tgt_idx

    def _build_sequences(self, query: ParsedQuery, task_type: TaskType,
                         contexts: list[EntityContext], *,
                         target_sem: int = SEM_NUMBER
                         ) -> tuple[list[_Seq], float, float]:
        fk_to_parent = self._fk_to_parent()
        seqs: list[_Seq] = []
        all_labels: list[float] = []
        for ctx in contexts:
            seq, _, _, _ = self._build_ctx_seq(
                query, task_type, ctx, fk_to_parent, all_labels,
                target_sem=target_sem)
            seqs.append(seq)

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
        stats: dict[tuple[str, str], tuple[float, float]] = {}
        for ck, vals in num_vals.items():
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
                    seq.value[i] = (_days(v) - dt_mu) / dt_sd
                elif sem in (SEM_NUMBER, SEM_BOOLEAN):
                    mu, sd = stats[ck]
                    x = float(v) if not isinstance(v, bool) \
                        else (1.0 if v else 0.0)
                    seq.value[i] = (x - mu) / sd
                # text values stay as raw strings; embedded at collate

    def _forward(self, model: RtModel, seqs: list[_Seq], *,
                 want_text: bool = False):
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

        # schema phrases + text cells embed in one cached batch
        phrases = [f"{c} of {t}" for seq in seqs for (c, t) in seq.col]
        texts = [v for seq in seqs
                 for (sem, v) in zip(seq.sem, seq.value)
                 if sem == SEM_TEXT and isinstance(v, str)]
        self.embedder.encode(list(dict.fromkeys(phrases + texts)))

        for b, seq in enumerate(seqs):
            for s in range(len(seq)):
                ck, table, sem = seq.col[s], seq.tab[s], seq.sem[s]
                node_idxs[b, s] = seq.node[s]
                f2p[b, s] = seq.f2p[s]
                col_idxs[b, s] = col_vocab.setdefault(ck, len(col_vocab))
                table_idxs[b, s] = tab_vocab.setdefault(table, len(tab_vocab))
                is_padding[b, s] = 0
                is_target[b, s] = 1 if seq.is_tgt[s] else 0
                col_name_v[b, s] = self.embedder.encode_one(
                    f"{ck[0]} of {ck[1]}")
                v = seq.value[s]
                if sem == SEM_TEXT:
                    if isinstance(v, str):
                        text_v[b, s] = self.embedder.encode_one(v)
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
        if want_text:
            return model.forward_ex(**kw)   # (scores[B], target_text[B, 384])
        return model.forward(**kw)

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

    # -- ranking (CONTRACT.md §3) -------------------------------------------
    def _score_ranking(self, query: ParsedQuery,
                       contexts: list[EntityContext],
                       model_uri: str) -> list[EntityPrediction]:
        t = query.target
        if not isinstance(t, Aggregation):
            raise RtNativeError(
                "ranking target must be LIST_DISTINCT(table.fk)")
        fk_ref = t.column
        link = None
        if self.schema is not None:
            link = next((l for l in self.schema.links_from(fk_ref.table)
                         if l.fk_column == fk_ref.column), None)
        if link is None:
            raise RtNativeError(
                f"ranking requires LIST_DISTINCT over a foreign-key column: "
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
        fk_to_parent = self._fk_to_parent()
        preds: list[EntityPrediction] = []
        for ctx in contexts:
            # one existence context per candidate parent row: attach the
            # candidate as the masked target cell's parent (§3.2). If the
            # candidate isn't already a context row, emit its feature cells as a
            # fresh node so the model can actually tell candidates apart — an
            # edge to an empty node scores identically for every candidate.
            all_labels: list[float] = []
            base, node_of, entity_node, tgt_idx = self._build_ctx_seq(
                query, TaskType.MULTILABEL_RANKING, ctx, fk_to_parent,
                all_labels, target_sem=SEM_NUMBER)
            cand_seqs: list[_Seq] = []
            for row in candidates:
                s = base.clone()
                cnode = node_of.get((parent_table, row.id))
                if cnode is None:
                    cnode = len(node_of)          # fresh node for this candidate
                    for col, v in row.cells.items():
                        sem = self._sem_for_cell(parent_table, col, v)
                        if sem is not None:
                            s.add(cnode, [], col, parent_table, sem, v)
                s.f2p[tgt_idx] = ([entity_node, cnode]
                                  + [-1] * MAX_F2P)[:MAX_F2P]
                cand_seqs.append(s)
            self._normalize(cand_seqs, 0.0, 1.0)
            logits = self._forward(model, cand_seqs)
            probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64)))
            order = sorted(range(len(candidates)),
                           key=lambda i: (-probs[i], i))   # ties: low cand idx
            ranked = tuple(str(candidates[i].id) for i in order[:k])
            preds.append(EntityPrediction(ctx.entity_id, ranked=ranked))
        return preds
