"""Model-agnostic scoring orchestration: contexts -> token sequences -> a
:class:`Scorer`.

Everything that decides WHAT the model sees lives here, in pure Python +
numpy: token sequence assembly (one token per feature cell, node graph,
FK-parent links, the synthetic task row and self-label history), per-entity
or reference normalization, task routing (binary / regression / forecasting /
multiclass / ranking), and the mapping from raw model outputs back to
:class:`~relativedb.engine.EntityPrediction`.

What does NOT live here is the model itself. A :class:`Scorer` supplies two
primitives — ``forward`` over a collated :class:`TokenBatch` and ``embed``
for text — and this module never imports a native library or an ML runtime:

* ``relativedb.rt.RelationalScorer`` (the torch-backed ``relativedb.rt``
  package) runs both in-process: librt_c for the transformer, the pinned
  MiniLM encoder for text.
* :class:`relativedb.remote.RemoteScorer` ships the same token batch to a
  scoring service (the C++ ``rt_serve`` backend or compatible), which embeds
  text and runs the forward on its side.

The token mapping mirrors rt/data.py (the reference): the arrays are RAW
PRE-SORT; the engine sorts and builds its own attention masks. Text cells
travel as raw strings — embedding them is the scorer's job, because the
embedding model belongs with the checkpoint, not with context creation.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from .engine import EntityContext, EntityPrediction
from .evaluate import eval_bool, eval_value
from relational_transformers_utils.normalize import ColumnStats as _ColumnStats
from relational_transformers_utils.normalize import NormalizationError
from relational_transformers_utils.normalize import bf16_as_f32  # noqa: F401 (re-export)
from relational_transformers_utils.normalize import normalize_sequence
from relational_transformers_utils.normalize import mean_std as _mean_std

from .model import ModelConfig, NormalizationMode
from .relql.ast import (AggFunc, Aggregation, Arith, Case, ColumnRef,
                        Condition, Func, LogicalOp, Not, ParsedQuery, TaskType)
from .retrieve import RetrieverWiring, Row, TemporalBound
from relational_transformers_utils.schema import Schema, ValueType
from .task import TaskSpec, TaskSpecFactory

__all__ = ["ScoringError", "ContextConnectivityWarning",
           "ContextTruncationWarning", "ColumnStats", "TokenBatch",
           "ForwardResult", "Scorer", "SequenceBackend",
           "D_TEXT", "D_MODEL", "MAX_F2P",
           "SEM_NUMBER", "SEM_TEXT", "SEM_DATETIME", "SEM_BOOLEAN",
           "MAX_MULTICLASS_CLASSES", "MAX_RANK_CANDIDATES", "T_SOFTMAX",
           "FT_BINARY", "FT_REGRESSION", "FT_MULTICLASS", "FT_RANKING",
           "bf16_as_f32"]

D_TEXT = 384                    # MiniLM-L12-v2 embedding width
D_MODEL = 512                   # frozen-backbone feature width (rt_c.h)
MAX_F2P = 5

# SemType enum from cpp/src/rt.hpp
SEM_NUMBER, SEM_TEXT, SEM_DATETIME, SEM_BOOLEAN = 0, 1, 2, 3

# Fine-tune task codes (rt_c.h). These are the wire values the C ABI expects,
# not the engine's TaskType. They live here so a trained head produced by the
# engine package can be routed without importing it.
FT_BINARY, FT_REGRESSION, FT_MULTICLASS, FT_RANKING = 0, 1, 2, 3

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

# Heads fitted by relational_transformers_utils.heads carry problem-type
# strings; serving matches the query's task type against them.
_HEAD_TASK_OF = {
    TaskType.BINARY_CLASSIFICATION: "binary",
    TaskType.REGRESSION: "regression",
    TaskType.FORECASTING: "regression",
    TaskType.MULTICLASS_CLASSIFICATION: "multiclass",
    TaskType.MULTILABEL_RANKING: "ranking",
}

_SEM_OF_VALUE_TYPE = {
    ValueType.NUMBER: SEM_NUMBER,
    ValueType.TEXT: SEM_TEXT,
    ValueType.DATETIME: SEM_DATETIME,
    ValueType.BOOLEAN: SEM_BOOLEAN,
}


class ScoringError(RuntimeError):
    """An error in sequence assembly or reported by the scorer."""


class ContextConnectivityWarning(UserWarning):
    """A context row that other rows hang off emits no tokens, so nothing
    below it can reach the prediction. Declare a feature column on that table
    — or declare its primary key as a column when the key carries meaning."""


class ContextTruncationWarning(UserWarning):
    """A context exceeded ``max_seq_len`` and lost cells. Raised because the
    cap bites hardest on the busiest entities, so silent truncation skews a
    comparison rather than merely shrinking it."""


class ColumnStats(_ColumnStats):
    """Per-column ``(mean, std)`` statistics, shared with
    relational-transformers-utils; this subclass adds the wiring-facing
    ``fit`` and reports missing task statistics as :class:`ScoringError`."""

    __slots__ = ()

    @classmethod
    def fit(cls, schema: Schema, wiring: RetrieverWiring,
            bound: Optional[TemporalBound] = None) -> "ColumnStats":
        bound = TemporalBound.unbounded() if bound is None else bound
        scalar = (ValueType.NUMBER, ValueType.BOOLEAN, ValueType.DATETIME)
        tables = {}
        for table in schema.tables:
            wanted = (any(c.type in scalar for c in table.columns)
                      or any(link.feature_type in scalar
                             for link in schema.links_from(table.name)))
            if not wanted:
                continue
            scanner = wiring.scanner(table.name)  # raises a precise error
            tables[table.name] = scanner(table.name, bound)
        base = _ColumnStats.fit(schema, tables, bound)
        return cls(base.stats, dt=base.dt, bound=base.bound,
                   task_stats=base.task_stats)

    def task(self, task) -> tuple[float, float]:
        try:
            return super().task(task)
        except NormalizationError as e:
            raise ScoringError(str(e)) from e


# ---------------------------------------------------------------------------
# collated token batch + the scorer protocol
# ---------------------------------------------------------------------------

@dataclass
class TokenBatch:
    """One collated forward's inputs, text still as raw strings.

    The integer/number channels are exactly the rt_c.h arrays (RAW PRE-SORT;
    ``number_v``/``datetime_v`` already bfloat16-rounded). The two 384-d text
    channels are NOT materialized here: ``col_phrases[col_idxs[b, s]]`` is the
    token's frozen ``"<column> of <table>"`` schema phrase, and
    ``text_idx[b, s]`` indexes ``texts`` (-1 = not a text cell). A scorer
    embeds the distinct strings once, scatters them into ``[B, S, 384]``
    buffers, applies :func:`bf16_as_f32`, and zero-fills ``boolean_v``
    (booleans route through the number channel; F52).
    """
    node_idxs: np.ndarray           # [B, S] int64
    f2p: np.ndarray                 # [B, S, MAX_F2P] int64, -1 padded
    col_idxs: np.ndarray            # [B, S] int64
    table_idxs: np.ndarray          # [B, S] int64
    is_padding: np.ndarray          # [B, S] uint8
    sem_types: np.ndarray           # [B, S] int64
    is_target: np.ndarray           # [B, S] uint8
    number_v: np.ndarray            # [B, S] float32 (bf16-rounded)
    datetime_v: np.ndarray          # [B, S] float32 (bf16-rounded)
    col_phrases: list[str] = field(default_factory=list)   # per col vocab id
    texts: list[str] = field(default_factory=list)          # distinct strings
    text_idx: Optional[np.ndarray] = None    # [B, S] int32, -1 = no text

    @property
    def b(self) -> int:
        return int(self.node_idxs.shape[0])

    @property
    def s(self) -> int:
        return int(self.node_idxs.shape[1])


@dataclass
class ForwardResult:
    """What a scorer hands back; which fields are set follows ``output``.

    * ``"target_scores"``: ``scores`` is [B] — the number-head output summed
      over each row's target cells.
    * ``"token_scores"``: ``scores`` is [B, S] in the caller's pre-sort order.
    * ``"target_scores_and_text"``: ``scores`` [B] plus ``target_text``
      [B, 384] — the text decoder head at the target cell, NOT L2-normalized.
    * ``"target_features"``: ``features`` is [B, 512] — the frozen backbone's
      output-normalized target-cell state (what task heads train on).
    """
    scores: Optional[np.ndarray] = None
    target_text: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None


class Scorer(Protocol):
    """The two model primitives scoring needs; everything else is Python.

    ``embed`` returns [N, 384] float32 MiniLM-L12-v2 embeddings (the pinned
    encoder; RT-J is frozen against it). ``normalize=True`` L2-normalizes —
    used for multiclass class labels; the default matches training for text
    CELL values. Implementations must be deterministic for equal inputs.
    """

    def forward(self, batch: TokenBatch, *, model_uri: str,
                output: str = "target_scores") -> ForwardResult: ...

    def embed(self, texts: Sequence[str], *,
              normalize: bool = False) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# context -> token sequence conversion
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
        if len(parents) > MAX_F2P:
            raise ScoringError(
                f"node {node} has {len(parents)} foreign-key parents; "
                f"RT supports at most {MAX_F2P}")
        self.node.append(node)
        self.f2p.append(list(parents) + [-1] * (MAX_F2P - len(parents)))
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


class SequenceBackend:
    """A :class:`~relativedb.engine.ModelBackend` over any :class:`Scorer`.

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
    * text cells and ``"<column> of <table>"`` schema phrases stay raw strings
      in the :class:`TokenBatch`; the scorer embeds them (F13/F14).

    Classification scores are logits -> sigmoid -> probability; regression
    scores are normalized -> denormalized with the in-context label stats.

    Multiclass classification masks the target cell as TEXT, reads the text
    decoder head (``output="target_scores_and_text"``), and nearest-neighbor-
    decodes it against L2-normalized MiniLM embeddings of the distinct
    target-column values (CONTRACT.md §2). Ranking scores each candidate
    parent id's existence context through the number head and takes the top-k
    (§3). Both need a ``wiring`` with a ``TableScanner`` to enumerate the
    class/candidate domain.

    ``head`` is an optional trained task head over the frozen backbone's
    target-cell features (``relativedb.rt.FineTunedHead``): any object
    with ``predict(features [N,512]) -> logits [N,K]``, ``task`` (FT_* code),
    ``classes`` and ``column_stats``/``normalization_mode`` attributes.
    """

    def __init__(self, scorer: Scorer, *, schema: Optional[Schema] = None,
                 wiring: Optional[RetrieverWiring] = None,
                 n_threads: int = 0,
                 num_history_windows: int = 3,
                 max_seq_len: int = 2048,      # reference eval uses 8192
                 column_stats: Optional["ColumnStats"] = None,
                 normalization_mode: Optional[NormalizationMode | str] = None,
                 task_spec_factory: Optional[TaskSpecFactory] = None,
                 head: Optional[Any] = None,
                 batch_size: Optional[int] = None):
        self.scorer = scorer
        self.schema = schema
        self.wiring = wiring
        self.n_threads = n_threads
        self.num_history_windows = max(1, num_history_windows)
        self.max_seq_len = max_seq_len
        self.normalization_mode = (None if normalization_mode is None else
                                   NormalizationMode.coerce(normalization_mode))
        self.task_spec_factory = task_spec_factory or TaskSpec.from_query
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")
        # Bound the physical forward without changing the query cohort or the
        # context length. Engine.execute() may resolve thousands of entities
        # at once; passing that whole cohort through one forward makes the
        # public product path unusable and can OOM the scorer even when a
        # small physical batch would fit comfortably.
        self.batch_size = batch_size
        self.column_stats = column_stats
        # A head fitted under fitted-statistics normalization must be served
        # the same way, so the head's own stats win over anything passed here.
        if head is not None and getattr(head, "column_stats", None) is not None:
            self.column_stats = head.column_stats
            if self.normalization_mode is None:
                self.normalization_mode = NormalizationMode.coerce(
                    head.normalization_mode)
        # A fine-tuned head replaces the checkpoint's zero-shot head for the
        # task it was trained on; every other task still scores zero-shot.
        self.head: Optional[Any] = head

    def _mode(self, config: Optional[ModelConfig] = None) -> NormalizationMode:
        return (self.normalization_mode
                or (config.normalization_mode if config is not None else
                    NormalizationMode.ZERO_SHOT))

    def task_spec(self, query: ParsedQuery, task_type: TaskType) -> TaskSpec:
        spec = self.task_spec_factory(query, task_type)
        if not isinstance(spec, TaskSpec):
            raise TypeError("task_spec_factory must return a TaskSpec")
        return spec

    def _head_for(self, task_type: TaskType) -> Optional[Any]:
        """The fine-tuned head, when it was trained for this task type."""
        if self.head is None:
            return None
        return (self.head if _HEAD_TASK_OF.get(task_type) == self.head.task
                else None)

    def _encode(self, seqs: list["_Seq"], model_uri: str) -> np.ndarray:
        """Frozen-backbone features ``[len(seqs), 512]`` for these sequences."""
        return self._forward_batched(seqs, model_uri, output="target_features")

    def score_shared(self, query: ParsedQuery, task_type: TaskType,
                     ctx: EntityContext,
                     targets: list[tuple[Any, tuple[str, Any]]],
                     model_uri: str,
                     config: ModelConfig) -> list[EntityPrediction]:
        """Score every cohort target inside one shared context.

        ``targets`` is ``[(entity_id, target_row_key), ...]`` with the first
        entry being the entity the context was assembled for. Each target's
        label cell is masked (flipped when the row carries one, appended when
        it does not), and one forward reads each prediction at its own token
        through the per-token number head. Peers' labels are therefore never
        visible to each other — strictly less outcome exposure than scoring
        the same cohort one entity at a time.
        """
        payload = self.prepare_shared(query, task_type, ctx, targets,
                                      model_uri, config)
        yhat = self.scorer.forward(payload["batch"],
                                   model_uri=payload["model_uri"],
                                   output="token_scores").scores[0]
        return self.finish_shared(payload, yhat)

    def prepare_shared(self, query: ParsedQuery, task_type: TaskType,
                       ctx: EntityContext,
                       targets: list[tuple[Any, tuple[str, Any]]],
                       model_uri: str, config: ModelConfig) -> dict:
        """The assembly half of shared-context scoring: sequence assembly,
        target masking, normalization, and collation. The returned payload's
        ``batch``/``model_uri`` feed a ``token_scores`` forward;
        :meth:`finish_shared` shapes that forward's output. Split so a caller
        can overlap the next cohort's preparation with the current forward.
        """
        if task_type not in (TaskType.BINARY_CLASSIFICATION,
                             TaskType.REGRESSION):
            raise ScoringError(
                "shared-context scoring supports binary classification and "
                "regression targets")
        ret = query.ret
        ret_kind = ret.kind if ret is not None else None
        mode = self._mode(config)
        task_spec = self.task_spec(query, task_type)
        fk_to_parent = self._fk_to_parent()
        labels: list[float] = []
        seq, node_of, _, tgt_idx = self._build_ctx_seq(
            query, task_type, ctx, fk_to_parent, labels, task_spec=task_spec)
        stats = self._label_stats(seq, labels, task_spec, mode)
        self._normalize_one(seq, task_spec, stats, mode)

        tcol = (task_spec.target_column, task_spec.table_name)
        node_to_entity = {}
        for eid, key in targets:
            node = node_of.get(key)
            if node is None:
                raise ScoringError(
                    f"shared-context target row {key!r} is not part of the "
                    f"assembled context")
            node_to_entity[node] = eid
        # Mask existing label cells; remember which target nodes have one.
        seen_nodes = set()
        first_cell_of_node: dict[int, int] = {}
        for s in range(len(seq)):
            n = seq.node[s]
            if n in node_to_entity and n not in first_cell_of_node:
                first_cell_of_node[n] = s
            if n in node_to_entity and seq.col[s] == tcol:
                seq.is_tgt[s] = True
                seq.value[s] = 0.0
                seen_nodes.add(n)
        # Targets whose row carries no label cell (synthesized target rows)
        # get a fresh masked cell right after the focal target, with the f2p
        # of a sibling cell of the same row (captured before inserting, since
        # inserts shift positions).
        pending = [(n, eid) for n, eid in node_to_entity.items()
                   if n not in seen_nodes]
        for n, eid in pending:
            if first_cell_of_node.get(n) is None:
                raise ScoringError(
                    f"shared-context target for entity {eid!r} emitted no "
                    f"cells; its row was truncated out of the context")
        pending_f2p = {n: list(seq.f2p[first_cell_of_node[n]])
                       for n, _ in pending}
        for n, eid in pending:
            at = tgt_idx + 1
            seq.node.insert(at, n)
            seq.f2p.insert(at, pending_f2p[n])
            seq.col.insert(at, tcol)
            seq.tab.insert(at, task_spec.table_name)
            seq.sem.insert(at, SEM_NUMBER)
            seq.is_tgt.insert(at, True)
            seq.value.insert(at, 0.0)

        batch = self._collate([seq])
        return {
            "batch": batch, "model_uri": model_uri, "seq": seq,
            "node_to_entity": node_to_entity, "stats": stats,
            "targets": targets, "task_type": task_type,
            "ret_kind": ret_kind,
        }

    def finish_shared(self, payload: dict,
                      yhat: np.ndarray) -> list[EntityPrediction]:
        seq = payload["seq"]
        node_to_entity = payload["node_to_entity"]
        task_type = payload["task_type"]
        stats = payload["stats"]
        ret_kind = payload["ret_kind"]
        preds: list[EntityPrediction] = []
        emitted = set()
        for s in range(len(seq)):
            if not seq.is_tgt[s]:
                continue
            eid = node_to_entity.get(seq.node[s])
            if eid is None or eid in emitted:
                continue
            emitted.add(eid)
            v = float(yhat[s])
            if task_type is TaskType.BINARY_CLASSIFICATION:
                p = 1.0 / (1.0 + math.exp(-v))
                preds.append(self._shape_binary(eid, ret_kind, p))
            else:
                preds.append(EntityPrediction(eid, value=v * stats[1] + stats[0]))
        missing = [eid for eid, _ in payload["targets"] if eid not in emitted]
        if missing:
            raise ScoringError(
                f"shared-context scoring produced no prediction for "
                f"{missing[:3]!r}")
        return preds

    # Physical forward bound when the caller sets none. Text now travels as
    # strings, but the scorer still materializes two [B, S, 384] float32
    # channels — ~6 MB per sequence at S=2048 — so an unbounded cohort is an
    # unbounded allocation on its side: 4201 entities through one collate
    # reached ~40 GB and got the serving process OOM-killed. 64 caps a chunk
    # near 400 MB; per-sequence normalization makes chunking bit-identical to
    # one call.
    DEFAULT_PHYSICAL_BATCH = 64

    def _forward_batched(self, seqs: list["_Seq"], model_uri: str, *,
                         output: str = "target_scores"):
        """Run bounded physical forwards while preserving logical ordering."""
        bs = self.batch_size or self.DEFAULT_PHYSICAL_BATCH
        if not seqs or len(seqs) <= bs:
            return self._forward(seqs, model_uri, output=output)
        chunks = [self._forward(seqs[i:i + bs], model_uri, output=output)
                  for i in range(0, len(seqs), bs)]
        if output == "target_scores_and_text":
            scores = np.concatenate([c[0] for c in chunks], axis=0)
            text = np.concatenate([c[1] for c in chunks], axis=0)
            return scores, text
        return np.concatenate(chunks, axis=0)

    # -- ModelBackend -------------------------------------------------------
    def score(self, query: ParsedQuery, task_type: TaskType,
              contexts: list[EntityContext], model_uri: str,
              config: ModelConfig) -> list[EntityPrediction]:
        ret = query.ret
        ret_kind = ret.kind if ret is not None else None
        if not contexts:
            return []
        mode = self._mode(config)
        if task_type is TaskType.MULTICLASS_CLASSIFICATION:
            return self._by_anchor(
                contexts,
                lambda cs: self._score_multiclass(query, cs, model_uri, mode))
        if task_type is TaskType.MULTILABEL_RANKING:
            return self._by_anchor(
                contexts,
                lambda cs: self._score_ranking(query, cs, model_uri, mode))
        head = self._head_for(task_type)
        if task_type is TaskType.FORECASTING and (query.num_forecasts or 1) > 1:
            return self._score_forecast(query, task_type, contexts, model_uri,
                                        mode, head)
        seqs, label_mu, label_sd = self._build_sequences(
            query, task_type, contexts, normalization_mode=mode)
        if head is not None:
            # trained head over the frozen backbone's target-cell features
            scores = head.predict(self._encode(seqs, model_uri))[:, 0]
        else:
            scores = self._forward_batched(seqs, model_uri)
        preds: list[EntityPrediction] = []
        for ctx, s, mu, sd in zip(contexts, scores, label_mu, label_sd):
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
                v = s if head is not None else s * sd + mu
                if task_type is TaskType.FORECASTING:
                    preds.append(EntityPrediction(ctx.entity_id, value=v,
                                                  forecast=(v,)))
                else:
                    preds.append(EntityPrediction(ctx.entity_id, value=v))
        return preds

    @staticmethod
    def _by_anchor(contexts: list[EntityContext], run):
        """Run a scorer once per distinct context anchor.

        Class domains and ranking candidates are scanned at a batch-wide
        bound; under per-entity anchors that bound used to be max(anchors),
        so an early-anchored entity's candidate rows included other
        entities' futures. Grouping keeps every scan at its own anchor."""
        anchors = {c.anchor for c in contexts}
        if len(anchors) <= 1:
            return run(contexts)
        by_index: dict[int, EntityPrediction] = {}
        for anchor in sorted(anchors, key=lambda a: (a is not None, a)):
            picked = [(i, c) for i, c in enumerate(contexts)
                      if c.anchor == anchor]
            for (i, _), pred in zip(picked, run([c for _, c in picked])):
                by_index[i] = pred
        return [by_index[i] for i in range(len(contexts))]

    def _score_forecast(self, query: ParsedQuery, task_type: TaskType,
                        contexts: list[EntityContext], model_uri: str, mode,
                        head) -> list[EntityPrediction]:
        """HORIZONS N [STEP d]: one forward per horizon.

        Horizon k re-asks the same masked question with the target token
        stamped at ``anchor + k*step`` — the model's task row encodes *when*
        the window starts, so moving its timestamp is the native way to ask
        about a later window. Context rows and self-label history stay at
        the base anchor: the evidence is factual, only the question moves.
        (Previously HORIZONS N returned one prediction copied N times.)"""
        n = query.num_forecasts or 1
        window = next((a.window for a in query.target_aggregations
                       if a.window is not None), None)
        if window is None:
            raise ScoringError(
                "HORIZONS > 1 needs a windowed aggregation target")
        step = (window.unit.delta(window.step) if window.step is not None
                else window.span())
        if step is None:
            raise ScoringError(
                "HORIZONS > 1 over an unbounded window needs an explicit "
                "STEP; there is no frame width to stride by")
        per_ctx: list[list[float]] = [[] for _ in contexts]
        for k in range(n):
            seqs, label_mu, label_sd = self._build_sequences(
                query, task_type, contexts, normalization_mode=mode,
                target_time_shift=step * k if k else None)
            if head is not None:
                scores = head.predict(self._encode(seqs, model_uri))[:, 0]
            else:
                scores = self._forward_batched(seqs, model_uri)
            for i, (s, mu, sd) in enumerate(zip(scores, label_mu, label_sd)):
                per_ctx[i].append(float(s) if head is not None
                                  else float(s) * sd + mu)
        return [EntityPrediction(ctx.entity_id, value=vals[0],
                                 forecast=tuple(vals))
                for ctx, vals in zip(contexts, per_ctx)]

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
        rows_by_table = ctx.focal_rows_by_table()
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
                       ctx: EntityContext, fk_to_parent: dict, labels: list,
                       *, target_sem: int = SEM_NUMBER,
                       task_spec: Optional[TaskSpec] = None,
                       target_time_shift: Optional[timedelta] = None,
                       ) -> tuple[_Seq, dict, int, int]:
        """Assemble one entity's context into a token sequence. Returns
        ``(seq, node_of, entity_node, tgt_idx)``. ``target_sem`` overrides the
        masked target cell's sem-type (SEM_TEXT for multiclass, §2.1); ``tgt_idx``
        is that cell's position (used by ranking to rewire its f2p, §3.2)."""
        entity_table = query.entity_key.table
        task_spec = task_spec or self.task_spec(query, task_type)
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
        node_of: dict[tuple[str, Any], int] = dict(ctx.node_ids)
        next_node = [max(node_of.values(), default=-1) + 1]

        def node(key: tuple[str, Any]) -> int:
            if key not in node_of:
                node_of[key] = next_node[0]
                next_node[0] += 1
            return node_of[key]

        # rows first claim node ids so f2p links resolve in any order
        for r in ctx.rows:
            node(r.key)
        by_id: dict[Any, list[tuple[str, Any]]] = {}
        for r in ctx.rows:
            by_id.setdefault(r.id, []).append(r.key)

        entity_node = node((entity_table, ctx.entity_id))

        def row_parents(r) -> list[int]:
            parents: list[int] = []
            for fk, pid in r.parents.items():
                if (fk == "__entity__" and r.table == task_spec.table_name):
                    parents.append(node((entity_table, pid)))
                    continue
                ptable = fk_to_parent.get(r.table, {}).get(fk)
                if ptable is not None:
                    for one in (pid if isinstance(pid, (list, tuple)) else (pid,)):
                        pkey = (ptable, one)
                        if pkey in node_of:
                            parents.append(node_of[pkey])
                    continue
                cands = by_id.get(pid, [])
                if len(cands) == 1:
                    parents.append(node_of[cands[0]])
            return parents

        # -- the target task row (masked label) --
        if task_spec.direct_target:
            tgt_node = entity_node
            focal = next((r for r in ctx.rows
                          if r.key == (entity_table, ctx.entity_id)), None)
            tgt_parents = row_parents(focal) if focal is not None else []
        else:
            focal_task = next((r for r in ctx.rows
                               if r.table == task_spec.table_name
                               and r.key in ctx.focal_row_keys
                               and task_spec.target_column not in r.cells), None)
            tgt_node = node(focal_task.key if focal_task is not None else
                            (task_spec.table_name,
                             f"__target__:{task_spec.id}"))
            tgt_parents = (row_parents(focal_task) if focal_task is not None
                           else [entity_node])
        tgt_idx = len(seq)
        seq.add(tgt_node, tgt_parents, task_spec.target_column,
                task_spec.table_name, target_sem, None, target=True)
        # The reference contract is target cell first, including for a derived
        # task row; its timestamp follows the masked target token. A horizon
        # shift moves ONLY this token: the question is asked about a later
        # window while the evidence stays at the base anchor.
        tgt_time = (ctx.anchor + target_time_shift
                    if target_time_shift is not None and ctx.anchor is not None
                    else ctx.anchor)
        if (not task_spec.direct_target and tgt_time is not None
                and (target_time_shift is not None or focal_task is None
                     or task_spec.time_column not in focal_task.cells)):
            seq.add(tgt_node, tgt_parents, task_spec.time_column,
                    task_spec.table_name, SEM_DATETIME, tgt_time)

        # -- past outcomes of the same task (self labels, F65) --
        materialized_labels = [
            r for r in ctx.rows
            if r.table == task_spec.table_name
            and task_spec.target_column in r.cells
        ]
        if materialized_labels:
            labels.extend(r.cells[task_spec.target_column]
                          for r in materialized_labels)
        else:
            for ts, label in self._self_labels(query, task_type, ctx):
                hnode = node((task_spec.table_name, (task_spec.id, ts)))
                seq.add(hnode, [entity_node], task_spec.target_column,
                        task_spec.table_name, SEM_NUMBER, label)
                seq.add(hnode, [entity_node], task_spec.time_column,
                        task_spec.table_name, SEM_DATETIME, ts)
                labels.append(label)

        # -- one token per feature cell of every context row --
        # Under a horizon shift the focal task row's own time cell is
        # suppressed: the shifted token above already answers "when", and
        # emitting both timestamps would give the model two contradictory
        # anchors for one masked question.
        shift_key = (focal_task.key
                     if (target_time_shift is not None
                         and not task_spec.direct_target
                         and focal_task is not None) else None)
        for r in ctx.rows:
            parents = row_parents(r)
            rnode = node_of[r.key]
            is_entity_row = r.key == (entity_table, ctx.entity_id)
            for col, v in r.cells.items():
                if r.key == shift_key and col == task_spec.time_column:
                    continue
                if len(seq) >= self.max_seq_len:
                    # Truncation is never silent: it falls hardest on the
                    # busiest entities, which in an imbalanced task tend to be
                    # the positive class, so a quiet clip biases the result.
                    truncated[0] = True
                    break
                if is_entity_row and col in suppressed:
                    continue
                if self.schema is not None:
                    tdef = self.schema.table(r.table)
                    if (tdef is not None and (col == tdef.primary_key
                                               or tdef.column(col) is None)):
                        continue
                sem = self._sem_for_cell(r.table, col, v)
                if sem is None:
                    continue
                seq.add(rnode, parents, col, r.table, sem, v)
            if (r.table == task_spec.table_name and r.timestamp is not None
                    and task_spec.time_column not in r.cells
                    and rnode != tgt_node):
                seq.add(rnode, parents, task_spec.time_column,
                        task_spec.table_name, SEM_DATETIME, r.timestamp)
            if self.schema is not None:
                for link in self.schema.links_from(r.table):
                    if link.feature_type is None:
                        continue
                    value = r.parents.get(link.fk_column)
                    if value is None:
                        continue
                    if isinstance(value, (list, tuple)):
                        if link.feature_type is not ValueType.TEXT:
                            raise ScoringError(
                                f"list-valued FK feature {r.table}."
                                f"{link.fk_column} must use ValueType.TEXT")
                        value = json.dumps(list(value), separators=(",", ":"),
                                           ensure_ascii=True)
                    seq.add(rnode, parents, link.fk_column, r.table,
                            _SEM_OF_VALUE_TYPE[link.feature_type], value)
        if truncated[0]:
            warnings.warn(
                f"context for {entity_table}={ctx.entity_id!r} was truncated at "
                f"max_seq_len={self.max_seq_len}; the tail of its history did "
                f"not reach the model", ContextTruncationWarning, stacklevel=2)
        return seq, node_of, entity_node, tgt_idx

    def _build_sequences(self, query: ParsedQuery, task_type: TaskType,
                         contexts: list[EntityContext], *,
                         target_sem: int = SEM_NUMBER,
                         normalization_mode: Optional[NormalizationMode | str] = None,
                         task_spec: Optional[TaskSpec] = None,
                         target_time_shift: Optional[timedelta] = None,
                         ) -> tuple[list[_Seq], list[float], list[float]]:
        fk_to_parent = self._fk_to_parent()
        task_spec = task_spec or self.task_spec(query, task_type)
        mode = (self._mode() if normalization_mode is None else
                NormalizationMode.coerce(normalization_mode))
        seqs: list[_Seq] = []
        labels_by_seq: list[list[float]] = []
        severed: set = set()
        for ctx in contexts:
            labels: list[float] = []
            seq, node_of, _, _ = self._build_ctx_seq(
                query, task_type, ctx, fk_to_parent, labels,
                target_sem=target_sem, task_spec=task_spec,
                target_time_shift=target_time_shift)
            # A node can be token-less because the configured sequence budget
            # clipped its cells, not because its schema has no features. Do
            # not misdiagnose an intentional context cap as a connectivity
            # error; the explicit truncation warning above is the right one.
            if len(seq) < self.max_seq_len:
                severed |= self._severed_parents(seq, node_of)
            seqs.append(seq)
            labels_by_seq.append(labels)
        if severed:
            tables = ", ".join(sorted(repr(t) for t in severed))
            warnings.warn(
                f"context is disconnected: {tables} rows carry no feature "
                f"cells, so nothing linked through them can reach the "
                f"prediction through those rows. Declare a "
                f"feature column on those tables — or, when the primary key "
                f"itself carries meaning, declare it as a column too.",
                ContextConnectivityWarning, stacklevel=4)

        label_stats = [self._label_stats(seq, labels, task_spec, mode)
                       for seq, labels in zip(seqs, labels_by_seq)]
        for seq, stats in zip(seqs, label_stats):
            self._normalize_one(seq, task_spec, stats, mode)
        return (seqs, [x[0] for x in label_stats],
                [x[1] for x in label_stats])

    def _label_stats(self, seq: _Seq, labels: list[float],
                     task_spec: TaskSpec,
                     mode: NormalizationMode) -> tuple[float, float]:
        target_sem = next((sem for sem, tgt in zip(seq.sem, seq.is_tgt) if tgt),
                          SEM_NUMBER)
        if target_sem not in (SEM_NUMBER, SEM_BOOLEAN):
            return (0.0, 1.0)
        if mode is NormalizationMode.REFERENCE:
            if self.column_stats is None:
                raise ScoringError(
                    "reference normalization requires ColumnStats; fit with "
                    "ColumnStats.fit(...) and attach it to the backend")
            if (task_spec.direct_target
                    and self.column_stats.has(task_spec.table_name,
                                              task_spec.target_column)):
                return self.column_stats.stats[(task_spec.table_name,
                                                task_spec.target_column)]
            return self.column_stats.task(task_spec)
        if task_spec.direct_target:
            key = (task_spec.target_column, task_spec.table_name)
            values = [float(v) if not isinstance(v, bool) else float(v)
                      for ck, sem, v, tgt in zip(
                          seq.col, seq.sem, seq.value, seq.is_tgt)
                      if ck == key and not tgt and v is not None
                      and sem in (SEM_NUMBER, SEM_BOOLEAN)]
            return _mean_std(values)
        return _mean_std(labels)

    def _normalize_one(self, seq: _Seq, task_spec: TaskSpec,
                       label_stats: tuple[float, float],
                       mode: NormalizationMode) -> None:
        """Normalize one entity independently or from persisted reference stats.

        The arithmetic lives in
        :func:`relational_transformers_utils.normalize.normalize_sequence`;
        this wrapper flips the sequence's ``(column, table)`` keys into the
        shared ``(table, column)`` order and leaves text cells as raw
        strings for the scorer to embed.
        """
        keys = [(ck[1], ck[0]) for ck in seq.col]
        try:
            normalized = normalize_sequence(
                keys, seq.sem, seq.value, seq.is_tgt, mode=mode,
                column_stats=self.column_stats, label_stats=label_stats,
                target_key=(task_spec.table_name, task_spec.target_column))
        except NormalizationError as e:
            raise ScoringError(str(e)) from e
        for i, (sem, v, tgt) in enumerate(zip(seq.sem, seq.value, seq.is_tgt)):
            if tgt or v is None or sem in (SEM_NUMBER, SEM_BOOLEAN,
                                           SEM_DATETIME):
                seq.value[i] = normalized[i]
            # text values stay as raw strings; embedded by the scorer

    def _collate(self, seqs: list[_Seq]) -> TokenBatch:
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
        text_idx = np.full((B, S), -1, np.int32)
        text_vocab: dict[str, int] = {}

        for b, seq in enumerate(seqs):
            for s in range(len(seq)):
                ck, table, sem = seq.col[s], seq.tab[s], seq.sem[s]
                node_idxs[b, s] = seq.node[s]
                f2p[b, s] = seq.f2p[s]
                col_idxs[b, s] = col_vocab.setdefault(ck, len(col_vocab))
                table_idxs[b, s] = tab_vocab.setdefault(table, len(tab_vocab))
                is_padding[b, s] = 0
                is_target[b, s] = 1 if seq.is_tgt[s] else 0
                v = seq.value[s]
                if sem == SEM_TEXT:
                    if isinstance(v, str):
                        text_idx[b, s] = text_vocab.setdefault(
                            v, len(text_vocab))
                    sem_types[b, s] = SEM_TEXT
                elif sem == SEM_DATETIME:
                    datetime_v[b, s] = float(v)
                    sem_types[b, s] = SEM_DATETIME
                else:  # number/boolean -> number channel (bool_as_num, F52)
                    number_v[b, s] = float(v)
                    sem_types[b, s] = SEM_NUMBER
        return TokenBatch(
            node_idxs=node_idxs, f2p=f2p, col_idxs=col_idxs,
            table_idxs=table_idxs, is_padding=is_padding,
            sem_types=sem_types, is_target=is_target,
            number_v=bf16_as_f32(number_v),
            datetime_v=bf16_as_f32(datetime_v),
            col_phrases=[f"{c} of {t}" for (c, t) in col_vocab],
            texts=list(text_vocab),
            text_idx=text_idx)

    def _forward(self, seqs: list[_Seq], model_uri: str, *,
                 output: str = "target_scores"):
        batch = self._collate(seqs)
        result = self.scorer.forward(batch, model_uri=model_uri, output=output)
        if output == "target_features":
            return result.features
        if output == "target_scores_and_text":
            return result.scores, result.target_text
        return result.scores

    # -- multiclass / ranking domain enumeration ----------------------------
    def _require_wiring(self, what: str) -> RetrieverWiring:
        if self.wiring is None:
            raise ScoringError(
                f"{what} requires a wiring with a TableScanner to enumerate "
                f"the domain; construct the backend with schema=..., wiring=...")
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
        raise ScoringError(
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
        if len(labels) > MAX_MULTICLASS_CLASSES:
            warnings.warn(
                f"multiclass domain for {table}.{column} has {len(labels)} "
                f"distinct values; only the first {MAX_MULTICLASS_CLASSES} "
                f"(byte-sorted) can ever be predicted — the rest are "
                f"unreachable classes", stacklevel=3)
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
        if len(ids) > MAX_RANK_CANDIDATES:
            warnings.warn(
                f"ranking candidate set for {parent_table!r} has {len(ids)} "
                f"rows; only the first {MAX_RANK_CANDIDATES} (sorted) can be "
                f"ranked — candidates past the cap can never appear in "
                f"results", stacklevel=3)
        return [rows_by_id[i] for i in ids[:MAX_RANK_CANDIDATES]]

    # -- multiclass (CONTRACT.md §2) ----------------------------------------
    def _score_multiclass(self, query: ParsedQuery,
                          contexts: list[EntityContext],
                          model_uri: str,
                          mode: NormalizationMode) -> list[EntityPrediction]:
        col = self._target_column(query)
        decorative = next((a for a in query.target_aggregations
                           if a.window is not None or a.filter is not None
                           or a.func not in (AggFunc.FIRST, AggFunc.LAST)),
                          None)
        if decorative is not None:
            warnings.warn(
                "multiclass scoring reduces the target to its raw column: "
                "the aggregation's function, window, and inline filter do "
                "not affect the class domain or the prediction",
                stacklevel=3)
        head = self._head_for(TaskType.MULTICLASS_CLASSIFICATION)
        if head is not None:
            return self._score_multiclass_head(query, contexts, model_uri,
                                               head, mode)
        bound = self._batch_bound(contexts)
        labels = self._class_domain(col.table, col.column, bound)
        if not labels:
            raise ScoringError(
                f"multiclass: target column {col} has no observed values "
                f"at or before the anchor to form a class domain")
        # L2-normalized MiniLM embeddings of the raw class strings (E[K, 384]).
        # The scorer computes them: the embedding model lives with the model.
        E = np.asarray(self.scorer.embed(labels, normalize=True), np.float32)

        # masked-TEXT target cell -> text decoder head at each entity's target.
        seqs, _, _ = self._build_sequences(
            query, TaskType.MULTICLASS_CLASSIFICATION, contexts,
            target_sem=SEM_TEXT, normalization_mode=mode)
        _, pred_text = self._forward_batched(
            seqs, model_uri, output="target_scores_and_text")

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
                               contexts: list[EntityContext], model_uri: str,
                               head: Any,
                               mode: NormalizationMode) -> list[EntityPrediction]:
        """Multiclass through a trained head: logits over the class list the
        head was fitted on, rather than nearest-neighbour over MiniLM text."""
        labels = list(head.classes)
        if not labels:
            raise ScoringError(
                "the fine-tuned multiclass head carries no class list; "
                "re-run Engine.fit_head to regenerate it")
        seqs, _, _ = self._build_sequences(
            query, TaskType.MULTICLASS_CLASSIFICATION, contexts,
            normalization_mode=mode)
        logits = head.predict(self._encode(seqs, model_uri))
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

    def ranking_parent_table(self, query: ParsedQuery) -> str:
        """The parent table a ranking query's FK target points at."""
        t = query.target
        if not isinstance(t, Aggregation):
            raise ScoringError(
                "ranking target must be LIST_DISTINCT(table.fk) or "
                "ARRAY_AGG(table.fk)")
        link = None
        if self.schema is not None:
            link = next((l for l in self.schema.links_from(t.column.table)
                         if l.fk_column == t.column.column), None)
        if link is None:
            raise ScoringError(
                f"ranking requires LIST_DISTINCT/ARRAY_AGG over a foreign-key "
                f"column: {t.column} is not an FK to a parent table")
        return link.to_table

    def candidate_seqs(self, query: ParsedQuery, ctx: EntityContext,
                       parent_table: str, candidates: list, *,
                       normalization_mode: Optional[NormalizationMode | str] = None,
                       ) -> list["_Seq"]:
        """One existence sequence per candidate parent row (§3.2).

        The candidate is attached as the masked target cell's parent; if it is
        not already a context row its feature cells are emitted as a fresh
        node, because an edge to an empty node scores identically for every
        candidate. Shared by scoring and fine-tuning so the head is trained on
        exactly the inputs it will later be served."""
        fk_to_parent = self._fk_to_parent()
        all_labels: list[float] = []
        task_spec = self.task_spec(query, TaskType.MULTILABEL_RANKING)
        mode = (self._mode() if normalization_mode is None else
                NormalizationMode.coerce(normalization_mode))
        base, node_of, entity_node, tgt_idx = self._build_ctx_seq(
            query, TaskType.MULTILABEL_RANKING, ctx, fk_to_parent,
            all_labels, target_sem=SEM_NUMBER, task_spec=task_spec)
        out: list[_Seq] = []
        for row in candidates:
            s = base.clone()
            cnode = node_of.get((parent_table, row.id))
            if cnode is None:
                cnode = max(node_of.values(), default=-1) + 1
                for col, v in row.cells.items():
                    sem = self._sem_for_cell(parent_table, col, v)
                    if sem is not None:
                        s.add(cnode, [], col, parent_table, sem, v)
            s.f2p[tgt_idx] = ([entity_node, cnode] + [-1] * MAX_F2P)[:MAX_F2P]
            out.append(s)
        for seq in out:
            label_stats = self._label_stats(seq, all_labels, task_spec, mode)
            self._normalize_one(seq, task_spec, label_stats, mode)
        return out

    # -- ranking (CONTRACT.md §3) -------------------------------------------
    def _score_ranking(self, query: ParsedQuery,
                       contexts: list[EntityContext],
                       model_uri: str,
                       mode: NormalizationMode) -> list[EntityPrediction]:
        t = query.target
        if not isinstance(t, Aggregation):
            raise ScoringError(
                "ranking target must be LIST_DISTINCT(table.fk) or "
                "ARRAY_AGG(table.fk)")
        fk_ref = t.column
        link = None
        if self.schema is not None:
            link = next((l for l in self.schema.links_from(fk_ref.table)
                         if l.fk_column == fk_ref.column), None)
        if link is None:
            raise ScoringError(
                f"ranking requires LIST_DISTINCT/ARRAY_AGG over a foreign-key "
                f"column: "
                f"{fk_ref} is not an FK to a parent table")
        parent_table = link.to_table
        if query.top_k is None:
            warnings.warn(
                "LIST_DISTINCT/ARRAY_AGG target without RANK TOP K returns "
                "only the single top-ranked candidate; write RANK TOP K for "
                "a K-element result", stacklevel=3)
        k = query.top_k or 1
        bound = self._batch_bound(contexts)
        candidates = self._rank_candidates(parent_table, bound)
        if not candidates:
            raise ScoringError(
                f"ranking: parent table {parent_table!r} has no candidate ids "
                f"at or before the anchor")

        rank_head = self._head_for(TaskType.MULTILABEL_RANKING)
        preds: list[EntityPrediction] = []
        for ctx in contexts:
            cand_seqs = self.candidate_seqs(query, ctx, parent_table,
                                            candidates,
                                            normalization_mode=mode)
            if rank_head is not None:
                logits = rank_head.predict(
                    self._encode(cand_seqs, model_uri))[:, 0]
            else:
                logits = self._forward_batched(cand_seqs, model_uri)
            probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64)))
            order = sorted(range(len(candidates)),
                           key=lambda i: (-probs[i], i))   # ties: low cand idx
            ranked = tuple(str(candidates[i].id) for i in order[:k])
            preds.append(EntityPrediction(ctx.entity_id, ranked=ranked))
        return preds
