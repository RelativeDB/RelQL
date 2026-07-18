"""The execution engine: planning, context assembly, model routing.

Two traversal strategies (:class:`SamplerMode`):

* ``RETRIEVER`` (default) — pull-per-hop through Entity/Link retrievers.
* ``CSC`` — a materialized in-memory CSC index built from TableScanners
  (:mod:`relativedb.csc`); refresh with :meth:`Engine.refresh`.

Both enforce the temporal bound defensively: every row returned by user code
is re-checked and dropped if it is newer than the bound (F24 — a buggy
retriever must not leak the future into context).
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, Union

from .csc import CscIndex
from .evaluate import eval_bool, eval_value
from .model import ModelConfig
from .pql.ast import (AggFunc, Aggregation, ColumnRef, ParsedQuery, TaskType,
                      Window)
from .pql.parser import parse, validate
from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import Schema

__all__ = [
    "SamplerMode", "ContextPolicy", "ExecutionInput", "EntityContext",
    "EntityPrediction", "PredictionResult", "ModelBackend",
    "HistoryBaselineBackend", "Engine", "ExecutionError",
    "ContextTruncationWarning",
]

# Aggregations whose value grows with the number of rows in the window, and so
# are biased low when the fanout cap drops children. (FIRST/LAST/MIN/MAX and
# the DISTINCT/LIST variants are far less sensitive to a dropped tail.)
_COUNTING_AGGS = frozenset({AggFunc.COUNT, AggFunc.SUM, AggFunc.AVG})


class ExecutionError(RuntimeError):
    pass


class ContextTruncationWarning(UserWarning):
    """A windowed COUNT/SUM/AVG was computed over a fanout-truncated context,
    biasing the aggregate low. Raise ContextPolicy limits to silence."""


class SamplerMode(Enum):
    RETRIEVER = "retriever"   # pull-per-hop through retrievers (default)
    CSC = "csc"               # materialized in-memory CSC index (scanners)


@dataclass(frozen=True)
class ContextPolicy:
    """Context assembly knobs (storage-agnostic).

    ``fanouts`` are per-hop child caps; when unset, a
    uniform ``bfs_width`` per hop is used (RT geometry). ``max_context_cells``
    is the global cell budget.
    """

    max_context_cells: int = 8192
    bfs_width: int = 32
    fanouts: Optional[tuple[int, ...]] = None
    max_hops: int = 2
    cohort_size: int = 0
    prefer_latest: bool = True

    def __post_init__(self) -> None:
        if self.fanouts is not None:
            object.__setattr__(self, "fanouts", tuple(self.fanouts))

    def fanout_at(self, hop: int) -> int:
        if self.fanouts:
            return self.fanouts[min(hop, len(self.fanouts) - 1)]
        return self.bfs_width

    @property
    def effective_hops(self) -> int:
        if self.fanouts:
            return min(self.max_hops, len(self.fanouts))
        return self.max_hops


@dataclass(frozen=True)
class ExecutionInput:
    query: Union[str, ParsedQuery]
    anchor_time: Optional[datetime] = None       # "now"; None = unbounded
    per_entity_anchor: bool = False              # anchor_time="entity" semantics
    context_anchor_time: Optional[datetime] = None  # decouple context "now"
    entity_ids: Optional[Sequence[Any]] = None   # overrides FOR ... IN (...)


@dataclass
class EntityContext:
    """The assembled per-entity context: seed entity row + traversed rows."""

    entity_id: Any
    anchor: Optional[datetime]
    rows: list[Row] = field(default_factory=list)
    truncated_children: int = 0     # children dropped by the fanout cap (F-trunc)
    hit_cell_budget: bool = False   # assembly stopped on max_context_cells

    @property
    def row_keys(self) -> set[tuple[str, Any]]:
        return {r.key for r in self.rows}

    @property
    def cell_count(self) -> int:
        return sum(len(r.cells) + (1 if r.timestamp is not None else 0)
                   for r in self.rows)

    def rows_by_table(self) -> dict[str, list[Row]]:
        out: dict[str, list[Row]] = {}
        for r in self.rows:
            out.setdefault(r.table, []).append(r)
        return out

    def entity_cells(self, entity_table: str) -> dict[str, Any]:
        for r in self.rows:
            if r.table == entity_table and r.id == self.entity_id:
                return r.cells
        return {}


@dataclass(frozen=True)
class EntityPrediction:
    id: Any
    value: Optional[float] = None                 # regression / score
    probability: Optional[float] = None           # binary classification
    class_probs: dict[str, float] = field(default_factory=dict)
    ranked: tuple = ()                            # RANK TOP K
    forecast: tuple = ()                          # FORECAST N
    predicted_class: Optional[str] = None         # RETURN CLASS hard label
    quantiles: dict[float, float] = field(default_factory=dict)  # RETURN QUANTILES
    interval: Optional[tuple[float, float]] = None  # RETURN INTERVAL (lo, hi)


@dataclass(frozen=True)
class PredictionResult:
    task_type: TaskType
    predictions: tuple[EntityPrediction, ...]
    model_uri: str = ""
    stats: dict = field(default_factory=dict)   # context-assembly diagnostics

class ModelBackend(Protocol):
    """Anything that can score assembled contexts. The built-in
    :class:`HistoryBaselineBackend` is a model-free reference; real backends
    load the checkpoint at ``model_uri`` (routed by task type)."""

    def score(self, query: ParsedQuery, task_type: TaskType,
              contexts: list[EntityContext], model_uri: str,
              config: ModelConfig) -> list[EntityPrediction]: ...


# ---------------------------------------------------------------------------
# Samplers: the two traversal strategies behind one tiny surface
# ---------------------------------------------------------------------------

class _RetrieverSampler:
    def __init__(self, schema: Schema, wiring: RetrieverWiring):
        self.schema = schema
        self.wiring = wiring

    def entities(self, table: str, ids: Sequence[Any],
                 bound: TemporalBound) -> list[Row]:
        return list(self.wiring.entity_retriever(table)(table, list(ids), bound))

    def children(self, link, parent_id, bound: TemporalBound,
                 limit: int) -> list[Row]:
        return list(self.wiring.link_retriever(link.from_table)(
            link, parent_id, bound, limit))

    def cohort(self, table: str, anchor: Any, bound: TemporalBound,
               limit: int) -> list[Any]:
        r = self.wiring.cohort_retriever(table)
        return list(r(table, anchor, bound, limit)) if r else []

    def all_ids(self, table: str) -> Optional[list[Any]]:
        if table in self.wiring.scanners:
            scanner = self.wiring.scanner(table)
            return [r.id for r in scanner(table, TemporalBound.unbounded())]
        return None


class _CscSampler:
    def __init__(self, index: CscIndex):
        self.index = index

    def entities(self, table, ids, bound):
        return self.index.entities(table, ids, bound)

    def children(self, link, parent_id, bound, limit):
        return self.index.children(link, parent_id, bound, limit)

    def cohort(self, table, anchor, bound, limit):
        return self.index.cohort(table, anchor, bound, limit)

    def all_ids(self, table):
        return self.index.all_ids(table)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _newest_first_key(r: Row):
    return (r.timestamp is None,
            -(r.timestamp.timestamp() if r.timestamp else 0.0))


class Engine:
    def __init__(self, schema: Schema, wiring: RetrieverWiring, *,
                 model_config: Optional[ModelConfig] = None,
                 model_backend: Optional[ModelBackend] = None,
                 context_policy: Optional[ContextPolicy] = None,
                 sampler_mode: SamplerMode = SamplerMode.RETRIEVER):
        self.schema = schema
        self.wiring = wiring
        self.model_config = model_config or ModelConfig.defaults()
        self.model_backend: ModelBackend = model_backend or HistoryBaselineBackend()
        self.context_policy = context_policy or ContextPolicy()
        self.sampler_mode = sampler_mode
        self._csc_index: Optional[CscIndex] = None
        if sampler_mode is SamplerMode.CSC:
            self.refresh()

    def refresh(self) -> None:
        """(Re)build the CSC snapshot from the wired TableScanners."""
        self._csc_index = CscIndex.build(self.schema, self.wiring)

    def _sampler(self):
        if self.sampler_mode is SamplerMode.CSC:
            if self._csc_index is None:
                self.refresh()
            return _CscSampler(self._csc_index)
        return _RetrieverSampler(self.schema, self.wiring)

    # -- context assembly ---------------------------------------------------
    def assemble_context(self, entity_table: str, entity_id: Any,
                         anchor: Optional[datetime],
                         policy: Optional[ContextPolicy] = None) -> EntityContext:
        """The hop loop: seed -> parents (always) -> children (fanout-capped,
        newest-first), every row re-checked against the temporal bound."""
        policy = policy or self.context_policy
        sampler = self._sampler()
        bound = (TemporalBound.at_or_before(anchor) if anchor is not None
                 else TemporalBound.unbounded())
        ctx = EntityContext(entity_id=entity_id, anchor=bound.as_of)
        visited: set[tuple[str, Any]] = set()

        def admit(rows: list[Row]) -> list[Row]:
            fresh = []
            for r in rows:
                if not bound.admits_row(r):
                    continue  # defensive leakage guard (F24)
                if r.key in visited:
                    continue
                visited.add(r.key)
                ctx.rows.append(r)
                fresh.append(r)
            return fresh

        seed = admit(sampler.entities(entity_table, [entity_id], bound))
        if not seed:
            return ctx
        frontier: list[Row] = list(seed)

        # optional cohort seeds (similar entities, Tier 1)
        if policy.cohort_size > 0:
            cohort_ids = sampler.cohort(entity_table, entity_id, bound,
                                        policy.cohort_size)
            if cohort_ids:
                frontier += admit(sampler.entities(entity_table, cohort_ids, bound))

        fk_to_parent = {t.name: {l.fk_column: l.to_table
                                 for l in self.schema.links_from(t.name)}
                        for t in self.schema.tables}

        for hop in range(policy.effective_hops):
            if ctx.cell_count >= policy.max_context_cells:
                break
            fanout = policy.fanout_at(hop)
            next_frontier: list[Row] = []
            # parents: always followed, batched per table
            wanted: dict[str, list[Any]] = {}
            for row in frontier:
                for fk, pid in row.parents.items():
                    ptable = fk_to_parent.get(row.table, {}).get(fk)
                    if ptable is not None and (ptable, pid) not in visited:
                        wanted.setdefault(ptable, []).append(pid)
            for ptable, pids in wanted.items():
                next_frontier += admit(sampler.entities(ptable, pids, bound))
            # children: width-bounded, newest-first
            for row in frontier:
                for link in self.schema.links_to(row.table):
                    # request one extra so we can *detect* (not just silently
                    # apply) truncation: if the sampler returns more than the
                    # fanout, a windowed COUNT/SUM over this context is biased
                    # low. Surfaced via ctx.truncated_children (F-trunc).
                    kids = sampler.children(link, row.id, bound, fanout + 1)
                    kids = [k for k in kids if bound.admits_row(k)]
                    if len(kids) > fanout:
                        ctx.truncated_children += len(kids) - fanout
                    if policy.prefer_latest:
                        kids.sort(key=_newest_first_key)
                    next_frontier += admit(kids[:fanout])
                if ctx.cell_count >= policy.max_context_cells:
                    ctx.hit_cell_budget = True
                    break
            frontier = next_frontier
            if not frontier:
                break
        return ctx

    # -- execution ----------------------------------------------------------
    def execute(self, input: Union[ExecutionInput, str], **kwargs) -> PredictionResult:
        if isinstance(input, str):
            input = ExecutionInput(query=input, **kwargs)
        pq = (parse(input.query) if isinstance(input.query, str)
              else input.query)
        validate(pq, self.schema)
        task_type = pq.task_type(self.schema)
        model_uri = self.model_config.model_uri_for(task_type)
        entity_table = pq.entity_key.table
        ids = self._resolve_entity_ids(pq, input)
        contexts: list[EntityContext] = []
        for eid in ids:
            anchor = self._anchor_for(entity_table, eid, input)
            ctx = self.assemble_context(entity_table, eid, anchor)
            if pq.where is not None and not self._where_ok(pq, ctx, entity_table):
                continue
            contexts.append(ctx)
        stats = self._collect_stats(pq, task_type, contexts)
        preds = self.model_backend.score(pq, task_type, contexts, model_uri,
                                         self.model_config)
        return PredictionResult(task_type=task_type,
                                predictions=tuple(preds),
                                model_uri=model_uri, stats=stats)

    def _collect_stats(self, pq: ParsedQuery, task_type: TaskType,
                       contexts: list[EntityContext]) -> dict:
        """Surface silent context truncation. A windowed COUNT/SUM whose
        window can hold more rows than the fanout cap is biased low when
        ``truncated_children`` is nonzero — previously invisible to callers."""
        # truncated_children counts *truncation events* (child-expansions that
        # hit the cap); the true number of dropped rows is unknown and larger,
        # since detection only requests one past the cap. So we report affected
        # entities, not a false-precision drop total.
        n_trunc = sum(1 for c in contexts if c.truncated_children)
        stats = {
            "entities_scored": len(contexts),
            "contexts_truncated": n_trunc,
            "truncation_events": sum(c.truncated_children for c in contexts),
            "contexts_hit_cell_budget": sum(1 for c in contexts if c.hit_cell_budget),
            "fanout_per_hop": self.context_policy.fanout_at(0),
        }
        # Truncation only distorts *count-like* aggregates (COUNT/SUM/AVG over a
        # window). Warn once per execute when it actually bites those.
        windowed = any(a.window is not None and a.func in _COUNTING_AGGS
                       for a in pq.target_aggregations)
        if n_trunc and windowed:
            warnings.warn(
                f"context truncation: {n_trunc}/{len(contexts)} entities hit "
                f"the fanout cap ({stats['fanout_per_hop']}); windowed "
                f"COUNT/SUM/AVG in this query are biased low for them. Raise "
                f"ContextPolicy.bfs_width/fanouts or max_context_cells.",
                ContextTruncationWarning, stacklevel=2)
        return stats

    def _where_ok(self, pq: ParsedQuery, ctx: EntityContext,
                  entity_table: str) -> bool:
        return eval_bool(pq.where, ctx.rows_by_table(),
                         ctx.entity_cells(entity_table), ctx.anchor)

    def _resolve_entity_ids(self, pq: ParsedQuery,
                            input: ExecutionInput) -> list[Any]:
        if input.entity_ids is not None:
            return list(input.entity_ids)
        if pq.entity_ids:
            return list(pq.entity_ids)
        sampler = self._sampler()
        ids = sampler.all_ids(pq.entity_key.table)
        if ids is None:
            raise ExecutionError(
                f"FOR EACH over all {pq.entity_key.table!r} entities needs "
                f"either explicit entity_ids, a pinned FOR ... IN (...) "
                f"selector, or a TableScanner wired for the entity table "
                f"(retrievers alone cannot enumerate a table)")
        return ids

    def _anchor_for(self, entity_table: str, entity_id: Any,
                    input: ExecutionInput) -> Optional[datetime]:
        anchor = input.context_anchor_time or input.anchor_time
        if input.per_entity_anchor:
            rows = self._sampler().entities(entity_table, [entity_id],
                                            TemporalBound.unbounded())
            if rows and rows[0].timestamp is not None:
                return rows[0].timestamp
        return anchor


# ---------------------------------------------------------------------------
# Built-in model-free backend: predicts from the entity's own history
# ---------------------------------------------------------------------------

class HistoryBaselineBackend:
    """Reference backend, no checkpoint: evaluates the target expression over
    trailing historical windows of the assembled context ("self labels",
    F65 — the strongest zero-shot signal). Real model backends implement the
    same :class:`ModelBackend` protocol and load ``model_uri``.

    ASSUMING clauses are parsed and carried on the query but ignored here
    (counterfactual injection is a context-transformer concern; see the
    design's open questions).
    """

    def __init__(self, num_history_windows: int = 3):
        self.num_history_windows = max(1, num_history_windows)

    def score(self, query: ParsedQuery, task_type: TaskType,
              contexts: list[EntityContext], model_uri: str,
              config: ModelConfig) -> list[EntityPrediction]:
        return [self._score_one(query, task_type, ctx) for ctx in contexts]

    def _score_one(self, query: ParsedQuery, task_type: TaskType,
                   ctx: EntityContext) -> EntityPrediction:
        rows_by_table = ctx.rows_by_table()
        cells = ctx.entity_cells(query.entity_key.table)
        aggs = query.target_aggregations
        window = next((a.window for a in aggs if a.window is not None), None)
        span = window.span() if window is not None else None

        ret = query.ret
        ret_kind = ret.kind if ret is not None else None

        if task_type is TaskType.FORECASTING:
            base = self._history_mean(query, rows_by_table, cells, ctx.anchor, span)
            n = query.num_forecasts or 1
            forecast = tuple([base] * n)
            # Forecasting keeps its series; RETURN QUANTILES/INTERVAL/EXPECTED
            # shape the summary value/quantiles/interval alongside it.
            if ret_kind in ("QUANTILES", "INTERVAL"):
                values = self._history_values(query, rows_by_table, cells,
                                              ctx.anchor, span)
                return EntityPrediction(
                    ctx.entity_id, value=base, forecast=forecast,
                    quantiles=self._quantiles_for(ret, values),
                    interval=self._interval_for(ret, values))
            return EntityPrediction(ctx.entity_id, value=base,
                                    forecast=forecast)
        if task_type is TaskType.MULTILABEL_RANKING:
            return self._rank(query, rows_by_table, ctx)
        if task_type is TaskType.BINARY_CLASSIFICATION:
            p = self._history_prob(query, rows_by_table, cells, ctx.anchor, span)
            if ret_kind == "CLASS":
                # Hard decision (threshold at 0.5), not the score.
                label = "true" if p >= 0.5 else "false"
                return EntityPrediction(ctx.entity_id, predicted_class=label)
            if ret_kind == "DISTRIBUTION":
                return EntityPrediction(
                    ctx.entity_id,
                    class_probs={"true": p, "false": 1.0 - p})
            if ret_kind == "EXPECTED_VALUE":
                # Expected value of the 0/1 indicator is p.
                return EntityPrediction(ctx.entity_id, value=p)
            # PROBABILITY (explicit) or default.
            return EntityPrediction(ctx.entity_id, probability=p)
        if task_type is TaskType.MULTICLASS_CLASSIFICATION:
            v = self._latest_value(query, rows_by_table, cells, ctx.anchor, span)
            if v is None:
                return EntityPrediction(ctx.entity_id)
            if ret_kind == "CLASS":
                return EntityPrediction(ctx.entity_id, predicted_class=str(v))
            # DISTRIBUTION / MULTICLASS / default: the degenerate distribution.
            return EntityPrediction(ctx.entity_id, class_probs={str(v): 1.0})
        # regression
        v = self._history_mean(query, rows_by_table, cells, ctx.anchor, span)
        if ret_kind in ("QUANTILES", "INTERVAL"):
            values = self._history_values(query, rows_by_table, cells,
                                          ctx.anchor, span)
            return EntityPrediction(
                ctx.entity_id, value=v,
                quantiles=self._quantiles_for(ret, values),
                interval=self._interval_for(ret, values))
        return EntityPrediction(ctx.entity_id, value=v)

    def _quantiles_for(self, ret, values: list[float]) -> dict[float, float]:
        if ret is None or ret.kind != "QUANTILES" or not values:
            return {}
        return {float(q): self._empirical_quantile(values, float(q))
                for q in ret.quantiles}

    def _interval_for(self, ret, values: list[float]):
        if ret is None or ret.kind != "INTERVAL" or not values:
            return None
        pct = ret.interval or 0
        lo = (1.0 - pct / 100.0) / 2.0
        hi = 1.0 - lo
        return (self._empirical_quantile(values, lo),
                self._empirical_quantile(values, hi))

    @staticmethod
    def _empirical_quantile(values: list[float], q: float) -> float:
        """Linear-interpolation empirical quantile (numpy "linear" semantics),
        pure Python."""
        xs = sorted(values)
        n = len(xs)
        if n == 1:
            return xs[0]
        idx = q * (n - 1)
        lo = int(idx)              # floor
        hi = min(lo + 1, n - 1)    # ceil, clamped
        frac = idx - lo
        return xs[lo] + (xs[hi] - xs[lo]) * frac

    def _pseudo_anchors(self, anchor: Optional[datetime], span) -> list[Optional[datetime]]:
        if anchor is None or span is None:
            return [anchor]
        return [anchor - span * k for k in range(1, self.num_history_windows + 1)]

    def _history_values(self, query, rows_by_table, cells, anchor, span) -> list[float]:
        """Per-pseudo-anchor numeric target evaluations (bools -> 0.0/1.0)."""
        vals = []
        for pa in self._pseudo_anchors(anchor, span):
            v = eval_value(query.target, rows_by_table, cells, pa)
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    def _history_mean(self, query, rows_by_table, cells, anchor, span) -> Optional[float]:
        vals = self._history_values(query, rows_by_table, cells, anchor, span)
        return sum(vals) / len(vals) if vals else None

    def _history_prob(self, query, rows_by_table, cells, anchor, span) -> float:
        outcomes = [eval_bool(query.target, rows_by_table, cells, pa)
                    for pa in self._pseudo_anchors(anchor, span)]
        return sum(1.0 for o in outcomes if o) / len(outcomes)

    def _latest_value(self, query, rows_by_table, cells, anchor, span):
        pa = self._pseudo_anchors(anchor, span)[0]
        v = eval_value(query.target, rows_by_table, cells, pa)
        if isinstance(v, list):
            return v[-1] if v else None
        return v

    def _rank(self, query: ParsedQuery, rows_by_table, ctx) -> EntityPrediction:
        agg = next(iter(query.target_aggregations), None)
        if agg is None:
            return EntityPrediction(ctx.entity_id)
        rows = rows_by_table.get(agg.column.table, [])
        # FK targets (the recommendation pattern) live in Row.parents, never in
        # cells (F17); fall back accordingly.
        def _val(r):
            v = r.cells.get(agg.column.column)
            return v if v is not None else r.parents.get(agg.column.column)
        counts = Counter(v for r in rows if (v := _val(r)) is not None)
        k = query.top_k or 10
        ranked = tuple(v for v, _ in counts.most_common(k))
        return EntityPrediction(ctx.entity_id, ranked=ranked)
