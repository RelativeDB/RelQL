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

import json
import math
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, Union

from .csc import CscIndex
from .evaluate import eval_bool
from .model import ModelConfig
from .pql.ast import (AggFunc, Aggregation, Arith, Case, ColumnRef, Condition,
                      Explain, Func, Lit, LogicalOp, Not, ParsedQuery, TaskType,
                      Window, _find_aggregations)
from .pql.parser import parse, validate
from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import Schema

__all__ = [
    "SamplerMode", "ContextPolicy", "ExecutionInput", "EntityContext",
    "EntityPrediction", "PredictionResult", "ExplainResult", "ModelBackend",
    "Engine", "ExecutionError",
    "ContextTruncationWarning",
]

# Aggregations whose value grows with the number of rows in the window, and so
# are biased low when the fanout cap drops children. (FIRST/LAST/MIN/MAX and
# the DISTINCT/LIST variants are far less sensitive to a dropped tail.)
_COUNTING_AGGS = frozenset({AggFunc.COUNT, AggFunc.SUM, AggFunc.AVG})

# Default output form per task type (contract Part B, `plan.output`), used
# when the query has no explicit RETURN spec.
_TASK_DEFAULT_OUTPUT = {
    TaskType.REGRESSION: "value",
    TaskType.BINARY_CLASSIFICATION: "probability",
    TaskType.MULTICLASS_CLASSIFICATION: "class",
    TaskType.MULTILABEL_RANKING: "ranked",
    TaskType.FORECASTING: "value-per-horizon",
}


def _num(x: float):
    """JSON-safe numeric: pass finite floats through, stringify infinities."""
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _window_str(w: Window) -> str:
    s = f"OVER ({_num(w.start)}, {_num(w.end)}] {w.unit.value}"
    if w.horizons > 1:
        s += f" HORIZONS {w.horizons}"
    return s


def _lit_str(v: Any) -> str:
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, tuple):
        return "(" + ", ".join(_lit_str(x) for x in v) + ")"
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def _expr_str(e: Any) -> str:
    """Normalized human-readable rendering of a target/where expression."""
    if isinstance(e, ColumnRef):
        return str(e)
    if isinstance(e, Lit):
        return _lit_str(e.value)
    if isinstance(e, Aggregation):
        s = f"{e.func.value}({e.column})"
        if e.window is not None:
            s += " " + _window_str(e.window)
        return s
    if isinstance(e, Condition):
        rhs = (_expr_str(e.right_expr) if e.right_expr is not None
               else _lit_str(e.right))
        return f"{_expr_str(e.left)} {e.op.value} {rhs}"
    if isinstance(e, LogicalOp):
        return f"({_expr_str(e.left)} {e.op.value} {_expr_str(e.right)})"
    if isinstance(e, Not):
        return f"NOT ({_expr_str(e.expr)})"
    if isinstance(e, Arith):
        return f"({_expr_str(e.left)} {e.op} {_expr_str(e.right)})"
    if isinstance(e, Func):
        return f"{e.name}(" + ", ".join(_expr_str(a) for a in e.args) + ")"
    if isinstance(e, Case):
        parts = " ".join(f"WHEN {_expr_str(c)} THEN {_expr_str(t)}"
                         for c, t in e.whens)
        els = f" ELSE {_expr_str(e.else_)}" if e.else_ is not None else ""
        return f"CASE {parts}{els} END"
    return str(e)


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
    params: Optional[dict[str, datetime]] = None  # AS OF :name bindings


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


def _pred_to_dict(p: "EntityPrediction") -> dict:
    return {
        "id": p.id,
        "value": p.value,
        "probability": p.probability,
        "class_probs": dict(p.class_probs),
        "ranked": list(p.ranked),
        "forecast": list(p.forecast),
        "predicted_class": p.predicted_class,
        "quantiles": {str(k): v for k, v in p.quantiles.items()},
        "interval": list(p.interval) if p.interval is not None else None,
    }


@dataclass(frozen=True)
class ExplainResult:
    """The result of :meth:`Engine.explain`. ``plan`` is always present;
    ``context`` is populated for CONTEXT/ANALYZE; ``predictions`` only for
    ANALYZE. ``render()`` produces the human TEXT or machine JSON form."""

    mode: str                        # PLAN | CONTEXT | ANALYZE | ABLATION
    format: str                      # TEXT | JSON
    plan: dict
    context: Optional[dict] = None
    predictions: Optional[tuple["EntityPrediction", ...]] = None

    def to_dict(self) -> dict:
        d: dict = {
            "mode": self.mode,
            "format": self.format,
            "plan": self.plan,
        }
        if self.context is not None:
            d["context"] = self.context
        if self.predictions is not None:
            d["predictions"] = [_pred_to_dict(p) for p in self.predictions]
        return d

    def render(self) -> str:
        if self.format.upper() == "JSON":
            return json.dumps(self.to_dict(), indent=2, default=str)
        return self._render_text()

    def _render_text(self) -> str:
        p = self.plan
        lines: list[str] = []
        lines.append(f"EXPLAIN {self.mode}")
        lines.append("")
        lines.append("PLAN")
        lines.append(f"  target:    {p.get('target')}")
        lines.append(f"  task_type: {p.get('task_type')}")
        ent = p.get("entity", {})
        lines.append(
            f"  entity:    {ent.get('table')}.{ent.get('pk')} "
            f"[{ent.get('selector')}]")
        lines.append(f"  output:    {p.get('output')}")
        ao = p.get("as_of", {})
        lines.append(
            f"  as_of:     {ao.get('source')} ({ao.get('value')})")
        lines.append(
            f"  where:     {'present' if p.get('where_present') else 'none'}")
        lines.append(
            f"  assuming:  "
            f"{'carried, not applied' if p.get('assuming_present') else 'none'}")
        windows = p.get("windows", [])
        if windows:
            lines.append("  windows:")
            for w in windows:
                lines.append(
                    f"    - [{w['role']}] {w['table']}.{w['time_column']} "
                    f"({w['start']}, {w['end']}] {w['unit']} "
                    f"horizons={w['horizons']} step={w['step']}")
        ablations = p.get("ablations", [])
        if ablations:
            lines.append("  ablations:")
            for a in ablations:
                lines.append(f"    - {a['table']}: {a['note']}")
        warnings_ = p.get("warnings", [])
        if warnings_:
            lines.append("  warnings:")
            for w in warnings_:
                lines.append(f"    - {w}")
        if self.context is not None:
            c = self.context
            lines.append("")
            lines.append("CONTEXT")
            lines.append(f"  entities_covered: {c.get('entities_covered')}")
            lines.append(f"  total_rows:       {c.get('total_rows')}")
            lines.append(f"  total_cells:      {c.get('total_cells')}")
            lines.append(f"  links_traversed:  {c.get('links_traversed')}")
            lines.append(
                f"  tables_unreachable: {c.get('tables_unreachable')}")
            tables = c.get("tables", {})
            if tables:
                lines.append("  per-table:")
                for name, t in tables.items():
                    lines.append(
                        f"    - {name}: rows={t['rows']} cells={t['cells']} "
                        f"min_time={t['min_time']} max_time={t['max_time']}")
        if self.predictions is not None:
            lines.append("")
            lines.append("PREDICTIONS")
            for pr in self.predictions:
                lines.append(f"  - {_pred_to_dict(pr)}")
        return "\n".join(lines)


class ModelBackend(Protocol):
    """Anything that can score assembled contexts. The real backend
    (:class:`~relativedb.rt_native.RtNativeBackend`) loads the checkpoint at
    ``model_uri`` (routed by task type); engine tests use a tiny deterministic
    test double. There is no built-in model-free scorer."""

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
        self.model_backend: Optional[ModelBackend] = model_backend
        self.context_policy = context_policy or ContextPolicy()
        self.sampler_mode = sampler_mode
        self._csc_index: Optional[CscIndex] = None
        if sampler_mode is SamplerMode.CSC:
            self.refresh()

    def refresh(self) -> None:
        """(Re)build the CSC snapshot from the wired TableScanners."""
        self._csc_index = CscIndex.build(self.schema, self.wiring)

    def _require_backend(self) -> ModelBackend:
        """Scoring paths (execute, EXPLAIN ANALYZE) need a model backend; the
        engine ships none. Parse/validate/EXPLAIN PLAN/CONTEXT/AS OF never
        reach here."""
        if self.model_backend is None:
            raise ExecutionError(
                "Engine requires a model backend (e.g. RtNativeBackend); "
                "there is no built-in model-free scorer.")
        return self.model_backend

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
        if pq.explain is not None:
            # An EXPLAIN-prefixed query must never silently score; route it.
            return self.explain(replace(input, query=pq))
        validate(pq, self.schema)
        # Bind the effective anchor (AS OF) before any assembly/scoring so it
        # threads through the temporal bound and pseudo-anchors unchanged.
        input = replace(input, anchor_time=self._effective_anchor(pq, input))
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
        preds = self._require_backend().score(pq, task_type, contexts,
                                              model_uri, self.model_config)
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

    # -- AS OF: resolve the effective anchor --------------------------------
    def _effective_anchor(self, pq: ParsedQuery,
                          input: ExecutionInput) -> Optional[datetime]:
        """Resolve the effective anchor from the query's ``AS OF`` clause
        (contract Part A). Absent/NOW -> ``anchor_time`` (the execution
        anchor, NOT wall clock); DATE -> parsed date (overrides); PARAM ->
        ``params[name]``, else ``anchor_time``, else a clear error."""
        ao = pq.as_of
        if ao is None or ao.kind == "now":
            return input.anchor_time
        if ao.kind == "date":
            return self._parse_anchor_date(ao.value)
        if ao.kind == "param":
            name = ao.value
            params = input.params or {}
            if name in params:
                return self._coerce_anchor(params[name])
            if input.anchor_time is not None:
                return input.anchor_time
            raise ExecutionError(
                f"AS OF :{name} — no value bound for parameter {name!r} "
                f"(supply ExecutionInput.params={{{name!r}: <datetime>}}) and "
                f"no anchor_time fallback is available")
        raise ExecutionError(f"unknown AS OF kind {ao.kind!r}")

    @staticmethod
    def _coerce_anchor(v: Any) -> datetime:
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if isinstance(v, str):
            return Engine._parse_anchor_date(v)
        raise ExecutionError(
            f"AS OF param must bind to a datetime or date string, got {v!r}")

    @staticmethod
    def _parse_anchor_date(text: Optional[str]) -> datetime:
        s = (text or "").strip()
        fmt = "%Y-%m-%d %H:%M:%S" if " " in s else "%Y-%m-%d"
        try:
            d = datetime.strptime(s, fmt)
        except ValueError as e:
            raise ExecutionError(
                f"AS OF: cannot parse date {text!r} "
                f"(expected YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'): {e}")
        return d.replace(tzinfo=timezone.utc)

    # -- EXPLAIN ------------------------------------------------------------
    def explain(self, input: Union[ExecutionInput, str], **kwargs) -> ExplainResult:
        """Explain a query without (PLAN/CONTEXT) or with (ANALYZE) scoring.
        A non-EXPLAIN query is explained as PLAN by default."""
        if isinstance(input, str):
            input = ExecutionInput(query=input, **kwargs)
        pq = (parse(input.query) if isinstance(input.query, str)
              else input.query)
        validate(pq, self.schema)
        ex = pq.explain or Explain()
        mode = (ex.mode or "PLAN").upper()
        fmt = (ex.format or "TEXT").upper()
        effective = self._effective_anchor(pq, input)

        plan = self._build_plan(pq, input, effective)
        context: Optional[dict] = None
        predictions: Optional[tuple[EntityPrediction, ...]] = None

        if mode == "ABLATION":
            plan["warnings"] = list(plan.get("warnings", [])) + [
                "ablation not implemented — declared ABLATE tables are not "
                "applied"]

        if mode in ("CONTEXT", "ANALYZE"):
            context, contexts = self._assemble_report(pq, input, effective)
            if mode == "ANALYZE":
                task_type = pq.task_type(self.schema)
                model_uri = self.model_config.model_uri_for(task_type)
                predictions = tuple(self._require_backend().score(
                    pq, task_type, contexts, model_uri, self.model_config))

        return ExplainResult(mode=mode, format=fmt, plan=plan,
                             context=context, predictions=predictions)

    def _build_plan(self, pq: ParsedQuery, input: ExecutionInput,
                    effective: Optional[datetime]) -> dict:
        task_type = pq.task_type(self.schema)
        # entity selector
        if input.entity_ids is not None:
            selector = list(input.entity_ids)
        elif pq.entity_ids:
            selector = list(pq.entity_ids)
        else:
            selector = "FOR EACH"
        # output form
        if pq.ret is not None:
            output = pq.ret.kind.lower()
        else:
            output = _TASK_DEFAULT_OUTPUT[task_type]
        # windows across target / where / assuming
        windows: list[dict] = []
        self._collect_windows(pq.target_aggregations, "target", windows)
        if pq.where is not None:
            self._collect_windows(_find_aggregations(pq.where), "where", windows)
        if pq.assuming is not None:
            self._collect_windows(_find_aggregations(pq.assuming), "assuming",
                                  windows)
        # as_of provenance
        ao = pq.as_of
        if ao is None or ao.kind == "now":
            source = "execution-anchor"
        elif ao.kind == "date":
            source = "query-date"
        else:
            source = "query-param"
        as_of = {
            "source": source,
            "value": effective.isoformat() if effective is not None else None,
        }
        if ao is not None and ao.kind == "param":
            as_of["param"] = ao.value

        warnings_: list[str] = []
        if pq.assuming is not None:
            warnings_.append("ASSUMING clause carried, not applied")

        return {
            "target": _expr_str(pq.target),
            "task_type": task_type.value,
            "entity": {
                "table": pq.entity_key.table,
                "pk": pq.entity_key.column,
                "selector": selector,
            },
            "output": output,
            "windows": windows,
            "where_present": pq.where is not None,
            "assuming_present": pq.assuming is not None,
            "as_of": as_of,
            "ablations": [{"table": a.name, "note": "declared, not applied"}
                          for a in pq.ablations],
            "warnings": warnings_,
        }

    def _collect_windows(self, aggs, role: str, out: list[dict]) -> None:
        for a in aggs:
            w = a.window
            if w is None:
                continue
            t = self.schema.table(a.column.table)
            out.append({
                "table": a.column.table,
                "time_column": t.time_column if t is not None else None,
                "start": _num(w.start),
                "end": _num(w.end),
                "unit": w.unit.value,
                "horizons": w.horizons,
                "step": _num(w.step) if w.step is not None else None,
                "role": role,
            })

    def _assemble_report(self, pq: ParsedQuery, input: ExecutionInput,
                         effective: Optional[datetime]):
        """Assemble per-entity context via the normal path (no scoring) and
        summarize it (contract: EXPLAIN CONTEXT)."""
        eff_input = replace(input, anchor_time=effective)
        entity_table = pq.entity_key.table
        ids = self._resolve_entity_ids(pq, eff_input)
        contexts: list[EntityContext] = []
        for eid in ids:
            anchor = self._anchor_for(entity_table, eid, eff_input)
            ctx = self.assemble_context(entity_table, eid, anchor)
            if pq.where is not None and not self._where_ok(pq, ctx, entity_table):
                continue
            contexts.append(ctx)

        tables: dict[str, dict] = {}
        links_traversed = 0
        for ctx in contexts:
            for r in ctx.rows:
                t = tables.setdefault(
                    r.table, {"rows": 0, "cells": 0,
                              "min_time": None, "max_time": None})
                t["rows"] += 1
                t["cells"] += len(r.cells) + (1 if r.timestamp is not None else 0)
                links_traversed += len(r.parents)
                if r.timestamp is not None:
                    ts = r.timestamp.isoformat()
                    if t["min_time"] is None or ts < t["min_time"]:
                        t["min_time"] = ts
                    if t["max_time"] is None or ts > t["max_time"]:
                        t["max_time"] = ts

        reachable = set(tables)
        unreachable = sorted(tbl.name for tbl in self.schema.tables
                             if tbl.name not in reachable)
        report = {
            "entities_covered": len(contexts),
            "anchor": effective.isoformat() if effective is not None else None,
            "total_rows": sum(t["rows"] for t in tables.values()),
            "total_cells": sum(t["cells"] for t in tables.values()),
            "links_traversed": links_traversed,
            "tables": tables,
            "tables_unreachable": unreachable,
            "truncated_children": sum(c.truncated_children for c in contexts),
            "contexts_hit_cell_budget": sum(1 for c in contexts
                                            if c.hit_cell_budget),
        }
        return report, contexts
