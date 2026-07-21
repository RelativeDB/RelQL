"""The execution engine: planning, context assembly, model routing.

Two traversal strategies (:class:`SamplerMode`):

* ``RETRIEVER`` (default) — pull-per-hop through Entity/Link retrievers.
* ``CSC`` — a materialized in-memory CSC index built from TableScanners
  (:mod:`relativedb.csc`), snapshotted once at construction.

Both enforce the temporal bound defensively: every row returned by user code
is re-checked and dropped if it is newer than the bound (F24 — a buggy
retriever must not leak the future into context).
"""
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Union

import numpy as np

from .csc import CscIndex
from .evaluate import eval_bool, eval_value
from .model import ModelConfig
from .relql.ast import (AggFunc, Aggregation, Arith, BoolOp, Case, ColumnRef,
                      Condition, Explain, Func, Lit, LogicalOp, Not, Operator,
                      ParsedQuery, TaskType, Window, _find_aggregations)
from .relql.parser import parse, validate
from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import Schema
from .traversal import (BreadthFirstTraversal, GraphTraversal,
                        ReferenceTraversal)

__all__ = [
    "SamplerMode", "ContextPolicy", "ExecutionInput", "EntityContext",
    "EntityPrediction", "PredictionResult", "ExplainResult", "ModelBackend",
    "Engine", "ExecutionError",
    "ContextTruncationWarning", "AssumptionNotAppliedWarning",
    "GraphTraversal", "BreadthFirstTraversal", "ReferenceTraversal",
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


def _target_span(pq: ParsedQuery):
    """How far past the anchor the target's window reaches — the bound the
    label context needs. None when the frame is unbounded."""
    for a in pq.target_aggregations:
        if a.window is not None:
            return a.window.span()
    return None


def _pinned_ids(where: Any, entity_key: ColumnRef) -> Optional[list[Any]]:
    """The cohort a WHERE clause pins the primary key to, or None if it
    doesn't pin one.

    Only conjunctive top-level predicates count: `pk = v` and `pk IN (...)`
    joined by AND. Under an OR (or a NOT) the clause no longer restricts the
    cohort on its own, so there is nothing safe to push down and we fall back
    to enumerating. Several ANDed pk predicates intersect.
    """
    if where is None:
        return None
    if isinstance(where, LogicalOp) and where.op is BoolOp.AND:
        left = _pinned_ids(where.left, entity_key)
        right = _pinned_ids(where.right, entity_key)
        if left is None:
            return right
        if right is None:
            return left
        keep = set(right)
        return [v for v in left if v in keep]      # intersect, order-stable
    if not isinstance(where, Condition):
        return None
    if not (isinstance(where.left, ColumnRef)
            and where.left.table == entity_key.table
            and where.left.column == entity_key.column):
        return None
    if where.right_expr is not None:
        return None                                 # unbound param / expression
    if where.op is Operator.EQ:
        return [where.right]
    if where.op is Operator.IN:
        return list(where.right)
    return None


def _assumptions(expr: Any) -> list[tuple[str, str, Any]]:
    """The `(table, column, value)` assignments an ASSUMING clause states.

    Only shapes with one concrete answer qualify: `column = literal`, and those
    joined by AND. An inequality, an `IN`, an `OR`/`NOT`, or an aggregate
    condition constrains the world without saying what it *is* — there is no
    single context that satisfies it — so those raise rather than being quietly
    dropped.
    """
    if isinstance(expr, LogicalOp) and expr.op is BoolOp.AND:
        return _assumptions(expr.left) + _assumptions(expr.right)
    if (isinstance(expr, Condition) and expr.op is Operator.EQ
            and isinstance(expr.left, ColumnRef)
            and expr.left.column != "*"
            and expr.right_expr is None
            and not isinstance(expr.right, tuple)):
        return [(expr.left.table, expr.left.column, expr.right)]
    raise ExecutionError(
        f"ASSUMING {_expr_str(expr)!r} cannot be applied: a counterfactual "
        f"must assign concrete values — `column = literal`, optionally joined "
        f"by AND. Inequalities, IN, OR/NOT and aggregate conditions describe a "
        f"set of possible worlds rather than one, so the engine cannot build "
        f"the context they imply.")


def _assuming_plan(expr: Any) -> Optional[str]:
    """How EXPLAIN renders the counterfactual. EXPLAIN must describe any query
    that parses, so an inapplicable clause is reported, not raised."""
    if expr is None:
        return None
    try:
        return ", ".join(f"{t}.{c} := {_lit_str(v)}"
                         for t, c, v in _assumptions(expr))
    except ExecutionError:
        return "cannot be applied (see warnings)"


def _apply_assumptions(assignments: list[tuple[str, str, Any]],
                       ctx: "EntityContext") -> "EntityContext":
    """A copy of ``ctx`` in which the assumed values hold.

    Every context row of an assigned table gets the new cell value — assuming
    `orders.status = 'shipped'` means *these* orders are shipped. Rows are
    replaced rather than mutated: retrievers and the CSC index hand out shared
    Row objects, so writing through one would corrupt every other entity's
    context.
    """
    if not assignments:
        return ctx
    by_table: dict[str, dict[str, Any]] = {}
    for table, col, value in assignments:
        by_table.setdefault(table, {})[col] = value
    rows = [replace(r, cells={**r.cells, **by_table[r.table]})
            if r.table in by_table else r
            for r in ctx.rows]
    return EntityContext(entity_id=ctx.entity_id, anchor=ctx.anchor, rows=rows,
                         truncated_children=ctx.truncated_children,
                         hit_cell_budget=ctx.hit_cell_budget,
                         focal_row_keys=ctx.focal_row_keys,
                         node_ids=dict(ctx.node_ids))


def _warn_inert_assumptions(assignments: list[tuple[str, str, Any]],
                            contexts: list["EntityContext"]) -> None:
    """An assumption about a table that never appears in any context changes
    nothing. That is the failure ASSUMING used to have wholesale, so say it."""
    if not assignments or not contexts:
        return
    present = {r.table for c in contexts for r in c.rows}
    missing = sorted({t for t, _, _ in assignments} - present)
    if missing:
        warnings.warn(
            f"ASSUMING has no effect on {', '.join(repr(t) for t in missing)}: "
            f"no rows of that table are in the assembled context, so the "
            f"assumption changes nothing that reaches the model",
            AssumptionNotAppliedWarning, stacklevel=3)


class AssumptionNotAppliedWarning(UserWarning):
    """An ASSUMING assignment targeted a table absent from the context."""


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

    # Reference evaluation defaults (eval_utils.build_evaluator).
    max_context_cells: int = 8192
    bfs_width: int = 32
    fanouts: Optional[tuple[int, ...]] = None
    max_hops: int = 2
    cohort_size: int = 256
    prefer_latest: bool = True
    local_context_cells: int = 256
    num_walks: int = 10_000
    walk_length: int = 20
    seed: int = 0
    # Number of prior task windows materialized when a RelQL aggregate defines
    # a derived target table. The reference consumes pre-materialized task
    # rows; this is the query-runtime equivalent.
    num_history_windows: int = 3

    def __post_init__(self) -> None:
        if self.fanouts is not None:
            object.__setattr__(self, "fanouts", tuple(self.fanouts))
        if self.max_context_cells <= 0:
            raise ValueError("max_context_cells must be positive")
        if self.local_context_cells <= 0:
            raise ValueError("local_context_cells must be positive")
        if self.num_walks < 0 or self.walk_length < 0:
            raise ValueError("num_walks and walk_length cannot be negative")
        if self.num_history_windows < 0:
            raise ValueError("num_history_windows cannot be negative")

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
    # `:name` bindings for the query text — the anchor (`AS OF :t`), the
    # cohort (`WHERE t.pk IN :ids`), and any other parameterized literal.
    params: Optional[dict[str, Any]] = None


@dataclass
class EntityContext:
    """The assembled per-entity context: seed entity row + traversed rows."""

    entity_id: Any
    anchor: Optional[datetime]
    rows: list[Row] = field(default_factory=list)
    truncated_children: int = 0     # children dropped by the fanout cap (F-trunc)
    hit_cell_budget: bool = False   # assembly stopped on max_context_cells
    # Rows belonging to the focal entity's local graph, excluding global peer
    # context. This keeps self-label construction invariant across traversals.
    focal_row_keys: frozenset[tuple[str, Any]] = frozenset()
    node_ids: dict[tuple[str, Any], int] = field(default_factory=dict)

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

    def focal_rows_by_table(self) -> dict[str, list[Row]]:
        selected = ([r for r in self.rows if r.key in self.focal_row_keys]
                    if self.focal_row_keys else self.rows)
        out: dict[str, list[Row]] = {}
        for r in selected:
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
            f"{p.get('assuming') or 'none'}")
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
        """Cohort ids for the context window.

        A wiring may supply its own retriever — a similarity index, say. When
        it does not, the cohort is derived from the table scanner using the
        same definition :meth:`CscIndex.cohort` uses, so the two sampler modes
        assemble identical contexts. Returning nothing here instead would let
        RETRIEVER mode quietly drop the cohort that CSC mode includes.
        """
        r = self.wiring.cohort_retriever(table)
        if r is not None:
            return list(r(table, anchor, bound, limit))
        scanner = self.wiring.scanners.get(table)
        if scanner is None:
            return []
        out: list[Any] = []
        for row in scanner(table, bound):
            if row.id != anchor:
                out.append(row.id)
                if len(out) >= limit:
                    break
        return out

    def all_ids(self, table: str) -> Optional[list[Any]]:
        if table in self.wiring.scanners:
            scanner = self.wiring.scanner(table)
            return [r.id for r in scanner(table, TemporalBound.unbounded())]
        return None

    def all_rows(self, table: str) -> Optional[list[Row]]:
        scanner = self.wiring.scanners.get(table)
        return (None if scanner is None else
                list(scanner(table, TemporalBound.unbounded())))


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

    def all_rows(self, table):
        return list(self.index.rows.get(table, ()))


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
                 sampler_mode: SamplerMode = SamplerMode.RETRIEVER,
                 traversal: Optional[GraphTraversal] = None):
        self.schema = schema
        self.wiring = wiring
        self.model_config = model_config or ModelConfig.defaults()
        self.model_backend: Optional[ModelBackend] = model_backend
        self.context_policy = context_policy or ContextPolicy()
        self.sampler_mode = sampler_mode
        self.traversal = traversal or ReferenceTraversal()
        # The CSC snapshot is built once, here. It is immutable for the life of
        # the engine: to pick up changed data, construct a new Engine.
        self._csc_index: Optional[CscIndex] = None
        # Reference sampling is defined over one immutable bidirectional graph
        # snapshot. Build it even in RETRIEVER mode; explicit legacy BFS keeps
        # the pull-per-hop retriever semantics.
        if sampler_mode is SamplerMode.CSC:
            self._csc_index = CscIndex.build(self.schema, self.wiring)
        elif (isinstance(self.traversal, ReferenceTraversal)
              and self.wiring.scanners):
            # A schema-only table with no scanner is an empty table in the
            # snapshot. At least one scanner is required to distinguish this
            # from legacy pull-only wiring.
            self._csc_index = CscIndex.build(
                self.schema, self.wiring, allow_missing_scanners=True)

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
        if isinstance(self.traversal, ReferenceTraversal):
            if self._csc_index is None:
                raise ExecutionError(
                    "reference traversal requires a TableScanner for every "
                    "schema table so it can build one immutable graph snapshot")
            return _CscSampler(self._csc_index)
        if self.sampler_mode is SamplerMode.CSC:
            if self._csc_index is None:
                # Only reachable by flipping sampler_mode after construction.
                # Say so rather than silently draining every scanner mid-query.
                raise ExecutionError(
                    "CSC mode has no index: the snapshot is built once, in the "
                    "Engine constructor. Construct the engine with "
                    "sampler_mode=SamplerMode.CSC instead of setting it later.")
            return _CscSampler(self._csc_index)
        return _RetrieverSampler(self.schema, self.wiring)

    # -- context assembly ---------------------------------------------------
    def assemble_context(self, entity_table: str, entity_id: Any,
                         anchor: Optional[datetime],
                         policy: Optional[ContextPolicy] = None, *,
                         query: Optional[ParsedQuery] = None) -> EntityContext:
        """Assemble a bounded context with the configured graph traversal."""
        policy = policy or self.context_policy
        sampler = self._sampler()
        bound = (TemporalBound.at_or_before(anchor) if anchor is not None
                 else TemporalBound.unbounded())
        result = self.traversal.traverse(
            self.schema, sampler, entity_table, entity_id, bound, policy,
            query=query)
        return EntityContext(
            entity_id=entity_id, anchor=bound.as_of, rows=list(result.rows),
            truncated_children=result.truncated_children,
            hit_cell_budget=result.hit_cell_budget,
            focal_row_keys=result.focal_row_keys,
            node_ids=dict(result.node_ids))

    # -- execution ----------------------------------------------------------
    def execute(self, input: Union[ExecutionInput, str], **kwargs) -> PredictionResult:
        if isinstance(input, str):
            input = ExecutionInput(query=input, **kwargs)
        pq = (parse(input.query) if isinstance(input.query, str)
              else input.query)
        if pq.explain is not None:
            # An EXPLAIN-prefixed query must never silently score; route it.
            return self.explain(replace(input, query=pq))
        pq = validate(pq, self.schema).query   # binds the population's pk
        pq = pq.bind_params(input.params)      # substitutes :name -> values
        # Bind the effective anchor (AS OF) before any assembly/scoring so it
        # threads through the temporal bound and pseudo-anchors unchanged.
        input = replace(input, anchor_time=self._effective_anchor(pq, input))
        task_type = pq.task_type(self.schema)
        model_uri = self.model_config.model_uri_for(task_type)
        entity_table = pq.entity_key.table
        ids = self._resolve_entity_ids(pq, input)
        assumed = _assumptions(pq.assuming) if pq.assuming is not None else []
        contexts: list[EntityContext] = []
        for eid in ids:
            anchor = self._anchor_for(entity_table, eid, input)
            ctx = self.assemble_context(entity_table, eid, anchor, query=pq)
            # WHERE selects who to score and is factual; the counterfactual is
            # applied afterwards, to the context that actually gets scored.
            if pq.where is not None and not self._where_ok(pq, ctx, entity_table):
                continue
            contexts.append(_apply_assumptions(assumed, ctx))
        _warn_inert_assumptions(assumed, contexts)
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
        cells = dict(ctx.entity_cells(entity_table))
        # The primary key is the row's identity, not one of its cells, so a
        # predicate on it (`WHERE t.pk IN :ids`) would otherwise see NULL and
        # drop every entity.
        pk = pq.entity_key.column
        if pk:
            cells.setdefault(pk, ctx.entity_id)
        return eval_bool(pq.where, ctx.rows_by_table(), cells, ctx.anchor)

    def _resolve_entity_ids(self, pq: ParsedQuery,
                            input: ExecutionInput) -> list[Any]:
        """The cohort to score.

        A WHERE clause that pins the primary key — `pk = :id`, `pk IN :ids`, or
        a literal equivalent — names the cohort outright, so it is pushed down
        and no enumeration happens. Anything else needs the table enumerated
        and filtered, which requires a TableScanner.
        """
        pinned = _pinned_ids(pq.where, pq.entity_key)
        if pinned is not None:
            return pinned
        ids = self._sampler().all_ids(pq.entity_key.table)
        if ids is None:
            raise ExecutionError(
                f"FROM {pq.entity_key.table!r} scores every entity in the "
                f"table, which needs a TableScanner wired for it (retrievers "
                f"alone cannot enumerate a table). To score a specific cohort "
                f"instead, pin the key: WHERE {pq.entity_key.table}."
                f"{pq.entity_key.column} IN :ids")
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
    # -- task-head adaptation ----------------------------------------------
    def finetune(self, query: Union[str, ParsedQuery], anchors: Sequence[datetime],
                 *, output_dir: Union[str, os.PathLike] = "finetuned_model",
                 entity_ids: Optional[Sequence[Any]] = None,
                 params: Optional[dict[str, Any]] = None,
                 labels: Optional[dict] = None,
                 epochs: int = 1, batch_size: int = 1,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 1e-2,
                 grad_clip_norm: float = 1.0,
                 model_uri: Optional[str] = None):
        """Fine-tune the complete RT-J checkpoint with native C++/MPS.

        Unlike :meth:`fit_head`, this differentiates through all transformer
        blocks, encoders, learned masks, normalization scales, and the number
        decoder. It currently supports scalar binary/regression tasks, whose
        labels use the reference bool-as-number/Huber training contract.
        """
        from .model import NormalizationMode
        from .rt_native import (ColumnStats, FineTunedCheckpoint,
                                RT_DEVICE_MPS, RtNativeError, load_lib,
                                resolve_model_path)
        backend = self._require_backend()
        if not hasattr(backend, "_build_sequences"):
            raise ExecutionError(
                "full-backbone fine-tuning requires RtNativeBackend")
        lib = load_lib(backend._lib_path)
        if not lib.device_available(RT_DEVICE_MPS):
            raise ExecutionError("full-model fine-tuning requires Apple MPS")
        if not anchors:
            raise ExecutionError("full-model fine-tuning needs at least one anchor")
        if epochs <= 0 or batch_size <= 0:
            raise ExecutionError("epochs and batch_size must be positive")
        anchors = [self._coerce_anchor(a) for a in anchors]
        pq = parse(query) if isinstance(query, str) else query
        pq = validate(pq, self.schema).query.bind_params(params)
        task_type = pq.task_type(self.schema)
        if task_type not in (TaskType.BINARY_CLASSIFICATION,
                             TaskType.REGRESSION):
            raise ExecutionError(
                "native full-model fine-tuning currently supports binary "
                "classification and regression; use fit_head for multiclass/ranking")
        model_uri = model_uri or self.model_config.model_uri_for(task_type)
        normalization_mode = backend._mode(self.model_config)
        task_spec = backend.task_spec(pq, task_type)
        if normalization_mode is NormalizationMode.REFERENCE:
            backend.column_stats = ColumnStats.fit(
                self.schema, self.wiring,
                TemporalBound.at_or_before(max(anchors)))

        examples: list[tuple[EntityContext, float]] = []
        span = _target_span(pq)
        entity_table = pq.entity_key.table
        for anchor in anchors:
            ids = (list(entity_ids) if entity_ids is not None else
                   self._resolve_entity_ids(
                       pq, ExecutionInput(query=pq, anchor_time=anchor)))
            for eid in ids:
                if labels is not None and (eid, anchor) not in labels:
                    continue
                ctx = self.assemble_context(entity_table, eid, anchor, query=pq)
                label_ctx = (None if labels is not None else
                             self.assemble_context(
                                 entity_table, eid,
                                 None if span is None else anchor + span,
                                 query=pq))
                y = self._scalar_label(pq, task_type, label_ctx, anchor,
                                       labels, eid, [])
                if y is not None:
                    examples.append((ctx, float(y)))
        if not examples:
            raise ExecutionError(
                "full-model fine-tuning produced no training examples")
        if normalization_mode is NormalizationMode.REFERENCE:
            backend.column_stats = backend.column_stats.with_task_values(
                task_spec, [y for _, y in examples])

        seqs = []
        for ctx, y in examples:
            one, mus, sds = backend._build_sequences(
                pq, task_type, [ctx], normalization_mode=normalization_mode,
                task_spec=task_spec)
            seq = one[0]
            target = (y - mus[0]) / sds[0]
            for i, is_target in enumerate(seq.is_tgt):
                if is_target:
                    seq.value[i] = target
            seqs.append(seq)

        source_path = resolve_model_path(model_uri)
        model = lib.load_model(source_path)  # independent mutable checkpoint
        losses: list[float] = []
        grad_norms: list[float] = []
        total_seconds = 0.0
        try:
            for _ in range(epochs):
                for start in range(0, len(seqs), batch_size):
                    arrays = backend._collate(seqs[start:start + batch_size])
                    result = model.finetune_step(
                        **arrays, learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        grad_clip_norm=grad_clip_norm)
                    losses.append(result["loss"])
                    grad_norms.append(result["grad_norm"])
                    total_seconds += result["seconds"]
        except RtNativeError as exc:
            raise ExecutionError(str(exc)) from exc

        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        model.save(str(out / "model.safetensors"))
        source_config = Path(source_path).with_name("config.json")
        config = (json.loads(source_config.read_text())
                  if source_config.exists() else {})
        config["checkpoint_file"] = "model.safetensors"
        config["finetune"] = {
            "backend": "native-mps", "full_model": True,
            "source_model": model_uri, "steps": len(losses),
            "examples": len(examples), "final_loss": losses[-1],
            "normalization_mode": normalization_mode.value,
        }
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return FineTunedCheckpoint(
            out, tuple(losses), tuple(grad_norms), total_seconds,
            len(examples), len(losses), backend.column_stats,
            normalization_mode)

    def fit_head(self, query: Union[str, ParsedQuery], anchors: Sequence[datetime],
                 *, entity_ids: Optional[Sequence[Any]] = None,
                 params: Optional[dict[str, Any]] = None,
                 labels: Optional[dict] = None,
                 epochs: int = 100, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 model_uri: Optional[str] = None):
        """Fit a task head for ``query`` over the frozen backbone.

        The transformer is not updated; each training example is encoded once
        into its target-cell state and a small head is fitted on those. Returns
        a :class:`~relativedb.rt_native.FineTunedHead` — inspect its losses,
        ``save(path)`` it, and serve it by passing ``head=`` to
        :class:`~relativedb.rt_native.RtNativeBackend`.

        ``anchors`` are past cut-off times. For each one the context is bounded
        at the anchor — exactly as at prediction time — while the **label** is
        read from what actually happened in the target's window after it. So
        the query defines its own supervision and no labels need supplying.

        Pass ``labels`` to override that: ``{(entity_id, anchor): value}``, or
        for ranking ``{(entity_id, anchor): {candidate_id: relevance}}``. When
        given, it also *selects* the examples — a pair it does not name is
        skipped rather than derived — so passing every row's own timestamp as
        an anchor and labelling only the diagonal trains each entity at its own
        cut-off instead of at a cut-off shared with rows it predates.
        """
        from .rt_native import (FT_MULTICLASS, FT_RANKING, FineTunedHead,
                                RtNativeError)
        backend = self._require_backend()
        if not hasattr(backend, "candidate_seqs"):
            raise ExecutionError(
                "task-head fitting requires the native RT backend (RtNativeBackend)")
        if not anchors:
            raise ExecutionError("task-head fitting needs at least one anchor")
        # Row timestamps are UTC-aware; naive anchors would fail to compare.
        anchors = [self._coerce_anchor(a) for a in anchors]

        pq = parse(query) if isinstance(query, str) else query
        pq = validate(pq, self.schema).query.bind_params(params)
        task_type = pq.task_type(self.schema)
        if task_type not in (TaskType.MULTICLASS_CLASSIFICATION,
                             TaskType.MULTILABEL_RANKING):
            raise ExecutionError(
                "frozen task-head fitting is limited to multiclass and "
                "multilabel-ranking adapters; scalar binary/regression tasks "
                "require full-backbone fine-tuning")
        model_uri = model_uri or self.model_config.model_uri_for(task_type)
        model = backend._model_for(model_uri)

        from .model import NormalizationMode
        from .rt_native import ColumnStats
        normalization_mode = backend._mode(self.model_config)
        if normalization_mode is NormalizationMode.REFERENCE:
            # Reference preprocessing is fitted only on rows knowable during
            # training. Target stats are added after labels are collected.
            backend.column_stats = ColumnStats.fit(
                self.schema, self.wiring,
                TemporalBound.at_or_before(max(anchors)))

        entity_table = pq.entity_key.table
        span = _target_span(pq)
        ys: list[float] = []
        groups: list[int] = [0]
        classes: list[Any] = []
        skipped = 0
        scalar_examples: list[tuple[EntityContext, float]] = []
        ranking_examples: list[tuple[EntityContext, str, list, list[float]]] = []

        for t in anchors:
            ids = (list(entity_ids) if entity_ids is not None
                   else self._resolve_entity_ids(
                       pq, ExecutionInput(query=pq, anchor_time=t)))
            for eid in ids:
                # A supplied ``labels`` dict IS the training set: it names the
                # (entity, anchor) pairs that are examples, and pairs it does
                # not name are not examples. That is what lets every row carry
                # its own anchor -- pass each row's timestamp and label the
                # diagonal -- which is the shape a RelBench train table has.
                # Without it a shared anchor either hides the entities after it
                # or shows the ones before it their own outcome.
                if labels is not None and (eid, t) not in labels:
                    continue
                # features see only what was knowable at the anchor...
                ctx = self.assemble_context(entity_table, eid, t, query=pq)
                # ...the label reads the window after it, but only when the
                # label has to be *derived*. A supplied label needs no context,
                # and assembling one anyway doubled the cost of every fit --
                # unbounded, when the target names no window.
                label_ctx = (None if labels is not None
                             else self.assemble_context(
                                 entity_table, eid,
                                 None if span is None else t + span,
                                 query=pq))
                if task_type is TaskType.MULTILABEL_RANKING:
                    parent = backend.ranking_parent_table(pq)
                    cands = backend._rank_candidates(
                        parent, TemporalBound.at_or_before(t) if t
                        else TemporalBound.unbounded())
                    if not cands:
                        continue
                    rel = self._ranking_relevance(pq, label_ctx, t, cands,
                                                  labels, eid)
                    if not any(r > 0 for r in rel):
                        # listwise cross-entropy needs a positive in the group;
                        # an entity that interacted with nothing in the window
                        # carries no ranking signal, so it is not an example.
                        skipped += 1
                        continue
                    ranking_examples.append((ctx, parent, cands, rel))
                    ys.extend(rel)
                    groups.append(len(ys))
                else:
                    y = self._scalar_label(pq, task_type, label_ctx, t,
                                           labels, eid, classes)
                    if y is None:
                        continue
                    scalar_examples.append((ctx, float(y)))
                    ys.append(float(y))

        if not ys:
            extra = (f" ({skipped} ranking groups had no positive relevance in "
                     f"the target window)" if skipped else "")
            raise ExecutionError(
                f"task-head fitting produced no training examples — check the anchors "
                f"and that the cohort resolves at them{extra}")
        if skipped:
            warnings.warn(
                f"task-head fitting skipped {skipped} ranking group(s) with no "
                f"positive relevance in the target window; listwise loss needs "
                f"at least one relevant candidate per group",
                UserWarning, stacklevel=2)
        task_spec = backend.task_spec(pq, task_type)
        if normalization_mode is NormalizationMode.REFERENCE:
            # A derived target is materialized as a real task column by the
            # reference pipeline, and its transform is persisted alongside
            # physical column transforms. Ranking relevance is binary and
            # deliberately keeps the identity scale.
            task_values = ([y for _, y in scalar_examples]
                           if scalar_examples else [0.0, 1.0])
            backend.column_stats = backend.column_stats.with_task_values(
                task_spec, task_values)

        feats: list = []
        for ctx, _ in scalar_examples:
            seqs, _, _ = backend._build_sequences(
                pq, task_type, [ctx], normalization_mode=normalization_mode,
                task_spec=task_spec)
            feats.append(backend._encode(model, seqs))
        for ctx, parent, cands, _ in ranking_examples:
            seqs = backend.candidate_seqs(
                pq, ctx, parent, cands,
                normalization_mode=normalization_mode)
            feats.append(backend._encode(model, seqs))
        features = np.concatenate(feats, axis=0).astype(np.float32)
        y = np.asarray(ys, np.float32)
        n_outputs = len(classes) if task_type is TaskType.MULTICLASS_CLASSIFICATION else 1
        if task_type is TaskType.MULTICLASS_CLASSIFICATION and n_outputs < 2:
            raise ExecutionError(
                f"multiclass task-head fitting needs at least two observed classes, "
                f"saw {n_outputs}")
        group_off = (np.asarray(groups, np.int32)
                     if task_type is TaskType.MULTILABEL_RANKING
                     else np.zeros(1, np.int32))
        n_groups = (len(groups) - 1
                    if task_type is TaskType.MULTILABEL_RANKING else 0)
        return backend.fit_head(
            model, task_type, features, y, group_off, n_groups,
            epochs=epochs, learning_rate=learning_rate,
            weight_decay=weight_decay, classes=classes,
            normalization_mode=normalization_mode)

    def _scalar_label(self, pq, task_type, label_ctx, t, labels, eid, classes):
        """The outcome the query asks about, as it actually turned out."""
        if labels is not None and (eid, t) in labels:
            v = labels[(eid, t)]
        else:
            rows = label_ctx.rows_by_table()
            cells = label_ctx.entity_cells(pq.entity_key.table)
            if task_type is TaskType.BINARY_CLASSIFICATION:
                v = 1.0 if eval_bool(pq.target, rows, cells, t) else 0.0
            else:
                v = eval_value(pq.target, rows, cells, t)
        if task_type is TaskType.MULTICLASS_CLASSIFICATION:
            if v is None:
                return None
            if v not in classes:
                classes.append(v)
            return float(classes.index(v))
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if not isinstance(v, (int, float)):
            return None
        return float(v)

    def _ranking_relevance(self, pq, label_ctx, t, candidates, labels, eid):
        """Per-candidate relevance: which candidate ids actually turned up in
        the target's window after the anchor."""
        if labels is not None and (eid, t) in labels:
            given = labels[(eid, t)] or {}
            return [float(given.get(c.id, 0.0)) for c in candidates]
        observed = eval_value(pq.target, label_ctx.rows_by_table(),
                              label_ctx.entity_cells(pq.entity_key.table), t)
        seen = {str(x) for x in (observed or [])}
        return [1.0 if str(c.id) in seen else 0.0 for c in candidates]

    def explain(self, input: Union[ExecutionInput, str], **kwargs) -> ExplainResult:
        """Explain a query without (PLAN/CONTEXT) or with (ANALYZE) scoring.
        A non-EXPLAIN query is explained as PLAN by default."""
        if isinstance(input, str):
            input = ExecutionInput(query=input, **kwargs)
        pq = (parse(input.query) if isinstance(input.query, str)
              else input.query)
        pq = validate(pq, self.schema).query   # binds the population's pk
        pq = pq.bind_params(input.params)      # substitutes :name -> values
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
        # entity selector: the cohort a pinned primary key names, else the
        # whole table
        pinned = _pinned_ids(pq.where, pq.entity_key)
        selector = pinned if pinned is not None else "ALL"
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
            # surfaces as an error at execute() if it cannot be applied
            try:
                _assumptions(pq.assuming)
            except ExecutionError as e:
                warnings_.append(str(e))

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
            "assuming": _assuming_plan(pq.assuming),
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
        assumed = _assumptions(pq.assuming) if pq.assuming is not None else []
        contexts: list[EntityContext] = []
        for eid in ids:
            anchor = self._anchor_for(entity_table, eid, eff_input)
            ctx = self.assemble_context(entity_table, eid, anchor, query=pq)
            if pq.where is not None and not self._where_ok(pq, ctx, entity_table):
                continue
            contexts.append(_apply_assumptions(assumed, ctx))

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
