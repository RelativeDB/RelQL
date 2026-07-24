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
import os
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, Union

from . import strategies as _strategies
from . import training as _training
from .anchors import coerce_anchor, effective_anchor, parse_anchor_date
from .csc import CscIndex
from .errors import ExecutionError
from .evaluate import eval_bool
from .model import ModelConfig
from .plan import QueryPlan, assumptions, build_plan, pinned_ids, pure_pin
from .relql.ast import AggFunc, Explain, ParsedQuery, TaskType
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

    # Geometry follows the reference evaluator (eval_utils.build_evaluator);
    # the default cell budget is 2048 — pass max_context_cells=8192 to match
    # the reference evaluation context size.
    max_context_cells: int = 2048
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
    # Score the whole cohort inside ONE shared context: every entity's target
    # is masked in the same sequence and read from its own token, so cohort
    # members never see each other's outcomes (strictly less exposure than
    # per-entity scoring) and a cohort costs one forward instead of N.
    # Scalar tasks with a shared anchor and a key-pinned WHERE only.
    shared_context: bool = False
    # Zero-inflation hurdle for regression targets: the engine also scores
    # the derived existence question (target > 0, RETURN PROBABILITY) on the
    # same cohort and predicts exactly 0 wherever that probability falls
    # below this gate. The gate is a protocol hyperparameter — tune it on a
    # validation split, never on test.
    hurdle_gate: Optional[float] = None


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
        entity_table = pq.entity_key.table
        ids = self._resolve_entity_ids(pq, input)
        assumed = assumptions(pq.assuming) if pq.assuming is not None else []
        backend = self._require_backend()
        # One plan, built here and acted on below. EXPLAIN builds the same one
        # from the same inputs, so what it prints is what runs.
        plan = self._plan(pq, input, input.anchor_time, cohort_size=len(ids))

        # Which strategy runs, and how it batches, is the plan's decision --
        # see relativedb.strategies for the registry it selects from.
        return _strategies.run(self, _strategies.ExecutionRequest(
            pq=pq, input=input, plan=plan, entity_table=entity_table,
            ids=ids, assumed=assumed, backend=backend))

    def _execute_hurdle(self, pq: ParsedQuery, input: ExecutionInput,
                        task_type: TaskType, model_uri: str,
                        entity_table: str, ids: list[Any]) -> PredictionResult:
        """Zero-inflation hurdle: gate the regression prediction with the
        derived existence probability.

        Two passes over the same cohort: the regression query as written,
        and a synthesized ``<target> > 0 ... RETURN PROBABILITY`` variant.
        The final value is the regression prediction where the existence
        probability clears ``hurdle_gate``, else exactly 0 — the composition
        that lets a continuous head and a calibrated-ranking head jointly
        model a zero-inflated target."""
        from .relql.ast import Condition, Operator, ReturnSpec
        if task_type is not TaskType.REGRESSION:
            raise ExecutionError(
                "hurdle_gate applies to regression targets only")
        reg = self.execute(replace(input, query=pq, hurdle_gate=None))
        clf_pq = replace(
            pq,
            target=Condition(left=pq.target, op=Operator.GT, right=0),
            ret=ReturnSpec("PROBABILITY"))
        clf = self.execute(replace(input, query=clf_pq, hurdle_gate=None))
        prob = {p.id: p.probability for p in clf.predictions}
        gated = tuple(
            EntityPrediction(p.id,
                             value=(p.value
                                    if prob.get(p.id, 1.0) > input.hurdle_gate
                                    else 0.0))
            for p in reg.predictions)
        stats = dict(reg.stats)
        stats["hurdle_gate"] = input.hurdle_gate
        stats["hurdle_zeroed"] = sum(1 for p in gated if p.value == 0.0)
        return PredictionResult(task_type=task_type, predictions=gated,
                                model_uri=model_uri, stats=stats)

    def _prepare_shared(self, pq: ParsedQuery, input: ExecutionInput,
                        task_type: TaskType, model_uri: str,
                        entity_table: str, ids: list[Any], backend):
        """Python half of shared-context execution: validate, assemble the
        shared context, inject cohort rows, and build the collated payload.
        Returns ``("ok", payload, stats)``, ``("empty", None, None)``, or
        ``("fallback", None, None)`` when no shared state matches."""
        if task_type not in (TaskType.BINARY_CLASSIFICATION,
                             TaskType.REGRESSION):
            raise ExecutionError(
                "shared_context supports binary classification and "
                "regression targets")
        if pq.assuming is not None:
            raise ExecutionError("shared_context does not support ASSUMING")
        # per_entity_anchor cohorts are handled by the caller: it groups ids
        # whose anchors are exactly equal and prepares each group separately,
        # so within this call every id shares the seed's anchor.
        if pq.where is not None and not pure_pin(pq.where, pq.entity_key):
            raise ExecutionError(
                "shared_context supports WHERE only as a primary-key pin "
                "(pk = :id / pk IN :ids); other filters need per-entity "
                "scoring")
        if not ids:
            return ("empty", None, None)
        if (not hasattr(self.traversal, "cohort_targets")
                or not hasattr(backend, "prepare_shared")):
            raise ExecutionError(
                "shared_context requires the reference traversal and the "
                "native RT backend")
        seed = ids[0]
        anchor = self._anchor_for(entity_table, seed, input)
        ctx = self.assemble_context(entity_table, seed, anchor, query=pq)
        task_spec = backend.task_spec(pq, task_type)
        got = self.traversal.cohort_targets(
            entity_table, list(ids), anchor, task_spec,
            history=self.context_policy.num_history_windows)
        if got is None:
            return ("fallback", None, None)
        targets, inject, extra_node_ids = got
        # Guarantee cohort rows a place in the context: right after the focal
        # target, so the sequence budget truncates low-relevance tail rows
        # instead of them. Rows already present move to the front (a copy
        # left in the tail would be the one truncation cuts).
        front: list[Row] = []
        front_keys: set = set()
        for row in inject:
            if row.key not in front_keys:
                front.append(row)
                front_keys.add(row.key)
        if front:
            head = [r for r in ctx.rows[:1] if r.key not in front_keys]
            rest = [r for r in ctx.rows[1:] if r.key not in front_keys]
            ctx.rows = head + front + rest
        if extra_node_ids:
            ctx.node_ids.update(extra_node_ids)
        # Experiment knob: cap the shared context's cell count. The cohort
        # block sits at the front, so only the low-relevance tail is dropped.
        trim = int(os.environ.get("RELATIVEDB_SHARED_TRIM_CELLS", "0") or 0)
        if trim > 0:
            cells = 0
            kept: list[Row] = []
            for row in ctx.rows:
                kept.append(row)
                cells += len(row.cells) + (1 if row.timestamp is not None
                                           else 0)
                if cells >= trim:
                    break
            ctx.rows = kept
        payload = backend.prepare_shared(pq, task_type, ctx, targets,
                                         model_uri, self.model_config)
        stats = self._collect_stats(pq, task_type, [ctx])
        stats["shared_context_forwards"] = 1
        return ("ok", payload, stats)

    def _shared_anchor_groups(self, pq: ParsedQuery, input: ExecutionInput,
                              entity_table: str, ids: list[Any],
                              backend) -> list[list[Any]]:
        """Split a cohort into groups whose anchors are exactly equal, then
        into chunks whose injected block fits the context.

        With a shared anchor there is one anchor group. With per-entity
        anchors, entities whose anchors compare equal — e.g. every row of
        one event — legitimately share a context (first-seen order); anchors
        that differ by any amount stay separate.

        Chunk sizing is measured, not guessed: each member's injected rows
        (target, entity row, recent task history) are fetched once per
        anchor group and their actual cell cost caps the chunk so the
        injected block stays within a third of the context budget — wide
        entity tables previously filled most of the context with the cohort
        itself, starving the walk-selected rows (labeled peers included)
        that the prediction feeds on.
        """
        if not input.per_entity_anchor:
            groups = [ids]
        else:
            by_anchor: dict[Any, list[Any]] = {}
            for eid in ids:
                by_anchor.setdefault(
                    self._anchor_for(entity_table, eid, input), []).append(eid)
            groups = list(by_anchor.values())
        out: list[list[Any]] = []
        for group in groups:
            limit = self._cohort_chunk_limit(pq, input, entity_table, group,
                                             backend)
            out.extend(group[i:i + limit]
                       for i in range(0, len(group), limit))
        return out

    def _cohort_chunk_limit(self, pq: ParsedQuery, input: ExecutionInput,
                            entity_table: str, group: list[Any],
                            backend) -> int:
        """Members per shared-context chunk, from measured injection cost."""
        budget = max(1, self.context_policy.max_context_cells // 3)
        if len(group) <= 1:
            return 1
        try:
            seed = group[0]
            anchor = self._anchor_for(entity_table, seed, input)
            # Warms the traversal's shared state for this anchor; chunk 1
            # reuses the same cached build.
            self.assemble_context(entity_table, seed, anchor, query=pq)
            task_spec = backend.task_spec(pq, pq.task_type(self.schema))
            got = self.traversal.cohort_targets(
                entity_table, list(group), anchor, task_spec,
                history=self.context_policy.num_history_windows)
        except Exception:
            got = None
        if got is None:
            # No shared state: the caller will fall back to per-entity
            # scoring anyway; any split works.
            return max(1, budget // 32)
        targets, inject, _ = got
        cells = sum(len(r.cells) + (1 if r.timestamp is not None else 0)
                    for r in inject)
        avg = max(1.0, cells / max(1, len(targets)))
        return max(1, int(budget // avg))

    def _execute_shared(self, pq: ParsedQuery, input: ExecutionInput,
                        task_type: TaskType, model_uri: str,
                        entity_table: str, ids: list[Any],
                        backend) -> Optional[PredictionResult]:
        """Score the cohort in shared contexts (one forward per anchor group).

        Returns None when the traversal has no matching shared state, in
        which case the caller falls back to per-entity scoring.
        """
        preds: list[EntityPrediction] = []
        stats: dict = {"entities_scored": 0, "shared_context_forwards": 0}
        for group in self._shared_anchor_groups(pq, input, entity_table,
                                                ids, backend):
            tag, payload, group_stats = self._prepare_shared(
                pq, input, task_type, model_uri, entity_table, group, backend)
            if tag == "empty":
                continue
            if tag == "fallback":
                return None
            yhat = payload["model"].forward_tokens(
                device=backend._resolve_device(), **payload["kw"])[0]
            group_preds = backend.finish_shared(payload, yhat)
            preds.extend(group_preds)
            for key, value in group_stats.items():
                if isinstance(value, (int, float)):
                    stats[key] = stats.get(key, 0) + value
            stats["entities_scored"] += len(group_preds)
        return PredictionResult(task_type=task_type,
                                predictions=tuple(preds),
                                model_uri=model_uri, stats=stats)

    def execute_many(self, inputs) -> list[PredictionResult]:
        """Execute several inputs, overlapping each shared-context cohort's
        Python-side preparation (context assembly, masking, collation) with
        the previous cohort's model forward. Inputs that are not
        shared-context — or any preparation that cannot use the shared path —
        are executed serially through :meth:`execute`.
        """
        inputs = [ExecutionInput(query=i) if isinstance(i, str) else i
                  for i in inputs]
        backend = self.model_backend
        if (backend is None or not hasattr(backend, "prepare_shared")
                or len(inputs) < 2
                or not all(i.shared_context for i in inputs)):
            return [self.execute(i) for i in inputs]

        import queue as _queue
        import threading
        feed: _queue.Queue = _queue.Queue(maxsize=2)
        stop = threading.Event()

        def resolve(inp):
            pq = (parse(inp.query) if isinstance(inp.query, str)
                  else inp.query)
            if pq.explain is not None:
                return None
            pq = validate(pq, self.schema).query.bind_params(inp.params)
            inp = replace(inp, anchor_time=self._effective_anchor(pq, inp))
            task_type = pq.task_type(self.schema)
            model_uri = self.model_config.model_uri_for(task_type)
            ids = self._resolve_entity_ids(pq, inp)
            return pq, inp, task_type, model_uri, pq.entity_key.table, ids

        def produce():
            try:
                for idx, inp in enumerate(inputs):
                    if stop.is_set():
                        return
                    resolved = resolve(inp)
                    if resolved is None:
                        feed.put(("serial", idx, None, None, None))
                        continue
                    pq, inp, task_type, model_uri, table, ids = resolved
                    prepared = []
                    fell_back = False
                    for group in self._shared_anchor_groups(
                            pq, inp, table, ids, self.model_backend):
                        tag, payload, stats = self._prepare_shared(
                            pq, inp, task_type, model_uri, table, group,
                            self.model_backend)
                        if tag == "fallback":
                            fell_back = True
                            break
                        if tag == "ok":
                            prepared.append((payload, stats))
                    if fell_back:
                        feed.put(("serial", idx, None, None, None))
                    elif not prepared:
                        feed.put(("empty", idx, None, None,
                                  (task_type, model_uri)))
                    else:
                        feed.put(("ok", idx, prepared, None,
                                  (task_type, model_uri)))
                feed.put(("done", None, None, None, None))
            except BaseException as error:
                feed.put(("error", error, None, None, None))

        results: list[Optional[PredictionResult]] = [None] * len(inputs)
        serial: list[int] = []
        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        try:
            while True:
                tag, idx, payload, stats, meta = feed.get()
                if tag == "error":
                    raise idx
                if tag == "done":
                    break
                if tag == "serial":
                    serial.append(idx)
                    continue
                task_type, model_uri = meta
                if tag == "empty":
                    results[idx] = PredictionResult(
                        task_type=task_type, predictions=(),
                        model_uri=model_uri, stats={"entities_scored": 0})
                    continue
                preds: list[EntityPrediction] = []
                merged: dict = {"entities_scored": 0,
                                "shared_context_forwards": 0}
                for group_payload, group_stats in payload:
                    yhat = group_payload["model"].forward_tokens(
                        device=backend._resolve_device(),
                        **group_payload["kw"])[0]
                    group_preds = backend.finish_shared(group_payload, yhat)
                    preds.extend(group_preds)
                    for key, value in group_stats.items():
                        if isinstance(value, (int, float)):
                            merged[key] = merged.get(key, 0) + value
                    merged["entities_scored"] += len(group_preds)
                results[idx] = PredictionResult(
                    task_type=task_type, predictions=tuple(preds),
                    model_uri=model_uri, stats=merged)
        finally:
            stop.set()
            while producer.is_alive():
                try:
                    feed.get_nowait()
                except _queue.Empty:
                    producer.join(timeout=0.05)
        # Inputs the shared path could not serve run serially afterwards,
        # with the producer thread quiesced (traversal state is not
        # thread-safe against concurrent assembly).
        for idx in serial:
            results[idx] = self.execute(inputs[idx])
        return results

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
        pinned = pinned_ids(pq.where, pq.entity_key)
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
    # The rules live in relativedb.anchors (pure, no engine state). These stay
    # as methods because callers and subclasses already use them.
    def _effective_anchor(self, pq: ParsedQuery,
                          input: ExecutionInput) -> Optional[datetime]:
        return effective_anchor(pq.as_of, input.anchor_time, input.params)

    _coerce_anchor = staticmethod(coerce_anchor)
    _parse_anchor_date = staticmethod(parse_anchor_date)

    # -- EXPLAIN ------------------------------------------------------------
    # -- task-head adaptation ----------------------------------------------
    # -- task-head adaptation ----------------------------------------------
    # Implemented in relativedb.training; these stay methods so the public
    # API (engine.fit_head(...), engine.finetune(...)) is unchanged.
    def finetune(self, *args, **kwargs):
        """Finetune the full model on labelled anchors. See
        :func:`relativedb.training.finetune`."""
        return _training.finetune(self, *args, **kwargs)

    def fit_head(self, *args, **kwargs):
        """Fit a task head on labelled anchors. See
        :func:`relativedb.training.fit_head`."""
        return _training.fit_head(self, *args, **kwargs)

    def _scalar_label(self, *args, **kwargs):
        return _training._scalar_label(self, *args, **kwargs)

    def _ranking_relevance(self, *args, **kwargs):
        return _training._ranking_relevance(self, *args, **kwargs)

    def _sampler_kind(self) -> str:
        """Which sampler _sampler() will hand back, as a plan-facing name."""
        if isinstance(self.traversal, ReferenceTraversal):
            return "csc"
        return "csc" if self.sampler_mode is SamplerMode.CSC else "retriever"

    def _plan(self, pq: ParsedQuery, input: ExecutionInput,
              effective: Optional[datetime], *,
              cohort_size: Optional[int] = None) -> QueryPlan:
        """Build the plan for one query. Both execute() and explain() go
        through here, which is what keeps EXPLAIN honest.

        ``cohort_size`` is passed by execute(), which has already resolved the
        entity ids. explain() leaves it None unless the WHERE pins the cohort,
        because enumerating a table to describe a query would be a surprising
        cost -- the plan reports those fields as unknown instead of guessing.
        """
        task_type = pq.task_type(self.schema)
        backend = self.model_backend
        batch = getattr(backend, "batch_size", None) if backend else None
        return build_plan(
            self.schema, pq, input,
            effective=effective,
            traversal=self.traversal,
            sampler=self._sampler_kind(),
            model_uri=self.model_config.model_uri_for(task_type),
            cohort_size=cohort_size,
            scoring_batch_size=batch,
        )

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

        # The same planner execute() uses, so EXPLAIN describes the run that
        # would actually happen rather than a separately-derived description.
        plan = self._plan(pq, input, effective).to_dict()
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

    def _assemble_report(self, pq: ParsedQuery, input: ExecutionInput,
                         effective: Optional[datetime]):
        """Assemble per-entity context via the normal path (no scoring) and
        summarize it (contract: EXPLAIN CONTEXT)."""
        eff_input = replace(input, anchor_time=effective)
        entity_table = pq.entity_key.table
        ids = self._resolve_entity_ids(pq, eff_input)
        assumed = assumptions(pq.assuming) if pq.assuming is not None else []
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
