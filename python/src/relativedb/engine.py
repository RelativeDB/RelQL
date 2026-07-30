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
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, Union

from . import strategies as _strategies
from . import training as _training
from .anchors import coerce_anchor, effective_anchor, parse_anchor_date
from .csc import CscIndex
from .errors import ExecutionError
from .evaluate import eval_bool
from .model import ModelConfig
from .plan import (QueryPlan, aggregate_assumptions, assumptions, build_plan,
                   pinned_ids, pure_pin)
from .relql.ast import (Ablation, AggFunc, Explain, Operator, ParsedQuery,
                        TaskType, _find_aggregations)
from .relql.parser import parse, validate
from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import Schema, TableDef, ValueType
from .traversal import (BreadthFirstTraversal, GraphTraversal,
                        ReferenceTraversal)

__all__ = [
    "SamplerMode", "ContextPolicy", "ExecutionInput", "EntityContext",
    "EntityPrediction", "PredictionResult", "ExplainResult", "ModelBackend",
    "Engine", "ExecutionError",
    "ContextTruncationWarning", "AssumptionNotAppliedWarning",
    "ContextCompositionWarning", "InvisibleTableWarning",
    "GraphTraversal", "BreadthFirstTraversal", "ReferenceTraversal",
]

# Aggregations whose value grows with the number of rows in the window, and so
# are biased low when the fanout cap drops children. (FIRST/LAST/MIN/MAX and
# the DISTINCT/LIST variants are far less sensitive to a dropped tail.)
_COUNTING_AGGS = frozenset({AggFunc.COUNT, AggFunc.SUM, AggFunc.AVG})


def _apply_assumptions(assignments: list[tuple[str, str, Any]],
                       ctx: "EntityContext",
                       entity_table: Optional[str] = None) -> "EntityContext":
    """A copy of ``ctx`` in which the assumed values hold.

    An assignment on the **entity table** intervenes on the entity's own row
    only: ``ASSUMING c.plan = 'premium'`` changes *this* customer, while any
    sibling rows of the same table stay factual. That is more than semantics —
    zero-shot normalization derives column statistics inside each context, so
    overwriting every row of the table flattens the column to a constant, and
    a constant numeric column normalizes to zero *regardless of the constant*:
    the assumed value would be erased before it reached the model. Keeping the
    siblings factual is what preserves the scale the assumed value is measured
    against.

    An assignment on any other table keeps the broad meaning: every context
    row of that table gets the value — assuming ``orders.status = 'shipped'``
    means *these* orders are shipped.

    Rows are replaced rather than mutated: retrievers and the CSC index hand
    out shared Row objects, so writing through one would corrupt every other
    entity's context.
    """
    if not assignments:
        return ctx
    by_table: dict[str, dict[str, Any]] = {}
    for table, col, value in assignments:
        by_table.setdefault(table, {})[col] = value
    entity_key = ((entity_table, ctx.entity_id)
                  if entity_table is not None else None)

    def patched(r):
        patch = by_table.get(r.table)
        if patch is None:
            return r
        if r.table == entity_table and r.key != entity_key:
            return r                              # siblings stay factual
        return replace(r, cells={**r.cells, **patch})

    rows = [patched(r) for r in ctx.rows]
    return EntityContext(entity_id=ctx.entity_id, anchor=ctx.anchor, rows=rows,
                         truncated_children=ctx.truncated_children,
                         hit_cell_budget=ctx.hit_cell_budget,
                         focal_row_keys=ctx.focal_row_keys,
                         node_ids=dict(ctx.node_ids))


def _apply_ablations(ablations, ctx: "EntityContext") -> "EntityContext":
    """A copy of ``ctx`` with every row of each ablated table removed —
    ``ABLATE TABLE t`` scores the query as if ``t`` did not exist. The engine
    rejects ablating the entity table before this runs, so the entity's own
    row is never at risk. Rows are dropped, not blanked: a blanked row would
    still spend context budget and carry its FK structure."""
    if not ablations:
        return ctx
    drop = {a.name for a in ablations}
    rows = [r for r in ctx.rows if r.table not in drop]
    return EntityContext(entity_id=ctx.entity_id, anchor=ctx.anchor, rows=rows,
                         truncated_children=ctx.truncated_children,
                         hit_cell_budget=ctx.hit_cell_budget,
                         focal_row_keys=frozenset(
                             k for k in ctx.focal_row_keys if k[0] not in drop),
                         node_ids=dict(ctx.node_ids))


def _apply_aggregate_assumptions(agg_conds, ctx: "EntityContext",
                                 entity_table: str,
                                 schema: Optional[Schema],
                                 fetch_template=None) -> "EntityContext":
    """Reshape the context so an ``ASSUMING COUNT(t.*) OVER (...) >= k``
    (or EXISTS / NOT EXISTS) bound holds for the entity.

    The counterfactual is realized structurally: the entity's own in-window
    rows of ``t`` are counted, and the context gains cloned copies of its
    most recent row (re-timestamped inside the window) or loses its oldest
    rows until the bound is met. Clones inherit the template's cells, so the
    synthetic history looks like "more of the same" rather than empty rows
    the model cannot see. Every unsatisfiable case raises — a counterfactual
    that silently did not hold would poison the comparison it exists for.
    """
    from .evaluate import _rows_in_window, _window_bounds

    if not agg_conds:
        return ctx
    rows = list(ctx.rows)
    focal = set(ctx.focal_row_keys)
    for n, (agg, op, bound) in enumerate(agg_conds):
        table = agg.column.table
        if agg.func not in (AggFunc.COUNT, AggFunc.EXISTS):
            raise ExecutionError(
                f"ASSUMING supports COUNT and EXISTS bounds; "
                f"{agg.func.name} describes a value the engine cannot build "
                f"rows for")
        if agg.filter is not None:
            raise ExecutionError(
                "ASSUMING a filtered COUNT(t.* WHERE ...) is not supported: "
                "synthesized rows would have to satisfy the filter. Assume "
                "the plain count, or assign the filtered column explicitly.")
        if table == entity_table:
            raise ExecutionError(
                f"cannot assume a count over the entity table {table!r}: "
                f"each context holds this one entity, not a population of "
                f"them")
        if float(bound) != int(bound) or bound < 0:
            raise ExecutionError(
                f"ASSUMING COUNT(...) needs a non-negative integer bound, "
                f"got {bound!r}")
        bound = int(bound)

        mine = [r for r in rows if r.table == table
                and (not focal or r.key in focal)]
        col = agg.column.column
        counted = [r for r in _rows_in_window(mine, agg.window, ctx.anchor)
                   if col == "*" or r.cells.get(col) is not None]
        c = len(counted)
        target = {Operator.GE: max(c, bound),
                  Operator.GT: max(c, bound + 1),
                  Operator.EQ: bound,
                  Operator.LE: min(c, bound),
                  Operator.LT: min(c, bound - 1)}[op]
        if target < 0:
            raise ExecutionError(
                f"ASSUMING COUNT(...) {op.value} {bound} demands a negative "
                f"count")

        if target > c:
            # newest in-window row, else the entity's newest row of the
            # table, else any context row of the table re-parented to the
            # entity, else a row fetched from the database — the clone must
            # carry real cells to be visible.
            def usable(candidates):
                return [r for r in candidates
                        if col == "*" or r.cells.get(col) is not None]
            pool = (usable(counted) or usable(mine)
                    or usable(r for r in rows if r.table == table))
            if not pool and fetch_template is not None:
                pool = usable(fetch_template(table))
            if not pool:
                raise ExecutionError(
                    f"ASSUMING a count on {table!r} needs at least one row "
                    f"of that table to clone — none in the context or the "
                    f"database (an empty synthetic row would be invisible "
                    f"to the model)")
            template = max(pool, key=lambda r: (r.timestamp is not None,
                                                r.timestamp or ctx.anchor))
            lo = hi = None
            if agg.window is not None and ctx.anchor is not None:
                lo, hi = _window_bounds(agg.window, ctx.anchor)
            base_ts = hi or ctx.anchor or template.timestamp
            fk = next((l.fk_column for l in (schema.links_from(table)
                                             if schema else [])
                       if l.to_table == entity_table), None)
            tdef = schema.table(table) if schema else None
            for i in range(1, target - c + 1):
                ts = (base_ts - timedelta(seconds=i)
                      if base_ts is not None else None)
                if ts is not None and lo is not None and ts <= lo:
                    raise ExecutionError(
                        f"ASSUMING COUNT window on {table!r} is too narrow "
                        f"to hold {target} distinct rows")
                cells = dict(template.cells)
                if tdef is not None and tdef.time_column in cells:
                    cells[tdef.time_column] = ts
                parents = dict(template.parents)
                if fk is not None:
                    parents[fk] = ctx.entity_id
                clone = Row(table, f"__assumed__:{table}:{n}:{i}",
                            cells, ts, parents)
                rows.append(clone)
                if focal:
                    focal.add(clone.key)
        elif target < c:
            # keep the newest `target` in-window rows; drop the oldest
            drop = {r.key for r in counted[:c - target]}
            rows = [r for r in rows if r.key not in drop]
            focal -= drop

    return EntityContext(entity_id=ctx.entity_id, anchor=ctx.anchor,
                         rows=rows,
                         truncated_children=ctx.truncated_children,
                         hit_cell_budget=ctx.hit_cell_budget,
                         focal_row_keys=frozenset(focal),
                         node_ids=dict(ctx.node_ids))


def _warn_inert_assumptions(assignments: list[tuple[str, str, Any]],
                            contexts: list["EntityContext"],
                            entity_table: Optional[str] = None,
                            schema: Optional[Schema] = None) -> None:
    """Say it when an ASSUMING assignment cannot reach the model.

    Three ways an assumption goes silently inert, each reported rather than
    swallowed:

    * the assigned table never appears in any context;
    * the assignment targets the entity table but the entity's own row is
      missing from the context (only siblings made it in);
    * the assigned column is numeric/boolean and ends up **constant** within
      every context — zero-shot normalization maps a constant column to zero,
      erasing which constant it was.
    """
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

    if entity_table is not None:
        entity_assigned = [c for t, c, _ in assignments if t == entity_table]
        if entity_assigned:
            absent = sum(
                1 for c in contexts
                if not any(r.key == (entity_table, c.entity_id)
                           for r in c.rows))
            if absent:
                warnings.warn(
                    f"ASSUMING on {entity_table}.{entity_assigned[0]}: the "
                    f"entity's own row is missing from {absent} of "
                    f"{len(contexts)} contexts, so the assumption cannot be "
                    f"applied there", AssumptionNotAppliedWarning,
                    stacklevel=3)

    for table, col, _ in assignments:
        if schema is not None:
            tdef = schema.table(table)
            cdef = tdef.column(col) if tdef is not None else None
            if cdef is None or cdef.type not in (ValueType.NUMBER,
                                                 ValueType.BOOLEAN):
                continue                # text embeds; datetime pools globally
        flattened = 0
        populated = 0
        for c in contexts:
            vals = {r.cells.get(col) for r in c.rows
                    if r.table == table and col in r.cells}
            if vals:
                populated += 1
                if len(vals) == 1:
                    flattened += 1
        if populated and flattened == populated:
            warnings.warn(
                f"ASSUMING leaves {table}.{col} constant across the context "
                f"in all {populated} contexts; zero-shot normalization maps a "
                f"constant numeric column to zero, so the assumed value is "
                f"erased before it reaches the model. Keep factual sibling "
                f"rows of {table} in the context, or use reference "
                f"normalization (column_stats)",
                AssumptionNotAppliedWarning, stacklevel=3)


class AssumptionNotAppliedWarning(UserWarning):
    """An ASSUMING assignment could not (fully) reach the model input."""


# Loud-degradation thresholds for assembled contexts. A single truncated
# context is routine; a cohort where most contexts truncate, or where one
# table swallows the budget, is a schema problem the caller should hear about
# once, at the end, instead of piecing it together from per-entity warnings.
_TRUNCATION_WARN_SHARE = 0.5
_DOMINANCE_WARN_SHARE = 0.6
# Below this many cells across the cohort, dominance says nothing: tiny test
# fixtures and single-entity probes are trivially lopsided.
_DOMINANCE_MIN_CELLS = 256


class ContextCompositionWarning(UserWarning):
    """Assembled contexts truncated en masse or dominated by one table."""


class InvisibleTableWarning(UserWarning):
    """A schema table whose rows serialize to zero feature tokens.

    The serialization contract emits one token per feature cell, and a row's
    foreign-key links ride on the tokens it emits — so a table of pure FK
    rows (a follows/likes edge table with no payload columns) enters the
    context, costs nothing against the budget, and never reaches the model
    at all. The connection the row exists to represent is silently lost.
    """


def _invisible_fix_hint(table: str, schema: Schema) -> str:
    links = [l.fk_column for l in schema.links_from(table)]
    if links:
        return (f"If a key itself carries signal (a handle, a name, a "
                f"category), set feature_type on that link — e.g. "
                f"LinkDef({table!r}, {links[0]!r}, ..., "
                f"feature_type=ValueType.TEXT) — so its value is emitted as "
                f"a feature token. Otherwise add a payload column or "
                f"summarize the edges into a table with measured columns.")
    return "Add a payload column so its rows have something to emit."


def _warn_invisible_tables(schema: Schema) -> None:
    """Statically provable at construction: no non-key columns and no
    feature-typed link means every row of the table emits zero tokens."""
    for tbl in schema.tables:
        if any(c.name != tbl.primary_key for c in tbl.columns):
            continue
        if any(l.feature_type is not None for l in schema.links_from(tbl.name)):
            continue
        warnings.warn(
            f"every row of table {tbl.name!r} will serialize to zero tokens: "
            f"it has no non-key columns and none of its links sets "
            f"feature_type. Its rows can enter the context but the model "
            f"never sees them — or the link structure they carry. "
            f"{_invisible_fix_hint(tbl.name, schema)}",
            InvisibleTableWarning, stacklevel=3)


def _emitted_cells(r: Row, tdef: "TableDef",
                   fk_feature_cols: Sequence[str]) -> int:
    """Feature tokens this row actually contributes to the model input:
    declared non-key non-null cells plus feature-typed FK values. A bare
    row timestamp is not emitted for database rows and does not count."""
    n = sum(1 for col, v in r.cells.items()
            if col != tdef.primary_key and tdef.column(col) is not None
            and v is not None)
    return n + sum(1 for fk in fk_feature_cols
                   if r.parents.get(fk) is not None)


def _warn_context_health(contexts: list["EntityContext"],
                         schema: Optional[Schema],
                         policy: "ContextPolicy") -> None:
    """One aggregate report on what actually reached the model.

    Found the hard way: an experiment ran for hours with 100% of contexts
    truncated and one event table holding 69% of every context's cells — the
    model never saw the columns that mattered, and nothing said so except a
    per-entity warning stream nobody reads at cohort scale. This is the
    end-of-run summary that makes those two failure shapes loud.
    """
    if not contexts:
        return
    hit = sum(1 for c in contexts if c.hit_cell_budget)
    if hit and hit / len(contexts) >= _TRUNCATION_WARN_SHARE:
        warnings.warn(
            f"{hit} of {len(contexts)} contexts hit the cell budget "
            f"(max_context_cells={policy.max_context_cells}); the tail of "
            f"every truncated context never reached the model. Raise the "
            f"budget or reshape the schema — EXPLAIN CONTEXT shows where "
            f"the cells go", ContextCompositionWarning, stacklevel=3)

    fk_features = ({tbl.name: [l.fk_column
                               for l in schema.links_from(tbl.name)
                               if l.feature_type is not None]
                    for tbl in schema.tables} if schema is not None else {})
    cells: dict[str, int] = {}
    rows_seen: dict[str, int] = {}
    total = 0
    for c in contexts:
        for r in c.rows:
            # Virtual task rows (task_*) are engine-generated; dominance is a
            # statement about the USER's schema, so only its tables count.
            if schema is not None:
                tdef = schema.table(r.table)
                if tdef is None:
                    continue
                n = _emitted_cells(r, tdef, fk_features.get(r.table, ()))
            else:
                n = len(r.cells) + (1 if r.timestamp is not None else 0)
            cells[r.table] = cells.get(r.table, 0) + n
            rows_seen[r.table] = rows_seen.get(r.table, 0) + 1
            total += n
    if schema is not None:
        for table, nrows in rows_seen.items():
            if cells.get(table, 0) == 0:
                warnings.warn(
                    f"table {table!r} contributed {nrows} context row(s) but "
                    f"zero feature cells — none of it reached the model "
                    f"(all-FK rows, undeclared columns, or all-null values). "
                    f"{_invisible_fix_hint(table, schema)}",
                    ContextCompositionWarning, stacklevel=3)
    if total >= _DOMINANCE_MIN_CELLS and len(cells) > 1:
        table, top = max(cells.items(), key=lambda kv: kv[1])
        if top / total >= _DOMINANCE_WARN_SHARE:
            warnings.warn(
                f"table {table!r} holds {top / total:.0%} of all context "
                f"cells; one table is crowding the rest of the schema out of "
                f"the model's context. Summarize high-cardinality event "
                f"tables into period tables whose rows carry several "
                f"meaningful columns, or cap that link's fanout",
                ContextCompositionWarning, stacklevel=3)


class ProtocolFallbackWarning(UserWarning):
    """Execution silently took a different protocol than the one requested.

    These fallbacks change predictions -- a cohort scored per-entity is not the
    cohort scored in one shared context, and a different chunk split gives
    every member a different context. They used to happen quietly, which is how
    a run could be compared against a recording made under a different protocol
    and read as a regression. Set RELATIVEDB_STRICT=1 to make them raise.
    """


def _strict() -> bool:
    """RELATIVEDB_STRICT=1 turns a silent protocol fallback into an error.

    Benchmarks and conformance runs should set it: there, a fallback is a bug
    to find, not a degradation to absorb.
    """
    return os.environ.get("RELATIVEDB_STRICT", "").lower() in ("1", "true", "yes")


def _fallback(message: str) -> None:
    if _strict():
        raise ExecutionError(f"{message} (RELATIVEDB_STRICT=1)")
    warnings.warn(message, ProtocolFallbackWarning, stacklevel=3)



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
    # Members per shared-context chunk. None sizes it from the measured cost of
    # injecting the cohort's rows; 0 puts the whole cohort in one forward; a
    # positive value fixes the split. Chunking has no counterpart in the
    # reference implementation -- it scores each item in its own context -- so
    # this is a relativedb protocol knob, and how a cohort is split changes
    # every member's context and therefore its prediction.
    cohort_chunk: Optional[int] = None

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
        if self.cohort_chunk is not None and self.cohort_chunk < 0:
            raise ValueError("cohort_chunk cannot be negative (0 = one chunk)")

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
        """Raw cell tally for diagnostics. Schema-blind: it cannot see
        feature-typed FK values or declared-column filtering, so it is an
        approximation of what serialization emits, not a promise."""
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
    ``context`` is populated for CONTEXT/ANALYZE; ``predictions`` for
    ANALYZE (scored as written) and ABLATION (the baseline); ``ablation``
    is the EXPLAIN ABLATE impact ranking. ``render()`` produces the human
    TEXT or machine JSON form."""

    mode: str                        # PLAN | CONTEXT | ANALYZE | ABLATION
    format: str                      # TEXT | JSON
    plan: dict
    context: Optional[dict] = None
    predictions: Optional[tuple["EntityPrediction", ...]] = None
    ablation: Optional[dict] = None

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
        if self.ablation is not None:
            d["ablation"] = self.ablation
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
        if self.ablation is not None:
            a = self.ablation
            lines.append("")
            lines.append("ABLATION")
            base = a.get("baseline")
            lines.append(f"  metric:   {a.get('metric')}")
            if base:
                lines.append(f"  baseline: mean={base['mean']:.6g} "
                             f"n={base['n']}")
            lines.append(f"  forwards: {a.get('model_forwards')}")
            for e in a.get("candidates", []):
                lines.append(
                    f"  - [{e['kind']}] {e['name']}: "
                    f"mean_abs_delta={e['mean_abs_delta']:.6g} "
                    f"mean_delta={e['mean_delta']:+.6g} "
                    f"max_abs_delta={e['max_abs_delta']:.6g}")
            if a.get("note"):
                lines.append(f"  note: {a['note']}")
        if self.predictions is not None:
            lines.append("")
            lines.append("PREDICTIONS")
            for pr in self.predictions:
                lines.append(f"  - {_pred_to_dict(pr)}")
        return "\n".join(lines)


class ModelBackend(Protocol):
    """Anything that can score assembled contexts. The real backend
    (:class:`~relativedb_engine.RtNativeBackend`) loads the checkpoint at
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
                 model_backend: Optional[Union[ModelBackend, str]] = None,
                 context_policy: Optional[ContextPolicy] = None,
                 sampler_mode: SamplerMode = SamplerMode.RETRIEVER,
                 traversal: Optional[GraphTraversal] = None):
        self.schema = schema
        self.wiring = wiring
        if isinstance(model_backend, str):
            # A URL is the cloud backend: sequence assembly stays here,
            # embeddings + the forward run on the service (see .remote).
            from .remote import RemoteBackend
            model_backend = RemoteBackend(model_backend, schema=schema,
                                          wiring=wiring)
        _warn_invisible_tables(schema)
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

    def _check_ablations(self, pq: ParsedQuery) -> None:
        """ABLATE TABLE names must be real and must not be the entity table:
        dropping the entity's own row leaves nothing to predict about, and a
        typo'd table name would silently ablate nothing."""
        for ab in pq.ablations:
            if self.schema.table(ab.name) is None:
                raise ExecutionError(
                    f"ABLATE TABLE {ab.name!r}: unknown table")
            if ab.name == pq.entity_key.table:
                raise ExecutionError(
                    f"cannot ablate the entity table {ab.name!r}: the "
                    f"prediction is about its rows. Ablate the tables around "
                    f"it, or use EXPLAIN ABLATE to rank per-column impact.")

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
        if not result.rows:
            # A typo'd id, an int/str key mismatch, or a temporally
            # inadmissible row all land here — and used to be scored as a
            # normal prediction over an empty context, indistinguishable
            # from a real entity with no history.
            _fallback(f"entity {entity_table}={entity_id!r} assembled an "
                      f"EMPTY context (unknown id, key-type mismatch, or "
                      f"row not admissible at the anchor); scoring it would "
                      f"predict from nothing")
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
        # One trip into the C++ front end: validate, bind :params, infer the
        # task and build the logical plan. Binding there is what lets the plan
        # see a cohort pinned through `IN :ids`.
        _vq = validate(pq, self.schema, input.params or {})
        pq, _logical = _vq.query, _vq.plan
        # Bind the effective anchor (AS OF) before any assembly/scoring so it
        # threads through the temporal bound and pseudo-anchors unchanged.
        input = replace(input, anchor_time=self._effective_anchor(pq, input))
        entity_table = pq.entity_key.table
        self._check_ablations(pq)
        ids = self._resolve_entity_ids(pq, input)
        assumed = assumptions(pq.assuming) if pq.assuming is not None else []
        backend = self._require_backend()
        # One plan, built here and acted on below. EXPLAIN builds the same one
        # from the same inputs, so what it prints is what runs.
        plan = self._plan(pq, input, input.anchor_time, cohort_size=len(ids),
                          logical=_logical)

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
        # A derived query: built by AST surgery, not written by a user. Its
        # source text belongs to the regression query above, so it is cleared
        # -- re-analyzing that text would validate the wrong query and hand
        # back a regression task. `_task_type` is cleared for the same reason:
        # inherited from pq it would say regression, and this asks a binary
        # question. See validate()'s already-bound path.
        clf_pq = replace(
            pq,
            target=Condition(left=pq.target, op=Operator.GT, right=0),
            ret=ReturnSpec("PROBABILITY"),
            text="", _task_type=None)
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
        if pq.ablations:
            raise ExecutionError("shared_context does not support ABLATE; "
                                 "ablated queries score per-entity")
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
        # Chunk sizing just assembled this exact context (the first chunk's
        # seed is the group's seed); consume it instead of assembling twice.
        memo = getattr(self, "_seed_ctx_memo", None)
        if memo is not None and memo[0] == (entity_table, seed, anchor,
                                            id(pq)):
            ctx = memo[1]
            self._seed_ctx_memo = None
        else:
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
                              backend):
        """Yield cohort chunks: ids grouped by exactly-equal anchor, then
        split so each chunk's injected block fits the context.

        This is a GENERATOR on purpose: chunk sizing assembles the group
        seed's context and parks it in ``_seed_ctx_memo`` for the first
        chunk's ``_prepare_shared`` to pop, so sizing must interleave with
        preparation — a materialized list would run every sizing first and
        the memo would only survive for the last group.

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
        for group in groups:
            limit = self._cohort_chunk_limit(pq, input, entity_table, group,
                                             backend)
            for i in range(0, len(group), limit):
                yield group[i:i + limit]

    def _cohort_chunk_limit(self, pq: ParsedQuery, input: ExecutionInput,
                            entity_table: str, group: list[Any],
                            backend) -> int:
        """Members per shared-context chunk, from measured injection cost
        unless ``ContextPolicy.cohort_chunk`` fixes it."""
        configured = self.context_policy.cohort_chunk
        if configured is not None:
            return len(group) if configured <= 0 else configured
        budget = max(1, self.context_policy.max_context_cells // 3)
        if len(group) <= 1:
            return 1
        try:
            seed = group[0]
            anchor = self._anchor_for(entity_table, seed, input)
            # Warms the traversal's shared state for this anchor. The first
            # chunk's seed is this same entity, so park the assembled context
            # for _prepare_shared to pop — assembly was measured at half the
            # client-side cost of a cohort, and this halves it back.
            ctx = self.assemble_context(entity_table, seed, anchor, query=pq)
            self._seed_ctx_memo = (
                (entity_table, seed, anchor, id(pq)), ctx)
            task_spec = backend.task_spec(pq, pq.task_type(self.schema))
            got = self.traversal.cohort_targets(
                entity_table, list(group), anchor, task_spec,
                history=self.context_policy.num_history_windows)
        except Exception as exc:
            _fallback(f"shared-context chunk sizing failed ({type(exc).__name__}: "
                      f"{exc}); falling back to a guessed split, which changes "
                      f"every member's context")
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
            yhat = backend.scorer.forward(
                payload["batch"], model_uri=payload["model_uri"],
                output="token_scores").scores[0]
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

        # Forwards for different cohorts are independent, so they may run
        # concurrently: the service coalesces simultaneous requests into one
        # GPU batch, and overlapping them hides each request's CPU phases.
        # Serial scorers keep concurrency 1 (in-process forwards already own
        # the whole machine); remote scorers default to 4 in flight. The
        # semaphore bounds how many prepared payloads wait in the executor so
        # a fast producer cannot buffer every cohort in memory.
        conc = int(os.environ.get("RELATIVEDB_FORWARD_CONCURRENCY", "0") or 0)
        if conc <= 0:
            conc = 4 if hasattr(getattr(backend, "scorer", None), "url") else 1

        import concurrent.futures as _futures
        executor = (_futures.ThreadPoolExecutor(max_workers=conc)
                    if conc > 1 else None)
        inflight = threading.Semaphore(conc * 2)

        def score_group(group_payload):
            try:
                yhat = backend.scorer.forward(
                    group_payload["batch"],
                    model_uri=group_payload["model_uri"],
                    output="token_scores").scores[0]
                return backend.finish_shared(group_payload, yhat)
            finally:
                if executor is not None:
                    inflight.release()

        # (idx, task_type, model_uri, [(future-or-preds, group_stats), ...])
        scored: list[tuple] = []
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
                groups = []
                for group_payload, group_stats in payload:
                    if executor is not None:
                        inflight.acquire()
                        groups.append((executor.submit(
                            score_group, group_payload), group_stats))
                    else:
                        groups.append((score_group(group_payload),
                                       group_stats))
                scored.append((idx, task_type, model_uri, groups))
            for idx, task_type, model_uri, groups in scored:
                preds: list[EntityPrediction] = []
                merged: dict = {"entities_scored": 0,
                                "shared_context_forwards": 0}
                for got, group_stats in groups:
                    group_preds = (got.result() if executor is not None
                                   else got)
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
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
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
        # Truncation only distorts *count-like* aggregates (COUNT/SUM/AVG over
        # a window) in the TARGET. WHERE aggregations are evaluated against
        # the database (`_where_agg_rows`), so truncation cannot bias the
        # cohort. Warn once per execute when it actually bites.
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

    def _assumption_template_fetch(self, ctx: EntityContext,
                                   entity_table: str):
        """Database fallback for ASSUMING COUNT clone templates: the entity's
        own newest rows of the table, else the newest row anywhere in it."""
        def fetch(table: str) -> list[Row]:
            bound = (TemporalBound.at_or_before(ctx.anchor)
                     if ctx.anchor is not None else TemporalBound.unbounded())
            link = next((l for l in self.schema.links_from(table)
                         if l.to_table == entity_table), None)
            if link is not None:
                rows = list(self._sampler().children(link, ctx.entity_id,
                                                     bound, 64))
                if rows:
                    return rows
            scanner = self.wiring.scanners.get(table)
            if scanner is None:
                return []
            newest: Optional[Row] = None
            for r in scanner(table, bound):
                if newest is None or _newest_first_key(r) < _newest_first_key(newest):
                    newest = r
            return [newest] if newest is not None else []
        return fetch

    # WHERE aggregations are evaluated against the database, capped here so a
    # pathological fan-out fails loudly instead of silently saturating.
    _WHERE_FETCH_LIMIT = 1_000_000

    def _label_rows(self, pq: ParsedQuery, entity_table: str, eid: Any,
                    anchor: Optional[datetime]
                    ) -> tuple[dict[str, list[Row]], dict[str, Any]]:
        """Database-exact rows and entity cells for evaluating the target as
        GROUND TRUTH (training labels, ranking relevance).

        Labels used to be derived from an assembled context — a sample built
        for the model that carries peer entities' rows (a peer's event
        counted into this entity's label) and is fanout/budget truncated (a
        heavy entity's own events under-counted). A label is a fact: fetch
        the entity's rows through the wiring, bounded at each window's far
        edge (``end_delta``, not span — an offset window's tail was being
        cut off).
        """
        aggs = _find_aggregations(pq.target)
        ent = self._sampler().entities(entity_table, [eid],
                                       TemporalBound.unbounded())
        cells = dict(ent[0].cells) if ent else {}
        horizon: dict[str, Optional[datetime]] = {}
        for a in aggs:
            table = a.column.table
            if table == entity_table:
                continue
            if a.window is None or anchor is None:
                horizon[table] = None            # unbounded fetch
                continue
            end = a.window.end_delta()
            hi = None if end == timedelta.max else anchor + max(
                end, timedelta(0))
            prev = horizon.get(table, anchor)
            horizon[table] = (None if prev is None or hi is None
                              else max(prev, hi))
        out: dict[str, list[Row]] = {entity_table: list(ent)}
        sampler = self._sampler()
        for table, hi in horizon.items():
            link = next((l for l in self.schema.links_from(table)
                         if l.to_table == entity_table), None)
            if link is None:
                raise ExecutionError(
                    f"the target aggregates {table!r}, which has no direct "
                    f"link to {entity_table!r}; a derived label cannot "
                    f"attribute its rows to one entity. Supply labels "
                    f"explicitly.")
            bound = (TemporalBound.at_or_before(hi) if hi is not None
                     else TemporalBound.unbounded())
            rows = list(sampler.children(link, eid, bound,
                                         self._WHERE_FETCH_LIMIT))
            if len(rows) >= self._WHERE_FETCH_LIMIT:
                raise ExecutionError(
                    f"label fetch for {table!r} hit the row cap for entity "
                    f"{eid!r}; the derived label would be silently wrong")
            out[table] = rows
        return out, cells

    def _where_ok(self, pq: ParsedQuery, ctx: EntityContext,
                  entity_table: str,
                  anchor_override: Optional[datetime] = None) -> bool:
        # WHERE is a factual filter at the query's anchor. When
        # context_anchor_time decouples the context "now", the override keeps
        # WHERE at the true anchor instead of the context's.
        anchor = anchor_override or ctx.anchor
        cells = dict(ctx.entity_cells(entity_table))
        # The primary key is the row's identity, not one of its cells, so a
        # predicate on it (`WHERE t.pk IN :ids`) would otherwise see NULL and
        # drop every entity.
        pk = pq.entity_key.column
        if pk:
            cells.setdefault(pk, ctx.entity_id)
        return eval_bool(
            pq.where, self._where_agg_rows(pq, ctx, entity_table, anchor),
            cells, anchor, entity_table=entity_table)

    def _where_agg_rows(self, pq: ParsedQuery, ctx: EntityContext,
                        entity_table: str,
                        anchor: Optional[datetime] = None
                        ) -> dict[str, list[Row]]:
        """Database-exact rows for each table a WHERE aggregation touches.

        WHERE is a factual cohort filter, so its counts must be facts. They
        used to be computed over the assembled context, which is a *sample*
        built for the model — it carries peer entities' rows (so
        ``COUNT(orders.*) > 0`` passed for a customer who never ordered) and,
        depending on the task shape, can omit the entity's own children (so
        the same count read zero for a customer with orders). Fetching the
        entity's rows through the wiring answers from the data instead.
        Aggregations on tables with no direct link to the entity raise: a
        multi-hop count needs a join this filter does not do, and a wrong
        cohort is worse than a failed statement.
        """
        aggs = _find_aggregations(pq.where)
        for a in aggs:
            if a.window is not None and a.window.end > 0:
                raise ExecutionError(
                    f"WHERE window on {a.column.table!r} faces the future "
                    f"(FOLLOWING); a cohort filter is factual and its rows "
                    f"are fetched at or before the anchor, so this window "
                    f"can never match anything. Use a PRECEDING window.")
        tables = {a.column.table for a in aggs}
        if not tables:
            return {}
        anchor = anchor if anchor is not None else ctx.anchor
        bound = (TemporalBound.at_or_before(anchor)
                 if anchor is not None else TemporalBound.unbounded())
        sampler = self._sampler()
        out: dict[str, list[Row]] = {}
        for table in tables:
            if table == entity_table:
                out[table] = [r for r in ctx.rows
                              if r.key == (entity_table, ctx.entity_id)]
                continue
            link = next((l for l in self.schema.links_from(table)
                         if l.to_table == entity_table), None)
            if link is None:
                raise ExecutionError(
                    f"WHERE aggregates over {table!r}, which has no direct "
                    f"link to the entity table {entity_table!r}; the filter "
                    f"cannot attribute its rows to one entity")
            rows = list(sampler.children(link, ctx.entity_id, bound,
                                         self._WHERE_FETCH_LIMIT))
            if len(rows) >= self._WHERE_FETCH_LIMIT:
                raise ExecutionError(
                    f"WHERE aggregation over {table!r} hit the "
                    f"{self._WHERE_FETCH_LIMIT}-row fetch cap for entity "
                    f"{ctx.entity_id!r}; the count would be silently wrong")
            out[table] = rows
        return out

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
        if input.params and "ids" in input.params:
            # The caller clearly meant to score a cohort, but nothing in the
            # query consumes the binding. Scoring the whole table instead
            # once collated thousands of unrequested contexts into a single
            # forward and took the host to its memory limit. validate()
            # rejects this for text queries; this covers the bound-AST path.
            raise ExecutionError(
                f"params['ids'] was provided but the query never references "
                f":ids, which would score the whole {pq.entity_key.table!r} "
                f"table. Pin the cohort with WHERE {pq.entity_key.table}."
                f"{pq.entity_key.column} IN :ids")
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
            ts = rows[0].timestamp if rows else None
            if ts is not None:
                # An entity's own timestamp never overrides a declared AS OF
                # forward: that would score past the anchor the user pinned.
                return min(ts, anchor) if anchor is not None else ts
            if anchor is None:
                raise ExecutionError(
                    f"per_entity_anchor: entity {entity_id!r} has no dated "
                    f"row and no anchor_time was given — its context would "
                    f"be unbounded and read the future")
            _fallback(f"per_entity_anchor: entity {entity_id!r} has no "
                      f"dated row; falling back to the shared anchor "
                      f"{anchor.isoformat()}")
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

    def fit_xgboost(self, *args, **kwargs):
        """Fit an XGBoost model on labelled anchors — the third adaptation
        path next to :meth:`fit_head` and :meth:`finetune`, for queries the
        flat-feature planner accepts. Returns the fitted
        :class:`~relativedb.xgb.XgboostBackend`; assign it to
        ``model_backend`` to serve. See
        :func:`relativedb_engine.xgb.fit_xgboost`."""
        try:                                      # optional dependency
            from relativedb_engine import xgb as _xgb
        except ImportError as e:
            raise ExecutionError(
                "fit_xgboost runs in the native engine; install the optional "
                "packages: pip install 'relativedb-engine[xgboost]'") from e
        return _xgb.fit_xgboost(self, *args, **kwargs)

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
              cohort_size: Optional[int] = None,
              logical: Optional[dict] = None) -> QueryPlan:
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
            logical=logical,
        )

    def explain(self, input: Union[ExecutionInput, str], **kwargs) -> ExplainResult:
        """Explain a query without (PLAN/CONTEXT) or with (ANALYZE) scoring.
        A non-EXPLAIN query is explained as PLAN by default."""
        if isinstance(input, str):
            input = ExecutionInput(query=input, **kwargs)
        pq = (parse(input.query) if isinstance(input.query, str)
              else input.query)
        # One trip into the C++ front end: validate, bind :params, infer the
        # task and build the logical plan. Binding there is what lets the plan
        # see a cohort pinned through `IN :ids`.
        _vq = validate(pq, self.schema, input.params or {})
        pq, _logical = _vq.query, _vq.plan
        ex = pq.explain or Explain()
        mode = (ex.mode or "PLAN").upper()
        fmt = (ex.format or "TEXT").upper()
        effective = self._effective_anchor(pq, input)

        # The same planner execute() uses, so EXPLAIN describes the run that
        # would actually happen rather than a separately-derived description.
        self._check_ablations(pq)
        plan = self._plan(pq, input, effective, logical=_logical).to_dict()
        context: Optional[dict] = None
        predictions: Optional[tuple[EntityPrediction, ...]] = None
        ablation: Optional[dict] = None

        if mode == "ABLATION":
            ablation, predictions = self._ablation_report(pq, input, effective)

        if mode in ("CONTEXT", "ANALYZE"):
            context, contexts = self._assemble_report(pq, input, effective)
            if mode == "ANALYZE":
                if input.shared_context or input.hurdle_gate is not None:
                    plan["warnings"] = list(plan.get("warnings", [])) + [
                        "EXPLAIN ANALYZE scores per-entity; the "
                        "shared-context/hurdle strategy named in this plan "
                        "is not applied to these predictions"]
                task_type = pq.task_type(self.schema)
                model_uri = self.model_config.model_uri_for(task_type)
                predictions = tuple(self._require_backend().score(
                    pq, task_type, contexts, model_uri, self.model_config))

        return ExplainResult(mode=mode, format=fmt, plan=plan,
                             context=context, predictions=predictions,
                             ablation=ablation)

    def _score_ready_contexts(self, pq: ParsedQuery, input: ExecutionInput,
                              effective: Optional[datetime]
                              ) -> list[EntityContext]:
        """Contexts exactly as scoring would see them: assembled, WHERE-
        filtered, declared ablations dropped, assumptions applied."""
        eff_input = replace(input, anchor_time=effective)
        entity_table = pq.entity_key.table
        ids = self._resolve_entity_ids(pq, eff_input)
        assumed = assumptions(pq.assuming) if pq.assuming is not None else []
        agg_assumed = aggregate_assumptions(pq.assuming)
        where_anchor = (eff_input.anchor_time
                        if eff_input.context_anchor_time is not None else None)
        contexts: list[EntityContext] = []
        for eid in ids:
            anchor = self._anchor_for(entity_table, eid, eff_input)
            ctx = self.assemble_context(entity_table, eid, anchor, query=pq)
            if pq.where is not None and not self._where_ok(
                    pq, ctx, entity_table, where_anchor):
                continue
            ctx = _apply_ablations(pq.ablations, ctx)
            ctx = _apply_aggregate_assumptions(
                agg_assumed, ctx, entity_table, self.schema,
                self._assumption_template_fetch(ctx, entity_table))
            contexts.append(_apply_assumptions(assumed, ctx, entity_table))
        return contexts

    def _ablation_report(self, pq: ParsedQuery, input: ExecutionInput,
                         effective: Optional[datetime]):
        """Contract: EXPLAIN ABLATE. Score the query as written, then once
        per candidate ablation — every non-entity schema table present in the
        contexts, and every declared non-key column carrying values — and
        rank candidates by how far dropping them moves the predictions.
        Large movement means the model leans on it; near zero means it is
        not earning its context budget and is a good ablation. One cohort
        forward per candidate, so cost scales with schema width, not data."""
        entity_table = pq.entity_key.table
        contexts = self._score_ready_contexts(pq, input, effective)
        task_type = pq.task_type(self.schema)
        metric = ("probability"
                  if task_type is TaskType.BINARY_CLASSIFICATION
                  else "expected_value")
        if not contexts:
            return ({"metric": metric, "baseline": None, "candidates": [],
                     "model_forwards": 0}, ())
        model_uri = self.model_config.model_uri_for(task_type)
        backend = self._require_backend()

        def score(ctxs):
            preds = list(backend.score(pq, task_type, ctxs, model_uri,
                                       self.model_config))
            return preds, [p.probability if p.probability is not None
                           else p.value for p in preds]

        base_preds, base = score(contexts)

        # Candidates come from what is actually in these contexts, minus the
        # entity table (its rows are the task), anything already ablated by
        # the query, and time columns (a task-table timestamp is re-derived
        # at serialization, so dropping the cell measures nothing). The
        # target column stays IN: the entity's own cell is masked either
        # way, so its ablation measures reliance on sibling rows' past
        # outcomes — usually the most load-bearing input there is.
        already = {a.name for a in pq.ablations}
        tables: set = set()
        columns: set = set()
        for c in contexts:
            for r in c.rows:
                tdef = self.schema.table(r.table)
                if tdef is None:
                    continue
                if r.table != entity_table and r.table not in already:
                    tables.add(r.table)
                for col, v in r.cells.items():
                    if (v is None or col == tdef.primary_key
                            or col == tdef.time_column
                            or tdef.column(col) is None):
                        continue
                    columns.add((r.table, col))

        def drop_column(table, col):
            return [replace(c, rows=[
                replace(r, cells={k: v for k, v in r.cells.items()
                                  if k != col})
                if r.table == table else r for r in c.rows])
                for c in contexts]

        entries: list[dict] = []
        forwards = 1

        def measure(kind, name, ctxs):
            nonlocal forwards
            _, vals = score(ctxs)
            forwards += 1
            deltas = [v - b for v, b in zip(vals, base)
                      if v is not None and b is not None]
            n = len(deltas)
            entries.append({
                "kind": kind, "name": name, "n": n,
                "mean_delta": sum(deltas) / n if n else 0.0,
                "mean_abs_delta": sum(abs(d) for d in deltas) / n if n else 0.0,
                "max_abs_delta": max((abs(d) for d in deltas), default=0.0),
            })

        for t in sorted(tables):
            measure("table", t,
                    [_apply_ablations([Ablation("table", t)], c)
                     for c in contexts])
        for t, col in sorted(columns):
            measure("column", f"{t}.{col}", drop_column(t, col))
        entries.sort(key=lambda e: -e["mean_abs_delta"])
        known = [b for b in base if b is not None]
        report = {
            "metric": metric,
            "baseline": {"n": len(known),
                         "mean": sum(known) / len(known) if known else None},
            "candidates": entries,
            "model_forwards": forwards,
            "note": ("mean_abs_delta: how far the prediction moves when this "
                     "is dropped. Near zero = the model is not using it — a "
                     "good ablation; large = load-bearing, keep it."),
        }
        return report, tuple(base_preds)

    def _assemble_report(self, pq: ParsedQuery, input: ExecutionInput,
                         effective: Optional[datetime]):
        """Assemble per-entity context via the normal path (no scoring) and
        summarize it (contract: EXPLAIN CONTEXT)."""
        contexts = self._score_ready_contexts(pq, input, effective)

        tables: dict[str, dict] = {}
        links_traversed = 0
        # FK values a link opts into emitting (feature_type) are model input
        # too. EXPLAIN once counted only r.cells, which made feature-bearing
        # links — and any table whose rows carry nothing but FKs — report zero
        # cells while the model saw their tokens. What EXPLAIN prints must be
        # what the model gets.
        fk_features = {
            tbl.name: [l.fk_column for l in self.schema.links_from(tbl.name)
                       if l.feature_type is not None]
            for tbl in self.schema.tables}
        for ctx in contexts:
            for r in ctx.rows:
                t = tables.setdefault(
                    r.table, {"rows": 0, "cells": 0,
                              "min_time": None, "max_time": None})
                t["rows"] += 1
                t["cells"] += len(r.cells) + (1 if r.timestamp is not None else 0)
                t["cells"] += sum(1 for fk in fk_features.get(r.table, ())
                                  if r.parents.get(fk) is not None)
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
