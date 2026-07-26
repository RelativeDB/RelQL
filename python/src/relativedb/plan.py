"""Query planning: derive, once, every decision execution and EXPLAIN need.

Before this module the two derived the same facts independently -- EXPLAIN
rebuilt task type, the pinned entity selector and the AS OF provenance in
``Engine._build_plan`` while ``Engine.execute`` worked them out inline. Nothing
kept the two in step, so EXPLAIN could drift away from what actually ran. Worse,
it described only the *logical* query: which execution strategy was chosen, the
sampler, the traversal, the scoring batch size and whether context assembly was
pipelined were all decided inside ``execute`` and invisible to the user asking
how their query would run.

:class:`QueryPlan` now holds both halves. ``Engine.execute`` builds one and acts
on it; ``Engine.explain`` builds one and renders it. A plan is a description, so
building one never touches data: the physical fields that genuinely need a
cohort size or a backend are ``Optional`` and read "unknown" when a caller (an
EXPLAIN PLAN, say) has not resolved them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .errors import ExecutionError
from .relql.ast import (AggFunc, Aggregation, Arith, BoolOp, Case, ColumnRef,
                        Condition, Func, Lit, LogicalOp, Not, Operator,
                        ParsedQuery, TaskType, Window)
from .schema import Schema

__all__ = ["QueryPlan", "build_plan"]

# Default output form per task type (contract Part B, `plan.output`), used
# when the query has no explicit RETURN spec.
TASK_DEFAULT_OUTPUT = {
    TaskType.REGRESSION: "value",
    TaskType.BINARY_CLASSIFICATION: "probability",
    TaskType.MULTICLASS_CLASSIFICATION: "class",
    TaskType.MULTILABEL_RANKING: "ranked",
    TaskType.FORECASTING: "value-per-horizon",
}

# Task types whose scoring path is a plain batched forward, so context
# assembly for the next chunk can overlap the current chunk's forward.
PIPELINEABLE_TASKS = (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION,
                      TaskType.FORECASTING)


# ---------------------------------------------------------------------------
# rendering helpers for query fragments
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# query-shape analysis
# ---------------------------------------------------------------------------

def target_span(pq: ParsedQuery):
    """How far past the anchor the target's window reaches — the bound the
    label context needs. None when the frame is unbounded."""
    for a in pq.target_aggregations:
        if a.window is not None:
            return a.window.span()
    return None


def pure_pin(where: Any, entity_key: ColumnRef) -> bool:
    """True when every WHERE leaf is a primary-key pin — i.e. the clause
    selects the cohort and nothing else, so shared-context scoring applies
    it fully by construction."""
    if isinstance(where, LogicalOp) and where.op is BoolOp.AND:
        return (pure_pin(where.left, entity_key)
                and pure_pin(where.right, entity_key))
    return pinned_ids(where, entity_key) is not None


def pinned_ids(where: Any, entity_key: ColumnRef) -> Optional[list[Any]]:
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
        left = pinned_ids(where.left, entity_key)
        right = pinned_ids(where.right, entity_key)
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


_AGG_ASSUMABLE_OPS = (Operator.GE, Operator.GT, Operator.EQ,
                      Operator.LE, Operator.LT)


def _walk_assuming(expr: Any):
    """Yield each applicable ASSUMING conjunct as ``("assign", (t, c, v))``
    or ``("agg", (aggregation, op, count))``; raise on anything else.

    Two shapes have one concrete answer the engine can build: a value
    assignment (`column = literal`) and an aggregate count bound
    (`COUNT(t.*) OVER (...) >= k`, or EXISTS/NOT EXISTS) — the latter is
    realized by adding or removing the entity's own event rows until the
    bound holds. An `IN`, an `OR`, or an inequality on a plain column still
    describes a set of possible worlds rather than one, so those raise
    rather than being quietly dropped.
    """
    if isinstance(expr, LogicalOp) and expr.op is BoolOp.AND:
        yield from _walk_assuming(expr.left)
        yield from _walk_assuming(expr.right)
        return
    if (isinstance(expr, Condition) and expr.op is Operator.EQ
            and isinstance(expr.left, ColumnRef)
            and expr.left.column != "*"
            and expr.right_expr is None
            and not isinstance(expr.right, tuple)):
        yield ("assign", (expr.left.table, expr.left.column, expr.right))
        return
    if (isinstance(expr, Condition) and isinstance(expr.left, Aggregation)
            and expr.left.func in (AggFunc.COUNT, AggFunc.EXISTS)
            and expr.op in _AGG_ASSUMABLE_OPS
            and expr.right_expr is None
            and isinstance(expr.right, (int, float))
            and not isinstance(expr.right, bool)):
        yield ("agg", (expr.left, expr.op, expr.right))
        return
    if isinstance(expr, Aggregation) and expr.func is AggFunc.EXISTS:
        yield ("agg", (expr, Operator.GE, 1))
        return
    if (isinstance(expr, Not) and isinstance(expr.expr, Aggregation)
            and expr.expr.func is AggFunc.EXISTS):
        yield ("agg", (expr.expr, Operator.EQ, 0))
        return
    raise ExecutionError(
        f"ASSUMING {_expr_str(expr)!r} cannot be applied: a counterfactual "
        f"must state one concrete world — `column = literal`, or a count "
        f"bound like `COUNT(t.*) OVER (...) >= k` / `EXISTS(t.*)`, joined "
        f"by AND. IN, OR, and inequalities on plain columns describe a set "
        f"of possible worlds, so the engine cannot build the context they "
        f"imply.")


def assumptions(expr: Any) -> list[tuple[str, str, Any]]:
    """The `(table, column, value)` assignments an ASSUMING clause states."""
    return [item for kind, item in _walk_assuming(expr) if kind == "assign"]


def aggregate_assumptions(expr: Any) -> list[tuple[Any, Operator, Any]]:
    """The `(aggregation, op, count)` bounds an ASSUMING clause states."""
    if expr is None:
        return []
    return [item for kind, item in _walk_assuming(expr) if kind == "agg"]


def assuming_plan(expr: Any) -> Optional[str]:
    """How EXPLAIN renders the counterfactual. EXPLAIN must describe any query
    that parses, so an inapplicable clause is reported, not raised."""
    if expr is None:
        return None
    try:
        parts = []
        for kind, item in _walk_assuming(expr):
            if kind == "assign":
                t, c, v = item
                parts.append(f"{t}.{c} := {_lit_str(v)}")
            else:
                agg, op, bound = item
                parts.append(f"{_expr_str(agg)} {op.value} {bound}")
        return ", ".join(parts)
    except ExecutionError:
        return None


def collect_windows(schema: Schema, aggs, role: str, out: list[dict]) -> None:
    for a in aggs:
        w = a.window
        if w is None:
            continue
        t = schema.table(a.column.table)
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


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryPlan:
    """One query's execution decisions: what to compute, and how.

    The logical half answers "what does this query mean" and needs only the
    parsed query and the schema. The physical half answers "how will it run"
    and is what EXPLAIN previously could not tell you.
    """

    # -- logical ---------------------------------------------------------
    target: str
    task_type: TaskType
    entity_table: str
    entity_pk: str
    entity_selector: Any                  # pinned id list, or "ALL"
    output: str
    windows: tuple
    where_present: bool
    assuming_present: bool
    assuming: Optional[str]
    as_of: dict
    ablations: tuple
    warnings: tuple

    # -- physical --------------------------------------------------------
    strategy: str                         # hurdle | shared-context | per-entity
    sampler: str                          # csc | retriever
    traversal: str
    model_uri: Optional[str] = None
    # None means "not resolved at plan time" rather than zero/false: an
    # EXPLAIN PLAN does not enumerate a cohort, so it cannot know these.
    cohort_size: Optional[int] = None
    scoring_batch_size: Optional[int] = None
    pipelined: Optional[bool] = None

    def to_dict(self) -> dict:
        """The EXPLAIN plan payload.

        The logical keys keep the exact names and shapes they had when this was
        assembled inline, so ``ExplainResult._render_text`` and any consumer
        reading ``plan[...]`` are unaffected. ``execution`` is additive.
        """
        return {
            "target": self.target,
            "task_type": self.task_type.value,
            "entity": {
                "table": self.entity_table,
                "pk": self.entity_pk,
                "selector": self.entity_selector,
            },
            "output": self.output,
            "windows": list(self.windows),
            "where_present": self.where_present,
            "assuming_present": self.assuming_present,
            "assuming": self.assuming,
            "as_of": self.as_of,
            "ablations": list(self.ablations),
            "warnings": list(self.warnings),
            "execution": {
                "strategy": self.strategy,
                "sampler": self.sampler,
                "traversal": self.traversal,
                "model_uri": self.model_uri,
                "cohort_size": self.cohort_size,
                "scoring_batch_size": self.scoring_batch_size,
                "pipelined": self.pipelined,
            },
        }


def _strategy_for(input) -> str:
    """Mirrors the dispatch in Engine.execute. hurdle wins over shared: the
    hurdle path runs its own sub-queries and never consults shared_context."""
    if input.hurdle_gate is not None:
        return "hurdle"
    if input.shared_context:
        return "shared-context"
    return "per-entity"



def _logical_from_ast(schema: Schema, pq: ParsedQuery) -> dict:
    """The logical plan derived from the AST, in the shape cpp/src/plan.cpp
    emits.

    Used only for a query built by AST surgery rather than written -- the
    engine's hurdle composition derives one, and it has no source text for the
    C++ pass to analyze. Everything written by a user goes through relql_analyze
    instead, so this is a fallback, not a second implementation in waiting.
    """
    task = pq.task_type(schema)
    pinned = pinned_ids(pq.where, pq.entity_key)
    windows: list[dict] = []
    collect_windows(schema, pq.target_aggregations, "target", windows)
    if pq.where is not None:
        from .relql.ast import _find_aggregations
        collect_windows(schema, _find_aggregations(pq.where), "where", windows)
    if pq.assuming is not None:
        from .relql.ast import _find_aggregations
        collect_windows(schema, _find_aggregations(pq.assuming), "assuming",
                        windows)
    warnings_: list[str] = []
    if pq.assuming is not None:
        try:
            assumptions(pq.assuming)
        except ExecutionError as e:
            warnings_.append(str(e))
    ao = pq.as_of
    if ao is None or ao.kind == "now":
        source = "execution-anchor"
    elif ao.kind == "date":
        source = "query-date"
    else:
        source = "query-param"
    return {
        "target": _expr_str(pq.target),
        "task_type": task.value,
        "entity_table": pq.entity_key.table,
        "entity_pk": pq.entity_key.column,
        "selector_all": pinned is None,
        "selector": pinned or [],
        "output": (pq.ret.kind.lower() if pq.ret is not None
                   else TASK_DEFAULT_OUTPUT[task]),
        "windows": windows,
        "where_present": pq.where is not None,
        "assuming_present": pq.assuming is not None,
        "assuming": assuming_plan(pq.assuming),
        "as_of_source": source,
        "as_of_param": (ao.value if ao is not None and ao.kind == "param"
                        else None),
        "ablations": [a.name for a in pq.ablations],
        "warnings": warnings_,
    }


def build_plan(schema: Schema, pq: ParsedQuery, input, *,
               effective: Optional[datetime] = None,
               traversal: Any = None,
               sampler: str = "retriever",
               model_uri: Optional[str] = None,
               cohort_size: Optional[int] = None,
               scoring_batch_size: Optional[int] = None,
               logical: Optional[dict] = None) -> QueryPlan:
    """Assemble the plan for one query. Never touches data.

    The logical half comes from the C++ pass (cpp/src/plan.cpp) so every
    frontend derives it identically; `logical` is that payload, as returned by
    relql_analyze. When it is absent -- a query built by AST surgery, which has
    no source text to re-analyze -- the fields it would have supplied are
    derived here from the AST instead.
    """
    if logical is None:
        logical = _logical_from_ast(schema, pq)

    task_type = TaskType(logical["task_type"])
    selector = "ALL" if logical["selector_all"] else list(logical["selector"])

    as_of = {"source": logical["as_of_source"],
             "value": effective.isoformat() if effective is not None else None}
    if logical.get("as_of_param"):
        as_of["param"] = logical["as_of_param"]

    # A pinned cohort is known without touching data; "ALL" is not.
    if cohort_size is None and selector != "ALL":
        cohort_size = len(selector)

    pipelined: Optional[bool] = None
    if scoring_batch_size is not None and cohort_size is not None:
        pipelined = (scoring_batch_size > 0
                     and cohort_size > scoring_batch_size
                     and task_type in PIPELINEABLE_TASKS)

    return QueryPlan(
        target=logical["target"],
        task_type=task_type,
        entity_table=logical["entity_table"],
        entity_pk=logical["entity_pk"],
        entity_selector=selector,
        output=logical["output"],
        windows=tuple(logical["windows"]),
        where_present=logical["where_present"],
        assuming_present=logical["assuming_present"],
        assuming=logical["assuming"],
        as_of=as_of,
        ablations=tuple({"table": name,
                         "note": "applied: rows dropped from every context"}
                        for name in logical["ablations"]),
        warnings=tuple(logical["warnings"]),
        strategy=_strategy_for(input),
        sampler=sampler,
        traversal=type(traversal).__name__,
        model_uri=model_uri,
        cohort_size=cohort_size,
        scoring_batch_size=scoring_batch_size,
        pipelined=pipelined,
    )
