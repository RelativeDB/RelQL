"""Flat-feature planning: can a RelQL query run as tree-model features, and
which features?

Gradient-boosted trees consume one fixed-width numeric vector per entity.
That representation cannot see graph structure, but for scalar targets it is
a strong technique — so the decision of WHICH queries qualify and the
derivation of the feature columns live here, in pure Python next to the
parser and planner. The numeric EVALUATION of those columns over assembled
contexts stays in C++ (``cpp/src/flat.*``): it receives this module's
feature-spec JSON — never RelQL text — plus the encoded contexts, and fills a
dense float matrix (see an external tree backend).

Eligibility is deliberately narrow: scalar regression / binary targets, one
horizon, no RANK/CLASSIFY, no ASSUMING (a fitted tree cannot honor a
counterfactual), no ABLATE. Everything else stays with the sequence model.

The feature columns are the classic tabular recipe:
  - the entity row's own scalar columns (dates become age-at-anchor,
    categoricals a stable hash),
  - the target aggregation mirrored into recent PAST windows (the
    autoregressive signal),
  - every windowed aggregation the WHERE clause already computes,
  - per linked table: COUNT over standard past windows, recency, and
    SUM/AVG/MAX of each numeric column.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Optional, Union

from .plan import _expr_str
from .relql.ast import (AggFunc, Aggregation, ColumnRef, Condition,
                        LogicalOp, Not, ParsedQuery, TaskType,
                        TimeUnit, Window, _find_aggregations)
from .schema import Schema, ValueType

__all__ = ["FlatAnalysis", "FlatFeature", "analyze_flat", "derive_flat_spec",
           "flat_spec_to_json"]

_PAST_DAYS = (7, 30, 90)


@dataclass(frozen=True)
class FlatFeature:
    """One feature column. ``kind`` is entity_column | aggregate |
    days_since_last; exactly one of the payload fields applies."""

    kind: str
    name: str
    column: str = ""                      # entity_column
    col_type: Optional[ValueType] = None  # entity_column
    agg: Optional[Aggregation] = None     # aggregate
    table: str = ""                       # days_since_last


@dataclass(frozen=True)
class FlatAnalysis:
    """The planner's answer to "can this query run as flat features"."""

    eligible: bool
    reason: str                       # why not, when ineligible
    task_type: str
    entity_table: str
    features: tuple[FlatFeature, ...] = ()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)


def _decline(task: TaskType, entity_table: str, why: str) -> FlatAnalysis:
    return FlatAnalysis(False, why, task.value, entity_table)


def derive_flat_spec(bound: ParsedQuery, schema: Schema) -> FlatAnalysis:
    """Derive the flat plan from a BOUND query (validate + bind_params first).
    Never raises for ineligibility — that is an answer, not an error."""
    entity_table = bound.entity_key.table
    task = bound.task_type(schema)

    if task not in (TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION):
        return _decline(task, entity_table,
                        "flat features cover scalar regression and binary "
                        f"classification; {task.value} needs the sequence "
                        f"model")
    if bound.rank is not None or bound.top_k is not None:
        return _decline(task, entity_table,
                        "RANK/CLASSIFY targets need the sequence model")
    if bound.num_forecasts is not None and bound.num_forecasts > 1:
        return _decline(task, entity_table,
                        "multi-horizon forecasting needs the sequence model")
    if bound.assuming is not None:
        return _decline(task, entity_table,
                        "ASSUMING is a counterfactual on the context; a "
                        "fitted tree model cannot honor it")
    if bound.ablations:
        return _decline(task, entity_table,
                        "ABLATE changes the context a sequence model reads; "
                        "flat features have no equivalent")
    if bound.ret is not None and bound.ret.kind not in ("EXPECTED_VALUE",
                                                        "PROBABILITY"):
        return _decline(task, entity_table,
                        "only RETURN EXPECTED VALUE / PROBABILITY map onto a "
                        "scalar tree prediction")

    target_aggs = _find_aggregations(bound.target)
    for a in target_aggs:
        if a.func in (AggFunc.ARRAY_AGG, AggFunc.LIST_DISTINCT):
            return _decline(task, entity_table,
                            "list-valued aggregations in the target need the "
                            "sequence model")
        if a.window is not None and a.window.horizons > 1:
            return _decline(task, entity_table,
                            "multi-horizon windows need the sequence model")

    features: list[FlatFeature] = []
    names: set[str] = set()

    def add(f: FlatFeature) -> None:
        if f.name not in names:
            names.add(f.name)
            features.append(f)

    # 1. The entity row's own scalar columns.
    entity = schema.table(entity_table)
    if entity is not None:
        for c in entity.columns:
            if c.name == entity.primary_key:
                continue
            if c.type in (ValueType.TEXT,):
                continue
            add(FlatFeature(
                kind="entity_column", column=c.name, col_type=c.type,
                name="entity." + c.name
                     + ("_age_days" if c.type is ValueType.DATETIME else "")))

    def windowed(agg: Aggregation, start: float, end: float,
                 unit: TimeUnit) -> Aggregation:
        return replace(agg, window=Window(start, end, unit))

    # 2. The target mirrored into the recent past — the autoregressive signal.
    for a in target_aggs:
        w = a.window
        finite = (w is not None and math.isfinite(w.start)
                  and math.isfinite(w.end))
        if finite:
            width = w.end - w.start
            if width <= 0:
                width = 1
            for i in (1, 2, 3):
                clone = windowed(a, w.start - i * width, w.end - i * width,
                                 w.unit)
                add(FlatFeature(kind="aggregate", agg=clone,
                                name=f"hist{i}:{_expr_str(clone)}"))
        else:
            for d in _PAST_DAYS:
                clone = windowed(a, -float(d), 0.0, TimeUnit.DAYS)
                add(FlatFeature(kind="aggregate", agg=clone,
                                name=f"hist:{_expr_str(clone)}"))

    # 3. Whatever the WHERE clause already computes over the past.
    for a in _find_aggregations(bound.where) if bound.where is not None else []:
        if a.func in (AggFunc.ARRAY_AGG, AggFunc.LIST_DISTINCT):
            continue
        if a.window is not None and a.window.horizons > 1:
            continue
        add(FlatFeature(kind="aggregate", agg=a,
                        name=f"where:{_expr_str(a)}"))

    # 4. The standard per-table recipe over every linked table. "Linked" =
    # within two link hops of the entity table, direction-blind.
    nearby = _reachable_tables(schema, entity_table)
    for t in schema.tables:
        if t.name not in nearby:
            continue
        for d in _PAST_DAYS:
            add(FlatFeature(
                kind="aggregate",
                agg=Aggregation(AggFunc.COUNT, ColumnRef(t.name, "*"),
                                window=Window(-float(d), 0.0, TimeUnit.DAYS)),
                name=f"{t.name}.count_{d}d"))
        # Unbounded frames stay windowless: assembly already bounds the past,
        # and a windowed frame would drop static (undated) rows entirely.
        add(FlatFeature(
            kind="aggregate",
            agg=Aggregation(AggFunc.COUNT, ColumnRef(t.name, "*")),
            name=f"{t.name}.count_all"))
        if t.time_column:
            add(FlatFeature(kind="days_since_last", table=t.name,
                            name=f"{t.name}.recency_days"))
        for c in t.columns:
            if c.type is not ValueType.NUMBER or c.name == t.primary_key:
                continue
            for func, tag in ((AggFunc.SUM, "sum"), (AggFunc.AVG, "avg"),
                              (AggFunc.MAX, "max")):
                add(FlatFeature(
                    kind="aggregate",
                    agg=Aggregation(func, ColumnRef(t.name, c.name),
                                    window=Window(-30.0, 0.0, TimeUnit.DAYS)),
                    name=f"{t.name}.{c.name}_{tag}_30d"))
                add(FlatFeature(
                    kind="aggregate",
                    agg=Aggregation(func, ColumnRef(t.name, c.name)),
                    name=f"{t.name}.{c.name}_{tag}_all"))

    return FlatAnalysis(True, "", task.value, entity_table, tuple(features))


def _reachable_tables(schema: Schema, entity_table: str) -> set[str]:
    """Tables whose rows can appear in an entity-scoped context: within two
    link hops of the entity table, direction-blind (children, parents,
    siblings)."""
    seen = {entity_table}
    frontier = [entity_table]
    for _ in range(2):
        nxt = []
        for t in frontier:
            for link in schema.links:
                if link.from_table == t:
                    other = link.to_table
                elif link.to_table == t:
                    other = link.from_table
                else:
                    continue
                if other not in seen:
                    seen.add(other)
                    nxt.append(other)
        frontier = nxt
    seen.discard(entity_table)
    return seen


def analyze_flat(query: Union[str, ParsedQuery], schema: Schema,
                 params: Optional[dict[str, Any]] = None) -> FlatAnalysis:
    """Whether ``query`` runs as flat features, and with which feature
    columns. Ineligibility is an answer, not an exception; a query that does
    not parse/validate at all still raises."""
    from .relql.parser import validate
    bound = validate(query, schema, params).query
    return derive_flat_spec(bound, schema)


# ---------------------------------------------------------------------------
# wire encoding for the C++ evaluator (cpp/src/flat.*)
# ---------------------------------------------------------------------------

def _lit_json(v: Any) -> Any:
    """Literal -> the flat evaluator's JSON: dates as {"date": iso-ish},
    lists (IN) as arrays, everything else JSON-native."""
    if isinstance(v, datetime):
        return {"date": v.strftime("%Y-%m-%dT%H:%M:%S")}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_lit_json(x) for x in v]
    return v


def _cond_json(e: Any) -> dict:
    """Encode an aggregation's inline filter: cond / logic / not over the
    aggregated table's own columns. This mirrors ``evaluate.eval_row_predicate``
    semantics; the C++ side re-implements exactly this subset."""
    if isinstance(e, Condition):
        if not isinstance(e.left, ColumnRef):
            raise ValueError(
                "flat filters compare the aggregated table's own columns")
        if e.right_expr is not None:
            raise ValueError(
                "flat filters need literal right-hand sides (bind :params "
                "before deriving features)")
        return {"kind": "cond", "column": e.left.column, "op": e.op.name,
                "right": _lit_json(e.right)}
    if isinstance(e, LogicalOp):
        return {"kind": "logic", "op": e.op.name,
                "left": _cond_json(e.left), "right": _cond_json(e.right)}
    if isinstance(e, Not):
        return {"kind": "not", "expr": _cond_json(e.expr)}
    raise ValueError(f"unsupported flat filter node: {type(e).__name__}")


def _agg_json(a: Aggregation) -> dict:
    out: dict[str, Any] = {
        "func": a.func.name,
        "table": a.column.table,
        "column": a.column.column,
    }
    if a.filter is not None:
        out["filter"] = _cond_json(a.filter)
    if a.window is not None:
        out["window"] = {"start": _num_json(a.window.start),
                         "end": _num_json(a.window.end),
                         "unit": a.window.unit.value}
    return out


def _num_json(x: float) -> Any:
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def flat_spec_to_json(analysis: FlatAnalysis) -> str:
    """The evaluator's input: no query text, no schema — just the features."""
    feats = []
    for f in analysis.features:
        if f.kind == "entity_column":
            feats.append({"kind": "entity_column", "name": f.name,
                          "column": f.column,
                          "col_type": f.col_type.value if f.col_type else None})
        elif f.kind == "aggregate":
            feats.append({"kind": "aggregate", "name": f.name,
                          "agg": _agg_json(f.agg)})
        else:
            feats.append({"kind": "days_since_last", "name": f.name,
                          "table": f.table})
    return json.dumps({
        "entity_table": analysis.entity_table,
        "task_type": analysis.task_type,
        "features": feats,
    }, separators=(",", ":"))
