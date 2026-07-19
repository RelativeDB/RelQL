"""RelQL parsing (single-sourced in C++) + schema-bound validation.

Parsing lives once in the native layer — ``librt_c``'s ``relql_parse`` (see
:mod:`relativedb.relql.native`), shared by the Python, Java, and Rust bindings.
``librt_c`` is a hard dependency: it is the same native library the RT-J model
requires to run. Only the schema-binding validation stays in Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ast import (Arith, Case, ColumnRef, Condition, Func, Lit, LogicalOp,
                  Not, ParsedQuery, TaskType)

__all__ = ["parse", "validate", "RelqlSyntaxError", "RelqlValidationError",
           "ValidatedQuery"]


class RelqlSyntaxError(ValueError):
    def __init__(self, message: str, pos: int = -1, text: str = ""):
        self.pos = pos
        loc = f" at position {pos}" if pos >= 0 else ""
        snippet = ""
        if text and 0 <= pos <= len(text):
            snippet = f": ...{text[max(0, pos - 10):pos]}>>>{text[pos:pos + 15]}"
        super().__init__(f"RelQL syntax error{loc}: {message}{snippet}")


class RelqlValidationError(ValueError):
    pass


def parse(query: str) -> ParsedQuery:
    """Parse a RelQL string via the shared C++ parser (``librt_c``). Raises
    :class:`RelqlSyntaxError` on malformed input, or
    :class:`~relativedb.relql.native.NativeParserUnavailable` if the native
    library cannot be loaded."""
    if not isinstance(query, str) or not query.strip():
        raise RelqlSyntaxError("empty query")
    from .native import parse_native
    return parse_native(query)


# ---------------------------------------------------------------------------
# Schema-bound validation (Python-side; the grammar itself is in C++)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatedQuery:
    query: ParsedQuery
    task_type: Any  # TaskType


def _check_column(col: ColumnRef, schema, *, allow_star: bool,
                  allow_pk: bool = False, allow_fk: bool = False) -> None:
    table = schema.table(col.table)
    if table is None:
        raise RelqlValidationError(f"unknown table {col.table!r}")
    if col.column == "*":
        if not allow_star:
            raise RelqlValidationError(f"{col}: '*' not allowed here")
        return
    if table.column(col.column) is None:
        if allow_pk and col.column == table.primary_key:
            return
        if allow_fk and any(l.fk_column == col.column
                            for l in schema.links_from(col.table)):
            return
        raise RelqlValidationError(
            f"unknown column {col.column!r} on table {col.table!r}")


def _walk_columns(expr, schema) -> None:
    from .ast import Aggregation as Agg, AggFunc
    if isinstance(expr, Agg):
        # FK columns are legal aggregation targets for set/count aggregations
        # only FIRST/LAST exclude them per the docs.
        fk_ok = expr.func in (AggFunc.LIST_DISTINCT, AggFunc.ARRAY_AGG,
                              AggFunc.COUNT, AggFunc.COUNT_DISTINCT)
        _check_column(expr.column, schema, allow_star=True, allow_fk=fk_ok)
        # Only a frame the query actually wrote needs a time column to cut rows
        # against. The implied default expresses no temporal intent, so a table
        # without a time column stays legal there — as it was before the
        # default existed.
        if expr.window is not None and not expr.window.implied:
            t = schema.table(expr.column.table)
            if t is not None and t.time_column is None:
                raise RelqlValidationError(
                    f"windowed aggregation over {expr.column.table!r}, "
                    f"which has no time_column")
        if expr.filter is not None:
            _walk_columns(expr.filter, schema)
    elif isinstance(expr, ColumnRef):
        # The primary key is a legal column reference even though tables do not
        # list it among their columns — pinning a cohort (`WHERE t.pk IN :ids`)
        # depends on it.
        _check_column(expr, schema, allow_star=False, allow_pk=True)
    elif isinstance(expr, Condition):
        _walk_columns(expr.left, schema)
        if expr.right_expr is not None:
            _walk_columns(expr.right_expr, schema)
    elif isinstance(expr, LogicalOp):
        _walk_columns(expr.left, schema)
        _walk_columns(expr.right, schema)
    elif isinstance(expr, Not):
        _walk_columns(expr.expr, schema)
    elif isinstance(expr, Arith):
        _walk_columns(expr.left, schema)
        _walk_columns(expr.right, schema)
    elif isinstance(expr, Func):
        for a in expr.args:
            _walk_columns(a, schema)
    elif isinstance(expr, Case):
        for cond, then in expr.whens:
            _walk_columns(cond, schema)
            _walk_columns(then, schema)
        if expr.else_ is not None:
            _walk_columns(expr.else_, schema)
    elif isinstance(expr, Lit):
        pass


def validate(query, schema) -> ValidatedQuery:
    """Parse + bind against a schema: tables/columns exist, the population's
    primary key is resolved, target windows are future-facing (start >= 0).

    ``ValidatedQuery.query`` is the *bound* query — same AST, with the
    population's primary key filled in — so callers should use it rather than
    the query they passed in."""
    pq = parse(query) if isinstance(query, str) else query
    ek = pq.entity_key
    table = schema.table(ek.table)
    if table is None:
        origin = ("inferred from the PREDICT target" if pq.entity_inferred
                  else "named by FROM")
        raise RelqlValidationError(
            f"unknown entity table {ek.table!r} ({origin})")
    if not table.primary_key:
        raise RelqlValidationError(
            f"table {ek.table!r} declares no primary key, so it cannot be a "
            f"population")
    if ek.column and table.primary_key != ek.column:
        raise RelqlValidationError(
            f"{ek}: {ek.column!r} is not the primary key of {ek.table!r} "
            f"(expected {table.primary_key!r})")
    pq = pq.bind_entity_key(schema)
    _walk_columns(pq.target, schema)
    for agg in pq.target_aggregations:
        if agg.window is not None and agg.window.start < 0:
            raise RelqlValidationError(
                f"target window ({agg.window.start}, {agg.window.end}] must "
                f"be future-facing (start >= 0)")
    from .ast import _find_aggregations
    for clause_name, clause in (("WHERE", pq.where), ("ASSUMING", pq.assuming)):
        if clause is None:
            continue
        _walk_columns(clause, schema)
        for agg in _find_aggregations(clause):
            if agg.window is not None and agg.window.horizons > 1:
                raise RelqlValidationError(
                    f"HORIZONS > 1 is only allowed on the PREDICT target, "
                    f"not in {clause_name}")
    task = pq.task_type(schema)
    if pq.ret is not None:
        _validate_return(pq.ret, task)
    return ValidatedQuery(pq, task)


# RETURN kind -> inferred task types it is compatible with (contract §1).
_RETURN_COMPATIBILITY = {
    "EXPECTED_VALUE": {TaskType.REGRESSION, TaskType.FORECASTING,
                       TaskType.BINARY_CLASSIFICATION},
    "PROBABILITY": {TaskType.BINARY_CLASSIFICATION},
    "CLASS": {TaskType.BINARY_CLASSIFICATION,
              TaskType.MULTICLASS_CLASSIFICATION},
    "DISTRIBUTION": {TaskType.BINARY_CLASSIFICATION,
                     TaskType.MULTICLASS_CLASSIFICATION},
    "MULTILABEL": {TaskType.MULTILABEL_RANKING},
    "MULTICLASS": {TaskType.MULTICLASS_CLASSIFICATION},
}


def _validate_return(ret, task: TaskType) -> None:
    """Enforce the RETURN compatibility matrix (contract §1) against the
    inferred task type."""
    allowed = _RETURN_COMPATIBILITY.get(ret.kind)
    if allowed is None:
        raise RelqlValidationError(f"unknown RETURN kind {ret.kind!r}")
    if task not in allowed:
        allowed_names = ", ".join(sorted(t.value for t in allowed))
        raise RelqlValidationError(
            f"RETURN {ret.kind} is not compatible with inferred task "
            f"{task.value!r} (allowed tasks: {allowed_names})")
