"""PQL parsing (single-sourced in C++) + schema-bound validation.

Parsing lives once in the native layer — ``librt_c``'s ``pql_parse`` (see
:mod:`relativedb.pql.native`), shared by the Python, Java, and Rust bindings.
``librt_c`` is a hard dependency: it is the same native library the RT-J model
requires to run. Only the schema-binding validation stays in Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ast import ColumnRef, Condition, LogicalOp, Not, ParsedQuery

__all__ = ["parse", "validate", "PqlSyntaxError", "PqlValidationError",
           "ValidatedQuery"]


class PqlSyntaxError(ValueError):
    def __init__(self, message: str, pos: int = -1, text: str = ""):
        self.pos = pos
        loc = f" at position {pos}" if pos >= 0 else ""
        snippet = ""
        if text and 0 <= pos <= len(text):
            snippet = f": ...{text[max(0, pos - 10):pos]}>>>{text[pos:pos + 15]}"
        super().__init__(f"PQL syntax error{loc}: {message}{snippet}")


class PqlValidationError(ValueError):
    pass


def parse(query: str) -> ParsedQuery:
    """Parse a PQL string via the shared C++ parser (``librt_c``). Raises
    :class:`PqlSyntaxError` on malformed input, or
    :class:`~relativedb.pql.native.NativeParserUnavailable` if the native
    library cannot be loaded."""
    if not isinstance(query, str) or not query.strip():
        raise PqlSyntaxError("empty query")
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
        raise PqlValidationError(f"unknown table {col.table!r}")
    if col.column == "*":
        if not allow_star:
            raise PqlValidationError(f"{col}: '*' not allowed here")
        return
    if table.column(col.column) is None:
        if allow_pk and col.column == table.primary_key:
            return
        if allow_fk and any(l.fk_column == col.column
                            for l in schema.links_from(col.table)):
            return
        raise PqlValidationError(
            f"unknown column {col.column!r} on table {col.table!r}")


def _walk_columns(expr, schema) -> None:
    from .ast import Aggregation as Agg, AggFunc
    if isinstance(expr, Agg):
        # FK columns are legal aggregation targets for set/count aggregations
        # (the Kumo recommendation pattern: LIST_DISTINCT over a foreign key,
        # "RANK on a foreign key"); only FIRST/LAST exclude them per the docs.
        fk_ok = expr.func in (AggFunc.LIST_DISTINCT, AggFunc.COUNT,
                              AggFunc.COUNT_DISTINCT)
        _check_column(expr.column, schema, allow_star=True, allow_fk=fk_ok)
        if expr.window is not None:
            t = schema.table(expr.column.table)
            if t is not None and t.time_column is None:
                raise PqlValidationError(
                    f"windowed aggregation over {expr.column.table!r}, "
                    f"which has no time_column")
        if expr.filter is not None:
            _walk_columns(expr.filter, schema)
    elif isinstance(expr, ColumnRef):
        _check_column(expr, schema, allow_star=False)
    elif isinstance(expr, Condition):
        _walk_columns(expr.left, schema)
    elif isinstance(expr, LogicalOp):
        _walk_columns(expr.left, schema)
        _walk_columns(expr.right, schema)
    elif isinstance(expr, Not):
        _walk_columns(expr.expr, schema)


def validate(query, schema) -> ValidatedQuery:
    """Parse + bind against a schema: tables/columns exist, the entity key is
    a primary key, target windows are future-facing (start >= 0)."""
    pq = parse(query) if isinstance(query, str) else query
    ek = pq.entity_key
    table = schema.table(ek.table)
    if table is None:
        raise PqlValidationError(f"unknown entity table {ek.table!r}")
    if table.primary_key != ek.column:
        raise PqlValidationError(
            f"FOR EACH {ek}: {ek.column!r} is not the primary key of "
            f"{ek.table!r} (expected {table.primary_key!r})")
    _walk_columns(pq.target, schema)
    for agg in pq.target_aggregations:
        if agg.window is not None and agg.window.start < 0:
            raise PqlValidationError(
                f"target window ({agg.window.start}, {agg.window.end}] must "
                f"be future-facing (start >= 0)")
    if pq.where is not None:
        _walk_columns(pq.where, schema)
    if pq.assuming is not None:
        _walk_columns(pq.assuming, schema)
    return ValidatedQuery(pq, pq.task_type(schema))
