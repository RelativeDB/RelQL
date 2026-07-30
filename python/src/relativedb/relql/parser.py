"""RelQL parsing + schema-bound validation, pure Python.

Parsing lives in :mod:`._parse` (the port of the former C++ parser); this
module carries the user-facing entry points and the schema-aware semantic
pass — table/column binding, frame direction rules, task-type inference and
RETURN compatibility — that used to run in ``cpp/src/analyze.cpp``. The
error messages are the user-facing contract for a rejected query; tests
assert on their wording.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .ast import (AggFunc, Aggregation, Arith, Case, ColumnRef, Condition,
                  Func, LogicalOp, Not, ParsedQuery, TaskType,
                  UnreferencedParameterError)

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
    """Parse a RelQL string with the pure-Python parser (:mod:`._parse`).
    Raises :class:`RelqlSyntaxError` on malformed input."""
    if not isinstance(query, str) or not query.strip():
        raise RelqlSyntaxError("empty query")
    from ._parse import parse_text
    return parse_text(query)


# ---------------------------------------------------------------------------
# Schema-bound validation (the semantic pass formerly in cpp/src/analyze.cpp).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatedQuery:
    query: ParsedQuery
    task_type: Any  # TaskType
    # The logical plan built from the bound query (relativedb.plan). None only
    # for a query taken as already-bound, which has no source text to analyze;
    # relativedb.plan derives that case from the AST instead.
    plan: Any = None


def _col_str(table: str, column: str) -> str:
    return f"{table}.{column}"


def _check_column(schema, table: str, column: str, *, allow_star: bool,
                  allow_pk: bool, allow_fk: bool) -> None:
    """The primary key is a legal reference even though tables do not list it
    among their columns; pinning a cohort (``WHERE t.pk IN :ids``) depends on
    it."""
    tdef = schema.table(table)
    if tdef is None:
        raise RelqlValidationError(f"unknown table '{table}'")
    if column == "*":
        if not allow_star:
            raise RelqlValidationError(
                f"{_col_str(table, column)}: '*' not allowed here")
        return
    if tdef.column(column) is not None:
        return
    if allow_pk and tdef.primary_key and column == tdef.primary_key:
        return
    if allow_fk:
        for link in schema.links_from(table):
            if link.fk_column == column:
                return
    raise RelqlValidationError(
        f"unknown column '{column}' on table '{table}'")


def _walk_columns(e, schema) -> None:
    if e is None:
        return
    if isinstance(e, Aggregation):
        # FK columns are legal aggregation targets for set/count aggregations;
        # FIRST/LAST exclude them per the docs.
        fk_ok = e.func in (AggFunc.LIST_DISTINCT, AggFunc.ARRAY_AGG,
                           AggFunc.COUNT, AggFunc.COUNT_DISTINCT)
        _check_column(schema, e.column.table, e.column.column,
                      allow_star=True, allow_pk=False, allow_fk=fk_ok)
        # Only a frame the query actually wrote needs a time column to cut
        # rows against. The implied default expresses no temporal intent, so
        # a table without a time column stays legal there.
        if e.window is not None and not e.window.implied:
            tdef = schema.table(e.column.table)
            if tdef is not None and not tdef.time_column:
                raise RelqlValidationError(
                    f"windowed aggregation over '{e.column.table}', which "
                    f"has no time_column")
        _walk_columns(e.filter, schema)
    elif isinstance(e, ColumnRef):
        _check_column(schema, e.table, e.column,
                      allow_star=False, allow_pk=True, allow_fk=False)
    elif isinstance(e, Condition):
        _walk_columns(e.left, schema)
        _walk_columns(e.right_expr, schema)
    elif isinstance(e, LogicalOp):
        _walk_columns(e.left, schema)
        _walk_columns(e.right, schema)
    elif isinstance(e, Not):
        _walk_columns(e.expr, schema)
    elif isinstance(e, Arith):
        _walk_columns(e.left, schema)
        _walk_columns(e.right, schema)
    elif isinstance(e, Func):
        for a in e.args:
            _walk_columns(a, schema)
    elif isinstance(e, Case):
        for c, t in e.whens:
            _walk_columns(c, schema)
            _walk_columns(t, schema)
        _walk_columns(e.else_, schema)
    # Lit / Param carry no columns.


# RETURN kind -> task types it is compatible with (contract section 1).
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
    allowed_tasks = _RETURN_COMPATIBILITY.get(ret.kind)
    if allowed_tasks is None:
        raise RelqlValidationError(f"unknown RETURN kind '{ret.kind}'")
    if task in allowed_tasks:
        return
    allowed = ", ".join(sorted(t.value for t in allowed_tasks))
    raise RelqlValidationError(
        f"RETURN {ret.kind} is not compatible with inferred task "
        f"'{task.value}' (allowed tasks: {allowed})")


def _fmt_bound(v: float) -> str:
    """Frame offsets in messages print as integers, like the C++ pass did."""
    import math
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return str(int(v))


def _validate_bound(pq: ParsedQuery, schema) -> tuple[ParsedQuery, TaskType]:
    """Validate + bind one parsed query against ``schema``; returns the bound
    query and its inferred task type."""
    tdef = schema.table(pq.entity_key.table)
    if tdef is None:
        origin = ("inferred from the PREDICT target" if pq.entity_inferred
                  else "named by FROM")
        raise RelqlValidationError(
            f"unknown entity table '{pq.entity_key.table}' ({origin})")
    if not tdef.primary_key:
        raise RelqlValidationError(
            f"table '{pq.entity_key.table}' declares no primary key, so it "
            f"cannot be a population")
    if (pq.entity_key.column
            and tdef.primary_key != pq.entity_key.column):
        raise RelqlValidationError(
            f"{pq.entity_key.table}.{pq.entity_key.column}: "
            f"'{pq.entity_key.column}' is not the primary key of "
            f"'{pq.entity_key.table}' (expected '{tdef.primary_key}')")

    bound = pq.bind_entity_key(schema)

    _walk_columns(bound.target, schema)
    for agg in bound.target_aggregations:
        if agg.window is not None and agg.window.start < 0:
            raise RelqlValidationError(
                f"target window ({_fmt_bound(agg.window.start)}, "
                f"{_fmt_bound(agg.window.end)}] must be future-facing "
                f"(start >= 0)")

    from .ast import _find_aggregations
    for name, clause in (("WHERE", bound.where), ("ASSUMING", bound.assuming)):
        if clause is None:
            continue
        _walk_columns(clause, schema)
        for agg in _find_aggregations(clause):
            if agg.window is not None and agg.window.horizons > 1:
                raise RelqlValidationError(
                    f"HORIZONS > 1 is only allowed on the PREDICT target, "
                    f"not in {name}")

    task = bound.task_type(schema)
    if bound.ret is not None:
        _validate_return(bound.ret, task)
    return bound, task


def validate(query, schema, params=None) -> ValidatedQuery:
    """Parse + bind against a schema: tables/columns exist, the population's
    primary key is resolved, target windows are future-facing (start >= 0).

    ``params`` distinguishes two things. ``None`` (the default) validates
    WITHOUT binding: a parameterized query is legitimate to validate before
    its values exist, and callers do that to get the bound primary key and the
    task type. A dict -- including an empty one -- binds now, so an unsupplied
    ``:name`` is reported here rather than surfacing much later. Binding also
    matters for planning: a cohort pinned through ``IN :ids`` is only visible
    once bound.

    ``ValidatedQuery.query`` is the *bound* query — same AST, with the
    population's primary key filled in — so callers should use it rather than
    the query they passed in."""
    text = query if isinstance(query, str) else query.text
    if params and text:
        # A supplied binding the query never references is a hard error, not
        # a no-op: the binder would drop it silently, and a dropped cohort pin
        # (`params={"ids": ...}` without `IN :ids`) turns into a whole-table
        # scan. Binding rebinds from the TEXT, so the check must parse the
        # text too — a ParsedQuery handed back in (the engine re-validates
        # bound queries) may already have its params folded into literals. The
        # no-text already-bound path is not checked here: its params are the
        # consumed bindings of the query it was derived from, and the engine's
        # cohort resolution guards the dangerous case.
        ref = parse(text)
        extra = set(params) - ref.param_names()
        if extra:
            raise UnreferencedParameterError(extra, ref.param_names())
    if not text:
        # An already-bound query built by AST surgery rather than written by a
        # user -- the engine's hurdle composition derives one. There is no
        # faithful source text to re-analyze (the text it inherited describes a
        # different query), so take it as given and infer the task from the
        # AST.
        if not isinstance(query, ParsedQuery) or not query.entity_key.column:
            raise RelqlValidationError(
                "validate() needs the query text; pass the query string or a "
                "ParsedQuery produced by parse()")
        return ValidatedQuery(query, query.task_type(schema), None)

    # Rebind from the text: a ParsedQuery handed back in may carry folded
    # params, and the semantic pass is defined over what the user wrote.
    bound, task = _validate_bound(parse(text), schema)
    if params is not None:
        # bind_params raises MissingParameterError for an unsupplied :name.
        # (The unreferenced check ran above, against the freshly parsed text.)
        bound = bound.bind_params(params)
    # Cache the schema-aware inference so ParsedQuery.task_type() answers from
    # this pass rather than re-deriving the same rules schema-less.
    object.__setattr__(bound, "_task_type", task)
    from ..plan import _logical_from_ast
    logical = _logical_from_ast(schema, bound)
    return ValidatedQuery(bound, task, logical)
