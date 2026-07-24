"""RelQL parsing (single-sourced in C++) + schema-bound validation.

Parsing lives once in the native layer — ``librt_c``'s ``relql_parse`` (see
:mod:`relativedb.relql.native`), shared by the Python, Java, and Rust bindings.
``librt_c`` is a hard dependency: it is the same native library the RT-J model
requires to run. Only the schema-binding validation stays in Python.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .ast import ParsedQuery, TaskType

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
# Schema-bound validation. Both the grammar and the semantic rules live in
# C++ (cpp/src/relql.cpp, cpp/src/analyze.cpp); this is the Python surface.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidatedQuery:
    query: ParsedQuery
    task_type: Any  # TaskType
    # The logical plan the C++ pass built from the bound query (cpp/src/plan.cpp).
    # None only for a query taken as already-bound, which has no source text to
    # analyze; relativedb.plan derives that case from the AST instead.
    plan: Any = None


def validate(query, schema, params=None) -> ValidatedQuery:
    """Parse + bind against a schema: tables/columns exist, the population's
    primary key is resolved, target windows are future-facing (start >= 0).

    The rules live in the C++ layer (``relql_analyze``, cpp/src/analyze.cpp) so
    every binding enforces the same ones; this is the Python surface over them.
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
    from .native import analyze_native, params_to_json

    text = query if isinstance(query, str) else query.text
    if not text:
        # An already-bound query built by AST surgery rather than written by a
        # user -- the engine's hurdle composition derives one. There is no
        # faithful source text to re-analyze (the text it inherited describes a
        # different query), so take it as given and infer the task from the
        # AST. Closing this properly needs an AST-input entry point on the C
        # ABI; until then this is the one path the C++ pass does not cover.
        if not isinstance(query, ParsedQuery) or not query.entity_key.column:
            raise RelqlValidationError(
                "validate() needs the query text; pass the query string or a "
                "ParsedQuery produced by parse()")
        return ValidatedQuery(query, query.task_type(schema), None)
    bound, task_name, logical = analyze_native(
        text, json.dumps(schema.to_json_dict()),
        "" if params is None else params_to_json(params), params)
    task = TaskType(task_name)
    # The C++ pass already inferred the task with the schema in hand; cache it
    # so ParsedQuery.task_type() answers from that rather than re-deriving the
    # same rules in Python.
    object.__setattr__(bound, "_task_type", task)
    return ValidatedQuery(bound, task, logical)
