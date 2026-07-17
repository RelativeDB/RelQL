"""Hand-written recursive-descent PQL parser, faithful to ``grammar/Pql.g4``.

Chosen over the ANTLR Python target to avoid a codegen/runtime dependency;
the grammar is small and stable. The full 44-query corpus in
``grammar/examples.pql`` is the conformance suite (see tests).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .ast import (AggFunc, Aggregation, BoolOp, ColumnRef, Condition,
                  LogicalOp, Not, Operator, ParsedQuery, RankKind, TargetExpr,
                  TimeUnit, Window)

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


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

_KEYWORDS = {
    "PREDICT", "FORECAST", "TIMEFRAMES", "FOR", "EACH", "WHERE", "ASSUMING",
    "CLASSIFY", "RANK", "TOP",
    "SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT_DISTINCT", "LIST_DISTINCT",
    "FIRST", "LAST",
    "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "CONTAINS", "STARTS",
    "ENDS", "WITH",
    "SECONDS", "MINUTES", "HOURS", "DAYS", "WEEKS", "MONTHS", "INF",
}

# Keywords permitted as identifiers (grammar softKeyword rule): everything
# except the structural clause words and boolean/null words.
_SOFT_KEYWORDS = _KEYWORDS - {
    "PREDICT", "FOR", "WHERE", "ASSUMING", "AND", "OR", "NOT", "NULL",
}

_AGG_FUNCS = {f.name for f in AggFunc}
_TIME_UNITS = {u.name for u in TimeUnit}

_TOKEN_RE = re.compile(r"""
      (?P<WS>\s+|--[^\r\n]*|/\*.*?\*/)
    | (?P<DATE>\d{4}-\d{2}-\d{2}(?:\ \d{2}:\d{2}:\d{2})?)
    | (?P<FLOAT>\d+\.\d+)
    | (?P<INT>\d+)
    | (?P<STRING>'(?:[^'\\]|\\.|'')*'|"(?:[^"\\]|\\.|"")*")
    | (?P<IDENT>[A-Za-z_][A-Za-z_0-9]*)
    | (?P<OP>>=|<=|!=|==|[><=(),.*+\-])
""", re.VERBOSE | re.DOTALL)


@dataclass(frozen=True)
class _Token:
    kind: str          # keyword name | IDENT | INT | FLOAT | STRING | DATE | op char | EOF
    value: Any
    pos: int


def _lex(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise PqlSyntaxError(f"unexpected character {text[pos]!r}", pos, text)
        if m.lastgroup == "WS":
            pos = m.end()
            continue
        val = m.group()
        if m.lastgroup == "DATE":
            tokens.append(_Token("DATE", val, pos))
        elif m.lastgroup == "FLOAT":
            tokens.append(_Token("FLOAT", float(val), pos))
        elif m.lastgroup == "INT":
            tokens.append(_Token("INT", int(val), pos))
        elif m.lastgroup == "STRING":
            q = val[0]
            body = val[1:-1].replace(q * 2, q)
            body = re.sub(r"\\(.)", r"\1", body)
            tokens.append(_Token("STRING", body, pos))
        elif m.lastgroup == "IDENT":
            upper = val.upper()
            if upper in _KEYWORDS:
                tokens.append(_Token(upper, val, pos))
            else:
                tokens.append(_Token("IDENT", val, pos))
        else:  # OP
            tokens.append(_Token(val, val, pos))
        pos = m.end()
    tokens.append(_Token("EOF", None, n))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_COMPARISON_SYMBOLS = {">": Operator.GT, "<": Operator.LT, "=": Operator.EQ,
                       "==": Operator.EQ, "!=": Operator.NEQ,
                       ">=": Operator.GE, "<=": Operator.LE}


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _lex(text)
        self.i = 0

    # -- token helpers ------------------------------------------------------
    def peek(self, offset: int = 0) -> _Token:
        j = min(self.i + offset, len(self.tokens) - 1)
        return self.tokens[j]

    def next(self) -> _Token:
        t = self.tokens[self.i]
        if t.kind != "EOF":
            self.i += 1
        return t

    def accept(self, kind: str) -> Optional[_Token]:
        if self.peek().kind == kind:
            return self.next()
        return None

    def expect(self, kind: str, what: str = "") -> _Token:
        t = self.peek()
        if t.kind != kind:
            raise PqlSyntaxError(
                f"expected {what or kind}, found {t.kind}", t.pos, self.text)
        return self.next()

    def error(self, message: str) -> PqlSyntaxError:
        return PqlSyntaxError(message, self.peek().pos, self.text)

    # -- grammar ------------------------------------------------------------
    def parse_query(self) -> ParsedQuery:
        self.expect("PREDICT", "'PREDICT'")
        target = self.parse_expr()
        rank: Optional[RankKind] = None
        top_k: Optional[int] = None
        if self.accept("CLASSIFY"):
            rank = RankKind.CLASSIFY
        elif self.peek().kind == "RANK" and self.peek(1).kind == "TOP":
            self.next()
            self.next()
            top_k = int(self.expect("INT", "an integer after RANK TOP").value)
            rank = RankKind.RANK
        num_forecasts: Optional[int] = None
        if self.peek().kind == "FORECAST" and self.peek(1).kind == "INT":
            self.next()
            num_forecasts = int(self.next().value)
            self.expect("TIMEFRAMES", "'TIMEFRAMES'")
        self.expect("FOR", "'FOR'")
        # EACH is a soft keyword: only consume it as the EACH marker when it
        # is not itself the table name of the entity columnRef (`EACH.x`).
        if self.peek().kind == "EACH" and self.peek(1).kind != ".":
            self.next()
        entity_key = self.parse_column_ref()
        entity_ids: tuple = ()
        if self.accept("="):
            entity_ids = (self.parse_literal(),)
        elif self.peek().kind == "IN" and self.peek(1).kind == "(":
            self.next()
            entity_ids = tuple(self.parse_list_literal())
        where = self.parse_expr() if self.accept("WHERE") else None
        assuming = self.parse_expr() if self.accept("ASSUMING") else None
        self.expect("EOF", "end of query")
        return ParsedQuery(target=target, entity_key=entity_key,
                           entity_ids=entity_ids, where=where,
                           assuming=assuming, rank=rank, top_k=top_k,
                           num_forecasts=num_forecasts, text=self.text)

    # expr precedence: parens > NOT > AND > OR
    def parse_expr(self) -> TargetExpr:
        left = self.parse_and()
        while self.accept("OR"):
            left = LogicalOp(left, BoolOp.OR, self.parse_and())
        return left

    def parse_and(self) -> TargetExpr:
        left = self.parse_not()
        while self.accept("AND"):
            left = LogicalOp(left, BoolOp.AND, self.parse_not())
        return left

    def parse_not(self) -> TargetExpr:
        if self.accept("NOT"):
            return Not(self.parse_not())
        return self.parse_primary()

    def parse_primary(self) -> TargetExpr:
        if self.accept("("):
            inner = self.parse_expr()
            self.expect(")", "')'")
            return inner
        return self.parse_predicate()

    def parse_predicate(self) -> TargetExpr:
        value = self.parse_value_expr()
        t = self.peek()
        # symbol comparisons
        if t.kind in _COMPARISON_SYMBOLS:
            self.next()
            return Condition(value, _COMPARISON_SYMBOLS[t.kind],
                             self.parse_literal())
        # word comparisons
        if t.kind == "STARTS":
            self.next()
            self.expect("WITH", "'WITH' after STARTS")
            return Condition(value, Operator.STARTS_WITH, self.parse_literal())
        if t.kind == "ENDS":
            self.next()
            self.expect("WITH", "'WITH' after ENDS")
            return Condition(value, Operator.ENDS_WITH, self.parse_literal())
        if t.kind == "CONTAINS":
            self.next()
            return Condition(value, Operator.CONTAINS, self.parse_literal())
        if t.kind == "LIKE":
            self.next()
            return Condition(value, Operator.LIKE, self.parse_literal())
        if t.kind == "NOT" and self.peek(1).kind in ("CONTAINS", "LIKE", "IN"):
            self.next()
            op_tok = self.next().kind
            if op_tok == "CONTAINS":
                return Condition(value, Operator.NOT_CONTAINS,
                                 self.parse_literal())
            if op_tok == "LIKE":
                return Condition(value, Operator.NOT_LIKE, self.parse_literal())
            return Condition(value, Operator.NOT_IN,
                             tuple(self.parse_list_literal()))
        if t.kind == "IN":
            self.next()
            return Condition(value, Operator.IN,
                             tuple(self.parse_list_literal()))
        if t.kind == "IS":
            if self.peek(1).kind == "IN":
                self.next()
                self.next()
                return Condition(value, Operator.IN,
                                 tuple(self.parse_list_literal()))
            self.next()
            negated = self.accept("NOT") is not None
            self.expect("NULL", "'NULL'")
            return Condition(value,
                             Operator.IS_NOT_NULL if negated else Operator.IS_NULL)
        return value  # bare value predicate (regression target)

    def parse_value_expr(self) -> TargetExpr:
        t = self.peek()
        if t.kind in _AGG_FUNCS and self.peek(1).kind == "(":
            return self.parse_aggregation()
        return self.parse_column_ref()

    def parse_aggregation(self) -> Aggregation:
        func = AggFunc[self.next().kind]
        self.expect("(", "'('")
        column = self.parse_column_ref()
        filt = self.parse_expr() if self.accept("WHERE") else None
        window: Optional[Window] = None
        if self.accept(","):
            start = self.parse_bound()
            self.expect(",", "',' between window bounds")
            end = self.parse_bound()
            unit = TimeUnit.DAYS
            if self.accept(","):
                ut = self.next()
                if ut.kind not in _TIME_UNITS:
                    raise PqlSyntaxError(f"expected a time unit, found {ut.kind}",
                                         ut.pos, self.text)
                unit = TimeUnit[ut.kind]
            window = Window(start, end, unit)
        self.expect(")", "')' to close aggregation")
        return Aggregation(func, column, filt, window)

    def parse_bound(self) -> float:
        sign = 1.0
        if self.accept("+"):
            sign = 1.0
        elif self.accept("-"):
            sign = -1.0
        t = self.next()
        if t.kind == "INT":
            return sign * t.value
        if t.kind == "INF":
            return sign * math.inf
        raise PqlSyntaxError(f"expected a window bound, found {t.kind}",
                             t.pos, self.text)

    def parse_column_ref(self) -> ColumnRef:
        table = self.parse_name("a table name")
        self.expect(".", "'.' in table.column reference")
        if self.accept("*"):
            return ColumnRef(table, "*")
        return ColumnRef(table, self.parse_name("a column name"))

    def parse_name(self, what: str) -> str:
        t = self.peek()
        if t.kind == "IDENT" or t.kind in _SOFT_KEYWORDS:
            self.next()
            return str(t.value)
        raise PqlSyntaxError(f"expected {what}, found {t.kind}", t.pos, self.text)

    def parse_list_literal(self) -> list:
        self.expect("(", "'(' to open a literal list")
        items = [self.parse_literal()]
        while self.accept(","):
            items.append(self.parse_literal())
        self.expect(")", "')' to close a literal list")
        return items

    def parse_literal(self) -> Any:
        t = self.next()
        if t.kind == "STRING":
            return t.value
        if t.kind == "DATE":
            fmt = "%Y-%m-%d %H:%M:%S" if " " in t.value else "%Y-%m-%d"
            return datetime.strptime(t.value, fmt)
        if t.kind == "NULL":
            return None
        if t.kind in ("+", "-"):
            sign = -1 if t.kind == "-" else 1
            n = self.next()
            if n.kind in ("INT", "FLOAT"):
                return sign * n.value
            raise PqlSyntaxError(f"expected a number after {t.kind!r}",
                                 n.pos, self.text)
        if t.kind in ("INT", "FLOAT"):
            return t.value
        raise PqlSyntaxError(f"expected a literal, found {t.kind}",
                             t.pos, self.text)


def parse(query: str) -> ParsedQuery:
    """Parse only — no schema needed. Raises :class:`PqlSyntaxError`."""
    if not isinstance(query, str) or not query.strip():
        raise PqlSyntaxError("empty query")
    return _Parser(query).parse_query()


# ---------------------------------------------------------------------------
# Schema-bound validation
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
