"""The RelQL lexer + recursive-descent parser, pure Python.

This is a line-for-line port of the C++ parser that previously lived in
``cpp/src/relql.cpp`` (removed when query parsing moved out of the native
layer). The grammar reference is the RelQL book and ``RelQL_EVOLUTION.md``;
the structure below mirrors the C++ (lexer, one method per production, the
post-parse passes in the same order) so a grammar change can be reviewed
against the corpus in ``python/tests/data/examples.relql``.

Parsing builds small MUTABLE nodes (like the C++ ``Expr``) because the
post-parse passes — window-ref resolution, alias/unqualified column binding,
implied frames — rewrite the tree in place; :func:`_freeze` converts the
result to the frozen dataclasses in :mod:`relativedb.relql.ast` once, at the
end.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

from .ast import (Ablation, AggFunc, Aggregation, Arith, AsOf, BoolOp, Case,
                  ColumnRef, Condition, Explain, Func, Lit, LogicalOp, Not,
                  Operator, Param, ParsedQuery, RankKind, ReturnSpec,
                  TimeUnit, Window)

__all__ = ["parse_text"]

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

_KEYWORDS = frozenset({
    # structural / clauses
    "PREDICT", "FROM", "WHERE", "ASSUMING", "CLASSIFY", "RANK", "TOP",
    "AS", "OF", "RETURN", "ABLATE", "TABLE", "EXPLAIN", "PLAN", "CONTEXT",
    "ANALYZE", "ABLATION", "FORMAT", "TEXT", "JSON", "WINDOW", "OVER",
    # frames
    "RANGE", "BETWEEN", "PRECEDING", "FOLLOWING", "UNBOUNDED", "NOW",
    "HORIZONS", "STEP",
    # aggregations
    "SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT_DISTINCT", "LIST_DISTINCT",
    "ARRAY_AGG", "FIRST", "LAST", "EXISTS",
    # value functions / literals
    "CASE", "WHEN", "THEN", "ELSE", "END", "COALESCE", "NULLIF", "ABS", "LOG",
    "EXP", "LEAST", "GREATEST", "TRUE", "FALSE",
    # RETURN outputs
    "EXPECTED", "VALUE", "PROBABILITY", "CLASS", "DISTRIBUTION", "QUANTILES",
    "INTERVAL", "MULTILABEL", "MULTICLASS",
    # boolean / predicate operators
    "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "CONTAINS", "STARTS",
    "ENDS", "WITH",
    # duration units (singular + plural)
    "SECOND", "SECONDS", "MINUTE", "MINUTES", "HOUR", "HOURS", "DAY", "DAYS",
    "WEEK", "WEEKS", "MONTH", "MONTHS", "YEAR", "YEARS"})

# Soft keywords may also appear as table/column identifiers. Everything except
# the truly structural/boolean words is soft, preserving backward compatibility
# with schemas whose names collide with keywords (e.g. a column named "value").
_SOFT_KEYWORDS = _KEYWORDS - {"PREDICT", "FROM", "WHERE", "ASSUMING", "AND",
                              "OR", "NOT", "NULL"}

_AGG_FUNC_NAMES = frozenset({
    "SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT_DISTINCT", "LIST_DISTINCT",
    "ARRAY_AGG", "FIRST", "LAST"})

_VALUE_FUNC_NAMES = frozenset({
    "COALESCE", "NULLIF", "ABS", "LOG", "EXP", "LEAST", "GREATEST"})

_UNIT_NAMES = {
    "SECOND": TimeUnit.SECONDS, "SECONDS": TimeUnit.SECONDS,
    "MINUTE": TimeUnit.MINUTES, "MINUTES": TimeUnit.MINUTES,
    "HOUR": TimeUnit.HOURS, "HOURS": TimeUnit.HOURS,
    "DAY": TimeUnit.DAYS, "DAYS": TimeUnit.DAYS,
    "WEEK": TimeUnit.WEEKS, "WEEKS": TimeUnit.WEEKS,
    "MONTH": TimeUnit.MONTHS, "MONTHS": TimeUnit.MONTHS,
    "YEAR": TimeUnit.YEARS, "YEARS": TimeUnit.YEARS}

# Keywords that open a trailing clause. They are soft elsewhere (a column may
# be named "value"), but in the alias slot they must not be mistaken for one.
_CLAUSE_STARTERS = frozenset({"AS", "ABLATE", "RETURN", "WINDOW", "WHERE",
                              "ASSUMING"})


class _Token:
    __slots__ = ("kind", "text", "ival", "dval", "pos")

    def __init__(self, kind: str, text: str = "", ival: int = 0,
                 dval: float = 0.0, pos: int = 0):
        self.kind = kind    # keyword name | IDENT | INT | FLOAT | STRING |
        self.text = text    # DATE | op string | EOF
        self.ival = ival
        self.dval = dval
        self.pos = pos


def _syntax_error(message: str, pos: int, text: str):
    from .parser import RelqlSyntaxError
    raise RelqlSyntaxError(message, pos, text)


def _is_digit(c: str) -> bool:
    return "0" <= c <= "9"


def _is_ident_start(c: str) -> bool:
    return c == "_" or "A" <= c <= "Z" or "a" <= c <= "z"


def _is_ident_part(c: str) -> bool:
    return _is_ident_start(c) or _is_digit(c)


def _match_date(t: str, pos: int) -> Optional[int]:
    """DATE at pos: \\d{4}-\\d{2}-\\d{2}( \\d{2}:\\d{2}:\\d{2})? -> length."""
    n = len(t)
    if pos + 10 > n:
        return None
    d = lambda i: _is_digit(t[i])  # noqa: E731
    if not (d(pos) and d(pos + 1) and d(pos + 2) and d(pos + 3)
            and t[pos + 4] == "-" and d(pos + 5) and d(pos + 6)
            and t[pos + 7] == "-" and d(pos + 8) and d(pos + 9)):
        return None
    length = 10
    # optional " HH:MM:SS"
    if (pos + 19 <= n and t[pos + 10] == " " and d(pos + 11) and d(pos + 12)
            and t[pos + 13] == ":" and d(pos + 14) and d(pos + 15)
            and t[pos + 16] == ":" and d(pos + 17) and d(pos + 18)):
        length = 19
    return length


_SINGLE_OPS = "><=(),.*+-/:%"
_TWO_OPS = (">=", "<=", "!=", "==")


def _lex(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    n = len(text)
    while pos < n:
        c = text[pos]
        # ---- whitespace / comments ----
        if c in " \t\r\n":
            pos += 1
            continue
        if c == "-" and pos + 1 < n and text[pos + 1] == "-":  # line comment
            pos += 2
            while pos < n and text[pos] not in "\r\n":
                pos += 1
            continue
        if c == "/" and pos + 1 < n and text[pos + 1] == "*":  # block comment
            end = text.find("*/", pos + 2)
            if end < 0:
                _syntax_error("unterminated block comment", pos, text)
            pos = end + 2
            continue
        # ---- DATE (before INT/FLOAT) ----
        if _is_digit(c):
            dlen = _match_date(text, pos)
            if dlen is not None:
                tokens.append(_Token("DATE", text[pos:pos + dlen], pos=pos))
                pos += dlen
                continue
        # ---- FLOAT / INT ----
        if _is_digit(c):
            start = pos
            while pos < n and _is_digit(text[pos]):
                pos += 1
            is_float = False
            if (pos < n and text[pos] == "." and pos + 1 < n
                    and _is_digit(text[pos + 1])):
                is_float = True
                pos += 1
                while pos < n and _is_digit(text[pos]):
                    pos += 1
            num = text[start:pos]
            if is_float:
                tokens.append(_Token("FLOAT", num, dval=float(num), pos=start))
            else:
                tokens.append(_Token("INT", num, ival=int(num), pos=start))
            continue
        # ---- STRING ----
        if c in "'\"":
            q = c
            start = pos
            i = pos + 1
            closed = False
            while i < n:
                ch = text[i]
                if ch == "\\" and i + 1 < n:
                    i += 2
                    continue
                if ch == q:
                    if i + 1 < n and text[i + 1] == q:  # doubled quote escape
                        i += 2
                        continue
                    closed = True
                    break
                i += 1
            if not closed:
                _syntax_error("unterminated string literal", start, text)
            inner = text[start + 1:i]
            # doubled-quote unescape, then backslash unescape (C++ order)
            body = []
            k = 0
            while k < len(inner):
                if k + 1 < len(inner) and inner[k] == q and inner[k + 1] == q:
                    body.append(q)
                    k += 2
                else:
                    body.append(inner[k])
                    k += 1
            body = "".join(body)
            out = []
            k = 0
            while k < len(body):
                if body[k] == "\\" and k + 1 < len(body):
                    out.append(body[k + 1])
                    k += 2
                else:
                    out.append(body[k])
                    k += 1
            tokens.append(_Token("STRING", "".join(out), pos=start))
            pos = i + 1
            continue
        # ---- IDENT / keyword ----
        if _is_ident_start(c):
            start = pos
            while pos < n and _is_ident_part(text[pos]):
                pos += 1
            word = text[start:pos]
            up = word.upper()
            tokens.append(_Token(up if up in _KEYWORDS else "IDENT", word,
                                 pos=start))
            continue
        # ---- operators ----
        if pos + 1 < n and text[pos:pos + 2] in _TWO_OPS:
            two = text[pos:pos + 2]
            tokens.append(_Token(two, two, pos=pos))
            pos += 2
            continue
        if c in _SINGLE_OPS:
            tokens.append(_Token(c, c, pos=pos))
            pos += 1
            continue
        _syntax_error(f"unexpected character '{c}'", pos, text)
    tokens.append(_Token("EOF", pos=n))
    return tokens


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

_COMPARISONS = {
    ">": Operator.GT, "<": Operator.LT, "=": Operator.EQ, "==": Operator.EQ,
    "!=": Operator.NEQ, ">=": Operator.GE, "<=": Operator.LE}


def _is_calendar_unit(u: TimeUnit) -> bool:
    return u in (TimeUnit.MONTHS, TimeUnit.YEARS)


_UNIT_SECONDS = {TimeUnit.SECONDS: 1.0, TimeUnit.MINUTES: 60.0,
                 TimeUnit.HOURS: 3600.0, TimeUnit.DAYS: 86400.0,
                 TimeUnit.WEEKS: 604800.0}


def _unit_seconds(u: TimeUnit) -> float:
    return _UNIT_SECONDS.get(u, 0.0)   # 0.0 for calendar


def _unit_months(u: TimeUnit) -> float:
    return 12.0 if u is TimeUnit.YEARS else 1.0   # MONTHS


class _Bound:
    """A single frame endpoint: an offset with an optional unit (NOW /
    UNBOUNDED carry no unit)."""

    __slots__ = ("off", "finite", "has_unit", "unit")

    def __init__(self):
        self.off = 0.0
        self.finite = True
        self.has_unit = False
        self.unit = TimeUnit.DAYS


class _MutWindow:
    __slots__ = ("start", "end", "unit", "horizons", "step", "has_step",
                 "top_k", "has_top_k", "implied")

    def __init__(self):
        self.start = 0.0
        self.end = 0.0
        self.unit = TimeUnit.DAYS
        self.horizons = 1
        self.step = 0.0
        self.has_step = False
        self.top_k = 0
        self.has_top_k = False
        self.implied = False


class _Expr:
    """Mutable parse node, mirroring the C++ ``Expr`` union-ish struct."""

    __slots__ = ("kind", "table", "column", "func", "filter", "window",
                 "window_ref", "left", "op", "right", "has_right",
                 "right_expr", "bop", "rleft", "rright", "inner", "arith_op",
                 "a_left", "a_right", "func_name", "args", "when_conds",
                 "when_thens", "case_else", "lit", "param_name")

    def __init__(self, kind: str):
        self.kind = kind             # col|agg|cond|logic|not|arith|func|case|
        self.table = ""              # lit|param
        self.column = ""
        self.func: Optional[AggFunc] = None
        self.filter: Optional[_Expr] = None
        self.window: Optional[_MutWindow] = None
        self.window_ref = ""
        self.left: Optional[_Expr] = None
        self.op: Optional[Operator] = None
        self.right: Any = None
        self.has_right = False
        self.right_expr: Optional[_Expr] = None
        self.bop: Optional[BoolOp] = None
        self.rleft: Optional[_Expr] = None
        self.rright: Optional[_Expr] = None
        self.inner: Optional[_Expr] = None
        self.arith_op = ""
        self.a_left: Optional[_Expr] = None
        self.a_right: Optional[_Expr] = None
        self.func_name = ""
        self.args: list[_Expr] = []
        self.when_conds: list[_Expr] = []
        self.when_thens: list[_Expr] = []
        self.case_else: Optional[_Expr] = None
        self.lit: Any = None         # ("int"|"float"|"str"|"bool"|"null"|
        self.param_name = ""         #  "date"|"list", value)


# Literals carry their kind so unary-minus folding and JSON-identical value
# conversion stay exact (a float keeps its source text; a date its string).
class _Lit:
    __slots__ = ("kind", "ival", "dval", "sval", "bval", "items")

    def __init__(self, kind: str):
        self.kind = kind
        self.ival = 0
        self.dval = 0.0
        self.sval = ""
        self.bval = False
        self.items: list["_Lit"] = []

    def value(self) -> Any:
        """The Python value the old JSON decoder produced for this literal."""
        if self.kind == "int":
            return self.ival
        if self.kind == "float":
            # The C++ emitter wrote the raw numeric text into JSON; parsing
            # that text is exactly float(sval).
            return float(self.sval)
        if self.kind == "str":
            return self.sval
        if self.kind == "bool":
            return self.bval
        if self.kind == "null":
            return None
        if self.kind == "date":
            fmt = "%Y-%m-%d %H:%M:%S" if " " in self.sval else "%Y-%m-%d"
            return datetime.strptime(self.sval, fmt)
        if self.kind == "list":
            return tuple(x.value() for x in self.items)
        raise AssertionError(self.kind)


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _lex(text)
        self.i = 0
        self.window_ref_sites: list[_Expr] = []  # aggs with OVER <name>
        self.from_alias = ""                     # FROM <table> <alias>

    # ---- token plumbing ----------------------------------------------------
    def peek(self, offset: int = 0) -> _Token:
        j = min(self.i + offset, len(self.tokens) - 1)
        return self.tokens[j]

    def next(self) -> _Token:
        t = self.tokens[self.i]
        if t.kind != "EOF":
            self.i += 1
        return t

    def accept(self, kind: str) -> bool:
        if self.peek().kind == kind:
            self.next()
            return True
        return False

    def expect(self, kind: str, what: str) -> _Token:
        t = self.peek()
        if t.kind != kind:
            _syntax_error(f"expected {what or kind}, found {t.kind}", t.pos,
                          self.text)
        return self.next()

    def error(self, message: str, pos: int):
        _syntax_error(message, pos, self.text)

    # ---- entry --------------------------------------------------------------
    def parse_query(self) -> ParsedQuery:
        q = _MutQuery()
        self.parse_explain_prefix(q)
        self.expect("PREDICT", "'PREDICT'")
        q.target = self.parse_expr()
        if self.accept("CLASSIFY"):
            q.rank = RankKind.CLASSIFY
        self.parse_from(q)
        self.parse_trailing_clauses(q)
        self.expect("EOF", "end of query")
        self.resolve_window_refs(q)
        self.resolve_names(q)
        self.apply_implied_windows(q)
        self.derive_forecasts(q)
        self.derive_rank(q)
        return _freeze_query(q, self.text)

    # ---- FROM ---------------------------------------------------------------
    # from_clause := FROM <table> [[AS] <alias>]
    #
    # Optional. Without it the population is the target's own table, which only
    # works when every column in the target agrees on one table (`PREDICT
    # issues.label`); an aggregation target names a *linked* table, so it can
    # never stand in for the population and requires an explicit FROM.
    def parse_from(self, q: "_MutQuery") -> None:
        if not self.accept("FROM"):
            return
        q.has_from = True
        q.entity_table = self.parse_name("a table name after FROM")
        # `AS` also opens `AS OF`; only treat it as an alias marker otherwise.
        if self.peek().kind == "AS" and self.peek(1).kind != "OF":
            self.next()
        t = self.peek()
        if ((t.kind == "IDENT" or t.kind in _SOFT_KEYWORDS)
                and t.kind not in _CLAUSE_STARTERS):
            self.from_alias = self.next().text

    # Infer the population from the target when there is no FROM clause.
    def infer_entity_table(self, q: "_MutQuery") -> None:
        aggs: list[_Expr] = []
        _collect_aggs(q.target, aggs)
        if aggs:
            self.error(
                "an aggregate target needs a FROM clause to name the "
                "population (the aggregation's table is a linked table, not "
                "the population)", 0)
        tables: list[str] = []
        _collect_tables(q.target, tables)
        if not tables:
            self.error("cannot infer the population: PREDICT names no table; "
                       "add a FROM clause", 0)
        for t in tables:
            if t != tables[0]:
                self.error(
                    f"cannot infer the population: PREDICT spans tables "
                    f"'{tables[0]}' and '{t}'; add a FROM clause", 0)
        q.entity_table = tables[0]

    # ---- post-parse: alias + unqualified column resolution ------------------
    # Column references parse before FROM is seen, so binding happens here:
    # `c.plan` -> `customers.plan` for `FROM customers c`, and a bare `label`
    # -> `issues.label`. After this pass the AST holds only real table names.
    def resolve_names(self, q: "_MutQuery") -> None:
        if not q.has_from:
            self.infer_entity_table(q)
        cols: list[_Expr] = []
        _collect_columns(q.target, cols)
        _collect_columns(q.where, cols)
        _collect_columns(q.assuming, cols)
        for e in cols:
            if not e.table:
                e.table = q.entity_table      # unqualified -> the population
            elif self.from_alias and e.table == self.from_alias:
                e.table = q.entity_table      # alias -> its table

    # ---- post-parse: implied unbounded frames -------------------------------
    # An aggregation with no OVER is unbounded, in the direction of the clause
    # it sits in: the future in PREDICT/ASSUMING, the past in WHERE. Filling the
    # frame in here keeps one downstream code path and makes the default visible
    # in EXPLAIN.
    def apply_implied_windows(self, q: "_MutQuery") -> None:
        self._implied_into(q.target, future=True)
        self._implied_into(q.assuming, future=True)
        self._implied_into(q.where, future=False)

    def _implied_into(self, root: Optional[_Expr], *, future: bool) -> None:
        aggs: list[_Expr] = []
        _collect_aggs(root, aggs)
        for a in aggs:
            if a.window is not None:
                continue
            w = _MutWindow()
            w.unit = TimeUnit.DAYS
            w.implied = True
            w.start = 0.0 if future else -math.inf
            w.end = math.inf if future else 0.0
            a.window = w

    # ---- prefix / trailing clauses ----------------------------------------
    def parse_explain_prefix(self, q: "_MutQuery") -> None:
        if self.peek().kind != "EXPLAIN":
            return
        self.next()
        q.explain_present = True
        q.explain_mode = "PLAN"
        if self.accept("PLAN"):
            q.explain_mode = "PLAN"
        elif self.accept("CONTEXT"):
            q.explain_mode = "CONTEXT"
        elif self.accept("ANALYZE"):
            q.explain_mode = "ANALYZE"
        elif self.accept("ABLATION") or self.accept("ABLATE"):
            q.explain_mode = "ABLATION"
        if self.accept("FORMAT"):
            if self.accept("TEXT"):
                q.explain_format = "TEXT"
            elif self.accept("JSON"):
                q.explain_format = "JSON"
            else:
                self.error("expected TEXT or JSON after FORMAT",
                           self.peek().pos)

    def parse_trailing_clauses(self, q: "_MutQuery") -> None:
        have_where = have_assuming = have_asof = have_return = False
        while True:
            k = self.peek().kind
            if k == "WHERE":
                if have_where:
                    self.error("duplicate WHERE clause", self.peek().pos)
                self.next()
                q.where = self.parse_expr()
                have_where = True
            elif k == "ASSUMING":
                if have_assuming:
                    self.error("duplicate ASSUMING clause", self.peek().pos)
                self.next()
                q.assuming = self.parse_expr()
                have_assuming = True
            elif k == "AS":
                if have_asof:
                    self.error("duplicate AS OF clause", self.peek().pos)
                self.next()
                self.expect("OF", "'OF' after AS")
                self.parse_as_of(q)
                have_asof = True
            elif k == "ABLATE":
                self.next()
                self.expect("TABLE", "'TABLE' after ABLATE")
                q.ablations.append(
                    self.parse_name("a table name after ABLATE TABLE"))
            elif k == "RETURN":
                if have_return:
                    self.error("duplicate RETURN clause", self.peek().pos)
                self.next()
                self.parse_return(q)
                have_return = True
            elif k == "WINDOW":
                self.next()
                self.parse_window_decl(q)
            else:
                break

    def parse_as_of(self, q: "_MutQuery") -> None:
        q.as_of_present = True
        if self.accept(":"):
            q.as_of_kind = "param"
            q.as_of_value = self.parse_name(
                "a bind parameter name after ':'")
        elif self.peek().kind == "DATE":
            q.as_of_kind = "date"
            q.as_of_value = self.next().text
        elif self.accept("NOW"):
            q.as_of_kind = "now"
        else:
            self.error("expected :param, a DATE, or NOW after AS OF",
                       self.peek().pos)

    def parse_return(self, q: "_MutQuery") -> None:
        q.ret_present = True
        t = self.next()
        k = t.kind
        if k == "EXPECTED":
            self.expect("VALUE", "'VALUE' after EXPECTED")
            q.ret_kind = "EXPECTED_VALUE"
        elif k == "PROBABILITY":
            q.ret_kind = "PROBABILITY"
        elif k == "CLASS":
            q.ret_kind = "CLASS"
        elif k == "DISTRIBUTION":
            q.ret_kind = "DISTRIBUTION"
        elif k == "MULTILABEL":
            q.ret_kind = "MULTILABEL"
        elif k == "MULTICLASS":
            q.ret_kind = "MULTICLASS"
        elif k in ("QUANTILES", "INTERVAL"):
            # Removed from the language: the checkpoint has a single point head
            # and exposes no empirical distribution, so these could never
            # execute. Named explicitly because queries written against the old
            # grammar are out there and deserve better than "unexpected token".
            self.error(
                f"RETURN {k} is not supported: the model exposes a single "
                f"point estimate, not a distribution. Use RETURN EXPECTED "
                f"VALUE for a regression target.", t.pos)
        else:
            self.error(f"expected a RETURN output type, found {k}", t.pos)

    def parse_window_decl(self, q: "_MutQuery") -> None:
        name = self.parse_name("a window name after WINDOW")
        if name in q.windows:
            self.error(f"window '{name}' declared more than once",
                       self.peek().pos)
        self.expect("AS", "'AS' in WINDOW declaration")
        self.expect("(", "'(' to open a window spec")
        w = self.parse_window_spec()
        self.expect(")", "')' to close a window spec")
        q.windows[name] = w

    # ---- boolean / value expression ---------------------------------------
    # expr precedence: OR > AND > NOT > predicate; predicate = value [cmp rhs].
    def parse_expr(self) -> _Expr:
        left = self.parse_and()
        while self.accept("OR"):
            e = _Expr("logic")
            e.bop = BoolOp.OR
            e.rleft = left
            e.rright = self.parse_and()
            left = e
        return left

    def parse_and(self) -> _Expr:
        left = self.parse_not()
        while self.accept("AND"):
            e = _Expr("logic")
            e.bop = BoolOp.AND
            e.rleft = left
            e.rright = self.parse_not()
            left = e
        return left

    def parse_not(self) -> _Expr:
        if self.accept("NOT"):
            e = _Expr("not")
            e.inner = self.parse_not()
            return e
        return self.parse_predicate()

    def parse_param(self) -> _Expr:
        self.expect(":", "':' to open a bind parameter")
        e = _Expr("param")
        e.param_name = self.parse_name("a bind parameter name after ':'")
        return e

    # The RHS of a word operator (LIKE, CONTAINS, STARTS WITH, ...): a literal,
    # or a `:name` parameter standing in for one.
    def parse_rhs_literal_or_param(self, cond: _Expr) -> None:
        if self.peek().kind == ":":
            cond.right_expr = self.parse_param()
        else:
            cond.right = self.parse_literal()
        cond.has_right = True

    # The RHS of IN / NOT IN: a literal list, or a `:name` parameter bound to
    # the whole list (so one query text serves any cohort size).
    def parse_rhs_list_or_param(self, cond: _Expr) -> None:
        if self.peek().kind == ":":
            cond.right_expr = self.parse_param()
        else:
            lst = _Lit("list")
            lst.items = self.parse_list_literal()
            cond.right = lst
        cond.has_right = True

    @staticmethod
    def make_cond(value: _Expr, op: Operator) -> _Expr:
        e = _Expr("cond")
        e.left = value
        e.op = op
        return e

    def parse_predicate(self) -> _Expr:
        value = self.parse_add_expr()
        t = self.peek()
        if t.kind in _COMPARISONS:
            op = _COMPARISONS[t.kind]
            self.next()
            e = self.make_cond(value, op)
            rhs = self.parse_add_expr()
            if rhs.kind == "lit":
                e.right = rhs.lit
                e.has_right = True
            else:
                e.right_expr = rhs
                e.has_right = True
            return e
        if t.kind == "STARTS":
            self.next()
            self.expect("WITH", "'WITH' after STARTS")
            e = self.make_cond(value, Operator.STARTS_WITH)
            self.parse_rhs_literal_or_param(e)
            return e
        if t.kind == "ENDS":
            self.next()
            self.expect("WITH", "'WITH' after ENDS")
            e = self.make_cond(value, Operator.ENDS_WITH)
            self.parse_rhs_literal_or_param(e)
            return e
        if t.kind == "CONTAINS":
            self.next()
            e = self.make_cond(value, Operator.CONTAINS)
            self.parse_rhs_literal_or_param(e)
            return e
        if t.kind == "LIKE":
            self.next()
            e = self.make_cond(value, Operator.LIKE)
            self.parse_rhs_literal_or_param(e)
            return e
        if t.kind == "NOT" and self.peek(1).kind in ("CONTAINS", "LIKE", "IN"):
            self.next()
            op_tok = self.next().kind
            if op_tok == "CONTAINS":
                e = self.make_cond(value, Operator.NOT_CONTAINS)
                self.parse_rhs_literal_or_param(e)
                return e
            if op_tok == "LIKE":
                e = self.make_cond(value, Operator.NOT_LIKE)
                self.parse_rhs_literal_or_param(e)
                return e
            e = self.make_cond(value, Operator.NOT_IN)
            self.parse_rhs_list_or_param(e)
            return e
        if t.kind == "IN":
            self.next()
            e = self.make_cond(value, Operator.IN)
            self.parse_rhs_list_or_param(e)
            return e
        if t.kind == "IS":
            if self.peek(1).kind == "IN":
                self.next()
                self.next()
                e = self.make_cond(value, Operator.IN)
                self.parse_rhs_list_or_param(e)
                return e
            self.next()
            negated = self.accept("NOT")
            self.expect("NULL", "'NULL'")
            return self.make_cond(value, Operator.IS_NOT_NULL if negated
                                  else Operator.IS_NULL)
        return value  # bare value predicate (regression / value target)

    # arithmetic: + - (lowest), * / (higher), unary -, then primary value.
    def parse_add_expr(self) -> _Expr:
        left = self.parse_mul_expr()
        while self.peek().kind in ("+", "-"):
            op = self.next().kind
            e = _Expr("arith")
            e.arith_op = op
            e.a_left = left
            e.a_right = self.parse_mul_expr()
            left = e
        return left

    def parse_mul_expr(self) -> _Expr:
        left = self.parse_unary()
        while self.peek().kind in ("*", "/"):
            op = self.next().kind
            e = _Expr("arith")
            e.arith_op = op
            e.a_left = left
            e.a_right = self.parse_unary()
            left = e
        return left

    def parse_unary(self) -> _Expr:
        if self.peek().kind == "-":
            self.next()
            inner = self.parse_unary()
            if inner.kind == "lit":  # fold -literal
                if inner.lit.kind == "int":
                    inner.lit.ival = -inner.lit.ival
                    return inner
                if inner.lit.kind == "float":
                    inner.lit.dval = -inner.lit.dval
                    inner.lit.sval = "-" + inner.lit.sval
                    return inner
            zero = _Expr("lit")
            zl = _Lit("int")
            zl.ival = 0
            zero.lit = zl
            e = _Expr("arith")
            e.arith_op = "-"
            e.a_left = zero
            e.a_right = inner
            return e
        if self.peek().kind == "+":
            self.next()  # unary plus, no-op
        return self.parse_primary_value()

    @staticmethod
    def lit_expr(l: _Lit) -> _Expr:
        e = _Expr("lit")
        e.lit = l
        return e

    def parse_primary_value(self) -> _Expr:
        t = self.peek()
        if t.kind == "(":
            self.next()
            inner = self.parse_expr()
            self.expect(")", "')'")
            return inner
        if t.kind == "CASE":
            return self.parse_case()
        if t.kind in _VALUE_FUNC_NAMES and self.peek(1).kind == "(":
            return self.parse_func()
        if t.kind == "EXISTS" and self.peek(1).kind == "(":
            return self.parse_aggregation()
        if t.kind in _AGG_FUNC_NAMES and self.peek(1).kind == "(":
            return self.parse_aggregation()
        if t.kind == "TRUE":
            self.next()
            l = _Lit("bool")
            l.bval = True
            return self.lit_expr(l)
        if t.kind == "FALSE":
            self.next()
            l = _Lit("bool")
            l.bval = False
            return self.lit_expr(l)
        if t.kind == "NULL":
            self.next()
            return self.lit_expr(_Lit("null"))
        if t.kind == ":":
            return self.parse_param()
        if t.kind in ("STRING", "DATE", "INT", "FLOAT"):
            return self.lit_expr(self.parse_literal())
        return self.parse_column_ref()

    def parse_case(self) -> _Expr:
        self.expect("CASE", "'CASE'")
        e = _Expr("case")
        if self.peek().kind != "WHEN":
            self.error("expected WHEN after CASE", self.peek().pos)
        while self.accept("WHEN"):
            cond = self.parse_expr()
            self.expect("THEN", "'THEN' in CASE")
            then = self.parse_add_expr()
            e.when_conds.append(cond)
            e.when_thens.append(then)
        if self.accept("ELSE"):
            e.case_else = self.parse_add_expr()
        self.expect("END", "'END' to close CASE")
        return e

    def parse_func(self) -> _Expr:
        e = _Expr("func")
        e.func_name = self.next().kind  # canonical uppercase keyword
        self.expect("(", f"'(' after {e.func_name}")
        e.args.append(self.parse_add_expr())
        while self.accept(","):
            e.args.append(self.parse_add_expr())
        self.expect(")", f"')' to close {e.func_name}")
        nargs = len(e.args)
        if e.func_name in ("ABS", "LOG", "EXP") and nargs != 1:
            self.error(f"{e.func_name} takes exactly 1 argument",
                       self.peek().pos)
        if e.func_name == "NULLIF" and nargs != 2:
            self.error("NULLIF takes exactly 2 arguments", self.peek().pos)
        return e

    def parse_aggregation(self) -> _Expr:
        e = _Expr("agg")
        e.func = AggFunc[self.next().kind]
        self.expect("(", "'('")
        e.table, e.column = self.parse_column_ref_into()
        if self.accept("WHERE"):
            e.filter = self.parse_expr()
        self.expect(")", "')' to close aggregation")
        if self.accept("OVER"):
            if self.peek().kind == "(":
                self.next()
                e.window = self.parse_window_spec()
                self.expect(")", "')' to close window spec")
            else:
                e.window_ref = self.parse_name("a window name after OVER")
                self.window_ref_sites.append(e)
        return e

    # ---- window frames -----------------------------------------------------
    # parse a duration `<positive-int> <unit>` -> (value, unit)
    def parse_duration(self) -> tuple[float, TimeUnit]:
        num = self.peek()
        if num.kind != "INT":
            self.error(
                f"expected a positive number in a duration, found {num.kind}",
                num.pos)
        if num.ival <= 0:
            self.error("durations must be positive", num.pos)
        self.next()
        u = self.next()
        unit = _UNIT_NAMES.get(u.kind)
        if unit is None:
            self.error(f"expected a duration unit (e.g. DAYS), found {u.kind}",
                       u.pos)
        return float(num.ival), unit

    def parse_bound(self) -> _Bound:
        b = _Bound()
        if self.accept("NOW"):
            return b
        if self.accept("UNBOUNDED"):
            if self.accept("PRECEDING"):
                b.finite = False
                b.off = -math.inf
                return b
            if self.accept("FOLLOWING"):
                b.finite = False
                b.off = math.inf
                return b
            self.error("expected PRECEDING or FOLLOWING after UNBOUNDED",
                       self.peek().pos)
        v, u = self.parse_duration()
        b.has_unit = True
        b.unit = u
        b.finite = True
        if self.accept("PRECEDING"):
            b.off = -v
            return b
        if self.accept("FOLLOWING"):
            b.off = v
            return b
        self.error("expected PRECEDING or FOLLOWING after a duration",
                   self.peek().pos)

    # window_spec := [frame [HORIZONS int [STEP duration]]] [RANK TOP int]
    #
    # The frame may be omitted when the spec carries only a RANK TOP directive
    # (`OVER (RANK TOP 12)`) — the frame then defaults to unbounded future, the
    # same default an aggregation with no OVER at all gets.
    def parse_window_spec(self) -> _MutWindow:
        if self.peek().kind == "RANK":
            w = _MutWindow()
            w.unit = TimeUnit.DAYS
            w.implied = True
            w.start = 0.0
            w.end = math.inf
            self.parse_rank_top(w)
            return w
        if self.accept("RANGE"):
            self.expect("BETWEEN", "'BETWEEN' after RANGE")
            lo = self.parse_bound()
            self.expect("AND", "'AND' between window bounds")
            hi = self.parse_bound()
        elif self.peek().kind == "UNBOUNDED":
            # shorthand: UNBOUNDED PRECEDING => (-inf, NOW]
            lo = self.parse_bound()
            hi = _Bound()
        else:
            # shorthand: <dur> PRECEDING => (-dur, NOW];
            #            <dur> FOLLOWING => (NOW, +dur]
            b = self.parse_bound()
            if b.off < 0:
                lo = b
                hi = _Bound()
            else:
                lo = _Bound()
                hi = b

        unit, lower, upper = self.normalize_frame(lo, hi)

        w = _MutWindow()
        w.start = lower
        w.end = upper
        w.unit = unit

        # validate ordering (extended reals: lower strictly below upper)
        if not (lower < upper):
            self.error(
                "invalid frame: lower bound must be strictly less than upper",
                self.peek().pos)

        if self.accept("HORIZONS"):
            h = self.expect("INT", "a positive integer after HORIZONS")
            if h.ival < 1:
                self.error("HORIZONS must be a positive integer", h.pos)
            w.horizons = h.ival
            if self.accept("STEP"):
                sv, su = self.parse_duration()
                # normalize step to the frame unit (same domain required)
                w.step = self.convert_to_unit(sv, su, unit, h.pos)
                w.has_step = True
            if w.horizons > 1:
                if math.isinf(lower) or math.isinf(upper):
                    self.error("a multi-horizon frame must have finite bounds",
                               h.pos)
                if not w.has_step:
                    # default stride = frame width; stays has_step=False so
                    # the AST keeps step=None (downstream uses span())
                    w.step = upper - lower

        self.parse_rank_top(w)
        return w

    def parse_rank_top(self, w: _MutWindow) -> None:
        if self.peek().kind != "RANK":
            return
        self.next()
        self.expect("TOP", "'TOP' after RANK")
        k = self.expect("INT", "an integer after RANK TOP")
        if k.ival < 1:
            self.error("RANK TOP must be a positive integer", k.pos)
        w.has_top_k = True
        w.top_k = k.ival

    # convert an offset expressed in `frm` to `to` (must share a domain)
    def convert_to_unit(self, v: float, frm: TimeUnit, to: TimeUnit,
                        pos: int) -> float:
        if _is_calendar_unit(frm) != _is_calendar_unit(to):
            self.error(
                "cannot mix fixed and calendar duration units in one frame",
                pos)
        if frm is to:
            return v
        if _is_calendar_unit(frm):
            return v * (_unit_months(frm) / _unit_months(to))
        return v * (_unit_seconds(frm) / _unit_seconds(to))

    # choose a common unit for the two bounds and express offsets in it
    def normalize_frame(self, lo: _Bound,
                        hi: _Bound) -> tuple[TimeUnit, float, float]:
        lo_u = lo.finite and lo.has_unit
        hi_u = hi.finite and hi.has_unit
        if not lo_u and not hi_u:
            # only NOW / UNBOUNDED bounds; unit irrelevant
            return TimeUnit.DAYS, lo.off, hi.off
        any_cal = ((lo_u and _is_calendar_unit(lo.unit))
                   or (hi_u and _is_calendar_unit(hi.unit)))
        any_fixed = ((lo_u and not _is_calendar_unit(lo.unit))
                     or (hi_u and not _is_calendar_unit(hi.unit)))
        if any_cal and any_fixed:
            self.error(
                "cannot mix fixed and calendar duration units in one frame",
                self.peek().pos)
        # pick target unit: smallest fixed present, or MONTHS for calendar
        if any_cal:
            unit = TimeUnit.MONTHS
        else:
            unit = TimeUnit.WEEKS  # start large, shrink to smallest present
            for b, used in ((lo, lo_u), (hi, hi_u)):
                if used and _unit_seconds(b.unit) < _unit_seconds(unit):
                    unit = b.unit
        pos = self.peek().pos
        lower = self.convert_to_unit(lo.off, lo.unit, unit, pos) if lo_u \
            else lo.off
        upper = self.convert_to_unit(hi.off, hi.unit, unit, pos) if hi_u \
            else hi.off
        return unit, lower, upper

    def resolve_window_refs(self, q: "_MutQuery") -> None:
        for e in self.window_ref_sites:
            w = q.windows.get(e.window_ref)
            if w is None:
                self.error(f"undeclared window '{e.window_ref}'", 0)
            e.window = w

    # ---- shared helpers ----------------------------------------------------
    def parse_column_ref(self) -> _Expr:
        e = _Expr("col")
        e.table, e.column = self.parse_column_ref_into()
        return e

    # `table.column`, `alias.column`, `table.*`, or a bare `column`. A bare
    # name leaves `table` empty for resolve_names to bind to the population.
    def parse_column_ref_into(self) -> tuple[str, str]:
        name = self.parse_name("a column or table name")
        if not self.accept("."):
            return "", name
        if self.accept("*"):
            return name, "*"
        return name, self.parse_name("a column name")

    def parse_name(self, what: str) -> str:
        t = self.peek()
        if t.kind == "IDENT" or t.kind in _SOFT_KEYWORDS:
            return self.next().text
        self.error(f"expected {what}, found {t.kind}", t.pos)

    def parse_list_literal(self) -> list[_Lit]:
        self.expect("(", "'(' to open a literal list")
        items = [self.parse_literal()]
        while self.accept(","):
            items.append(self.parse_literal())
        self.expect(")", "')' to close a literal list")
        return items

    def parse_literal(self) -> _Lit:
        t = self.next()
        if t.kind == "STRING":
            l = _Lit("str")
            l.sval = t.text
            return l
        if t.kind == "DATE":
            l = _Lit("date")
            l.sval = t.text
            return l
        if t.kind == "NULL":
            return _Lit("null")
        if t.kind == "TRUE":
            l = _Lit("bool")
            l.bval = True
            return l
        if t.kind == "FALSE":
            l = _Lit("bool")
            l.bval = False
            return l
        if t.kind in ("+", "-"):
            neg = t.kind == "-"
            num = self.next()
            if num.kind == "INT":
                l = _Lit("int")
                l.ival = -num.ival if neg else num.ival
                return l
            if num.kind == "FLOAT":
                l = _Lit("float")
                l.dval = -num.dval if neg else num.dval
                l.sval = ("-" + num.text) if neg else num.text
                return l
            self.error(f"expected a number after '{t.kind}'", num.pos)
        if t.kind == "INT":
            l = _Lit("int")
            l.ival = t.ival
            return l
        if t.kind == "FLOAT":
            l = _Lit("float")
            l.dval = t.dval
            l.sval = t.text
            return l
        self.error(f"expected a literal, found {t.kind}", t.pos)

    # ---- post-parse: derived query fields -----------------------------------
    def derive_forecasts(self, q: "_MutQuery") -> None:
        aggs: list[_Expr] = []
        _collect_aggs(q.target, aggs)
        for a in aggs:
            if a.window is not None and a.window.horizons > 1:
                q.num_forecasts = a.window.horizons
                break

    # Lift a target frame's RANK TOP k to the query level, where task-type
    # inference and the engine read it.
    def derive_rank(self, q: "_MutQuery") -> None:
        aggs: list[_Expr] = []
        _collect_aggs(q.target, aggs)
        for a in aggs:
            if a.window is None or not a.window.has_top_k:
                continue
            if q.top_k is not None and q.top_k != a.window.top_k:
                self.error("conflicting RANK TOP values in one target", 0)
            q.rank = RankKind.RANK
            q.top_k = a.window.top_k


# ---------------------------------------------------------------------------
# post-parse tree walks
# ---------------------------------------------------------------------------

# Every column-bearing node, including aggregation subjects and the columns
# inside an aggregation's inline filter.
def _collect_columns(e: Optional[_Expr], out: list[_Expr]) -> None:
    if e is None:
        return
    k = e.kind
    if k == "col":
        out.append(e)
    elif k == "agg":
        out.append(e)
        _collect_columns(e.filter, out)
    elif k == "cond":
        _collect_columns(e.left, out)
        _collect_columns(e.right_expr, out)
    elif k == "logic":
        _collect_columns(e.rleft, out)
        _collect_columns(e.rright, out)
    elif k == "not":
        _collect_columns(e.inner, out)
    elif k == "arith":
        _collect_columns(e.a_left, out)
        _collect_columns(e.a_right, out)
    elif k == "func":
        for a in e.args:
            _collect_columns(a, out)
    elif k == "case":
        for c in e.when_conds:
            _collect_columns(c, out)
        for th in e.when_thens:
            _collect_columns(th, out)
        _collect_columns(e.case_else, out)


# Distinct table names named by an expression's columns; unqualified columns
# contribute nothing, since they have no table yet.
def _collect_tables(e: Optional[_Expr], out: list[str]) -> None:
    cols: list[_Expr] = []
    _collect_columns(e, cols)
    for c in cols:
        if c.table and c.table not in out:
            out.append(c.table)


def _collect_aggs(e: Optional[_Expr], out: list[_Expr]) -> None:
    if e is None:
        return
    k = e.kind
    if k == "agg":
        out.append(e)
    elif k == "cond":
        _collect_aggs(e.left, out)
        _collect_aggs(e.right_expr, out)
    elif k == "logic":
        _collect_aggs(e.rleft, out)
        _collect_aggs(e.rright, out)
    elif k == "not":
        _collect_aggs(e.inner, out)
    elif k == "arith":
        _collect_aggs(e.a_left, out)
        _collect_aggs(e.a_right, out)
    elif k == "func":
        for a in e.args:
            _collect_aggs(a, out)
    elif k == "case":
        for c in e.when_conds:
            _collect_aggs(c, out)
        for th in e.when_thens:
            _collect_aggs(th, out)
        _collect_aggs(e.case_else, out)


# ---------------------------------------------------------------------------
# mutable query carrier + freeze to the public AST
# ---------------------------------------------------------------------------

class _MutQuery:
    __slots__ = ("target", "entity_table", "has_from", "where", "assuming",
                 "rank", "top_k", "num_forecasts", "explain_present",
                 "explain_mode", "explain_format", "as_of_present",
                 "as_of_kind", "as_of_value", "ablations", "ret_present",
                 "ret_kind", "windows")

    def __init__(self):
        self.target: Optional[_Expr] = None
        self.entity_table = ""
        self.has_from = False
        self.where: Optional[_Expr] = None
        self.assuming: Optional[_Expr] = None
        self.rank: Optional[RankKind] = None
        self.top_k: Optional[int] = None
        self.num_forecasts: Optional[int] = None
        self.explain_present = False
        self.explain_mode = "PLAN"
        self.explain_format = "TEXT"
        self.as_of_present = False
        self.as_of_kind = "now"
        self.as_of_value: Optional[str] = None
        self.ablations: list[str] = []
        self.ret_present = False
        self.ret_kind = ""
        self.windows: dict[str, _MutWindow] = {}


def _trunc(v: float) -> float:
    """Finite frame offsets round-trip through the C++ JSON emitter as
    ``(long long)v`` — integer truncation toward zero. Preserved so a query
    parses to the identical Window either way."""
    return v if math.isinf(v) else float(int(v))


def _freeze_window(w: Optional[_MutWindow]) -> Optional[Window]:
    if w is None:
        return None
    return Window(
        _trunc(w.start), _trunc(w.end), w.unit,
        horizons=int(w.horizons),
        step=(_trunc(w.step) if w.has_step else None),
        top_k=(w.top_k if w.has_top_k else None),
        implied=bool(w.implied))


def _freeze_expr(e: Optional[_Expr]):
    if e is None:
        return None
    k = e.kind
    if k == "col":
        return ColumnRef(e.table, e.column)
    if k == "agg":
        return Aggregation(e.func, ColumnRef(e.table, e.column),
                           _freeze_expr(e.filter), _freeze_window(e.window))
    if k == "cond":
        right_expr = _freeze_expr(e.right_expr)
        right = None if right_expr is not None else (
            e.right.value() if e.right is not None else None)
        return Condition(_freeze_expr(e.left), e.op, right, right_expr)
    if k == "logic":
        return LogicalOp(_freeze_expr(e.rleft), e.bop, _freeze_expr(e.rright))
    if k == "not":
        return Not(_freeze_expr(e.inner))
    if k == "arith":
        return Arith(e.arith_op, _freeze_expr(e.a_left),
                     _freeze_expr(e.a_right))
    if k == "func":
        return Func(e.func_name, tuple(_freeze_expr(a) for a in e.args))
    if k == "case":
        whens = tuple((_freeze_expr(c), _freeze_expr(t))
                      for c, t in zip(e.when_conds, e.when_thens))
        return Case(whens, _freeze_expr(e.case_else))
    if k == "lit":
        return Lit(e.lit.value())
    if k == "param":
        return Param(e.param_name)
    raise AssertionError(k)


def _freeze_query(q: _MutQuery, text: str) -> ParsedQuery:
    return ParsedQuery(
        target=_freeze_expr(q.target),
        entity_key=ColumnRef(q.entity_table, None),
        entity_inferred=not q.has_from,
        where=_freeze_expr(q.where),
        assuming=_freeze_expr(q.assuming),
        rank=q.rank,
        top_k=q.top_k,
        num_forecasts=q.num_forecasts,
        explain=(Explain(q.explain_mode, q.explain_format)
                 if q.explain_present else None),
        as_of=(AsOf(q.as_of_kind,
                    None if q.as_of_kind == "now" else q.as_of_value)
               if q.as_of_present else None),
        ablations=tuple(Ablation("table", name) for name in q.ablations),
        ret=ReturnSpec(q.ret_kind) if q.ret_present else None,
        windows={name: _freeze_window(w) for name, w in q.windows.items()},
        text=text,
    )


def parse_text(query: str) -> ParsedQuery:
    """Parse RelQL text into a :class:`~relativedb.relql.ast.ParsedQuery`."""
    return _Parser(query).parse_query()
