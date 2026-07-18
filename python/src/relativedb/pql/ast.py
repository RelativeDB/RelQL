"""Typed RelQL AST + task-type inference.

Mirrors the ``dev.rql.query`` records from the Java API design, pythonically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Union

__all__ = [
    "AggFunc", "TimeUnit", "Operator", "BoolOp", "TaskType",
    "ColumnRef", "Window", "Aggregation", "Condition", "LogicalOp", "Not",
    "Arith", "Func", "Case", "Lit",
    "Explain", "AsOf", "Ablation", "ReturnSpec",
    "TargetExpr", "ParsedQuery", "RankKind",
]


class AggFunc(Enum):
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT = "COUNT"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    LIST_DISTINCT = "LIST_DISTINCT"
    FIRST = "FIRST"
    LAST = "LAST"
    EXISTS = "EXISTS"       # boolean existence -> binary_classification


class TimeUnit(Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"         # defensive: the parser normalizes calendar frames
                            # to months, but accept years if ever emitted.

    def delta(self, n: float) -> timedelta:
        if self is TimeUnit.SECONDS:
            return timedelta(seconds=n)
        if self is TimeUnit.MINUTES:
            return timedelta(minutes=n)
        if self is TimeUnit.HOURS:
            return timedelta(hours=n)
        if self is TimeUnit.DAYS:
            return timedelta(days=n)
        if self is TimeUnit.WEEKS:
            return timedelta(weeks=n)
        if self is TimeUnit.YEARS:
            return timedelta(days=365 * n)
        # MONTHS: calendar months are irregular; 30-day approximation,
        # matching the engine's window arithmetic documented in README.
        return timedelta(days=30 * n)


class Operator(Enum):
    GT = ">"
    LT = "<"
    EQ = "="
    NEQ = "!="
    GE = ">="
    LE = "<="
    STARTS_WITH = "STARTS WITH"
    ENDS_WITH = "ENDS WITH"
    CONTAINS = "CONTAINS"
    NOT_CONTAINS = "NOT CONTAINS"
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


class BoolOp(Enum):
    AND = "AND"
    OR = "OR"


class RankKind(Enum):
    CLASSIFY = "CLASSIFY"
    RANK = "RANK"


class TaskType(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_RANKING = "multilabel_ranking"
    FORECASTING = "forecasting"

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.BINARY_CLASSIFICATION,
                        TaskType.MULTICLASS_CLASSIFICATION,
                        TaskType.MULTILABEL_RANKING)


@dataclass(frozen=True)
class ColumnRef:
    """``table.column`` — column may be ``"*"``."""

    table: str
    column: str

    def __str__(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(frozen=True)
class Window:
    """Aggregation window ``(start, end]`` in ``unit``.

    ``start`` is EXCLUDED, ``end`` is INCLUDED; ``±inf`` for unbounded.
    """

    start: float
    end: float
    unit: TimeUnit = TimeUnit.DAYS
    horizons: int = 1                       # 1 = single frame; >1 = forecasting
    step: Optional[float] = None            # horizon stride; None = frame width

    def start_delta(self) -> timedelta:
        if math.isinf(self.start):
            return timedelta.min if self.start < 0 else timedelta.max
        return self.unit.delta(self.start)

    def end_delta(self) -> timedelta:
        if math.isinf(self.end):
            return timedelta.max if self.end > 0 else timedelta.min
        return self.unit.delta(self.end)

    def span(self) -> Optional[timedelta]:
        if math.isinf(self.start) or math.isinf(self.end):
            return None
        return self.unit.delta(self.end - self.start)


@dataclass(frozen=True)
class Aggregation:
    func: AggFunc
    column: ColumnRef
    filter: Optional["TargetExpr"] = None   # inline `WHERE` inside the agg
    window: Optional[Window] = None         # None = static (windowless) agg


@dataclass(frozen=True)
class Condition:
    left: "TargetExpr"
    op: Operator
    right: Any = None  # literal | tuple of literals (IN) | None (IS NULL)
    right_expr: Optional["TargetExpr"] = None  # column/expression RHS; when
                                               # set, ``right`` is None


@dataclass(frozen=True)
class LogicalOp:
    left: "TargetExpr"
    op: BoolOp
    right: "TargetExpr"


@dataclass(frozen=True)
class Not:
    expr: "TargetExpr"


@dataclass(frozen=True)
class Arith:
    """Arithmetic combination of value expressions: ``left op right``."""

    op: str                       # "+" | "-" | "*" | "/"
    left: "TargetExpr"
    right: "TargetExpr"


@dataclass(frozen=True)
class Func:
    """Scalar function call: COALESCE/NULLIF/ABS/LOG/EXP/LEAST/GREATEST."""

    name: str
    args: tuple = ()


@dataclass(frozen=True)
class Case:
    """``CASE WHEN cond THEN then ... ELSE else END``."""

    whens: tuple = ()             # tuple of (cond, then) pairs
    else_: Optional["TargetExpr"] = None


@dataclass(frozen=True)
class Lit:
    """A literal appearing in value position (number, bool, string, date)."""

    value: Any = None


TargetExpr = Union[Aggregation, ColumnRef, Condition, LogicalOp, Not,
                   Arith, Func, Case, Lit]


def _find_aggregations(expr: TargetExpr) -> list[Aggregation]:
    if isinstance(expr, Aggregation):
        return [expr]
    if isinstance(expr, Condition):
        out = _find_aggregations(expr.left)
        if expr.right_expr is not None:
            out += _find_aggregations(expr.right_expr)
        return out
    if isinstance(expr, LogicalOp):
        return _find_aggregations(expr.left) + _find_aggregations(expr.right)
    if isinstance(expr, Not):
        return _find_aggregations(expr.expr)
    if isinstance(expr, Arith):
        return _find_aggregations(expr.left) + _find_aggregations(expr.right)
    if isinstance(expr, Func):
        out: list[Aggregation] = []
        for a in expr.args:
            out += _find_aggregations(a)
        return out
    if isinstance(expr, Case):
        out = []
        for cond, then in expr.whens:
            out += _find_aggregations(cond) + _find_aggregations(then)
        if expr.else_ is not None:
            out += _find_aggregations(expr.else_)
        return out
    return []


# ---------------------------------------------------------------------------
# Query-level clause carriers (represented in the AST; execution best-effort)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Explain:
    mode: str = "PLAN"            # PLAN | CONTEXT | ANALYZE | ABLATION
    format: str = "TEXT"          # TEXT | JSON


@dataclass(frozen=True)
class AsOf:
    kind: str                     # param | date | now
    value: Optional[str] = None   # param name / date string; None for NOW


@dataclass(frozen=True)
class Ablation:
    kind: str                     # "table"
    name: str = ""


@dataclass(frozen=True)
class ReturnSpec:
    kind: str                     # EXPECTED_VALUE | PROBABILITY | CLASS | ...
    quantiles: tuple = ()         # for QUANTILES
    interval: Optional[int] = None  # for INTERVAL <int>%


@dataclass(frozen=True)
class ParsedQuery:
    """The parse result — no schema needed. ``validate`` binds it to one."""

    target: TargetExpr
    entity_key: ColumnRef                       # FOR EACH table.pk
    where: Optional[TargetExpr] = None
    assuming: Optional[TargetExpr] = None
    rank: Optional[RankKind] = None             # CLASSIFY | RANK TOP K
    top_k: Optional[int] = None                 # RANK TOP K
    num_forecasts: Optional[int] = None         # derived from target HORIZONS>1
    explain: Optional[Explain] = None           # EXPLAIN [...] prefix
    as_of: Optional[AsOf] = None                # AS OF anchor
    ablations: tuple = ()                       # ABLATE TABLE ... (repeatable)
    ret: Optional[ReturnSpec] = None            # RETURN output spec
    windows: dict = field(default_factory=dict, compare=False)  # named WINDOW templates
    text: str = field(default="", compare=False)

    @property
    def target_aggregations(self) -> list[Aggregation]:
        return _find_aggregations(self.target)

    def task_type(self, schema=None) -> TaskType:
        """Infer the task type (design §4: execution semantics, step 1)."""
        if self.num_forecasts is not None:
            return TaskType.FORECASTING
        if self.rank is RankKind.RANK:
            return TaskType.MULTILABEL_RANKING
        if self.rank is RankKind.CLASSIFY:
            return TaskType.MULTICLASS_CLASSIFICATION
        t = self.target
        if isinstance(t, (Condition, LogicalOp, Not)):
            return TaskType.BINARY_CLASSIFICATION
        if isinstance(t, Lit):
            if isinstance(t.value, bool):
                return TaskType.BINARY_CLASSIFICATION
            return TaskType.REGRESSION
        if isinstance(t, (Arith, Func, Case)):
            return TaskType.REGRESSION
        if isinstance(t, Aggregation):
            if t.func is AggFunc.EXISTS:
                return TaskType.BINARY_CLASSIFICATION
            if t.func is AggFunc.LIST_DISTINCT:
                return TaskType.MULTILABEL_RANKING
            if t.func in (AggFunc.FIRST, AggFunc.LAST):
                return self._static_or_categorical(t.column, schema,
                                                   default=TaskType.MULTICLASS_CLASSIFICATION)
            return TaskType.REGRESSION
        # bare static column
        return self._static_or_categorical(t, schema,
                                           default=TaskType.MULTICLASS_CLASSIFICATION)

    @staticmethod
    def _static_or_categorical(col: ColumnRef, schema, default: TaskType) -> TaskType:
        if schema is None:
            return default
        from ..schema import ValueType
        table = schema.table(col.table)
        cdef = table.column(col.column) if table else None
        if cdef is None:
            return default
        if cdef.type is ValueType.NUMBER:
            return TaskType.REGRESSION
        if cdef.type is ValueType.BOOLEAN:
            return TaskType.BINARY_CLASSIFICATION
        if cdef.type is ValueType.DATETIME:
            return TaskType.REGRESSION
        return TaskType.MULTICLASS_CLASSIFICATION
