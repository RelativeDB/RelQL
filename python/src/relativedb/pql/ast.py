"""Typed PQL AST + task-type inference.

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


class TimeUnit(Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"

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


@dataclass(frozen=True)
class LogicalOp:
    left: "TargetExpr"
    op: BoolOp
    right: "TargetExpr"


@dataclass(frozen=True)
class Not:
    expr: "TargetExpr"


TargetExpr = Union[Aggregation, ColumnRef, Condition, LogicalOp, Not]


def _find_aggregations(expr: TargetExpr) -> list[Aggregation]:
    if isinstance(expr, Aggregation):
        return [expr]
    if isinstance(expr, Condition):
        return _find_aggregations(expr.left)
    if isinstance(expr, LogicalOp):
        return _find_aggregations(expr.left) + _find_aggregations(expr.right)
    if isinstance(expr, Not):
        return _find_aggregations(expr.expr)
    return []


@dataclass(frozen=True)
class ParsedQuery:
    """The parse result — no schema needed. ``validate`` binds it to one."""

    target: TargetExpr
    entity_key: ColumnRef                       # FOR [EACH] table.pk
    entity_ids: tuple = ()                      # FOR t.pk = v | IN (...); empty = all
    where: Optional[TargetExpr] = None
    assuming: Optional[TargetExpr] = None
    rank: Optional[RankKind] = None             # CLASSIFY | RANK TOP K
    top_k: Optional[int] = None                 # RANK TOP K
    num_forecasts: Optional[int] = None         # FORECAST N TIMEFRAMES
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
        if isinstance(t, Aggregation):
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
