"""Expression evaluation over an assembled context.

Used for WHERE-clause entity filtering and by the built-in history baseline
backend. Aggregation windows are ``(anchor + start, anchor + end]`` (start
excluded, end included), matching the grammar's window semantics.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from .pql.ast import (AggFunc, Aggregation, ColumnRef, Condition, LogicalOp,
                      Not, Operator, TargetExpr, Window)
from .retrieve import Row

__all__ = ["eval_value", "eval_bool", "eval_row_predicate", "EvalError"]


class EvalError(ValueError):
    pass


def _window_bounds(window: Window, anchor: datetime) -> tuple[Optional[datetime], Optional[datetime]]:
    lo = None if math.isinf(window.start) else anchor + window.start_delta()
    hi = None if math.isinf(window.end) else anchor + window.end_delta()
    return lo, hi


def _rows_in_window(rows: list[Row], window: Optional[Window],
                    anchor: Optional[datetime]) -> list[Row]:
    if window is None or anchor is None:
        picked = list(rows)
    else:
        lo, hi = _window_bounds(window, anchor)
        picked = []
        for r in rows:
            if r.timestamp is None:
                continue
            if lo is not None and not (r.timestamp > lo):   # start EXCLUDED
                continue
            if hi is not None and not (r.timestamp <= hi):  # end INCLUDED
                continue
            picked.append(r)
    picked.sort(key=lambda r: (r.timestamp is not None, r.timestamp or datetime.min))
    return picked


def eval_row_predicate(expr: TargetExpr, row: Row) -> bool:
    """Row-level inline filter, e.g. ``COUNT(t.* WHERE t.amount > 100, ...)``."""
    if isinstance(expr, LogicalOp):
        if expr.op.name == "AND":
            return eval_row_predicate(expr.left, row) and eval_row_predicate(expr.right, row)
        return eval_row_predicate(expr.left, row) or eval_row_predicate(expr.right, row)
    if isinstance(expr, Not):
        return not eval_row_predicate(expr.expr, row)
    if isinstance(expr, Condition):
        if not isinstance(expr.left, ColumnRef):
            raise EvalError("inline aggregation filters must compare columns")
        left = row.cells.get(expr.left.column) if expr.left.table == row.table else None
        return _compare(expr.op, left, expr.right)
    raise EvalError(f"unsupported row predicate: {expr!r}")


def _agg_rows(agg: Aggregation, rows_by_table: dict[str, list[Row]],
              anchor: Optional[datetime]) -> list[Row]:
    rows = rows_by_table.get(agg.column.table, [])
    rows = _rows_in_window(rows, agg.window, anchor)
    if agg.filter is not None:
        # Only conditions on the aggregated table's own columns are applied
        # row-wise; conditions on parent tables would need a join (not needed
        # by the baseline).
        rows = [r for r in rows if _row_filter_ok(agg.filter, r)]
    return rows


def _row_filter_ok(expr: TargetExpr, row: Row) -> bool:
    try:
        return eval_row_predicate(expr, row)
    except EvalError:
        return True  # unevaluable sub-filter: keep the row (best effort)


def eval_value(expr: TargetExpr, rows_by_table: dict[str, list[Row]],
               entity_cells: dict[str, Any],
               anchor: Optional[datetime]) -> Any:
    """Evaluate a valueExpr (aggregation or static column) over the context."""
    if isinstance(expr, ColumnRef):
        return entity_cells.get(expr.column)
    if not isinstance(expr, Aggregation):
        raise EvalError(f"not a value expression: {expr!r}")
    rows = _agg_rows(expr, rows_by_table, anchor)
    col = expr.column.column
    if expr.func is AggFunc.COUNT:
        if col == "*":
            return float(len(rows))
        return float(sum(1 for r in rows if r.cells.get(col) is not None))
    values = [r.cells.get(col) for r in rows]
    values = [v for v in values if v is not None] if col != "*" else values
    if expr.func is AggFunc.COUNT_DISTINCT:
        return float(len(set(values)))
    if expr.func is AggFunc.LIST_DISTINCT:
        seen: list[Any] = []
        for v in values:
            if v not in seen:
                seen.append(v)
        return seen
    if expr.func is AggFunc.FIRST:
        return values[0] if values else None
    if expr.func is AggFunc.LAST:
        return values[-1] if values else None
    nums = [float(v) for v in values if isinstance(v, (int, float, bool))]
    if expr.func is AggFunc.SUM:
        return float(sum(nums))
    if not nums:
        return None
    if expr.func is AggFunc.AVG:
        return sum(nums) / len(nums)
    if expr.func is AggFunc.MIN:
        return min(nums)
    if expr.func is AggFunc.MAX:
        return max(nums)
    raise EvalError(f"unsupported aggregation {expr.func}")


def _like_to_regex(pattern: str) -> "re.Pattern[str]":
    out = []
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return re.compile("^" + "".join(out) + "$", re.IGNORECASE)


def _coerce_pair(left: Any, right: Any) -> tuple[Any, Any]:
    if isinstance(left, bool) and isinstance(right, (int, float)):
        return (1.0 if left else 0.0), float(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left), float(right)
    return left, right


def _compare(op: Operator, left: Any, right: Any) -> bool:
    if op is Operator.IS_NULL:
        return left is None
    if op is Operator.IS_NOT_NULL:
        return left is not None
    if op is Operator.IN:
        return left in right if left is not None else False
    if op is Operator.NOT_IN:
        return left not in right if left is not None else False
    if left is None:
        return False
    if op in (Operator.STARTS_WITH, Operator.ENDS_WITH, Operator.CONTAINS,
              Operator.NOT_CONTAINS, Operator.LIKE, Operator.NOT_LIKE):
        s, pat = str(left), str(right)
        if op is Operator.STARTS_WITH:
            return s.startswith(pat)
        if op is Operator.ENDS_WITH:
            return s.endswith(pat)
        if op is Operator.CONTAINS:
            return pat in s
        if op is Operator.NOT_CONTAINS:
            return pat not in s
        matched = _like_to_regex(pat).match(s) is not None
        return matched if op is Operator.LIKE else not matched
    l, r = _coerce_pair(left, right)
    try:
        if op is Operator.EQ:
            return l == r
        if op is Operator.NEQ:
            return l != r
        if op is Operator.GT:
            return l > r
        if op is Operator.LT:
            return l < r
        if op is Operator.GE:
            return l >= r
        if op is Operator.LE:
            return l <= r
    except TypeError:
        return False
    raise EvalError(f"unsupported operator {op}")


def eval_bool(expr: TargetExpr, rows_by_table: dict[str, list[Row]],
              entity_cells: dict[str, Any],
              anchor: Optional[datetime]) -> bool:
    if isinstance(expr, LogicalOp):
        l = eval_bool(expr.left, rows_by_table, entity_cells, anchor)
        if expr.op.name == "AND":
            return l and eval_bool(expr.right, rows_by_table, entity_cells, anchor)
        return l or eval_bool(expr.right, rows_by_table, entity_cells, anchor)
    if isinstance(expr, Not):
        return not eval_bool(expr.expr, rows_by_table, entity_cells, anchor)
    if isinstance(expr, Condition):
        left = eval_value(expr.left, rows_by_table, entity_cells, anchor)
        return _compare(expr.op, left, expr.right)
    value = eval_value(expr, rows_by_table, entity_cells, anchor)
    return bool(value)
