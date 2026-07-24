"""AS OF resolution: turn a query's ``AS OF`` clause plus the caller's
``anchor_time`` into the one effective anchor everything downstream uses.

This is contract Part A and it is pure: given the parsed query and the
execution input it needs no schema, wiring, or model. Keeping it out of the
engine makes the precedence rules -- absent/NOW defers to the execution
anchor, DATE overrides it, PARAM binds from ``params`` and falls back to it --
directly testable without standing up an Engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .errors import ExecutionError

__all__ = ["effective_anchor", "coerce_anchor", "parse_anchor_date"]


def parse_anchor_date(text: Optional[str]) -> datetime:
    """``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS`` -> an aware UTC datetime."""
    s = (text or "").strip()
    fmt = "%Y-%m-%d %H:%M:%S" if " " in s else "%Y-%m-%d"
    try:
        d = datetime.strptime(s, fmt)
    except ValueError as e:
        raise ExecutionError(
            f"AS OF: cannot parse date {text!r} "
            f"(expected YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'): {e}")
    return d.replace(tzinfo=timezone.utc)


def coerce_anchor(v: Any) -> datetime:
    """A bound ``AS OF :param`` value -> an aware UTC datetime."""
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, str):
        return parse_anchor_date(v)
    raise ExecutionError(
        f"AS OF param must bind to a datetime or date string, got {v!r}")


def effective_anchor(as_of: Any, anchor_time: Optional[datetime],
                     params: Optional[dict]) -> Optional[datetime]:
    """Resolve the effective anchor from the query's ``AS OF`` clause.

    Absent/NOW -> ``anchor_time`` (the execution anchor, NOT wall clock);
    DATE -> the parsed date (overrides); PARAM -> ``params[name]``, else
    ``anchor_time``, else a clear error.
    """
    if as_of is None or as_of.kind == "now":
        return anchor_time
    if as_of.kind == "date":
        return parse_anchor_date(as_of.value)
    if as_of.kind == "param":
        name = as_of.value
        bound = params or {}
        if name in bound:
            return coerce_anchor(bound[name])
        if anchor_time is not None:
            return anchor_time
        raise ExecutionError(
            f"AS OF :{name} — no value bound for parameter {name!r} "
            f"(supply ExecutionInput.params={{{name!r}: <datetime>}}) and "
            f"no anchor_time fallback is available")
    raise ExecutionError(f"unknown AS OF kind {as_of.kind!r}")
