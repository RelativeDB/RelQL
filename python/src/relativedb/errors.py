"""Errors shared across the execution stack.

``ExecutionError`` lives here rather than in :mod:`relativedb.engine` so that
the planner and the anchor resolver -- both of which the engine imports -- can
raise it without importing the engine back. It is re-exported from
``relativedb.engine`` and ``relativedb``, so every existing import path keeps
working.
"""
from __future__ import annotations

__all__ = ["ExecutionError"]


class ExecutionError(RuntimeError):
    """A query cannot be executed as written (or as configured)."""
