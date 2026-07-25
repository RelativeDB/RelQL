"""Silent degradations must be loud.

Every bug that cost us a day this cycle degraded quietly rather than failing:
a context buffer that filled and returned a truncated sequence, a chunk-sizing
call that swallowed its exception and guessed a split, a shared-context request
that fell through to per-entity scoring. None of them raised, none of them
warned, and each changed predictions -- so they surfaced as an unexplained
accuracy gap weeks later instead of a stack trace immediately.

These tests assert the noise. A check that cannot fire is the same bug again,
so each one drives the condition rather than asserting on a mock.
"""
from __future__ import annotations

import warnings

import pytest

from relativedb import Engine, ExecutionInput, TaskType
from relativedb.engine import (EntityPrediction, ExecutionError,
                               ProtocolFallbackWarning, _fallback, _strict)

from conftest import churn_rows, dt, in_memory_wiring

CHURN = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")
ANCHOR = dt("2026-07-01")
IDS = ["C1", "C7", "C9"]


class _Stub:
    def score(self, query, task_type, contexts, model_uri, config):
        binary = task_type is TaskType.BINARY_CLASSIFICATION
        return [EntityPrediction(c.entity_id,
                                 probability=0.5 if binary else None,
                                 value=None if binary else 1.0)
                for c in contexts]


def _engine(churn_schema):
    return Engine(churn_schema, in_memory_wiring(churn_rows()),
                  model_backend=_Stub())


# --------------------------------------------------------------------------
# the strict switch itself
# --------------------------------------------------------------------------

def test_fallback_warns_by_default(monkeypatch):
    monkeypatch.delenv("RELATIVEDB_STRICT", raising=False)
    assert _strict() is False
    with pytest.warns(ProtocolFallbackWarning, match="something degraded"):
        _fallback("something degraded")


def test_fallback_raises_under_strict(monkeypatch):
    monkeypatch.setenv("RELATIVEDB_STRICT", "1")
    assert _strict() is True
    with pytest.raises(ExecutionError, match="something degraded"):
        _fallback("something degraded")


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True),
    ("0", False), ("", False), ("no", False),
])
def test_strict_switch_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("RELATIVEDB_STRICT", value)
    assert _strict() is expected


# --------------------------------------------------------------------------
# a shared-context request that quietly becomes per-entity scoring
# --------------------------------------------------------------------------

def test_declining_shared_context_warns(churn_schema, monkeypatch):
    """Per-entity scoring is not the cohort scored in one shared context; the
    two give different predictions, so the swap must not be silent."""
    from relativedb import strategies
    monkeypatch.setitem(strategies._REGISTRY, "shared-context",
                        lambda engine, req: None)
    monkeypatch.delenv("RELATIVEDB_STRICT", raising=False)
    eng = _engine(churn_schema)
    with pytest.warns(ProtocolFallbackWarning, match="declined"):
        eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                   params={"ids": IDS}, shared_context=True))


def test_declining_shared_context_raises_under_strict(churn_schema,
                                                      monkeypatch):
    from relativedb import strategies
    monkeypatch.setitem(strategies._REGISTRY, "shared-context",
                        lambda engine, req: None)
    monkeypatch.setenv("RELATIVEDB_STRICT", "1")
    eng = _engine(churn_schema)
    with pytest.raises(ExecutionError, match="declined"):
        eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                   params={"ids": IDS}, shared_context=True))


# --------------------------------------------------------------------------
# chunk sizing that swallowed its own failure
# --------------------------------------------------------------------------

def test_chunk_sizing_failure_is_reported(churn_schema, monkeypatch):
    """It used to catch every exception and guess a split. How a cohort is
    split changes every member's context, so a guess is a silent protocol
    change, not a detail."""
    eng = _engine(churn_schema)

    def boom(*a, **k):
        raise RuntimeError("cohort targets exploded")

    monkeypatch.setattr(eng.traversal, "cohort_targets", boom, raising=False)
    monkeypatch.delenv("RELATIVEDB_STRICT", raising=False)
    with pytest.warns(ProtocolFallbackWarning, match="chunk sizing failed"):
        eng._cohort_chunk_limit(
            __import__("relativedb").parse(CHURN),
            ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                           params={"ids": IDS}),
            "customers", list(IDS), eng.model_backend)


def test_chunk_sizing_failure_raises_under_strict(churn_schema, monkeypatch):
    eng = _engine(churn_schema)

    def boom(*a, **k):
        raise RuntimeError("cohort targets exploded")

    monkeypatch.setattr(eng.traversal, "cohort_targets", boom, raising=False)
    monkeypatch.setenv("RELATIVEDB_STRICT", "1")
    with pytest.raises(ExecutionError, match="chunk sizing failed"):
        eng._cohort_chunk_limit(
            __import__("relativedb").parse(CHURN),
            ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                           params={"ids": IDS}),
            "customers", list(IDS), eng.model_backend)


# --------------------------------------------------------------------------
# the context buffer that filled and returned a truncated sequence
# --------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.native
def test_context_truncation_is_an_error():
    """Nodes and cells are different quantities: a row whose feature columns
    are all null costs zero cells but still occupies a node slot. A buffer
    sized from the cell budget therefore binds on a real graph, and used to
    drop the tail of the context before the model saw it."""
    import numpy as np
    from conftest import require_native
    require_native()
    from relativedb.graph_native import ContextTruncated, NativeGraph

    n = 64
    ts = np.full(n, np.nan)
    cells = np.ones(n, dtype=np.int32)
    table = np.zeros(n, dtype=np.int32)
    is_task = np.zeros(n, dtype=np.uint8)
    parent = np.zeros(n - 1, dtype=np.int64)
    child = np.arange(1, n, dtype=np.int64)
    g = NativeGraph(ts, cells, table, is_task, parent, child)

    class _P:
        max_context_cells = 4096
        local_context_cells = 4096
        bfs_width = 64
        num_walks = 0
        walk_length = 0
        seed = 0
        prefer_latest = True

    # A buffer far smaller than the context must raise, not truncate.
    with pytest.raises(ContextTruncated, match="max_nodes|emitted-node buffer"):
        g.assemble(0, float("inf"), None, _P(), max_nodes=4)
