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
def test_context_truncation_is_an_error():
    """Nodes and cells are different quantities: a row whose feature columns
    are all null costs zero cells but still occupies a node slot. A buffer
    sized from the cell budget therefore binds on a real graph, and used to
    drop the tail of the context before the model saw it."""
    import numpy as np
    from relational_transformers_utils.graph import ContextGraph as NativeGraph
    from relational_transformers_utils.graph import ContextTruncated

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


# --------------------------------------------------------------------------
# cohort-level context health: mass truncation and one-table dominance
# --------------------------------------------------------------------------
# Found in the field: an experiment ran with 100% of contexts truncated and a
# single event table holding 69% of every context's cells. The model never saw
# the discriminating columns, and the only tell was a per-entity warning
# stream nobody reads at cohort scale. _warn_context_health is the aggregate
# report; these drive both failure shapes and the quiet path.

from relativedb import ContextCompositionWarning, ContextPolicy, Row
from relativedb.engine import EntityContext, _warn_context_health


def _ctx(rows, *, hit=False, entity_id="E1"):
    return EntityContext(entity_id=entity_id, anchor=ANCHOR, rows=tuple(rows),
                         hit_cell_budget=hit, focal_row_keys=frozenset(),
                         node_ids={})


def _like_rows(n):
    return [Row("likes", f"l{i}", {"at": dt("2026-06-01")},
                timestamp=dt("2026-06-01"), parents={"post_id": "p1"})
            for i in range(n)]


def _post_rows(n):
    return [Row("posts", f"p{i}", {"created_at": dt("2026-06-01"),
                                   "text_length": 100.0, "engagement": 5.0},
                timestamp=dt("2026-06-01"))
            for i in range(n)]


def test_mass_truncation_is_summarized_once(churn_schema):
    contexts = [_ctx(_post_rows(2), hit=True, entity_id=f"E{i}")
                for i in range(4)]
    with pytest.warns(ContextCompositionWarning,
                      match="4 of 4 contexts hit the cell budget"):
        _warn_context_health(contexts, None, ContextPolicy())


def test_dominant_table_is_called_out(churn_schema):
    # 200 like-rows at 2 cells each vs 10 post rows at 4: likes hold ~91%,
    # comfortably above both the share threshold and the cohort-size floor.
    contexts = [_ctx(_like_rows(200) + _post_rows(10))]
    with pytest.warns(ContextCompositionWarning, match="'likes' holds"):
        _warn_context_health(contexts, None, ContextPolicy())


def test_balanced_untruncated_contexts_stay_quiet(churn_schema):
    # 100x2 = 200 like cells vs 50x4 = 200 post cells: an even split, above
    # the cohort floor so silence means the threshold held, not the floor.
    contexts = [_ctx(_like_rows(100) + _post_rows(50))]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ContextCompositionWarning)
        _warn_context_health(contexts, None, ContextPolicy())


def test_single_table_context_never_reports_dominance(churn_schema):
    # A one-table schema is trivially "dominated"; that is not a finding.
    contexts = [_ctx(_like_rows(200))]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ContextCompositionWarning)
        _warn_context_health(contexts, None, ContextPolicy())


# --------------------------------------------------------------------------
# invisible tables: rows that serialize to zero tokens
# --------------------------------------------------------------------------
# Serialization emits one token per feature cell, and a row's FK links ride on
# the tokens it emits. A pure edge table (follows/likes with no payload
# columns) therefore enters the context at zero cell cost and never reaches
# the model at all — the connection each row exists to represent is lost.
# Found in the field: a follow graph that contributed nothing until its links
# got feature_type. Two layers of noise: provable at schema time, and observed
# at cohort time (rows arrived, no cells survived).

from relativedb import (InvisibleTableWarning, LinkDef, Schema, TableDef,
                        ValueType)
from relativedb.engine import _warn_invisible_tables


def _edge_schema(*, feature_type=None, payload=False):
    follows = TableDef.new_table("follows").primary_key("follow_id")
    if payload:
        follows = follows.column("weight", ValueType.NUMBER)
    return (Schema.new_schema()
            .table(TableDef.new_table("accounts")
                   .column("followers", ValueType.NUMBER)
                   .primary_key("account_id").build())
            .table(follows.build())
            .link(LinkDef("follows", "follower_id", "accounts", feature_type))
            .link(LinkDef("follows", "followee_id", "accounts"))
            .build())


def test_all_fk_table_warns_at_engine_construction():
    with pytest.warns(InvisibleTableWarning, match="'follows'.*zero tokens"):
        Engine(_edge_schema(),
               in_memory_wiring({"accounts": [], "follows": []}),
               model_backend=_Stub())


def test_feature_typed_link_makes_the_table_visible():
    # The FK value itself can be real signal (a handle, a name); feature_type
    # emits it, so the table is no longer invisible and must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InvisibleTableWarning)
        _warn_invisible_tables(_edge_schema(feature_type=ValueType.TEXT))


def test_payload_column_makes_the_table_visible():
    with warnings.catch_warnings():
        warnings.simplefilter("error", InvisibleTableWarning)
        _warn_invisible_tables(_edge_schema(payload=True))


def _follow_rows(n, *, schema_visible=False):
    return [Row("follows", f"f{i}",
                {"weight": 1.0} if schema_visible else {},
                parents={"follower_id": f"a{i}", "followee_id": "a0"})
            for i in range(n)]


def test_zero_emitted_cells_reported_at_cohort_time():
    schema = _edge_schema()
    contexts = [_ctx(_follow_rows(3))]
    with pytest.warns(ContextCompositionWarning,
                      match="'follows' contributed 3 context row"):
        _warn_context_health(contexts, schema, ContextPolicy())


def test_feature_typed_fk_counts_as_an_emitted_cell():
    # Same rows, but the link now emits the FK value: not invisible.
    schema = _edge_schema(feature_type=ValueType.TEXT)
    contexts = [_ctx(_follow_rows(3))]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ContextCompositionWarning)
        _warn_context_health(contexts, schema, ContextPolicy())


# --------------------------------------------------------------------------
# unreferenced bindings
# --------------------------------------------------------------------------

def test_unconsumed_ids_binding_fails_instead_of_scanning_the_table(churn_schema):
    """params={'ids': ...} against a query with no `IN :ids` once scored the
    whole table — thousands of unrequested contexts collated into one forward
    took the serving host to its memory limit. The mismatch must fail at
    validation, where the fix is a one-line WHERE clause."""
    from relativedb import UnreferencedParameterError
    eng = _engine(churn_schema)
    unpinned = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
                "FROM customers")
    with pytest.raises(UnreferencedParameterError, match="ids"):
        eng.execute(ExecutionInput(query=unpinned, anchor_time=ANCHOR,
                                   params={"ids": IDS}))


def test_pinned_ids_binding_still_scores(churn_schema):
    eng = _engine(churn_schema)
    res = eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                     params={"ids": IDS}))
    assert [p.id for p in res.predictions] == IDS
