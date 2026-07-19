"""Engine tests: temporal-leakage guard, fanout caps, CSC equivalence,
model routing."""
from __future__ import annotations

from datetime import timedelta

import pytest

from relativedb import (ContextPolicy, Engine, ExecutionInput, ModelConfig,
                      Row, SamplerMode, TaskType, TemporalBound)
from relativedb.csc import CscIndex
from relativedb.schema import LinkDef

from conftest import StubBackend, churn_rows, dt, in_memory_wiring

T0 = dt("2026-07-01")


# ---------------------------------------------------------------------------
# Temporal leakage
# ---------------------------------------------------------------------------

def test_future_row_never_enters_context(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring)
    ctx = eng.assemble_context("customers", "C7", T0)
    keys = ctx.row_keys
    assert ("orders", "O1") in keys
    assert ("orders", "O2") in keys
    assert ("orders", "O4") not in keys          # 2026-07-05 > t0
    assert ("products", "P3") not in keys        # only reachable via O4


def test_leaky_retriever_is_caught_by_engine(churn_schema):
    """A buggy retriever that ignores the bound must not leak the future:
    the engine re-checks every returned row (F24 defense in depth)."""
    leaky = in_memory_wiring(churn_rows(), honor_bound=False)
    eng = Engine(churn_schema, leaky)
    ctx = eng.assemble_context("customers", "C7", T0)
    assert ("orders", "O4") not in ctx.row_keys
    for r in ctx.rows:
        assert r.timestamp is None or r.timestamp <= ctx.anchor


def test_later_anchor_admits_the_row(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring)
    ctx = eng.assemble_context("customers", "C7", dt("2026-08-01"))
    assert ("orders", "O4") in ctx.row_keys
    assert ("products", "P3") in ctx.row_keys


def test_temporal_bound_semantics():
    b = TemporalBound.at_or_before(T0)
    assert b.admits(T0)                          # inclusive
    assert b.admits(T0 - timedelta(seconds=1))
    assert not b.admits(T0 + timedelta(seconds=1))
    assert b.admits(None)                        # static rows always admitted
    assert TemporalBound.unbounded().admits(T0 + timedelta(days=999))


# ---------------------------------------------------------------------------
# Fanouts / hop-loop shape
# ---------------------------------------------------------------------------

def test_fanout_caps_children_newest_first(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring,
                 context_policy=ContextPolicy(fanouts=(1, 0), max_hops=2))
    ctx = eng.assemble_context("customers", "C7", T0)
    order_rows = [r for r in ctx.rows if r.table == "orders"]
    assert [r.id for r in order_rows] == ["O2"]  # newest admitted child only


def test_parents_always_followed(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring)
    ctx = eng.assemble_context("customers", "C7", T0)
    # hop 1: orders O1/O2; hop 2: their product parents
    assert ("products", "P1") in ctx.row_keys
    assert ("products", "P2") in ctx.row_keys


def test_max_context_cells_budget(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring,
                 context_policy=ContextPolicy(max_context_cells=3))
    ctx = eng.assemble_context("customers", "C7", T0)
    assert ctx.rows[0].key == ("customers", "C7")   # seed always present
    assert len(ctx.rows) < 5                        # budget stopped expansion


# ---------------------------------------------------------------------------
# CSC mode
# ---------------------------------------------------------------------------

def test_csc_children_bound_and_limit(churn_schema, churn_wiring):
    idx = CscIndex.build(churn_schema, churn_wiring)
    link = LinkDef("orders", "customer_id", "customers")
    kids = idx.children(link, "C7", TemporalBound.at_or_before(T0), 10)
    assert [k.id for k in kids] == ["O2", "O1"]      # newest-first, O4 excluded
    kids = idx.children(link, "C7", TemporalBound.at_or_before(T0), 1)
    assert [k.id for k in kids] == ["O2"]
    kids = idx.children(link, "C7", TemporalBound.unbounded(), 10)
    assert [k.id for k in kids] == ["O4", "O2", "O1"]


def test_csc_context_equals_retriever_context(churn_schema, churn_wiring):
    """The two sampler modes must assemble the same context on a toy graph."""
    policy = ContextPolicy(fanouts=(8, 8), max_hops=2)
    ret = Engine(churn_schema, churn_wiring, context_policy=policy,
                 sampler_mode=SamplerMode.RETRIEVER)
    csc = Engine(churn_schema, churn_wiring, context_policy=policy,
                 sampler_mode=SamplerMode.CSC)
    for eid in ("C1", "C7", "C9"):
        for anchor in (T0, dt("2026-04-01"), dt("2026-08-01"), None):
            a = ret.assemble_context("customers", eid, anchor)
            b = csc.assemble_context("customers", eid, anchor)
            assert a.row_keys == b.row_keys, (eid, anchor)
            # and identical traversal order
            assert [r.key for r in a.rows] == [r.key for r in b.rows]


def test_csc_execute_end_to_end(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring, sampler_mode=SamplerMode.CSC,
                 model_backend=StubBackend())
    res = eng.execute(ExecutionInput(
        query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FOR EACH customers.customer_id",
        anchor_time=T0))
    assert res.task_type is TaskType.BINARY_CLASSIFICATION
    assert {p.id for p in res.predictions} == {"C1", "C7", "C9"}


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

class RecordingBackend:
    def __init__(self):
        self.calls = []

    def score(self, query, task_type, contexts, model_uri, config):
        from relativedb import EntityPrediction
        self.calls.append((task_type, model_uri))
        return [EntityPrediction(c.entity_id) for c in contexts]


def test_engine_routes_model_uri_by_task_type(churn_schema, churn_wiring):
    backend = RecordingBackend()
    eng = Engine(churn_schema, churn_wiring, model_backend=backend)
    cases = [
        ("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id",
         TaskType.REGRESSION, "hf://stanford-star/rt-j/regression"),
        ("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id",
         TaskType.BINARY_CLASSIFICATION, "hf://stanford-star/rt-j/classification"),
        ("PREDICT SUM(orders.qty) OVER (7 DAYS FOLLOWING HORIZONS 4) "
         "FOR EACH customers.customer_id",
         TaskType.FORECASTING, "hf://stanford-star/rt-j/regression"),
        ("PREDICT LIST_DISTINCT(orders.qty) OVER (30 DAYS FOLLOWING) RANK TOP 5 "
         "FOR EACH customers.customer_id",
         TaskType.MULTILABEL_RANKING, "hf://stanford-star/rt-j/classification"),
    ]
    for pql, expect_task, expect_uri in cases:
        res = eng.execute(ExecutionInput(query=pql, anchor_time=T0))
        assert res.task_type is expect_task
        assert res.model_uri == expect_uri
    assert [c[1] for c in backend.calls] == [c[2] for c in cases]


def test_execute_without_backend_raises_clear_error(churn_schema, churn_wiring):
    """The engine ships no model-free scorer; scoring with no backend set is a
    clear error (parse/validate/EXPLAIN PLAN/CONTEXT still work — see
    test_explain_asof)."""
    from relativedb import ExecutionError
    eng = Engine(churn_schema, churn_wiring)   # no model_backend
    with pytest.raises(ExecutionError, match="requires a model backend"):
        eng.execute(ExecutionInput(
            query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                  "FOR EACH customers.customer_id",
            entity_ids=['C7'], anchor_time=T0))


def test_for_each_without_scanner_raises():
    """FOR EACH over all entities needs enumeration; plain retrievers can't."""
    from relativedb import ExecutionError, RetrieverWiring
    rows = churn_rows()
    by_id = {t: {r.id: r for r in rs} for t, rs in rows.items()}
    wiring = (RetrieverWiring.new_wiring()
              .entities("customers", lambda t, ids, b:
                        [by_id[t][i] for i in ids if i in by_id[t]])
              .default_links(lambda l, p, b, lim: [])
              .build())
    eng = Engine(_make_schema(), wiring, model_backend=StubBackend())
    with pytest.raises(ExecutionError):
        eng.execute(ExecutionInput(
            query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                  "FOR EACH customers.customer_id", anchor_time=T0))
    # but pinned ids work
    res = eng.execute(ExecutionInput(
        query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FOR EACH customers.customer_id",
        entity_ids=['C7'], anchor_time=T0))
    assert len(res.predictions) == 1


def _make_schema():
    from relativedb import Schema, TableDef, ValueType, LinkDef
    return (Schema.new_schema()
            .table(TableDef.new_table("customers")
                   .column("age", ValueType.NUMBER)
                   .column("signup_date", ValueType.DATETIME)
                   .primary_key("customer_id").build())
            .table(TableDef.new_table("orders")
                   .column("qty", ValueType.NUMBER)
                   .column("order_date", ValueType.DATETIME)
                   .primary_key("order_id").time_column("order_date").build())
            .link(LinkDef("orders", "customer_id", "customers"))
            .build())


# ---------------------------------------------------------------------------
# RETURN clause execution (contract §3)
# ---------------------------------------------------------------------------

def test_return_quantiles_interval_unsupported(churn_schema, churn_wiring):
    """QUANTILES / INTERVAL parse+validate fine, but a single point-head
    checkpoint exposes no empirical distribution — execution raises. (The fake
    history-window quantile computation was deleted with the baseline.)"""
    from relativedb.rt_native import RtNativeBackend, RtNativeError
    eng = Engine(churn_schema, churn_wiring,
                 model_backend=RtNativeBackend(schema=churn_schema))
    for ret in ("QUANTILES (0.1, 0.5, 0.9)", "INTERVAL 80%"):
        with pytest.raises(RtNativeError, match="QUANTILES/INTERVAL"):
            eng.execute(ExecutionInput(
                query="PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                      f"FOR EACH customers.customer_id RETURN {ret}",
                entity_ids=['C7'], anchor_time=T0))


def test_ranking_over_non_fk_column_rejected(churn_schema, churn_wiring):
    """Ranking (LIST_DISTINCT ... RANK) is supported, but only over a real
    foreign-key column that names a parent table; ``orders.qty`` is an ordinary
    feature column, so ranking over it is a clear error. (Raised before the
    checkpoint is loaded, so this runs offline.)"""
    from relativedb.rt_native import RtNativeBackend, RtNativeError
    eng = Engine(churn_schema, churn_wiring,
                 model_backend=RtNativeBackend(schema=churn_schema,
                                               wiring=churn_wiring))
    with pytest.raises(RtNativeError, match="foreign-key"):
        eng.execute(ExecutionInput(
            query="PREDICT LIST_DISTINCT(orders.qty) OVER (30 DAYS FOLLOWING) "
                  "RANK TOP 5 FOR EACH customers.customer_id",
            entity_ids=['C7'], anchor_time=T0))


def test_return_quantiles_on_boolean_target_rejected(churn_schema, churn_wiring):
    from relativedb import PqlValidationError
    eng = Engine(churn_schema, churn_wiring)
    with pytest.raises(PqlValidationError):
        eng.execute(ExecutionInput(
            query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                  "FOR EACH customers.customer_id RETURN QUANTILES (0.1, 0.9)",
            entity_ids=['C7'], anchor_time=T0))


def test_return_probability_on_regression_target_rejected(churn_schema, churn_wiring):
    from relativedb import PqlValidationError
    eng = Engine(churn_schema, churn_wiring)
    with pytest.raises(PqlValidationError):
        eng.execute(ExecutionInput(
            query="PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                  "FOR EACH customers.customer_id RETURN PROBABILITY",
            entity_ids=['C7'], anchor_time=T0))


def test_return_quantile_out_of_range_rejected(churn_schema):
    from relativedb import PqlValidationError
    from relativedb.pql.parser import validate
    with pytest.raises(PqlValidationError):
        validate("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                 "FOR EACH customers.customer_id RETURN QUANTILES (0.0, 0.9)",
                 churn_schema)


def test_model_config_defaults_and_routing():
    cfg = ModelConfig.defaults()
    assert cfg.classification_model_uri == "hf://stanford-star/rt-j/classification"
    assert cfg.regression_model_uri == "hf://stanford-star/rt-j/regression"
    assert cfg.embedding_model == "all-MiniLM-L12-v2"
    assert cfg.d_text == 384
    assert cfg.model_uri_for(TaskType.REGRESSION) == cfg.regression_model_uri
    assert cfg.model_uri_for(TaskType.FORECASTING) == cfg.regression_model_uri
    for t in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION,
              TaskType.MULTILABEL_RANKING):
        assert cfg.model_uri_for(t) == cfg.classification_model_uri


def test_model_config_unified_uri_and_embedding_check():
    from relativedb import EmbeddingMismatchError
    cfg = ModelConfig.defaults().with_model_uri("file:///models/unified")
    assert cfg.model_uri_for(TaskType.REGRESSION) == "file:///models/unified"
    assert cfg.model_uri_for(TaskType.BINARY_CLASSIFICATION) == "file:///models/unified"
    with pytest.raises(EmbeddingMismatchError):
        cfg.check_checkpoint_embedding("all-mpnet-base-v2")
    relaxed = ModelConfig(allow_embedding_mismatch=True)
    relaxed.check_checkpoint_embedding("all-mpnet-base-v2")  # no raise
    cfg.check_checkpoint_embedding("all-MiniLM-L12-v2")      # match ok
