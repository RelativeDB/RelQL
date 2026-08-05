"""RelQL language coverage against the real model on CPU.

Every construct rel-studio can emit runs end to end here: RETURN kinds,
AS OF binding, ASSUMING, comparison targets, zero-shot multiclass and
ranking, and shared context — with small contexts (256 cells) so the tier
stays fast once checkpoints are cached.
"""

from __future__ import annotations

import math

import pytest
from conftest import (
    churn_rows,
    dt,
    in_memory_wiring,
    require_checkpoint,
    require_text_embedder,
)

from relativedb import (
    ContextPolicy,
    Engine,
    ExecutionInput,
    LinkDef,
    ModelConfig,
    Schema,
    TableDef,
    ValueType,
)

pytestmark = [pytest.mark.integration, pytest.mark.checkpoint]

T0 = "2026-07-01"
CHURN = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")


def _schema() -> Schema:
    return (Schema.new_schema()
            .table(TableDef.new_table("customers")
                   .column("age", ValueType.NUMBER)
                   .column("signup_date", ValueType.DATETIME)
                   .primary_key("customer_id").build())
            .table(TableDef.new_table("products")
                   .column("price", ValueType.NUMBER)
                   .column("name", ValueType.TEXT)
                   .primary_key("product_id").build())
            .table(TableDef.new_table("orders")
                   .column("qty", ValueType.NUMBER)
                   .column("order_date", ValueType.DATETIME)
                   .primary_key("order_id")
                   .time_column("order_date").build())
            .link(LinkDef("orders", "customer_id", "customers"))
            .link(LinkDef("orders", "product_id", "products"))
            .build())


def _engine(task: str) -> Engine:
    require_text_embedder()
    uri = require_checkpoint(task)
    schema = _schema()
    wiring = in_memory_wiring(churn_rows())
    from relativedb.rt import RtBackend

    return Engine(
        schema, wiring,
        model_backend=RtBackend(schema=schema, wiring=wiring,
                                inference_backend="torch"),
        context_policy=ContextPolicy(max_context_cells=256, seed=11),
        model_config=ModelConfig(classification_model_uri=uri,
                                 regression_model_uri=uri))


@pytest.fixture(scope="module")
def classifier() -> Engine:
    return _engine("classification")


@pytest.fixture(scope="module")
def regressor() -> Engine:
    return _engine("regression")


def _run(engine, query, **kwargs):
    defaults = dict(params={"ids": ["C7"]}, anchor_time=dt(T0))
    defaults.update(kwargs)
    return engine.execute(ExecutionInput(query=query, **defaults))


# -- RETURN kinds ------------------------------------------------------------

def test_binary_return_kinds_agree(classifier):
    probability = _run(classifier, CHURN + " RETURN PROBABILITY")
    hard = _run(classifier, CHURN + " RETURN CLASS")
    distribution = _run(classifier, CHURN + " RETURN DISTRIBUTION")

    p = probability.predictions[0].probability
    assert 0.0 <= p <= 1.0
    assert hard.predictions[0].predicted_class == ("true" if p >= 0.5 else "false")
    dist = distribution.predictions[0].class_probs
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-5)

def test_expected_value_of_a_binary_target_is_its_probability(classifier):
    expected = _run(classifier, CHURN + " RETURN EXPECTED VALUE")
    probability = _run(classifier, CHURN + " RETURN PROBABILITY")
    assert expected.predictions[0].value == pytest.approx(
        probability.predictions[0].probability, abs=1e-6)


def test_comparison_target_scores_as_binary(classifier):
    result = _run(classifier,
                  "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                  "FROM customers WHERE customers.customer_id IN :ids")
    assert 0.0 <= result.predictions[0].probability <= 1.0


def test_regression_returns_a_finite_expected_value(regressor):
    result = _run(regressor,
                  "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                  "FROM customers WHERE customers.customer_id IN :ids "
                  "RETURN EXPECTED VALUE")
    value = result.predictions[0].value
    assert value is not None and math.isfinite(value)


# -- anchors -----------------------------------------------------------------

def test_as_of_overrides_the_execution_anchor(classifier):
    base = _run(classifier, CHURN)
    later = _run(classifier, CHURN + " AS OF 2026-08-01")
    # O4 (2026-07-05) enters context only under the later anchor, so the
    # score must move; both runs stay deterministic.
    assert later.predictions[0].probability != base.predictions[0].probability
    assert _run(classifier, CHURN + " AS OF 2026-08-01").predictions[0] \
        .probability == later.predictions[0].probability


def test_as_of_param_binds_like_a_date_literal(classifier):
    literal = _run(classifier, CHURN + " AS OF 2026-08-01")
    bound = _run(classifier, CHURN + " AS OF :t",
                 params={"ids": ["C7"], "t": "2026-08-01"})
    assert bound.predictions[0].probability == pytest.approx(
        literal.predictions[0].probability)


# -- ASSUMING ----------------------------------------------------------------

def test_assuming_changes_the_entity_cell_and_the_score(classifier):
    base = _run(classifier, CHURN)
    assumed = _run(classifier, CHURN + " ASSUMING customers.age = 99")
    assert assumed.predictions[0].probability != base.predictions[0].probability


# -- multiclass and ranking (zero-shot heads) ----------------------------------

def test_zero_shot_multiclass_over_a_text_column(classifier):
    result = classifier.execute(ExecutionInput(
        query=("PREDICT products.name FROM products "
               "WHERE products.product_id IN :ids RETURN DISTRIBUTION"),
        params={"ids": ["P1"]}, anchor_time=dt(T0)))
    dist = result.predictions[0].class_probs
    observed = {"running shoes", "espresso machine", "yoga mat"}
    assert set(dist) <= observed
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-5)

    hard = classifier.execute(ExecutionInput(
        query=("PREDICT products.name FROM products "
               "WHERE products.product_id IN :ids RETURN CLASS"),
        params={"ids": ["P1"]}, anchor_time=dt(T0)))
    assert hard.predictions[0].predicted_class in observed


def test_zero_shot_ranking_returns_top_k(classifier):
    result = _run(classifier,
                  "PREDICT LIST_DISTINCT(orders.product_id) "
                  "OVER (90 DAYS FOLLOWING RANK TOP 2) "
                  "FROM customers WHERE customers.customer_id IN :ids")
    ranked = result.predictions[0].ranked
    assert ranked is not None and 1 <= len(ranked) <= 2
    candidate_ids = {"P1", "P2", "P3"}
    assert {str(c) for c in ranked} <= candidate_ids


# -- shared context ----------------------------------------------------------

def test_shared_context_scores_the_whole_cohort():
    # shared context requires the reference traversal
    require_text_embedder()
    uri = require_checkpoint("classification")
    schema = _schema()
    wiring = in_memory_wiring(churn_rows())
    from relativedb.rt import RtBackend
    from relativedb.traversal import ReferenceTraversal

    engine = Engine(
        schema, wiring,
        model_backend=RtBackend(schema=schema, wiring=wiring,
                                inference_backend="torch"),
        traversal=ReferenceTraversal(),
        context_policy=ContextPolicy(max_context_cells=256, seed=11),
        model_config=ModelConfig(classification_model_uri=uri))
    result = engine.execute(ExecutionInput(
        query=CHURN, params={"ids": ["C1", "C7", "C9"]},
        anchor_time=dt(T0), shared_context=True))
    assert {p.id for p in result.predictions} == {"C1", "C7", "C9"}
    for prediction in result.predictions:
        assert 0.0 <= prediction.probability <= 1.0


# -- windows and horizons ------------------------------------------------------

def test_forecast_horizons_return_one_value_per_step(regressor):
    result = _run(regressor,
                  "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING HORIZONS 3) "
                  "FROM customers WHERE customers.customer_id IN :ids "
                  "RETURN EXPECTED VALUE")
    forecast = result.predictions[0].forecast
    assert len(forecast) == 3
    assert all(math.isfinite(v) for v in forecast)


def test_preceding_target_windows_are_rejected(classifier):
    # PRECEDING belongs in WHERE; a backward-facing PREDICT target would
    # score the past as though it were the outcome.
    from relativedb import RelqlValidationError

    with pytest.raises(RelqlValidationError, match="future-facing"):
        _run(classifier,
             "PREDICT COUNT(orders.*) OVER (90 DAYS PRECEDING) > 0 "
             "FROM customers WHERE customers.customer_id IN :ids")


# -- WHERE and cohorts ---------------------------------------------------------

def test_where_combines_aggregates_and_scalar_predicates(classifier):
    result = classifier.execute(ExecutionInput(
        query=("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
               "FROM customers "
               "WHERE EXISTS(orders.*) OVER (365 DAYS PRECEDING) "
               "AND customers.age > :min_age"),
        params={"min_age": 40}, anchor_time=dt(T0)))
    # C7 (52, has orders) qualifies; C1 (34) fails the age bound and C9 has
    # no orders at all.
    assert {p.id for p in result.predictions} == {"C7"}


def test_entity_with_no_history_still_scores(classifier):
    result = _run(classifier, CHURN, params={"ids": ["C9"]})
    assert 0.0 <= result.predictions[0].probability <= 1.0


def test_entity_ids_as_data_match_the_where_cohort(classifier):
    by_where = _run(classifier, CHURN, params={"ids": ["C1", "C7"]})
    by_data = classifier.execute(ExecutionInput(
        query=("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
               "FROM customers"),
        entity_ids=["C1", "C7"], anchor_time=dt(T0)))
    assert ({p.id: p.probability for p in by_where.predictions}
            == {p.id: p.probability for p in by_data.predictions})


def test_per_entity_anchor_scores_rows_at_their_own_time(regressor):
    result = regressor.execute(ExecutionInput(
        query=("PREDICT orders.qty FROM orders "
               "WHERE orders.order_id IN :ids RETURN EXPECTED VALUE"),
        params={"ids": ["O1", "O3"]}, anchor_time=dt(T0),
        per_entity_anchor=True))
    assert {p.id for p in result.predictions} == {"O1", "O3"}
    assert all(math.isfinite(p.value) for p in result.predictions)


# -- samplers and explain --------------------------------------------------------

def test_csc_sampler_matches_the_retriever_sampler(classifier):
    from relativedb import SamplerMode

    uri = require_checkpoint("classification")
    schema = _schema()
    wiring = in_memory_wiring(churn_rows())
    from relativedb.rt import RtBackend

    csc = Engine(schema, wiring,
                 model_backend=RtBackend(schema=schema, wiring=wiring,
                                         inference_backend="torch"),
                 sampler_mode=SamplerMode.CSC,
                 context_policy=ContextPolicy(max_context_cells=256, seed=11),
                 model_config=ModelConfig(classification_model_uri=uri))
    over_csc = csc.execute(ExecutionInput(
        query=CHURN, params={"ids": ["C1", "C7", "C9"]}, anchor_time=dt(T0)))
    over_retriever = classifier.execute(ExecutionInput(
        query=CHURN, params={"ids": ["C1", "C7", "C9"]}, anchor_time=dt(T0)))
    assert ({p.id: p.probability for p in over_csc.predictions}
            == {p.id: p.probability for p in over_retriever.predictions})


def test_explain_analyze_runs_the_real_model(classifier):
    plan = classifier.explain(ExecutionInput(
        query="EXPLAIN PLAN " + CHURN, params={"ids": ["C7"]},
        anchor_time=dt(T0)))
    assert plan.plan
    context = classifier.explain(ExecutionInput(
        query="EXPLAIN CONTEXT " + CHURN, params={"ids": ["C7"]},
        anchor_time=dt(T0)))
    assert context.context["total_rows"] > 0
    analyze = classifier.explain(ExecutionInput(
        query="EXPLAIN ANALYZE " + CHURN, params={"ids": ["C7"]},
        anchor_time=dt(T0)))
    assert analyze.mode == "ANALYZE"
    assert analyze.predictions            # scored with the real model
    assert analyze.context["total_rows"] > 0


# -- normalization modes ---------------------------------------------------------

def test_reference_normalization_serves_with_fitted_stats():
    require_text_embedder()
    uri = require_checkpoint("classification")
    from relativedb.rt import RtBackend
    from relativedb.scoring import ColumnStats

    schema = _schema()
    wiring = in_memory_wiring(churn_rows())
    from relativedb import TaskSpec, parse, validate

    pq = validate(parse(CHURN), schema).query
    task_spec = TaskSpec.from_query(pq, pq.task_type(schema))
    stats = ColumnStats.fit(schema, wiring).with_task_values(
        task_spec, [0.0, 1.0])
    engine = Engine(
        schema, wiring,
        model_backend=RtBackend(schema=schema, wiring=wiring,
                                inference_backend="torch",
                                column_stats=stats,
                                normalization_mode="reference"),
        context_policy=ContextPolicy(max_context_cells=256, seed=11),
        model_config=ModelConfig(classification_model_uri=uri,
                                 normalization_mode="reference"))
    result = engine.execute(ExecutionInput(
        query=CHURN, params={"ids": ["C7"]}, anchor_time=dt(T0)))
    p = result.predictions[0].probability
    assert 0.0 <= p <= 1.0
    again = engine.execute(ExecutionInput(
        query=CHURN, params={"ids": ["C7"]}, anchor_time=dt(T0)))
    assert again.predictions[0].probability == p


# -- the columnar serving shape ------------------------------------------------

def test_columnar_store_serves_a_shared_cohort():
    """rel-studio's serving shape: frames -> ColumnarStore -> shared context."""
    import numpy as np

    require_text_embedder()
    uri = require_checkpoint("classification")
    from relativedb import TaskSpec, parse, validate
    from relativedb.rt import RtBackend
    from relativedb.traversal import ColumnarStore, ColumnarTraversal

    schema = _schema()
    pq = validate(parse(CHURN), schema).query
    task_spec = TaskSpec.from_query(pq, pq.task_type(schema))
    anchor = dt(T0)

    frames = {
        "customers": {"customer_id": np.asarray(["C1", "C7", "C9"]),
                      "age": np.asarray([34.0, 52.0, 27.0])},
        "products": {"product_id": np.asarray(["P1", "P2", "P3"]),
                     "price": np.asarray([25.0, 90.0, 35.0]),
                     "name": np.asarray(["running shoes", "espresso machine",
                                         "yoga mat"], dtype=object)},
        "orders": {"order_id": np.asarray(["O1", "O2", "O3"]),
                   "qty": np.asarray([1.0, 2.0, 1.0]),
                   "order_date": np.asarray([np.datetime64("2026-03-10", "us"),
                                             np.datetime64("2026-05-02", "us"),
                                             np.datetime64("2026-06-20", "us")]),
                   "customer_id": np.asarray(["C7", "C7", "C1"], dtype=object),
                   "product_id": np.asarray(["P2", "P1", "P3"], dtype=object)},
    }
    # One focal task row per entity at the anchor (label unknown), plus one
    # labeled history window per entity below it.
    entities = ["C1", "C7", "C9"]
    anchors = [np.datetime64(anchor.replace(tzinfo=None), "us")] * 3
    history_ts = [np.datetime64("2026-04-01", "us")] * 3
    task_frame = {
        "customer": np.asarray(entities * 2, dtype=object),
        "at": np.asarray(anchors + history_ts),
        task_spec.target_column: np.asarray(
            [np.nan, np.nan, np.nan, 0.0, 0.0, 1.0]),
    }
    store = ColumnarStore(
        schema, frames,
        task_frames={task_spec.table_name: task_frame},
        task_links={task_spec.table_name: ("customers", "customer", "at")})

    wiring = store.wiring()
    engine = Engine(
        schema, wiring,
        model_backend=RtBackend(schema=schema, wiring=wiring,
                                inference_backend="torch"),
        traversal=ColumnarTraversal(store),
        context_policy=ContextPolicy(max_context_cells=256, seed=11),
        model_config=ModelConfig(classification_model_uri=uri))
    result = engine.execute(ExecutionInput(
        query=CHURN, params={"ids": entities}, anchor_time=anchor,
        shared_context=True))
    assert {p.id for p in result.predictions} == set(entities)
    for prediction in result.predictions:
        assert 0.0 <= prediction.probability <= 1.0
