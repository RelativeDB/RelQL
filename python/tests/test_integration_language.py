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
