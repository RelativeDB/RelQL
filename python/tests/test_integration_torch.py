"""End-to-end integration over the shared torch runtime.

A real PREDICT query runs through retrieval, context assembly, MiniLM text
embedding in torch, and the rt-j checkpoint — the path a production caller
takes. Needs the downloaded checkpoint and encoder snapshot, so it lives in
the integration tier.
"""

from __future__ import annotations

import pytest
from conftest import (churn_rows, dt, in_memory_wiring, require_checkpoint,
                      require_text_embedder)

from relativedb import (ColumnDef, ContextPolicy, Engine, ExecutionInput,
                        LinkDef, ModelConfig, Schema, TableDef, ValueType)


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

pytestmark = [pytest.mark.integration, pytest.mark.checkpoint]

QUERY = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")


@pytest.fixture(scope="module")
def engine():
    require_text_embedder()
    classification = require_checkpoint("classification")
    from relativedb.rt import RtBackend

    schema = _schema()
    wiring = in_memory_wiring(churn_rows())
    backend = RtBackend(schema=schema, wiring=wiring,
                        inference_backend="torch")
    return Engine(schema, wiring, model_backend=backend,
                  context_policy=ContextPolicy(max_context_cells=512, seed=7),
                  model_config=ModelConfig(
                      classification_model_uri=classification))


def test_predict_scores_every_cohort_member(engine):
    result = engine.execute(ExecutionInput(
        query=QUERY, params={"ids": ["C7", "C1"]},
        anchor_time=dt("2026-06-01")))
    assert len(result.predictions) == 2
    for prediction in result.predictions:
        assert 0.0 <= prediction.probability <= 1.0


def test_predictions_are_deterministic_and_anchor_sensitive(engine):
    def run(anchor):
        result = engine.execute(ExecutionInput(
            query=QUERY, params={"ids": ["C7"]}, anchor_time=anchor))
        return result.predictions[0].probability

    first = run(dt("2026-06-01"))
    assert run(dt("2026-06-01")) == first
    # An earlier anchor hides later history, so the score should move.
    assert run(dt("2026-03-01")) != first
