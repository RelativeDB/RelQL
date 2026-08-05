"""Anchoring correctness against the real model.

The temporal guarantee, tested as an equivalence rather than a difference:
scoring at anchor T over the full dataset must be BIT-IDENTICAL to scoring
after physically deleting every row newer than T. If any future row leaks
into context, embeddings, or normalization statistics, the two runs diverge.
"""

from __future__ import annotations

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

T0 = dt("2026-07-01")      # O4 (2026-07-05) sits in T0's future
LATER = dt("2026-08-01")   # admits O4
CHURN = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")
COHORT = ["C1", "C7", "C9"]


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


def _engine(rows) -> Engine:
    require_text_embedder()
    uri = require_checkpoint("classification")
    schema = _schema()
    wiring = in_memory_wiring(rows)
    from relativedb.rt import RtBackend

    return Engine(
        schema, wiring,
        model_backend=RtBackend(schema=schema, wiring=wiring,
                                inference_backend="torch"),
        context_policy=ContextPolicy(max_context_cells=256, seed=11),
        model_config=ModelConfig(classification_model_uri=uri,
                                 regression_model_uri=uri))


def _without_future(rows, cutoff):
    return {table: [r for r in table_rows
                    if r.timestamp is None or r.timestamp <= cutoff]
            for table, table_rows in rows.items()}


def _scores(engine, anchor, **kwargs):
    result = engine.execute(ExecutionInput(
        query=CHURN, params={"ids": COHORT}, anchor_time=anchor, **kwargs))
    return {p.id: p.probability for p in result.predictions}


def test_scoring_at_an_anchor_equals_deleting_the_future():
    rows = churn_rows()
    assert any(r.timestamp and r.timestamp > T0
               for r in rows["orders"]), "the fixture must carry a future row"

    full = _engine(rows)
    trimmed = _engine(_without_future(rows, T0))

    assert _scores(full, T0) == _scores(trimmed, T0)
    # The future row is real signal: once the anchor admits it, the scores
    # must move for its owner (C7 owns O4).
    assert _scores(full, LATER)["C7"] != _scores(full, T0)["C7"]


def test_naive_and_aware_anchors_agree():
    engine = _engine(churn_rows())
    aware = _scores(engine, T0)
    naive = _scores(engine, T0.replace(tzinfo=None))
    assert naive == aware


def test_context_anchor_owns_context_assembly_entirely():
    """context_anchor_time decouples the context's "now" from the label
    anchor: at inference it governs everything the model sees — admitted
    rows AND the task row's own timestamp — so a decoupled run is
    bit-identical to running plainly at the context anchor."""
    engine = _engine(churn_rows())
    decoupled = _scores(engine, LATER, context_anchor_time=T0)
    plain = _scores(engine, T0)
    assert decoupled == plain
    assert decoupled != _scores(engine, LATER)


def test_per_entity_anchor_equals_each_rows_own_explicit_anchor():
    rows = churn_rows()
    engine = _engine(rows)
    query = ("PREDICT orders.qty FROM orders "
             "WHERE orders.order_id IN :ids RETURN EXPECTED VALUE")

    per_entity = engine.execute(ExecutionInput(
        query=query, params={"ids": ["O1", "O3"]}, anchor_time=LATER,
        per_entity_anchor=True))
    got = {p.id: p.value for p in per_entity.predictions}

    for order in rows["orders"]:
        if order.id not in got:
            continue
        explicit = engine.execute(ExecutionInput(
            query=query, params={"ids": [order.id]},
            anchor_time=order.timestamp))
        # Same context either way; the cohort run collates both entities
        # into one padded batch, and float32 reduction order shifts the
        # last bits relative to a single-context batch.
        assert got[order.id] == pytest.approx(
            explicit.predictions[0].value, rel=1e-4), order.id


def test_anchors_admit_history_monotonically():
    """Walking the anchor back strictly shrinks admitted history; every
    boundary crossing that hides one of C7's orders moves its score."""
    engine = _engine(churn_rows())
    # C7's orders: O1 (03-10), O2 (05-02); O4 (07-05) beyond both anchors.
    before_all = _scores(engine, dt("2026-02-01"))["C7"]
    after_one = _scores(engine, dt("2026-04-01"))["C7"]
    after_two = _scores(engine, dt("2026-06-01"))["C7"]
    assert before_all != after_one
    assert after_one != after_two
