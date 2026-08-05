"""Regression pins for the 2026-07 codebase review.

Every test here reproduces a bug that degraded silently: labels read from a
peer-contaminated sample, filters that compared against the wrong cell (or
none), typo'd cohorts scored on empty contexts, windows that vanished without
an anchor, forecasts that were one number copied N times. The fixed behavior
is either CORRECT (exact, entity-scoped, database-backed) or LOUD (raise or
warn) — never quietly wrong.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from relativedb import (Engine, ExecutionInput, LinkDef, Row,
                        Schema, TableDef, ValueType)
from relativedb.engine import (EntityPrediction, ExecutionError,
                               ProtocolFallbackWarning)
from relativedb.evaluate import EvalError, eval_bool, eval_value
from relativedb.relql.ast import (Aggregation, AggFunc, ColumnRef, Condition,
                                  Operator)
from relativedb.relql.parser import parse, validate

from conftest import churn_rows, dt, in_memory_wiring

ANCHOR = dt("2026-07-01")


def _churn_schema():
    return (Schema.new_schema()
            .table(TableDef.new_table("customers")
                   .column("age", ValueType.NUMBER)
                   .primary_key("customer_id").build())
            .table(TableDef.new_table("products")
                   .column("price", ValueType.NUMBER)
                   .column("name", ValueType.TEXT)
                   .primary_key("product_id").build())
            .table(TableDef.new_table("orders")
                   .column("qty", ValueType.NUMBER)
                   .column("order_date", ValueType.DATETIME)
                   .primary_key("order_id").time_column("order_date").build())
            .link(LinkDef("orders", "customer_id", "customers"))
            .link(LinkDef("orders", "product_id", "products"))
            .build())


class _Stub:
    def score(self, q, tt, cs, mu, cf):
        return [EntityPrediction(c.entity_id, probability=0.5, value=1.0)
                for c in cs]


def _engine():
    return Engine(_churn_schema(), in_memory_wiring(churn_rows()),
                  model_backend=_Stub())


# ---------------------------------------------------------------------------
# evaluation fails loud, and FK columns are comparable
# ---------------------------------------------------------------------------

def _orders(n=3):
    return [Row("orders", f"O{i}", {"qty": 1.0}, dt("2026-06-01"),
                parents={"product_id": "P1"}) for i in range(n)]


def test_inline_filter_reads_fk_columns():
    agg = Aggregation(func=AggFunc.COUNT, column=ColumnRef("orders", "*"),
                      filter=Condition(left=ColumnRef("orders", "product_id"),
                                       op=Operator.EQ, right="P1"),
                      window=None)
    assert eval_value(agg, {"orders": _orders()}, {}, ANCHOR) == 3.0
    agg_null = Aggregation(func=AggFunc.COUNT, column=ColumnRef("orders", "*"),
                           filter=Condition(
                               left=ColumnRef("orders", "product_id"),
                               op=Operator.IS_NULL, right=None),
                           window=None)
    assert eval_value(agg_null, {"orders": _orders()}, {}, ANCHOR) == 0.0


def test_cross_table_bare_column_in_where_raises():
    engine = _engine()
    with pytest.raises(EvalError, match="scalar position"):
        engine.execute(ExecutionInput(
            query=("PREDICT customers.age FROM customers "
                   "WHERE orders.qty >= 1 RETURN EXPECTED VALUE"),
            anchor_time=ANCHOR))


def test_windowed_aggregation_without_anchor_raises():
    agg = Aggregation(func=AggFunc.COUNT, column=ColumnRef("orders", "*"),
                      filter=None,
                      window=parse(
                          "PREDICT COUNT(orders.*) OVER (14 DAYS PRECEDING) "
                          "FROM customers RETURN EXPECTED VALUE"
                      ).target.window)
    with pytest.raises(EvalError, match="needs an anchor"):
        eval_value(agg, {"orders": _orders()}, {}, None)


def test_sum_over_non_numeric_raises():
    rows = [Row("orders", "O1", {"qty": "12.50"}, dt("2026-06-01"))]
    agg = Aggregation(func=AggFunc.SUM, column=ColumnRef("orders", "qty"),
                      filter=None, window=None)
    with pytest.raises(EvalError, match="non-numeric"):
        eval_value(agg, {"orders": rows}, {}, ANCHOR)


def test_cross_type_comparison_raises():
    cond = Condition(left=ColumnRef("customers", "age"),
                     op=Operator.GT, right=100)
    with pytest.raises(EvalError, match="cannot compare"):
        eval_bool(cond, {}, {"age": "85"}, ANCHOR)


def test_where_following_window_is_rejected():
    engine = _engine()
    with pytest.raises(ExecutionError, match="faces the future"):
        engine.execute(ExecutionInput(
            query=("PREDICT customers.age FROM customers "
                   "WHERE COUNT(orders.*) OVER (30 DAYS FOLLOWING) > 0 "
                   "RETURN EXPECTED VALUE"),
            anchor_time=ANCHOR))


# ---------------------------------------------------------------------------
# cohort hygiene: typos and anchors
# ---------------------------------------------------------------------------

def test_unknown_entity_id_warns_instead_of_scoring_silently():
    engine = _engine()
    with pytest.warns(ProtocolFallbackWarning, match="EMPTY context"):
        engine.execute(ExecutionInput(
            query=("PREDICT customers.age FROM customers "
                   "WHERE customers.customer_id IN :ids "
                   "RETURN EXPECTED VALUE"),
            anchor_time=ANCHOR, params={"ids": ["TYPO_ID"]}))


def test_per_entity_anchor_on_undated_entity_warns_and_falls_back():
    engine = _engine()
    with pytest.warns(ProtocolFallbackWarning, match="no dated row"):
        engine.execute(ExecutionInput(
            query=("PREDICT customers.age FROM customers "
                   "WHERE customers.customer_id IN :ids "
                   "RETURN EXPECTED VALUE"),
            anchor_time=ANCHOR, params={"ids": ["C7"]},
            per_entity_anchor=True))


def test_per_entity_anchor_without_any_anchor_is_an_error():
    engine = _engine()
    with pytest.raises(ExecutionError, match="unbounded"):
        engine.execute(ExecutionInput(
            query=("PREDICT customers.age FROM customers "
                   "WHERE customers.customer_id IN :ids "
                   "RETURN EXPECTED VALUE"),
            params={"ids": ["C7"]}, per_entity_anchor=True))


def test_fully_dangling_link_warns_when_the_index_is_built():
    # int primary keys, string FK values: every edge of the link severed
    schema = (Schema.new_schema()
              .table(TableDef.new_table("customers")
                     .column("age", ValueType.NUMBER)
                     .primary_key("customer_id").build())
              .table(TableDef.new_table("orders")
                     .column("qty", ValueType.NUMBER)
                     .column("order_date", ValueType.DATETIME)
                     .primary_key("order_id")
                     .time_column("order_date").build())
              .link(LinkDef("orders", "customer_id", "customers"))
              .build())
    rows = {
        "customers": [Row("customers", 1, {"age": 30.0})],
        "orders": [Row("orders", "O1", {"qty": 1.0,
                                        "order_date": dt("2026-06-01")},
                       dt("2026-06-01"), {"customer_id": "1"})],
    }
    # The index is built on first use now, not in the constructor, so the
    # warning arrives when the snapshot is actually assembled.
    eng = Engine(schema, in_memory_wiring(rows), model_backend=_Stub())
    with pytest.warns(UserWarning, match="dangling"):
        eng.csc_index()


# ---------------------------------------------------------------------------
# EXPLAIN tells the truth about aggregate ASSUMING
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# traversal temporal conformance: the declared anchor is the fallback cutoff
# ---------------------------------------------------------------------------

def test_undated_entity_gets_its_dated_children_up_to_the_anchor():
    """The old query-aware walk left an undated seed's cutoff as None and
    dropped every dated child — a customer's own orders never reached its
    context. The declared anchor is the fallback now: pre-anchor orders are
    admitted, the future one (O4, July 5) still is not."""
    engine = _engine()
    q = ("PREDICT customers.age FROM customers "
         "WHERE customers.customer_id IN :ids RETURN EXPECTED VALUE")
    pq = validate(parse(q), engine.schema,
                  {"ids": ["C7"]}).query.bind_params({"ids": ["C7"]})
    ctx = engine.assemble_context("customers", "C7", ANCHOR, query=pq)
    orders = {r.id for r in ctx.rows if r.table == "orders"}
    assert {"O1", "O2"} <= orders
    assert "O4" not in orders          # dated after the anchor


def test_future_dated_parent_is_not_pulled_into_context():
    """f2p edges used to be followed unconditionally: a parent row dated
    after the anchor was serialized into the context."""
    T = lambda d: datetime(2026, 7, d, tzinfo=timezone.utc)
    schema = (Schema.new_schema()
              .table(TableDef.new_table("sessions")
                     .column("closed_at", ValueType.DATETIME)
                     .column("score", ValueType.NUMBER)
                     .primary_key("session_id").time_column("closed_at")
                     .build())
              .table(TableDef.new_table("events")
                     .column("at", ValueType.DATETIME)
                     .column("kind", ValueType.TEXT)
                     .primary_key("event_id").time_column("at").build())
              .link(LinkDef("events", "session_id", "sessions"))
              .build())
    rows = {
        # the session row is stamped at close — AFTER the anchor
        "sessions": [Row("sessions", "s1",
                         {"closed_at": T(9), "score": 5.0}, T(9))],
        "events": [Row("events", f"e{i}", {"at": T(i), "kind": "click"},
                       T(i), {"session_id": "s1"}) for i in (1, 2, 3)],
    }
    engine = Engine(schema, in_memory_wiring(rows), model_backend=_Stub())
    q = ("PREDICT events.kind FROM events "
         "WHERE events.event_id IN :ids RETURN CLASS")
    pq = validate(parse(q), engine.schema,
                  {"ids": ["e2"]}).query.bind_params({"ids": ["e2"]})
    ctx = engine.assemble_context("events", "e2", T(5), query=pq)
    assert not any(r.table == "sessions" for r in ctx.rows), \
        "session closes at T9 > anchor T5: it is the future"


def test_explain_renders_aggregate_assuming():
    engine = _engine()
    res = engine.explain(ExecutionInput(
        query=("EXPLAIN PLAN PREDICT customers.age FROM customers "
               "WHERE customers.customer_id IN :ids "
               "ASSUMING COUNT(orders.*) OVER (90 DAYS PRECEDING) >= 3 "
               "RETURN EXPECTED VALUE"),
        anchor_time=ANCHOR, params={"ids": ["C7"]}))
    assert res.plan.get("assuming"), "aggregate ASSUMING must render"
    assert not any("cannot be applied" in w
                   for w in res.plan.get("warnings", []))
