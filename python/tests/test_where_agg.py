"""WHERE aggregations count THIS entity's rows, and evaluation fails loud.

The bug these tests pin down: ``_where_ok`` evaluated aggregations over every
row in the assembled context — but the context deliberately carries *peer*
entities' rows for in-context learning, so ``WHERE COUNT(orders.*) OVER
(14 DAYS PRECEDING) > 0`` passed for a customer who never ordered: any peer's
order in the context satisfied it. On a churn query that inverts the cohort —
the "recently active" precondition silently becomes "everyone". The fix scopes
WHERE evaluation to the entity's focal rows.

Second guarantee: no best-effort evaluation. An inline aggregation filter the
evaluator cannot run used to keep the row silently, turning
``COUNT(t.* WHERE ...)`` into ``COUNT(t.*)`` with no warning. It now raises.
"""
from __future__ import annotations

import warnings

import pytest

from relativedb import Engine, ExecutionInput
from relativedb.engine import EntityPrediction
from relativedb.evaluate import EvalError, eval_value
from relativedb.relql.ast import (Aggregation, AggFunc, ColumnRef, Condition,
                                  Lit, Operator)
from relativedb.relql.parser import parse, validate
from relativedb.retrieve import Row

from conftest import churn_rows, dt, in_memory_wiring

ANCHOR = dt("2026-07-01")

# The user-facing shape this file exists for: churn over recently-active
# players/customers. C7 and C1 ordered before the anchor; C9 never did.
CHURN = ("PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) = 0 "
         "FROM customers "
         "WHERE COUNT(orders.*) OVER (180 DAYS PRECEDING) > 0 "
         "RETURN PROBABILITY")


class _Stub:
    def score(self, query, task_type, contexts, model_uri, config):
        return [EntityPrediction(c.entity_id, probability=0.5, value=None)
                for c in contexts]


def _engine(churn_schema):
    return Engine(churn_schema, in_memory_wiring(churn_rows()),
                  model_backend=_Stub())


def test_where_count_scopes_to_the_entitys_own_rows(churn_schema):
    engine = _engine(churn_schema)
    res = engine.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR))
    assert sorted(p.id for p in res.predictions) == ["C1", "C7"]


def test_never_active_entity_is_excluded_despite_peer_rows(churn_schema):
    # C9's context DOES contain other customers' orders (peer context for
    # in-context learning) — exactly the rows that used to satisfy the WHERE.
    engine = _engine(churn_schema)
    ctx = engine.assemble_context("customers", "C9", ANCHOR)
    peer_orders = [r for r in ctx.rows if r.table == "orders"]
    assert peer_orders, "fixture must reproduce the peer-context trap"
    assert not engine._where_ok(
        validate(parse(CHURN), churn_schema, {}).query, ctx, "customers")


def test_zero_count_comparison_still_works(churn_schema):
    # The inverse cohort: customers with NO orders in the window.
    engine = _engine(churn_schema)
    q = ("PREDICT customers.age FROM customers "
         "WHERE COUNT(orders.*) OVER (180 DAYS PRECEDING) = 0 "
         "RETURN EXPECTED VALUE")
    res = engine.execute(ExecutionInput(query=q, anchor_time=ANCHOR))
    assert sorted(p.id for p in res.predictions) == ["C9"]


# ---------------------------------------------------------------------------
# no best-effort evaluation
# ---------------------------------------------------------------------------

def _order_row():
    return Row("orders", "O1", {"qty": 2.0}, timestamp=dt("2026-06-01"))


def test_unevaluable_inline_filter_raises():
    # left side is a literal, not a column: previously kept every row
    agg = Aggregation(func=AggFunc.COUNT,
                      column=ColumnRef("orders", "*"),
                      filter=Condition(left=Lit(1), op=Operator.EQ, right=1),
                      window=None)
    with pytest.raises(EvalError, match="must compare columns"):
        eval_value(agg, {"orders": [_order_row()]}, {}, ANCHOR)


def test_cross_table_inline_filter_raises():
    # filter names a different table's column: previously compared NULL and
    # silently dropped every row
    agg = Aggregation(func=AggFunc.COUNT,
                      column=ColumnRef("orders", "*"),
                      filter=Condition(left=ColumnRef("customers", "age"),
                                       op=Operator.GT, right=30),
                      window=None)
    with pytest.raises(EvalError, match="own columns"):
        eval_value(agg, {"orders": [_order_row()]}, {}, ANCHOR)


def test_failed_where_evaluation_fails_the_statement(churn_schema):
    engine = _engine(churn_schema)
    q = ("PREDICT customers.age FROM customers "
         "WHERE COUNT(orders.* WHERE customers.age > 30) "
         "OVER (180 DAYS PRECEDING) > 0 RETURN EXPECTED VALUE")
    with pytest.raises(EvalError):
        engine.execute(ExecutionInput(query=q, anchor_time=ANCHOR))


# ---------------------------------------------------------------------------
# WHERE counts are database-exact, not context-sampled
# ---------------------------------------------------------------------------

def test_where_count_is_exact_under_a_tiny_context_budget(churn_schema):
    # Starve the context: the cohort must not change, because WHERE counts
    # come from the wiring, not from whatever rows survived assembly.
    from relativedb import ContextPolicy
    engine = Engine(churn_schema, in_memory_wiring(churn_rows()),
                    model_backend=_Stub(),
                    context_policy=ContextPolicy(max_context_cells=4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = engine.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR))
    assert sorted(p.id for p in res.predictions) == ["C1", "C7"]


def test_where_agg_on_unlinked_table_is_an_error(churn_schema):
    from relativedb.engine import ExecutionError
    engine = _engine(churn_schema)
    # products has no direct link to customers: a count of it cannot be
    # attributed to one entity, and guessing would be silently wrong
    q = ("PREDICT customers.age FROM customers "
         "WHERE COUNT(products.*) > 0 RETURN EXPECTED VALUE")
    with pytest.raises(ExecutionError, match="no direct link"):
        engine.execute(ExecutionInput(query=q, anchor_time=ANCHOR))
