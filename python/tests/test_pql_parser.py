"""Grammar conformance: the full 54-query corpus parses; malformed input is
rejected; AST spot checks; task-type inference; schema validation."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pytest

from relativedb import PqlSyntaxError, PqlValidationError, parse, validate
from relativedb.pql import (AggFunc, Aggregation, Arith, AsOf, Case, ColumnRef,
                          Condition, Explain, Func, Lit, LogicalOp, Not,
                          Operator, RankKind, ReturnSpec, TaskType, TimeUnit)

CORPUS = [
    line.strip()
    for line in (Path(__file__).parent / "data" / "examples.pql").read_text().splitlines()
    if line.strip()
]


def test_corpus_has_54_queries():
    assert len(CORPUS) == 54


@pytest.mark.parametrize("query", CORPUS)
def test_corpus_query_parses(query):
    pq = parse(query)
    assert pq.entity_key.table
    assert pq.entity_key.column
    # every query has a well-defined task type even without a schema
    assert pq.task_type() in TaskType


MALFORMED = [
    "",
    "   ",
    "SELECT * FROM t",
    "PREDICT",
    "PREDICT FOR EACH t.id",
    "PREDICT SUM(t.c, 0, 30)",                              # positional windows removed
    "PREDICT SUM(t.c, 0, 30) FOR EACH t.id",               # positional windows removed
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t",  # entity not table.column
    "PREDICT SUM(c) OVER (30 DAYS FOLLOWING) FOR EACH t.id",  # agg operand not table.column
    "PREDICT SUM(t.c) OVER (30 DAYS) FOR EACH t.id",         # frame missing PRECEDING/FOLLOWING
    "PREDICT SUM(t.c) OVER (30 FORTNIGHTS FOLLOWING) FOR EACH t.id",  # bad unit
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING HORIZONS 0) FOR EACH t.id",  # HORIZONS must be positive
    "PREDICT SUM(t.c) OVER nope FOR EACH t.id",              # undeclared window
    "PREDICT SUM(t.c) OVER (RANGE BETWEEN 1 DAY FOLLOWING AND 2 MONTHS FOLLOWING) "
    "FOR EACH t.id",                                         # mixed fixed/calendar units
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t.id trailing junk",
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t.id WHERE",
    "PREDICT COUNT(t.*) OVER (30 DAYS FOLLOWING) = FOR EACH t.id",  # comparison w/o RHS
    "PREDICT t.c IN (1, 2 FOR EACH t.id",                    # unclosed list
    "PREDICT LIST_DISTINCT(t.c) OVER (30 DAYS FOLLOWING) RANK TOP FOR EACH t.id",  # RANK TOP w/o K
    "PREDICT SUM(t.c) OVER (1 DAY FOLLOWING) FORECAST 3 TIMEFRAMES FOR EACH t.id",  # FORECAST removed
    "PREDICT t.c IS FOR EACH t.id",
    "PREDICT (SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t.id",  # unbalanced paren
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t.id ASSUMING AND",
]


@pytest.mark.parametrize("query", MALFORMED)
def test_malformed_query_rejected(query):
    with pytest.raises(PqlSyntaxError):
        parse(query)


# ---------------------------------------------------------------------------
# AST spot checks
# ---------------------------------------------------------------------------

def test_churn_query_ast():
    pq = parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
               "FOR EACH customers.customer_id")
    assert isinstance(pq.target, Condition)
    assert pq.target.op is Operator.EQ
    assert pq.target.right == 0
    agg = pq.target.left
    assert isinstance(agg, Aggregation)
    assert agg.func is AggFunc.COUNT
    assert agg.column == ColumnRef("orders", "*")
    assert agg.window.start == 0 and agg.window.end == 90
    assert agg.window.unit is TimeUnit.DAYS
    assert agg.window.horizons == 1
    assert pq.entity_key == ColumnRef("customers", "customer_id")
    assert pq.task_type() is TaskType.BINARY_CLASSIFICATION


def test_entity_selectors():
    pq = parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
               "FOR users.user_id IN (42, 123)")
    assert pq.entity_ids == (42, 123)
    pq = parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
               "FOR users.user_id = 42 ASSUMING users.plan = 'premium'")
    assert pq.entity_ids == (42,)
    assert isinstance(pq.assuming, Condition)
    assert pq.assuming.right == "premium"


def test_rank_and_classify():
    pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID) OVER (30 DAYS FOLLOWING) "
               "RANK TOP 12 FOR EACH CUSTOMERS.CUSTOMER_ID")
    assert pq.rank is RankKind.RANK and pq.top_k == 12
    assert pq.task_type() is TaskType.MULTILABEL_RANKING
    pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID) OVER (30 DAYS FOLLOWING) "
               "CLASSIFY FOR EACH CUSTOMERS.CUSTOMER_ID")
    assert pq.rank is RankKind.CLASSIFY
    assert pq.task_type() is TaskType.MULTICLASS_CLASSIFICATION


def test_forecast_via_horizons():
    pq = parse("PREDICT SUM(usage.count) OVER (1 DAY FOLLOWING HORIZONS 28) "
               "FOR EACH accounts.account_id")
    assert pq.num_forecasts == 28
    assert pq.target.window.horizons == 28
    assert pq.task_type() is TaskType.FORECASTING
    # soft keyword `count` as a column name survived
    assert pq.target.column == ColumnRef("usage", "count")


def test_horizons_with_step():
    pq = parse("PREDICT SUM(sales.qty) OVER demand_projection FOR EACH stores.store_id "
               "WINDOW demand_projection AS (30 DAYS FOLLOWING HORIZONS 6 STEP 7 DAYS)")
    w = pq.target.window
    assert w.horizons == 6 and w.step == 7
    assert w.start == 0 and w.end == 30
    assert pq.num_forecasts == 6
    # the named window template is carried on the query, fully resolved
    assert "demand_projection" in pq.windows
    assert pq.windows["demand_projection"].horizons == 6


def test_inf_bound_and_not():
    pq = parse("PREDICT COUNT(transactions.*) OVER (UNBOUNDED PRECEDING) > 0 "
               "FOR EACH user.user_id")
    assert math.isinf(pq.target.left.window.start)
    assert pq.target.left.window.start < 0
    pq = parse("PREDICT NOT LAST(LOAN.AMOUNT) OVER (30 DAYS FOLLOWING) > 30 "
               "FOR EACH LOAN.id")
    assert isinstance(pq.target, Not)
    assert isinstance(pq.target.expr, Condition)


def test_word_operators_and_membership():
    pq = parse("PREDICT LAST(LOAN.STATUS) OVER (30 DAYS FOLLOWING) NOT LIKE '%DENIED' "
               "FOR EACH LOAN.id")
    assert pq.target.op is Operator.NOT_LIKE
    pq = parse("PREDICT LOAN.STATUS IN ('A', 'C') FOR EACH LOAN.id")
    assert pq.target.op is Operator.IN and pq.target.right == ("A", "C")
    pq = parse("PREDICT ARTICLES.DESCRIPTION IS NULL FOR EACH ARTICLES.id")
    assert pq.target.op is Operator.IS_NULL
    pq = parse("PREDICT MOVIE.TITLE STARTS WITH 'The' FOR EACH MOVIE.id")
    assert pq.target.op is Operator.STARTS_WITH


def test_inline_agg_filter_and_boolean_where():
    pq = parse("PREDICT COUNT(transaction.* WHERE transaction.amount > 100) "
               "FOR EACH user.user_id WHERE user.country = 'US'")
    agg = pq.target
    assert agg.window is None            # windowless static count
    assert isinstance(agg.filter, Condition)
    assert isinstance(pq.where, Condition)
    pq = parse("PREDICT SUM(TRANSACTIONS.PRICE) OVER (30 DAYS FOLLOWING) FOR EACH "
               "CUSTOMERS.CUSTOMER_ID WHERE (user.country = 'US' OR "
               "region.num_inhabitants < 10000) AND user.dietary = 'Vegetarian'")
    assert isinstance(pq.where, LogicalOp)
    assert pq.where.op.name == "AND"
    assert isinstance(pq.where.left, LogicalOp)  # the parenthesized OR


def test_date_literal():
    pq = parse("PREDICT CUSTOMERS.industry = 'IT' AND "
               "CUSTOMERS.date_of_birth <= 1990-01-01 FOR EACH CUSTOMERS.CUSTOMER_ID")
    cond = pq.target.right
    assert isinstance(cond, Condition)
    assert cond.right == datetime(1990, 1, 1)


def test_case_insensitive_keywords():
    pq = parse("predict sum(t.c) over (30 days following) for each t.id")
    assert pq.target.func is AggFunc.SUM
    assert pq.target.window.end == 30


# ---------------------------------------------------------------------------
# New v2 grammar surface
# ---------------------------------------------------------------------------

def test_exists_and_not_exists():
    pq = parse("PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) "
               "FOR EACH customers.customer_id")
    assert isinstance(pq.target, Aggregation)
    assert pq.target.func is AggFunc.EXISTS
    assert pq.task_type() is TaskType.BINARY_CLASSIFICATION

    pq = parse("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
               "FOR EACH customers.customer_id "
               "WHERE EXISTS(orders.*) OVER (90 DAYS PRECEDING) "
               "AS OF :prediction_time RETURN PROBABILITY")
    assert isinstance(pq.target, Not)
    assert pq.target.expr.func is AggFunc.EXISTS
    assert isinstance(pq.where, Aggregation) and pq.where.func is AggFunc.EXISTS
    assert pq.where.window.start == -90 and pq.where.window.end == 0
    assert pq.task_type() is TaskType.BINARY_CLASSIFICATION
    assert pq.as_of == AsOf("param", "prediction_time")
    assert pq.ret == ReturnSpec("PROBABILITY")


def test_named_window_resolution():
    pq = parse("PREDICT SUM(orders.revenue) OVER next_30_days "
               "- SUM(orders.cost) OVER next_30_days "
               "FOR EACH customers.customer_id "
               "WINDOW next_30_days AS (30 DAYS FOLLOWING)")
    assert isinstance(pq.target, Arith) and pq.target.op == "-"
    # both operands resolved to the same normalized frame
    assert pq.target.left.window.start == 0 and pq.target.left.window.end == 30
    assert pq.target.right.window.start == 0 and pq.target.right.window.end == 30
    assert "next_30_days" in pq.windows


def test_arithmetic_and_function_target():
    pq = parse("PREDICT GREATEST(SUM(orders.revenue) OVER (30 DAYS FOLLOWING), 0) * 2 "
               "FOR EACH customers.customer_id")
    assert isinstance(pq.target, Arith) and pq.target.op == "*"
    assert isinstance(pq.target.left, Func) and pq.target.left.name == "GREATEST"
    assert isinstance(pq.target.right, Lit) and pq.target.right.value == 2
    assert pq.task_type() is TaskType.REGRESSION

    pq = parse("PREDICT COALESCE(SUM(orders.revenue) OVER (30 DAYS FOLLOWING), 0) "
               "FOR EACH customers.customer_id RETURN EXPECTED VALUE")
    assert isinstance(pq.target, Func) and pq.target.name == "COALESCE"
    assert pq.ret == ReturnSpec("EXPECTED_VALUE")


def test_case_target():
    pq = parse("PREDICT CASE WHEN COUNT(orders.*) OVER (30 DAYS FOLLOWING) > 10 "
               "THEN 1 ELSE 0 END FOR EACH customers.customer_id")
    assert isinstance(pq.target, Case)
    assert len(pq.target.whens) == 1
    cond, then = pq.target.whens[0]
    assert isinstance(cond, Condition) and cond.op is Operator.GT
    assert isinstance(then, Lit) and then.value == 1
    assert isinstance(pq.target.else_, Lit) and pq.target.else_.value == 0


def test_return_specs():
    pq = parse("PREDICT SUM(orders.amount) OVER (RANGE BETWEEN 15 DAYS FOLLOWING "
               "AND 45 DAYS FOLLOWING) FOR customers.customer_id IN ('C7', 'C9') "
               "AS OF :prediction_time RETURN QUANTILES (0.10, 0.50, 0.90)")
    assert pq.ret.kind == "QUANTILES"
    assert pq.ret.quantiles == (0.10, 0.50, 0.90)
    pq = parse("PREDICT SUM(payments.amount) OVER (30 DAYS FOLLOWING) "
               "FOR EACH customers.customer_id AS OF :t RETURN INTERVAL 90%")
    assert pq.ret.kind == "INTERVAL" and pq.ret.interval == 90


def test_as_of_variants():
    pq = parse("EXPLAIN CONTEXT PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) "
               "FOR customers.customer_id = 'C7' AS OF 2026-07-01")
    assert pq.as_of == AsOf("date", "2026-07-01")


def test_explain_prefix_and_ablate():
    pq = parse("EXPLAIN PLAN FORMAT TEXT PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) "
               "FOR EACH customers.customer_id ABLATE TABLE support_tickets "
               "RETURN PROBABILITY")
    assert pq.explain == Explain("PLAN", "TEXT")
    assert len(pq.ablations) == 1
    assert pq.ablations[0].kind == "table"
    assert pq.ablations[0].name == "support_tickets"
    assert pq.ret == ReturnSpec("PROBABILITY")


# ---------------------------------------------------------------------------
# Schema-bound validation
# ---------------------------------------------------------------------------

def test_validate_ok(churn_schema):
    vq = validate("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                  "FOR EACH customers.customer_id", churn_schema)
    assert vq.task_type is TaskType.BINARY_CLASSIFICATION


def test_validate_static_column_types(churn_schema):
    vq = validate("PREDICT customers.age FOR EACH customers.customer_id",
                  churn_schema)
    assert vq.task_type is TaskType.REGRESSION
    vq = validate("PREDICT products.name FOR EACH products.product_id",
                  churn_schema)
    assert vq.task_type is TaskType.MULTICLASS_CLASSIFICATION


def test_validate_rejects_unknowns(churn_schema):
    with pytest.raises(PqlValidationError):
        validate("PREDICT COUNT(nope.*) OVER (90 DAYS FOLLOWING) = 0 "
                 "FOR EACH customers.customer_id", churn_schema)
    with pytest.raises(PqlValidationError):
        validate("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                 "FOR EACH customers.oops", churn_schema)
    with pytest.raises(PqlValidationError):
        validate("PREDICT orders.nope FOR EACH orders.order_id", churn_schema)


def test_validate_rejects_past_facing_target_window(churn_schema):
    with pytest.raises(PqlValidationError):
        validate("PREDICT COUNT(orders.*) OVER (30 DAYS PRECEDING) "
                 "FOR EACH customers.customer_id", churn_schema)


def test_validate_rejects_window_on_static_table(churn_schema):
    with pytest.raises(PqlValidationError):
        validate("PREDICT SUM(products.price) OVER (30 DAYS FOLLOWING) "
                 "FOR EACH customers.customer_id", churn_schema)


def test_validate_rejects_horizons_in_where(churn_schema):
    with pytest.raises(PqlValidationError):
        validate("PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) "
                 "FOR EACH customers.customer_id "
                 "WHERE COUNT(orders.*) OVER (7 DAYS FOLLOWING HORIZONS 3) > 0",
                 churn_schema)
