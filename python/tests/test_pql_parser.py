"""Grammar conformance: the full 44-query corpus parses; malformed input is
rejected; AST spot checks; task-type inference; schema validation."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pytest

from relativedb import PqlSyntaxError, PqlValidationError, parse, validate
from relativedb.pql import (AggFunc, Aggregation, ColumnRef, Condition,
                          LogicalOp, Not, Operator, RankKind, TaskType,
                          TimeUnit)

CORPUS = [
    line.strip()
    for line in (Path(__file__).parent / "data" / "examples.pql").read_text().splitlines()
    if line.strip()
]


def test_corpus_has_44_queries():
    assert len(CORPUS) == 44


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
    "PREDICT SUM(t.c, 0, 30)",                              # missing FOR
    "PREDICT SUM(t.c, 0, 30) FOR EACH t",                    # not table.column
    "PREDICT SUM(c, 0, 30) FOR EACH t.id",                   # agg operand not table.column
    "PREDICT SUM(t.c 0, 30) FOR EACH t.id",                  # missing comma
    "PREDICT SUM(t.c, 0) FOR EACH t.id",                     # window needs 2 bounds
    "PREDICT SUM(t.c, 0, 30, fortnights) FOR EACH t.id",     # bad time unit
    "PREDICT SUM(t.c, 0, 30) FOR EACH t.id trailing junk",
    "PREDICT SUM(t.c, 0, 30) FOR EACH t.id WHERE",
    "PREDICT COUNT(t.*, 0, 30) = FOR EACH t.id",             # comparison w/o literal
    "PREDICT t.c IN (1, 2 FOR EACH t.id",                    # unclosed list
    "PREDICT LIST_DISTINCT(t.c, 0, 30) RANK TOP FOR EACH t.id",  # RANK TOP w/o K
    "PREDICT SUM(t.c, 0, 30) FORECAST TIMEFRAMES FOR EACH t.id",
    "PREDICT t.c IS FOR EACH t.id",
    "PREDICT (SUM(t.c, 0, 30) FOR EACH t.id",                # unbalanced paren
    "PREDICT SUM(t.c, 0, 30) FOR EACH t.id ASSUMING AND",
]


@pytest.mark.parametrize("query", MALFORMED)
def test_malformed_query_rejected(query):
    with pytest.raises(PqlSyntaxError):
        parse(query)


# ---------------------------------------------------------------------------
# AST spot checks
# ---------------------------------------------------------------------------

def test_churn_query_ast():
    pq = parse("PREDICT COUNT(orders.*, 0, 90, days) = 0 "
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
    assert pq.entity_key == ColumnRef("customers", "customer_id")
    assert pq.task_type() is TaskType.BINARY_CLASSIFICATION


def test_entity_selectors():
    pq = parse("PREDICT COUNT(orders.*, 0, 90, days) = 0 "
               "FOR users.user_id IN (42, 123)")
    assert pq.entity_ids == (42, 123)
    pq = parse("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR users.user_id = 42 "
               "ASSUMING users.plan = 'premium'")
    assert pq.entity_ids == (42,)
    assert isinstance(pq.assuming, Condition)
    assert pq.assuming.right == "premium"


def test_rank_and_classify():
    pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID, 0, 30) "
               "RANK TOP 12 FOR EACH CUSTOMERS.CUSTOMER_ID")
    assert pq.rank is RankKind.RANK and pq.top_k == 12
    assert pq.task_type() is TaskType.MULTILABEL_RANKING
    pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID, 0, 30) "
               "CLASSIFY FOR EACH CUSTOMERS.CUSTOMER_ID")
    assert pq.rank is RankKind.CLASSIFY
    assert pq.task_type() is TaskType.MULTICLASS_CLASSIFICATION


def test_forecast():
    pq = parse("PREDICT SUM(usage.count, 0, 1, days) FORECAST 28 TIMEFRAMES "
               "FOR EACH accounts.account_id")
    assert pq.num_forecasts == 28
    assert pq.task_type() is TaskType.FORECASTING
    # soft keyword `count` as a column name survived
    assert pq.target.column == ColumnRef("usage", "count")


def test_inf_bound_and_not():
    pq = parse("PREDICT COUNT(transactions.*, -INF, 0) > 0 FOR EACH user.user_id")
    assert math.isinf(pq.target.left.window.start)
    assert pq.target.left.window.start < 0
    pq = parse("PREDICT NOT LAST(LOAN.AMOUNT, 0, 30) > 30 FOR EACH LOAN.id")
    assert isinstance(pq.target, Not)
    assert isinstance(pq.target.expr, Condition)


def test_word_operators_and_membership():
    pq = parse("PREDICT LAST(LOAN.STATUS, 0, 30) NOT LIKE '%DENIED' FOR EACH LOAN.id")
    assert pq.target.op is Operator.NOT_LIKE
    pq = parse("PREDICT LOAN.STATUS IS IN ('A', 'C') FOR EACH LOAN.id")
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
    pq = parse("PREDICT SUM(TRANSACTIONS.PRICE, 0, 30) FOR EACH "
               "CUSTOMERS.CUSTOMER_ID WHERE (user.country = 'US' OR "
               "region.num_inhabitants < 10000) AND user.dietary = 'Vegetarian'")
    assert isinstance(pq.where, LogicalOp)
    assert pq.where.op.name == "AND"
    assert isinstance(pq.where.left, LogicalOp)  # the parenthesized OR


def test_date_literal():
    pq = parse("PREDICT CUSTOMERS.industry = 'IT' AND "
               "CUSTOMERS.date_of_birth <= 1990-01-01 FOR EACH CUSTOMERS.CUSTOMER_ID")
    right = pq.target.right.right if False else pq.target.right
    cond = pq.target.right
    assert isinstance(cond, Condition)
    assert cond.right == datetime(1990, 1, 1)


def test_case_insensitive_keywords():
    pq = parse("predict sum(t.c, 0, 30) for each t.id")
    assert pq.target.func is AggFunc.SUM


# ---------------------------------------------------------------------------
# Schema-bound validation
# ---------------------------------------------------------------------------

def test_validate_ok(churn_schema):
    vq = validate("PREDICT COUNT(orders.*, 0, 90, days) = 0 "
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
        validate("PREDICT COUNT(nope.*, 0, 90) = 0 FOR EACH customers.customer_id",
                 churn_schema)
    with pytest.raises(PqlValidationError):
        validate("PREDICT COUNT(orders.*, 0, 90) = 0 FOR EACH customers.oops",
                 churn_schema)
    with pytest.raises(PqlValidationError):
        validate("PREDICT orders.nope FOR EACH orders.order_id", churn_schema)


def test_validate_rejects_past_facing_target_window(churn_schema):
    with pytest.raises(PqlValidationError):
        validate("PREDICT COUNT(orders.*, -30, 0) FOR EACH customers.customer_id",
                 churn_schema)


def test_validate_rejects_window_on_static_table(churn_schema):
    with pytest.raises(PqlValidationError):
        validate("PREDICT SUM(products.price, 0, 30) FOR EACH customers.customer_id",
                 churn_schema)
