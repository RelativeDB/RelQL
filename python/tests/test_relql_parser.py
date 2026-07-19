"""Grammar conformance: the full corpus parses; malformed input is rejected;
AST spot checks; task-type inference; schema validation."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import pytest

from relativedb import (MissingParameterError, RelqlSyntaxError,
                        RelqlValidationError, parse, validate)
from relativedb.relql import (AggFunc, Aggregation, Arith, AsOf, Case, ColumnRef,
                          Condition, Explain, Func, Lit, LogicalOp, Not,
                          Operator, Param, RankKind, ReturnSpec, TaskType,
                          TimeUnit)

CORPUS = [
    line.strip()
    for line in (Path(__file__).parent / "data" / "examples.relql").read_text().splitlines()
    if line.strip()
]


def test_corpus_has_67_queries():
    assert len(CORPUS) == 67


@pytest.mark.parametrize("query", CORPUS)
def test_corpus_query_parses(query):
    pq = parse(query)
    assert pq.entity_key.table
    # the primary key is schema knowledge; parsing leaves it unbound
    assert pq.entity_key.column is None
    # every query has a well-defined task type even without a schema
    assert pq.task_type() in TaskType


MALFORMED = [
    "",
    "   ",
    "SELECT * FROM t",
    "PREDICT",
    "PREDICT FROM t",
    "PREDICT SUM(t.c, 0, 30)",                              # positional windows removed
    "PREDICT SUM(t.c, 0, 30) FROM t",               # positional windows removed
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t.id",  # FOR EACH removed
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING)",   # aggregate target needs FROM
    "PREDICT a.x = b.y",                  # no FROM, target spans two tables
    "PREDICT 1 > 0",                      # no FROM, target names no table
    "PREDICT SUM(t.c) OVER (30 DAYS) FROM t",         # frame missing PRECEDING/FOLLOWING
    "PREDICT SUM(t.c) OVER (30 FORTNIGHTS FOLLOWING) FROM t",  # bad unit
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING HORIZONS 0) FROM t",  # HORIZONS must be positive
    "PREDICT SUM(t.c) OVER nope FROM t",              # undeclared window
    "PREDICT SUM(t.c) OVER (RANGE BETWEEN 1 DAY FOLLOWING AND 2 MONTHS FOLLOWING) "
    "FROM t",                                         # mixed fixed/calendar units
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FROM t trailing junk",
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FROM t WHERE",
    "PREDICT COUNT(t.*) OVER (30 DAYS FOLLOWING) = FROM t",  # comparison w/o RHS
    "PREDICT t.c IN (1, 2 FROM t",                    # unclosed list
    "PREDICT LIST_DISTINCT(t.c) OVER (30 DAYS FOLLOWING RANK TOP) FROM t",  # RANK TOP w/o K
    "PREDICT LIST_DISTINCT(t.c) OVER (30 DAYS FOLLOWING) RANK TOP 12 FROM t",  # RANK TOP is frame-only
    "PREDICT SUM(t.c) OVER (1 DAY FOLLOWING) FORECAST 3 TIMEFRAMES FROM t",  # FORECAST removed
    "PREDICT t.c IS FROM t",
    "PREDICT (SUM(t.c) OVER (30 DAYS FOLLOWING) FROM t",  # unbalanced paren
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FROM t ASSUMING AND",
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR t.id",       # bare FOR (needs EACH)
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR t.id = 42",  # pinned selector removed
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR t.id IN (1, 2)",  # IN selector removed
]


@pytest.mark.parametrize("query", MALFORMED)
def test_malformed_query_rejected(query):
    with pytest.raises(RelqlSyntaxError):
        parse(query)


# ---------------------------------------------------------------------------
# AST spot checks
# ---------------------------------------------------------------------------

def test_churn_query_ast():
    pq = parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
               "FROM customers")
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
    assert pq.entity_key == ColumnRef("customers", None)
    assert pq.entity_inferred is False
    assert pq.task_type() is TaskType.BINARY_CLASSIFICATION


def test_pinned_entity_selector_rejected():
    # The `FOR t.pk = v` / `FOR t.pk IN (...)` forms were removed; only
    # `FROM t` is valid. Concrete ids are supplied at execution time.
    for q in ("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FOR users.user_id IN (42, 123)",
              "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FOR users.user_id = 42"):
        with pytest.raises(RelqlSyntaxError):
            parse(q)


def test_assuming_with_from():
    pq = parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
               "FROM users WHERE users.user_id = 42 "
               "ASSUMING users.plan = 'premium'")
    assert pq.entity_key == ColumnRef("users", None)
    assert isinstance(pq.assuming, Condition)
    assert pq.assuming.right == "premium"


def test_rank_and_classify():
    pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID) "
               "OVER (30 DAYS FOLLOWING RANK TOP 12) FROM CUSTOMERS")
    # RANK TOP is a frame directive, lifted to the query for task inference
    assert pq.target.window.top_k == 12
    assert pq.rank is RankKind.RANK and pq.top_k == 12
    assert pq.task_type() is TaskType.MULTILABEL_RANKING
    pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID) OVER (30 DAYS FOLLOWING) "
               "CLASSIFY FROM CUSTOMERS")
    assert pq.rank is RankKind.CLASSIFY
    assert pq.task_type() is TaskType.MULTICLASS_CLASSIFICATION


def test_forecast_via_horizons():
    pq = parse("PREDICT SUM(usage.count) OVER (1 DAY FOLLOWING HORIZONS 28) "
               "FROM accounts")
    assert pq.num_forecasts == 28
    assert pq.target.window.horizons == 28
    assert pq.task_type() is TaskType.FORECASTING
    # soft keyword `count` as a column name survived
    assert pq.target.column == ColumnRef("usage", "count")


def test_horizons_with_step():
    pq = parse("PREDICT SUM(sales.qty) OVER demand_projection FROM stores "
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
               "FROM user")
    assert math.isinf(pq.target.left.window.start)
    assert pq.target.left.window.start < 0
    pq = parse("PREDICT NOT LAST(LOAN.AMOUNT) OVER (30 DAYS FOLLOWING) > 30 "
               "FROM LOAN")
    assert isinstance(pq.target, Not)
    assert isinstance(pq.target.expr, Condition)


def test_word_operators_and_membership():
    pq = parse("PREDICT LAST(LOAN.STATUS) OVER (30 DAYS FOLLOWING) NOT LIKE '%DENIED' "
               "FROM LOAN")
    assert pq.target.op is Operator.NOT_LIKE
    pq = parse("PREDICT LOAN.STATUS IN ('A', 'C') FROM LOAN")
    assert pq.target.op is Operator.IN and pq.target.right == ("A", "C")
    pq = parse("PREDICT ARTICLES.DESCRIPTION IS NULL FROM ARTICLES")
    assert pq.target.op is Operator.IS_NULL
    pq = parse("PREDICT MOVIE.TITLE STARTS WITH 'The' FROM MOVIE")
    assert pq.target.op is Operator.STARTS_WITH


def test_inline_agg_filter_and_boolean_where():
    pq = parse("PREDICT COUNT(transaction.* WHERE transaction.amount > 100) "
               "FROM user WHERE user.country = 'US'")
    agg = pq.target
    # no OVER in PREDICT => implied unbounded future frame
    assert agg.window.implied and agg.window.start == 0
    assert math.isinf(agg.window.end) and agg.window.end > 0
    assert isinstance(agg.filter, Condition)
    assert isinstance(pq.where, Condition)
    pq = parse("PREDICT SUM(TRANSACTIONS.PRICE) OVER (30 DAYS FOLLOWING) FROM "
               "CUSTOMERS WHERE (user.country = 'US' OR "
               "region.num_inhabitants < 10000) AND user.dietary = 'Vegetarian'")
    assert isinstance(pq.where, LogicalOp)
    assert pq.where.op.name == "AND"
    assert isinstance(pq.where.left, LogicalOp)  # the parenthesized OR


def test_date_literal():
    pq = parse("PREDICT CUSTOMERS.industry = 'IT' AND "
               "CUSTOMERS.date_of_birth <= 1990-01-01 FROM CUSTOMERS")
    cond = pq.target.right
    assert isinstance(cond, Condition)
    assert cond.right == datetime(1990, 1, 1)


def test_case_insensitive_keywords():
    pq = parse("predict sum(t.c) over (30 days following) from t")
    assert pq.target.func is AggFunc.SUM
    assert pq.target.window.end == 30


# ---------------------------------------------------------------------------
# New v2 grammar surface
# ---------------------------------------------------------------------------

def test_exists_and_not_exists():
    pq = parse("PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) "
               "FROM customers")
    assert isinstance(pq.target, Aggregation)
    assert pq.target.func is AggFunc.EXISTS
    assert pq.task_type() is TaskType.BINARY_CLASSIFICATION

    pq = parse("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
               "FROM customers "
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
               "FROM customers "
               "WINDOW next_30_days AS (30 DAYS FOLLOWING)")
    assert isinstance(pq.target, Arith) and pq.target.op == "-"
    # both operands resolved to the same normalized frame
    assert pq.target.left.window.start == 0 and pq.target.left.window.end == 30
    assert pq.target.right.window.start == 0 and pq.target.right.window.end == 30
    assert "next_30_days" in pq.windows


def test_arithmetic_and_function_target():
    pq = parse("PREDICT GREATEST(SUM(orders.revenue) OVER (30 DAYS FOLLOWING), 0) * 2 "
               "FROM customers")
    assert isinstance(pq.target, Arith) and pq.target.op == "*"
    assert isinstance(pq.target.left, Func) and pq.target.left.name == "GREATEST"
    assert isinstance(pq.target.right, Lit) and pq.target.right.value == 2
    assert pq.task_type() is TaskType.REGRESSION

    pq = parse("PREDICT COALESCE(SUM(orders.revenue) OVER (30 DAYS FOLLOWING), 0) "
               "FROM customers RETURN EXPECTED VALUE")
    assert isinstance(pq.target, Func) and pq.target.name == "COALESCE"
    assert pq.ret == ReturnSpec("EXPECTED_VALUE")


def test_case_target():
    pq = parse("PREDICT CASE WHEN COUNT(orders.*) OVER (30 DAYS FOLLOWING) > 10 "
               "THEN 1 ELSE 0 END FROM customers")
    assert isinstance(pq.target, Case)
    assert len(pq.target.whens) == 1
    cond, then = pq.target.whens[0]
    assert isinstance(cond, Condition) and cond.op is Operator.GT
    assert isinstance(then, Lit) and then.value == 1
    assert isinstance(pq.target.else_, Lit) and pq.target.else_.value == 0


def test_return_specs():
    pq = parse("PREDICT SUM(orders.amount) OVER (RANGE BETWEEN 15 DAYS FOLLOWING "
               "AND 45 DAYS FOLLOWING) FROM customers "
               "AS OF :prediction_time RETURN EXPECTED VALUE")
    assert pq.ret.kind == "EXPECTED_VALUE"
    pq = parse("PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) = 0 "
               "FROM customers RETURN PROBABILITY")
    assert pq.ret.kind == "PROBABILITY"


@pytest.mark.parametrize("clause", ["QUANTILES (0.1, 0.9)", "INTERVAL 90%"])
def test_return_quantiles_and_interval_are_removed(clause):
    """They never executed — a single point head exposes no distribution — so
    they are out of the grammar. The error names them rather than reporting an
    unexpected token, because queries written against the old grammar exist."""
    with pytest.raises(RelqlSyntaxError, match="not supported"):
        parse(f"PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
              f"FROM customers RETURN {clause}")


def test_as_of_variants():
    pq = parse("EXPLAIN CONTEXT PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) "
               "FROM customers WHERE customers.customer_id = 'C7' "
               "AS OF 2026-07-01")
    assert pq.as_of == AsOf("date", "2026-07-01")


def test_explain_prefix_and_ablate():
    pq = parse("EXPLAIN PLAN FORMAT TEXT PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) "
               "FROM customers ABLATE TABLE support_tickets "
               "RETURN PROBABILITY")
    assert pq.explain == Explain("PLAN", "TEXT")
    assert len(pq.ablations) == 1
    assert pq.ablations[0].kind == "table"
    assert pq.ablations[0].name == "support_tickets"
    assert pq.ret == ReturnSpec("PROBABILITY")


# ---------------------------------------------------------------------------
# FROM: aliases, unqualified columns, inferred population, implied frames
# ---------------------------------------------------------------------------

def test_unqualified_columns_bind_to_the_population():
    pq = parse("PREDICT label FROM issues WHERE label IS NULL")
    assert pq.target == ColumnRef("issues", "label")
    assert pq.where.left == ColumnRef("issues", "label")
    assert pq.entity_key.table == "issues"


def test_population_inferred_without_from():
    pq = parse("PREDICT issues.label WHERE issues.label IS NULL")
    assert pq.entity_key.table == "issues"
    assert pq.entity_inferred is True
    # ...and an explicit FROM is not inferred
    assert parse("PREDICT issues.label FROM issues").entity_inferred is False


def test_inferred_population_needs_one_unambiguous_table():
    for q in ("PREDICT a.x = b.y",      # two tables, no way to choose
              "PREDICT 1 > 0",          # no table at all
              "PREDICT SUM(orders.amount) OVER (30 DAYS FOLLOWING)"):
        with pytest.raises(RelqlSyntaxError):
            parse(q)


def test_alias_resolves_to_its_table():
    for q in ("PREDICT NOT EXISTS(orders.*) FROM customers c "
              "WHERE c.customer_id = 42 ASSUMING c.plan = 'premium'",
              "PREDICT NOT EXISTS(orders.*) FROM customers AS c "
              "WHERE c.customer_id = 42 ASSUMING c.plan = 'premium'"):
        pq = parse(q)
        assert pq.entity_key.table == "customers"
        # the alias is gone from the AST; only real table names survive
        assert pq.where.left == ColumnRef("customers", "customer_id")
        assert pq.assuming.left == ColumnRef("customers", "plan")


def test_alias_slot_does_not_swallow_a_clause_keyword():
    pq = parse("PREDICT SUM(payments.amount) OVER (30 DAYS FOLLOWING) "
               "FROM customers AS OF :t RETURN EXPECTED VALUE")
    assert pq.entity_key.table == "customers"
    assert pq.as_of == AsOf("param", "t")
    assert pq.ret.kind == "EXPECTED_VALUE"


def test_implied_frame_direction_follows_the_clause():
    pq = parse("PREDICT COUNT(orders.*) FROM customers "
               "WHERE COUNT(orders.*) > 5")
    target = pq.target.window                 # PREDICT -> the future
    assert target.implied and target.start == 0
    assert math.isinf(target.end) and target.end > 0
    where = pq.where.left.window              # WHERE -> the past
    assert where.implied and where.end == 0
    assert math.isinf(where.start) and where.start < 0


def test_explicit_frame_is_not_marked_implied():
    pq = parse("PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) FROM customers")
    assert pq.target.window.implied is False


def test_array_agg_keeps_duplicates_and_ranks():
    pq = parse("PREDICT ARRAY_AGG(transactions.article_id) "
               "OVER (30 DAYS FOLLOWING RANK TOP 12) FROM customers")
    assert pq.target.func is AggFunc.ARRAY_AGG
    assert pq.target.window.top_k == 12
    assert pq.task_type() is TaskType.MULTILABEL_RANKING


# ---------------------------------------------------------------------------
# Bind parameters
# ---------------------------------------------------------------------------

def test_parameter_in_equality_and_in():
    pq = parse("PREDICT NOT EXISTS(orders.*) FROM customers "
               "WHERE customers.customer_id = :id")
    assert pq.where.right_expr == Param("id")
    assert pq.has_params

    pq = parse("PREDICT NOT EXISTS(orders.*) FROM customers "
               "WHERE customers.customer_id IN :ids")
    assert pq.where.op is Operator.IN
    assert pq.where.right_expr == Param("ids")


def test_parameters_bind_to_literals():
    pq = parse("PREDICT NOT EXISTS(orders.*) FROM customers "
               "WHERE customers.customer_id IN :ids").bind_params(
                   {"ids": ["C7", "C8"]})
    # a bound parameter collapses onto the literal slot; nothing downstream
    # has to know a parameter was ever there
    assert pq.where.right == ("C7", "C8")
    assert pq.where.right_expr is None
    assert pq.has_params is False


def test_parameters_work_with_word_operators():
    pq = parse("PREDICT customers.churned FROM customers "
               "WHERE customers.plan LIKE :pat AND customers.age > :min_age")
    bound = pq.bind_params({"pat": "pro%", "min_age": 18})
    assert bound.where.left.right == "pro%"
    assert bound.where.right.right == 18


def test_missing_parameter_names_itself():
    pq = parse("PREDICT NOT EXISTS(orders.*) FROM customers "
               "WHERE customers.customer_id IN :ids")
    with pytest.raises(MissingParameterError, match="ids"):
        pq.bind_params({"other": 1})


def test_query_without_parameters_is_unchanged_by_binding():
    pq = parse("PREDICT NOT EXISTS(orders.*) FROM customers")
    assert pq.bind_params({"unused": 1}) is pq


# ---------------------------------------------------------------------------
# Schema-bound validation
# ---------------------------------------------------------------------------

def test_validate_ok(churn_schema):
    vq = validate("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                  "FROM customers", churn_schema)
    assert vq.task_type is TaskType.BINARY_CLASSIFICATION


def test_validate_binds_the_primary_key(churn_schema):
    vq = validate("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                  "FROM customers", churn_schema)
    assert vq.query.entity_key == ColumnRef("customers", "customer_id")


def test_validate_static_column_types(churn_schema):
    vq = validate("PREDICT customers.age FROM customers",
                  churn_schema)
    assert vq.task_type is TaskType.REGRESSION
    vq = validate("PREDICT products.name FROM products",
                  churn_schema)
    assert vq.task_type is TaskType.MULTICLASS_CLASSIFICATION


def test_validate_rejects_unknowns(churn_schema):
    with pytest.raises(RelqlValidationError):
        validate("PREDICT COUNT(nope.*) OVER (90 DAYS FOLLOWING) = 0 "
                 "FROM customers", churn_schema)
    with pytest.raises(RelqlValidationError):
        validate("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                 "FROM nope", churn_schema)          # unknown population
    with pytest.raises(RelqlValidationError):
        validate("PREDICT orders.nope FROM orders", churn_schema)


def test_validate_rejects_past_facing_target_window(churn_schema):
    with pytest.raises(RelqlValidationError):
        validate("PREDICT COUNT(orders.*) OVER (30 DAYS PRECEDING) "
                 "FROM customers", churn_schema)


def test_validate_rejects_window_on_static_table(churn_schema):
    with pytest.raises(RelqlValidationError):
        validate("PREDICT SUM(products.price) OVER (30 DAYS FOLLOWING) "
                 "FROM customers", churn_schema)


def test_validate_rejects_horizons_in_where(churn_schema):
    with pytest.raises(RelqlValidationError):
        validate("PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) "
                 "FROM customers "
                 "WHERE COUNT(orders.*) OVER (7 DAYS FOLLOWING HORIZONS 3) > 0",
                 churn_schema)
