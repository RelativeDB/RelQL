//! Grammar conformance, driven through the crate's public `parse()` — which is
//! now single-sourced on the shared C++ parser (`pql_parse` in `librt_c`). This
//! exercises the Rust JSON-AST deserialization end to end (the full 54-query
//! corpus parses; malformed input is rejected; AST spot checks), plus the
//! Rust-only task-type inference and schema validation the C ABI does not cover.
//! Mirrors the Python `test_pql_parser.py`.
//!
//! Requires `librt_c` to be discoverable (the sibling `cpp/build`, or
//! `RELATIVEDB_RT_LIB`); it is a hard runtime dependency of the crate.

mod common;

use relativedb::pql::ast::{AggFunc, CondRhs, Literal, Operator, RankKind, TargetExpr, TimeUnit};
use relativedb::{parse, validate, ColumnRef, LinkDef, Schema, TableDef, TaskType, ValueType};

const CORPUS_SRC: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/examples.pql"));

fn corpus() -> Vec<String> {
    CORPUS_SRC
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

#[test]
fn corpus_has_54_queries() {
    assert_eq!(corpus().len(), 54);
}

#[test]
fn every_corpus_query_parses() {
    for q in corpus() {
        let pq = parse(&q).unwrap_or_else(|e| panic!("failed to parse {:?}: {}", q, e));
        assert!(!pq.entity_key.table.is_empty(), "{:?}", q);
        assert!(!pq.entity_key.column.is_empty(), "{:?}", q);
        // a well-defined task type even without a schema
        let _ = pq.task_type(None);
    }
}

const MALFORMED: &[&str] = &[
    "",
    "   ",
    "SELECT * FROM t",
    "PREDICT",
    "PREDICT FOR EACH t.id",
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING)",             // missing FOR
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t",  // not table.column
    "PREDICT SUM(c) OVER (30 DAYS FOLLOWING) FOR EACH t.id", // agg operand not table.column
    "PREDICT SUM(t.c, 0, 30, days) FOR EACH t.id",           // positional windows removed
    "PREDICT SUM(t.c) OVER (30 DAYS) FOR EACH t.id",         // no PRECEDING/FOLLOWING
    "PREDICT SUM(t.c) OVER (0 DAYS FOLLOWING) FOR EACH t.id", // zero duration
    "PREDICT SUM(t.c) OVER (30 fortnights FOLLOWING) FOR EACH t.id", // bad time unit
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING HORIZONS 0) FOR EACH t.id", // HORIZONS 0
    "PREDICT SUM(t.c) OVER (RANGE BETWEEN 1 DAY FOLLOWING AND 1 MONTH FOLLOWING) FOR EACH t.id", // mixed unit domains
    "PREDICT SUM(t.c) OVER undeclared_win FOR EACH t.id",    // undeclared window name
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t.id trailing junk",
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t.id WHERE",
    "PREDICT COUNT(t.*) OVER (30 DAYS FOLLOWING) = FOR EACH t.id", // comparison w/o literal
    "PREDICT t.c IN (1, 2 FOR EACH t.id",                    // unclosed list
    "PREDICT LIST_DISTINCT(t.c) OVER (30 DAYS FOLLOWING) RANK TOP FOR EACH t.id", // RANK TOP w/o K
    "PREDICT t.c IS FOR EACH t.id",
    "PREDICT (SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t.id", // unbalanced paren
    "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FOR EACH t.id ASSUMING AND",
];

#[test]
fn malformed_queries_rejected() {
    for q in MALFORMED {
        assert!(parse(q).is_err(), "should reject: {:?}", q);
    }
}

#[test]
fn churn_query_ast() {
    let pq =
        parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id")
            .unwrap();
    match &pq.target {
        TargetExpr::Condition(c) => {
            assert_eq!(c.op, Operator::Eq);
            assert_eq!(c.right, CondRhs::One(Literal::Num(0.0)));
            match c.left.as_ref() {
                TargetExpr::Aggregation(a) => {
                    assert_eq!(a.func, AggFunc::Count);
                    assert_eq!(a.column, ColumnRef::new("orders", "*"));
                    let w = a.window.unwrap();
                    assert_eq!(w.start, 0.0);
                    assert_eq!(w.end, 90.0);
                    assert_eq!(w.unit, TimeUnit::Days);
                    assert_eq!(w.horizons, 1);
                    assert_eq!(w.step, None);
                }
                other => panic!("expected aggregation, got {:?}", other),
            }
        }
        other => panic!("expected condition, got {:?}", other),
    }
    assert_eq!(pq.entity_key, ColumnRef::new("customers", "customer_id"));
    assert_eq!(pq.task_type(None), TaskType::BinaryClassification);
}

#[test]
fn entity_selectors() {
    let pq =
        parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id IN (42, 123)")
            .unwrap();
    assert_eq!(pq.entity_ids, vec![Literal::Num(42.0), Literal::Num(123.0)]);
    let pq = parse(
        "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id = 42 ASSUMING users.plan = 'premium'",
    )
    .unwrap();
    assert_eq!(pq.entity_ids, vec![Literal::Num(42.0)]);
    match pq.assuming.unwrap() {
        TargetExpr::Condition(c) => assert_eq!(c.right, CondRhs::One(Literal::Str("premium".into()))),
        other => panic!("{:?}", other),
    }
}

#[test]
fn rank_and_classify() {
    let pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID) OVER (30 DAYS FOLLOWING) RANK TOP 12 FOR EACH CUSTOMERS.CUSTOMER_ID").unwrap();
    assert_eq!(pq.rank, Some(RankKind::Rank));
    assert_eq!(pq.top_k, Some(12));
    assert_eq!(pq.task_type(None), TaskType::MultilabelRanking);
    let pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID) OVER (30 DAYS FOLLOWING) CLASSIFY FOR EACH CUSTOMERS.CUSTOMER_ID").unwrap();
    assert_eq!(pq.rank, Some(RankKind::Classify));
    assert_eq!(pq.task_type(None), TaskType::MulticlassClassification);
}

#[test]
fn horizons_forecasting() {
    let pq =
        parse("PREDICT SUM(usage.count) OVER (1 DAY FOLLOWING HORIZONS 28) FOR EACH accounts.account_id")
            .unwrap();
    assert_eq!(pq.num_forecasts, Some(28));
    assert_eq!(pq.task_type(None), TaskType::Forecasting);
    match &pq.target {
        TargetExpr::Aggregation(a) => {
            assert_eq!(a.column, ColumnRef::new("usage", "count"));
            let w = a.window.unwrap();
            assert_eq!(w.horizons, 28);
            assert_eq!(w.start, 0.0);
            assert_eq!(w.end, 1.0);
        }
        other => panic!("{:?}", other),
    }
    // named window carrying HORIZONS + STEP
    let pq = parse("PREDICT SUM(sales.qty) OVER demand_projection FOR EACH stores.store_id WINDOW demand_projection AS (30 DAYS FOLLOWING HORIZONS 6 STEP 7 DAYS)").unwrap();
    assert_eq!(pq.num_forecasts, Some(6));
    assert!(pq.windows.contains_key("demand_projection"));
    match &pq.target {
        TargetExpr::Aggregation(a) => {
            let w = a.window.unwrap();
            assert_eq!(w.horizons, 6);
            assert_eq!(w.step, Some(7.0));
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn inf_bound_and_not() {
    let pq =
        parse("PREDICT COUNT(transactions.*) OVER (UNBOUNDED PRECEDING) > 0 FOR EACH user.user_id")
            .unwrap();
    match &pq.target {
        TargetExpr::Condition(c) => match c.left.as_ref() {
            TargetExpr::Aggregation(a) => {
                let w = a.window.unwrap();
                assert!(w.start.is_infinite() && w.start < 0.0);
                assert_eq!(w.end, 0.0);
            }
            other => panic!("{:?}", other),
        },
        other => panic!("{:?}", other),
    }
    let pq =
        parse("PREDICT NOT LAST(LOAN.AMOUNT) OVER (30 DAYS FOLLOWING) > 30 FOR EACH LOAN.id").unwrap();
    assert!(matches!(pq.target, TargetExpr::Not(_)));
}

#[test]
fn exists_and_not_exists() {
    // NOT EXISTS target + EXISTS in WHERE, plus AS OF and RETURN.
    let pq = parse("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FOR EACH customers.customer_id WHERE EXISTS(orders.*) OVER (90 DAYS PRECEDING) AS OF :prediction_time RETURN PROBABILITY").unwrap();
    assert_eq!(pq.task_type(None), TaskType::BinaryClassification);
    match &pq.target {
        TargetExpr::Not(inner) => match inner.as_ref() {
            TargetExpr::Aggregation(a) => assert_eq!(a.func, AggFunc::Exists),
            other => panic!("{:?}", other),
        },
        other => panic!("{:?}", other),
    }
    match pq.where_.as_ref().unwrap() {
        TargetExpr::Aggregation(a) => {
            assert_eq!(a.func, AggFunc::Exists);
            let w = a.window.unwrap();
            assert_eq!(w.start, -90.0); // PRECEDING
            assert_eq!(w.end, 0.0);
        }
        other => panic!("{:?}", other),
    }
    let as_of = pq.as_of.as_ref().unwrap();
    assert_eq!(as_of.kind, "param");
    assert_eq!(as_of.value.as_deref(), Some("prediction_time"));
    assert_eq!(pq.ret.as_ref().unwrap().kind, "PROBABILITY");

    // a bare EXISTS target infers binary directly.
    let pq =
        parse("PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id").unwrap();
    assert_eq!(pq.task_type(None), TaskType::BinaryClassification);
}

#[test]
fn arithmetic_target_and_named_window() {
    let pq = parse("PREDICT SUM(orders.revenue) OVER next_30_days - SUM(orders.cost) OVER next_30_days FOR EACH customers.customer_id WINDOW next_30_days AS (30 DAYS FOLLOWING)").unwrap();
    assert_eq!(pq.task_type(None), TaskType::Regression);
    assert!(pq.windows.contains_key("next_30_days"));
    match &pq.target {
        TargetExpr::Arith(a) => {
            assert_eq!(a.op, '-');
            // both operands resolved the named window to (0, 30] days.
            for side in [a.left.as_ref(), a.right.as_ref()] {
                match side {
                    TargetExpr::Aggregation(agg) => {
                        let w = agg.window.unwrap();
                        assert_eq!((w.start, w.end, w.unit), (0.0, 30.0, TimeUnit::Days));
                    }
                    other => panic!("{:?}", other),
                }
            }
        }
        other => panic!("{:?}", other),
    }
    // both aggregations are reachable from the arithmetic target.
    assert_eq!(pq.target_aggregations().len(), 2);
}

#[test]
fn func_case_and_lit_targets() {
    // COALESCE(...) wrapping an aggregation, with RETURN EXPECTED VALUE.
    let pq = parse("PREDICT COALESCE(SUM(orders.revenue) OVER (30 DAYS FOLLOWING), 0) FOR EACH customers.customer_id RETURN EXPECTED VALUE").unwrap();
    assert_eq!(pq.task_type(None), TaskType::Regression);
    assert_eq!(pq.ret.as_ref().unwrap().kind, "EXPECTED_VALUE");
    match &pq.target {
        TargetExpr::Func(f) => {
            assert_eq!(f.name, "COALESCE");
            assert_eq!(f.args.len(), 2);
        }
        other => panic!("{:?}", other),
    }
    // CASE WHEN ... THEN ... ELSE ... END.
    let pq = parse("PREDICT CASE WHEN COUNT(orders.*) OVER (30 DAYS FOLLOWING) > 10 THEN 1 ELSE 0 END FOR EACH customers.customer_id").unwrap();
    match &pq.target {
        TargetExpr::Case(c) => {
            assert_eq!(c.whens.len(), 1);
            assert!(c.else_.is_some());
        }
        other => panic!("{:?}", other),
    }
    // arithmetic on a GREATEST(...) function.
    let pq = parse("PREDICT GREATEST(SUM(orders.revenue) OVER (30 DAYS FOLLOWING), 0) * 2 FOR EACH customers.customer_id").unwrap();
    match &pq.target {
        TargetExpr::Arith(a) => {
            assert_eq!(a.op, '*');
            assert!(matches!(a.left.as_ref(), TargetExpr::Func(_)));
            assert_eq!(a.right.as_ref(), &TargetExpr::Lit(Literal::Num(2.0)));
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn explain_as_of_return() {
    // EXPLAIN prefix + ABLATE TABLE + RETURN.
    let pq = parse("EXPLAIN PLAN FORMAT TEXT PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id ABLATE TABLE support_tickets RETURN PROBABILITY").unwrap();
    let ex = pq.explain.as_ref().unwrap();
    assert_eq!(ex.mode, "PLAN");
    assert_eq!(ex.format, "TEXT");
    assert_eq!(pq.ablations.len(), 1);
    assert_eq!(pq.ablations[0].name, "support_tickets");
    assert_eq!(pq.ret.as_ref().unwrap().kind, "PROBABILITY");

    // EXPLAIN CONTEXT + AS OF date.
    let pq = parse("EXPLAIN CONTEXT PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR customers.customer_id = 'C7' AS OF 2026-07-01").unwrap();
    assert_eq!(pq.explain.as_ref().unwrap().mode, "CONTEXT");
    let as_of = pq.as_of.as_ref().unwrap();
    assert_eq!(as_of.kind, "date");
    assert_eq!(as_of.value.as_deref(), Some("2026-07-01"));

    // RETURN QUANTILES + INTERVAL forms.
    let pq = parse("PREDICT SUM(orders.amount) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING) FOR customers.customer_id IN ('C7', 'C9') AS OF :prediction_time RETURN QUANTILES (0.10, 0.50, 0.90)").unwrap();
    let ret = pq.ret.as_ref().unwrap();
    assert_eq!(ret.kind, "QUANTILES");
    assert_eq!(ret.quantiles, vec![0.10, 0.50, 0.90]);
    let pq = parse("PREDICT SUM(payments.amount) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id AS OF :t RETURN INTERVAL 90%").unwrap();
    let ret = pq.ret.as_ref().unwrap();
    assert_eq!(ret.kind, "INTERVAL");
    assert_eq!(ret.interval, Some(90));
}

#[test]
fn column_to_column_comparison() {
    let pq = parse("PREDICT SUM(transactions.value) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING) > 100 FOR EACH customers.customer_id WHERE customers.location NOT IN ('ALASKA', 'HAWAII')").unwrap();
    match pq.where_.unwrap() {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::NotIn),
        other => panic!("{:?}", other),
    }
}

#[test]
fn word_operators_and_membership() {
    let pq =
        parse("PREDICT LAST(LOAN.STATUS) OVER (30 DAYS FOLLOWING) NOT LIKE '%DENIED' FOR EACH LOAN.id")
            .unwrap();
    match pq.target {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::NotLike),
        other => panic!("{:?}", other),
    }
    let pq = parse("PREDICT LOAN.STATUS IN ('A', 'C') FOR EACH LOAN.id").unwrap();
    match pq.target {
        TargetExpr::Condition(c) => {
            assert_eq!(c.op, Operator::In);
            assert_eq!(c.right, CondRhs::List(vec![Literal::Str("A".into()), Literal::Str("C".into())]));
        }
        other => panic!("{:?}", other),
    }
    let pq = parse("PREDICT ARTICLES.DESCRIPTION IS NULL FOR EACH ARTICLES.id").unwrap();
    match pq.target {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::IsNull),
        other => panic!("{:?}", other),
    }
    let pq = parse("PREDICT MOVIE.TITLE STARTS WITH 'The' FOR EACH MOVIE.id").unwrap();
    match pq.target {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::StartsWith),
        other => panic!("{:?}", other),
    }
    let pq = parse("PREDICT ARTICLES.DESCRIPTION NOT CONTAINS 'refurbished' FOR EACH ARTICLES.id").unwrap();
    match pq.target {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::NotContains),
        other => panic!("{:?}", other),
    }
}

#[test]
fn inline_agg_filter_and_boolean_where() {
    let pq = parse("PREDICT COUNT(transaction.* WHERE transaction.amount > 100) FOR EACH user.user_id WHERE user.country = 'US'").unwrap();
    match &pq.target {
        TargetExpr::Aggregation(a) => {
            assert!(a.window.is_none()); // windowless static count
            assert!(matches!(a.filter.as_deref(), Some(TargetExpr::Condition(_))));
        }
        other => panic!("{:?}", other),
    }
    assert!(matches!(pq.where_, Some(TargetExpr::Condition(_))));
    let pq = parse("PREDICT SUM(TRANSACTIONS.PRICE) OVER (30 DAYS FOLLOWING) FOR EACH CUSTOMERS.CUSTOMER_ID WHERE (user.country = 'US' OR region.num_inhabitants < 10000) AND user.dietary = 'Vegetarian'").unwrap();
    match pq.where_.unwrap() {
        TargetExpr::LogicalOp(l) => {
            assert_eq!(l.op, relativedb::BoolOp::And);
            assert!(matches!(l.left.as_ref(), TargetExpr::LogicalOp(_))); // the parenthesized OR
        }
        other => panic!("{:?}", other),
    }
}

#[test]
fn date_literal_and_eqeq_alias() {
    let pq = parse("PREDICT CUSTOMERS.industry = 'IT' AND CUSTOMERS.date_of_birth <= 1990-01-01 FOR EACH CUSTOMERS.CUSTOMER_ID").unwrap();
    match pq.target {
        TargetExpr::LogicalOp(l) => match l.right.as_ref() {
            TargetExpr::Condition(c) => {
                assert_eq!(c.op, Operator::Le);
                assert_eq!(c.right, CondRhs::One(Literal::Date(common::dt("1990-01-01"))));
            }
            other => panic!("{:?}", other),
        },
        other => panic!("{:?}", other),
    }
    // == alias for =
    let pq = parse("PREDICT LAST(payments.amount) OVER (90 DAYS FOLLOWING) == 0 FOR EACH order.order_id WHERE order.location == 'US'").unwrap();
    match pq.target {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::Eq),
        other => panic!("{:?}", other),
    }
}

#[test]
fn coverage_gap_constructs() {
    // Aggregation functions and operators the corpus rarely exercises — the ones
    // most at risk of a JSON-AST deserialization gap in the native path.
    for (q, func) in [
        ("PREDICT AVG(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id", AggFunc::Avg),
        ("PREDICT MIN(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id", AggFunc::Min),
        ("PREDICT COUNT_DISTINCT(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id", AggFunc::CountDistinct),
    ] {
        match parse(q).unwrap().target {
            TargetExpr::Aggregation(a) => assert_eq!(a.func, func, "{}", q),
            other => panic!("{}: {:?}", q, other),
        }
    }
    match parse("PREDICT FIRST(t.s) OVER (30 DAYS FOLLOWING) = 'A' FOR EACH e.id").unwrap().target {
        TargetExpr::Condition(c) => match c.left.as_ref() {
            TargetExpr::Aggregation(a) => assert_eq!(a.func, AggFunc::First),
            other => panic!("{:?}", other),
        },
        other => panic!("{:?}", other),
    }
    for (q, op) in [
        ("PREDICT t.d IS NOT NULL FOR EACH e.id", Operator::IsNotNull),
        ("PREDICT t.name LIKE '%x%' FOR EACH e.id", Operator::Like),
        ("PREDICT t.name ENDS WITH 'ing' FOR EACH e.id", Operator::EndsWith),
    ] {
        match parse(q).unwrap().target {
            TargetExpr::Condition(c) => assert_eq!(c.op, op, "{}", q),
            other => panic!("{}: {:?}", q, other),
        }
    }
    // `!=` in a WHERE clause -> NEQ.
    match parse("PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) > 5 FOR EACH e.id WHERE t.a != 3").unwrap().where_.unwrap() {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::Neq),
        other => panic!("{:?}", other),
    }
}

#[test]
fn case_insensitive_keywords() {
    let pq = parse("predict sum(t.c) over (30 days following) for each t.id").unwrap();
    match pq.target {
        TargetExpr::Aggregation(a) => {
            assert_eq!(a.func, AggFunc::Sum);
            assert_eq!(a.window.unwrap().unit, TimeUnit::Days);
        }
        other => panic!("{:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Schema-bound validation
// ---------------------------------------------------------------------------

#[test]
fn validate_ok_and_types() {
    let schema = common::churn_schema();
    let vq = validate(&parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id").unwrap(), &schema).unwrap();
    assert_eq!(vq.task_type, TaskType::BinaryClassification);
    let vq = validate(&parse("PREDICT customers.age FOR EACH customers.customer_id").unwrap(), &schema).unwrap();
    assert_eq!(vq.task_type, TaskType::Regression);
    let vq = validate(&parse("PREDICT products.name FOR EACH products.product_id").unwrap(), &schema).unwrap();
    assert_eq!(vq.task_type, TaskType::MulticlassClassification);
}

#[test]
fn validate_rejects_unknowns_and_bad_windows() {
    let schema = common::churn_schema();
    assert!(validate(&parse("PREDICT COUNT(nope.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id").unwrap(), &schema).is_err());
    assert!(validate(&parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.oops").unwrap(), &schema).is_err());
    assert!(validate(&parse("PREDICT orders.nope FOR EACH orders.order_id").unwrap(), &schema).is_err());
    // past-facing target window (PRECEDING => start < 0)
    assert!(validate(&parse("PREDICT COUNT(orders.*) OVER (30 DAYS PRECEDING) FOR EACH customers.customer_id").unwrap(), &schema).is_err());
    // window on a static table (products has no time_column)
    assert!(validate(&parse("PREDICT SUM(products.price) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id").unwrap(), &schema).is_err());
}

#[test]
fn validate_rejects_horizons_outside_target() {
    let schema = common::churn_schema();
    // HORIZONS > 1 on the target is fine (forecasting) ...
    assert!(validate(&parse("PREDICT SUM(orders.qty) OVER (7 DAYS FOLLOWING HORIZONS 4) FOR EACH customers.customer_id").unwrap(), &schema).is_ok());
    // ... but not in a WHERE population filter.
    assert!(validate(&parse("PREDICT SUM(orders.qty) OVER (7 DAYS FOLLOWING) FOR EACH customers.customer_id WHERE COUNT(orders.*) OVER (7 DAYS PRECEDING HORIZONS 2) > 0").unwrap(), &schema).is_err());
}

#[test]
fn fk_column_is_legal_list_distinct_target() {
    // recommendation pattern: LIST_DISTINCT over a foreign key column.
    let schema = Schema::new_schema()
        .table(TableDef::new_table("users").column("age", ValueType::Number).primary_key("user_id").build())
        .table(
            TableDef::new_table("txns")
                .column("price", ValueType::Number)
                .column("ts", ValueType::Datetime)
                .primary_key("txn_id")
                .time_column("ts")
                .build(),
        )
        .table(TableDef::new_table("articles").column("name", ValueType::Text).primary_key("article_id").build())
        .link(LinkDef::link("txns", "user_id", "users"))
        .link(LinkDef::link("txns", "article_id", "articles"))
        .build();
    let vq = validate(
        &parse("PREDICT LIST_DISTINCT(txns.article_id) OVER (30 DAYS FOLLOWING) RANK TOP 12 FOR EACH users.user_id").unwrap(),
        &schema,
    )
    .unwrap();
    assert_eq!(vq.task_type, TaskType::MultilabelRanking);
}

#[test]
fn f17_invariant_rejects_pk_as_feature_column() {
    let bad = Schema::new_schema()
        .table(
            TableDef::new_table("users")
                .column("user_id", ValueType::Number) // PK also a feature column -> illegal
                .primary_key("user_id")
                .build(),
        )
        .try_build();
    assert!(bad.is_err());
}
