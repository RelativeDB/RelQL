//! Grammar conformance: the full 44-query corpus parses; malformed input is
//! rejected; AST spot checks; task-type inference; schema validation.
//! Mirrors the Python `test_pql_parser.py`.

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
fn corpus_has_44_queries() {
    assert_eq!(corpus().len(), 44);
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
    "PREDICT SUM(t.c, 0, 30)",                              // missing FOR
    "PREDICT SUM(t.c, 0, 30) FOR EACH t",                   // not table.column
    "PREDICT SUM(c, 0, 30) FOR EACH t.id",                  // agg operand not table.column
    "PREDICT SUM(t.c 0, 30) FOR EACH t.id",                 // missing comma
    "PREDICT SUM(t.c, 0) FOR EACH t.id",                    // window needs 2 bounds
    "PREDICT SUM(t.c, 0, 30, fortnights) FOR EACH t.id",    // bad time unit
    "PREDICT SUM(t.c, 0, 30) FOR EACH t.id trailing junk",
    "PREDICT SUM(t.c, 0, 30) FOR EACH t.id WHERE",
    "PREDICT COUNT(t.*, 0, 30) = FOR EACH t.id",            // comparison w/o literal
    "PREDICT t.c IN (1, 2 FOR EACH t.id",                   // unclosed list
    "PREDICT LIST_DISTINCT(t.c, 0, 30) RANK TOP FOR EACH t.id", // RANK TOP w/o K
    "PREDICT SUM(t.c, 0, 30) FORECAST TIMEFRAMES FOR EACH t.id",
    "PREDICT t.c IS FOR EACH t.id",
    "PREDICT (SUM(t.c, 0, 30) FOR EACH t.id",               // unbalanced paren
    "PREDICT SUM(t.c, 0, 30) FOR EACH t.id ASSUMING AND",
];

#[test]
fn malformed_queries_rejected() {
    for q in MALFORMED {
        assert!(parse(q).is_err(), "should reject: {:?}", q);
    }
}

#[test]
fn churn_query_ast() {
    let pq = parse("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id").unwrap();
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
    let pq = parse("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR users.user_id IN (42, 123)").unwrap();
    assert_eq!(pq.entity_ids, vec![Literal::Num(42.0), Literal::Num(123.0)]);
    let pq = parse(
        "PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR users.user_id = 42 ASSUMING users.plan = 'premium'",
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
    let pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID, 0, 30) RANK TOP 12 FOR EACH CUSTOMERS.CUSTOMER_ID").unwrap();
    assert_eq!(pq.rank, Some(RankKind::Rank));
    assert_eq!(pq.top_k, Some(12));
    assert_eq!(pq.task_type(None), TaskType::MultilabelRanking);
    let pq = parse("PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID, 0, 30) CLASSIFY FOR EACH CUSTOMERS.CUSTOMER_ID").unwrap();
    assert_eq!(pq.rank, Some(RankKind::Classify));
    assert_eq!(pq.task_type(None), TaskType::MulticlassClassification);
}

#[test]
fn forecast_and_soft_keyword_count() {
    let pq = parse("PREDICT SUM(usage.count, 0, 1, days) FORECAST 28 TIMEFRAMES FOR EACH accounts.account_id").unwrap();
    assert_eq!(pq.num_forecasts, Some(28));
    assert_eq!(pq.task_type(None), TaskType::Forecasting);
    match &pq.target {
        TargetExpr::Aggregation(a) => assert_eq!(a.column, ColumnRef::new("usage", "count")),
        other => panic!("{:?}", other),
    }
}

#[test]
fn inf_bound_and_not() {
    let pq = parse("PREDICT COUNT(transactions.*, -INF, 0) > 0 FOR EACH user.user_id").unwrap();
    match &pq.target {
        TargetExpr::Condition(c) => match c.left.as_ref() {
            TargetExpr::Aggregation(a) => {
                let w = a.window.unwrap();
                assert!(w.start.is_infinite() && w.start < 0.0);
            }
            other => panic!("{:?}", other),
        },
        other => panic!("{:?}", other),
    }
    let pq = parse("PREDICT NOT LAST(LOAN.AMOUNT, 0, 30) > 30 FOR EACH LOAN.id").unwrap();
    assert!(matches!(pq.target, TargetExpr::Not(_)));
}

#[test]
fn word_operators_and_membership() {
    let pq = parse("PREDICT LAST(LOAN.STATUS, 0, 30) NOT LIKE '%DENIED' FOR EACH LOAN.id").unwrap();
    match pq.target {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::NotLike),
        other => panic!("{:?}", other),
    }
    let pq = parse("PREDICT LOAN.STATUS IS IN ('A', 'C') FOR EACH LOAN.id").unwrap();
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
    // NOT IN and NOT CONTAINS
    let pq = parse("PREDICT SUM(transactions.value, 15, 45, days) > 100 FOR EACH customers.customer_id WHERE customers.location NOT IN ('ALASKA', 'HAWAII')").unwrap();
    match pq.where_.unwrap() {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::NotIn),
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
    let pq = parse("PREDICT SUM(TRANSACTIONS.PRICE, 0, 30) FOR EACH CUSTOMERS.CUSTOMER_ID WHERE (user.country = 'US' OR region.num_inhabitants < 10000) AND user.dietary = 'Vegetarian'").unwrap();
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
    let pq = parse("PREDICT LAST(payments.amount, 0, 90, days) == 0 FOR EACH order.order_id WHERE order.location == 'US'").unwrap();
    match pq.target {
        TargetExpr::Condition(c) => assert_eq!(c.op, Operator::Eq),
        other => panic!("{:?}", other),
    }
}

#[test]
fn case_insensitive_keywords() {
    let pq = parse("predict sum(t.c, 0, 30) for each t.id").unwrap();
    match pq.target {
        TargetExpr::Aggregation(a) => assert_eq!(a.func, AggFunc::Sum),
        other => panic!("{:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Schema-bound validation
// ---------------------------------------------------------------------------

#[test]
fn validate_ok_and_types() {
    let schema = common::churn_schema();
    let vq = validate(&parse("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id").unwrap(), &schema).unwrap();
    assert_eq!(vq.task_type, TaskType::BinaryClassification);
    let vq = validate(&parse("PREDICT customers.age FOR EACH customers.customer_id").unwrap(), &schema).unwrap();
    assert_eq!(vq.task_type, TaskType::Regression);
    let vq = validate(&parse("PREDICT products.name FOR EACH products.product_id").unwrap(), &schema).unwrap();
    assert_eq!(vq.task_type, TaskType::MulticlassClassification);
}

#[test]
fn validate_rejects_unknowns_and_bad_windows() {
    let schema = common::churn_schema();
    assert!(validate(&parse("PREDICT COUNT(nope.*, 0, 90) = 0 FOR EACH customers.customer_id").unwrap(), &schema).is_err());
    assert!(validate(&parse("PREDICT COUNT(orders.*, 0, 90) = 0 FOR EACH customers.oops").unwrap(), &schema).is_err());
    assert!(validate(&parse("PREDICT orders.nope FOR EACH orders.order_id").unwrap(), &schema).is_err());
    // past-facing target window
    assert!(validate(&parse("PREDICT COUNT(orders.*, -30, 0) FOR EACH customers.customer_id").unwrap(), &schema).is_err());
    // window on a static table (products has no time_column)
    assert!(validate(&parse("PREDICT SUM(products.price, 0, 30) FOR EACH customers.customer_id").unwrap(), &schema).is_err());
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
        &parse("PREDICT LIST_DISTINCT(txns.article_id, 0, 30) RANK TOP 12 FOR EACH users.user_id").unwrap(),
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
