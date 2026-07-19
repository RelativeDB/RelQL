//! Shared test fixtures: the worked churn example as a toy graph
//! (mirrors the Python `conftest.py`).

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, NaiveDate, Utc};

use relativedb::engine::ModelBackend;
use relativedb::{
    EntityContext, EntityId, EntityPrediction, Error, LinkDef, ModelConfig, ParsedQuery,
    RetrieverWiring, Row, Schema, TableDef, TaskType, TemporalBound, ValueType,
};

/// A tiny deterministic test double implementing [`ModelBackend`]. It ships no
/// real model — it just emits a fixed scalar per task type so the engine's own
/// plumbing (context assembly, temporal correctness, routing, RETURN shaping,
/// EXPLAIN, AS OF) can be exercised offline. Real-model behaviour is covered
/// separately by the native/golden tests.
pub struct StubBackend;

impl ModelBackend for StubBackend {
    fn score(
        &mut self,
        query: &ParsedQuery,
        task_type: TaskType,
        contexts: &[EntityContext],
        _model_uri: &str,
        _config: &ModelConfig,
        _aux: &relativedb::engine::ScoringAux,
    ) -> Result<Vec<EntityPrediction>, Error> {
        let n = query.num_forecasts.unwrap_or(1).max(1) as usize;
        Ok(contexts
            .iter()
            .map(|c| {
                let mut p = EntityPrediction::new(c.entity_id.clone());
                match task_type {
                    TaskType::BinaryClassification => p.probability = Some(0.5),
                    TaskType::Forecasting => {
                        p.value = Some(1.0);
                        p.forecast = vec![1.0; n];
                    }
                    _ => p.value = Some(1.0),
                }
                p
            })
            .collect())
    }
}

pub fn dt(s: &str) -> DateTime<Utc> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
}

pub fn churn_schema() -> Schema {
    Schema::new_schema()
        .table(
            TableDef::new_table("customers")
                .column("age", ValueType::Number)
                .column("signup_date", ValueType::Datetime)
                .primary_key("customer_id")
                .build(),
        )
        .table(
            TableDef::new_table("products")
                .column("price", ValueType::Number)
                .column("name", ValueType::Text)
                .primary_key("product_id")
                .build(),
        )
        .table(
            TableDef::new_table("orders")
                .column("qty", ValueType::Number)
                .column("order_date", ValueType::Datetime)
                .primary_key("order_id")
                .time_column("order_date")
                .build(),
        )
        .link(LinkDef::link("orders", "customer_id", "customers"))
        .link(LinkDef::link("orders", "product_id", "products"))
        .build()
}

/// The kb/example.md database. O4 (2026-07-05) is AFTER the anchor
/// t0 = 2026-07-01 and must never enter context.
pub fn churn_rows() -> HashMap<String, Vec<Row>> {
    let customers = vec![
        Row::new("customers", "C1").cell("age", 34.0).cell("signup_date", dt("2026-02-10")),
        Row::new("customers", "C7").cell("age", 52.0).cell("signup_date", dt("2026-01-20")),
        Row::new("customers", "C9").cell("age", 27.0).cell("signup_date", dt("2026-03-05")),
    ];
    let products = vec![
        Row::new("products", "P1").cell("price", 25.0).cell("name", "running shoes"),
        Row::new("products", "P2").cell("price", 90.0).cell("name", "espresso machine"),
        Row::new("products", "P3").cell("price", 35.0).cell("name", "yoga mat"),
    ];
    let orders = vec![
        Row::new("orders", "O1")
            .cell("qty", 1.0)
            .cell("order_date", dt("2026-03-10"))
            .timestamp(dt("2026-03-10"))
            .parent("customer_id", "C7")
            .parent("product_id", "P2"),
        Row::new("orders", "O2")
            .cell("qty", 2.0)
            .cell("order_date", dt("2026-05-02"))
            .timestamp(dt("2026-05-02"))
            .parent("customer_id", "C7")
            .parent("product_id", "P1"),
        Row::new("orders", "O3")
            .cell("qty", 1.0)
            .cell("order_date", dt("2026-06-20"))
            .timestamp(dt("2026-06-20"))
            .parent("customer_id", "C1")
            .parent("product_id", "P3"),
        Row::new("orders", "O4")
            .cell("qty", 1.0)
            .cell("order_date", dt("2026-07-05"))
            .timestamp(dt("2026-07-05")) // future of t0!
            .parent("customer_id", "C7")
            .parent("product_id", "P3"),
    ];
    let mut m = HashMap::new();
    m.insert("customers".to_string(), customers);
    m.insert("products".to_string(), products);
    m.insert("orders".to_string(), orders);
    m
}

fn newest_first(kids: &mut Vec<Row>) {
    kids.sort_by(|a, b| {
        let ka = (a.timestamp.is_none(), -(a.timestamp.map(|t| t.timestamp() as f64).unwrap_or(0.0)));
        let kb = (b.timestamp.is_none(), -(b.timestamp.map(|t| t.timestamp() as f64).unwrap_or(0.0)));
        ka.0.cmp(&kb.0).then(ka.1.partial_cmp(&kb.1).unwrap())
    });
}

/// Well-behaved (or, with `honor_bound = false`, deliberately leaky) retrievers
/// + scanners over an in-memory row map.
pub fn in_memory_wiring(rows: HashMap<String, Vec<Row>>, honor_bound: bool) -> RetrieverWiring {
    let rows = Arc::new(rows);
    let tables: Vec<String> = rows.keys().cloned().collect();
    let mut w = RetrieverWiring::new_wiring();

    let r_links = Arc::clone(&rows);
    w = w.default_links(move |link: &LinkDef, pid: &EntityId, bound: &TemporalBound, limit: usize| {
        let mut kids: Vec<Row> = r_links
            .get(&link.from_table)
            .map(|rs| {
                rs.iter()
                    .filter(|r| r.get_parent(&link.fk_column) == Some(pid))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        if honor_bound {
            kids.retain(|r| bound.admits_row(r));
        }
        newest_first(&mut kids);
        if honor_bound {
            kids.truncate(limit);
        }
        kids
    });

    for t in tables {
        let r_ent = Arc::clone(&rows);
        w = w.entities(t.clone(), move |table: &str, ids: &[EntityId], bound: &TemporalBound| {
            let mut out = Vec::new();
            if let Some(rs) = r_ent.get(table) {
                for id in ids {
                    if let Some(r) = rs.iter().find(|r| &r.id == id) {
                        if honor_bound && !bound.admits_row(r) {
                            continue;
                        }
                        out.push(r.clone());
                    }
                }
            }
            out
        });
        let r_sc = Arc::clone(&rows);
        w = w.scanner(t.clone(), move |table: &str, bound: &TemporalBound| {
            r_sc.get(table)
                .map(|rs| {
                    rs.iter()
                        .filter(|r| !honor_bound || bound.admits_row(r))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default()
        });
    }
    w.build()
}

pub fn churn_wiring() -> RetrieverWiring {
    in_memory_wiring(churn_rows(), true)
}
