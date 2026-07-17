//! Engine tests: temporal-leakage guard, fanout caps, CSC equivalence, model
//! routing. Mirrors the Python `test_engine.py`.

mod common;

use std::sync::{Arc, Mutex};

use chrono::{DateTime, Duration, Utc};

use relativedb::csc::CscIndex;
use relativedb::engine::ModelBackend;
use relativedb::{
    ContextPolicy, Engine, EntityId, EntityPrediction, ExecutionInput, LinkDef, ModelConfig,
    ParsedQuery, SamplerMode, TaskType, TemporalBound,
};

use common::{churn_schema, churn_rows, churn_wiring, dt, in_memory_wiring};

fn t0() -> DateTime<Utc> {
    dt("2026-07-01")
}

fn key(t: &str, id: &str) -> (String, EntityId) {
    (t.to_string(), EntityId::from(id))
}

// ---------------------------------------------------------------------------
// Temporal leakage
// ---------------------------------------------------------------------------

#[test]
fn future_row_never_enters_context() {
    let eng = Engine::new(churn_schema(), churn_wiring());
    let ctx = eng.assemble_context("customers", &EntityId::from("C7"), Some(t0())).unwrap();
    let keys = ctx.row_keys();
    assert!(keys.contains(&key("orders", "O1")));
    assert!(keys.contains(&key("orders", "O2")));
    assert!(!keys.contains(&key("orders", "O4"))); // 2026-07-05 > t0
    assert!(!keys.contains(&key("products", "P3"))); // only reachable via O4
}

#[test]
fn leaky_retriever_is_caught_by_engine() {
    let leaky = in_memory_wiring(churn_rows(), false);
    let eng = Engine::new(churn_schema(), leaky);
    let ctx = eng.assemble_context("customers", &EntityId::from("C7"), Some(t0())).unwrap();
    assert!(!ctx.row_keys().contains(&key("orders", "O4")));
    for r in &ctx.rows {
        assert!(r.timestamp.is_none() || r.timestamp.unwrap() <= ctx.anchor.unwrap());
    }
}

#[test]
fn later_anchor_admits_the_row() {
    let eng = Engine::new(churn_schema(), churn_wiring());
    let ctx = eng.assemble_context("customers", &EntityId::from("C7"), Some(dt("2026-08-01"))).unwrap();
    let keys = ctx.row_keys();
    assert!(keys.contains(&key("orders", "O4")));
    assert!(keys.contains(&key("products", "P3")));
}

#[test]
fn temporal_bound_semantics() {
    let b = TemporalBound::at_or_before(t0());
    assert!(b.admits(Some(t0()))); // inclusive
    assert!(b.admits(Some(t0() - Duration::seconds(1))));
    assert!(!b.admits(Some(t0() + Duration::seconds(1))));
    assert!(b.admits(None)); // static rows always admitted
    assert!(TemporalBound::unbounded().admits(Some(t0() + Duration::days(999))));
}

// ---------------------------------------------------------------------------
// Fanouts / hop-loop shape
// ---------------------------------------------------------------------------

#[test]
fn fanout_caps_children_newest_first() {
    let policy = ContextPolicy { fanouts: Some(vec![1, 0]), max_hops: 2, ..Default::default() };
    let eng = Engine::new(churn_schema(), churn_wiring()).context_policy(policy);
    let ctx = eng.assemble_context("customers", &EntityId::from("C7"), Some(t0())).unwrap();
    let order_ids: Vec<String> = ctx.rows.iter().filter(|r| r.table == "orders").map(|r| r.id.to_string()).collect();
    assert_eq!(order_ids, vec!["O2".to_string()]); // newest admitted child only
}

#[test]
fn parents_always_followed() {
    let eng = Engine::new(churn_schema(), churn_wiring());
    let ctx = eng.assemble_context("customers", &EntityId::from("C7"), Some(t0())).unwrap();
    let keys = ctx.row_keys();
    assert!(keys.contains(&key("products", "P1")));
    assert!(keys.contains(&key("products", "P2")));
}

#[test]
fn max_context_cells_budget() {
    let policy = ContextPolicy { max_context_cells: 3, ..Default::default() };
    let eng = Engine::new(churn_schema(), churn_wiring()).context_policy(policy);
    let ctx = eng.assemble_context("customers", &EntityId::from("C7"), Some(t0())).unwrap();
    assert_eq!(ctx.rows[0].key(), key("customers", "C7")); // seed always present
    assert!(ctx.rows.len() < 5); // budget stopped expansion
}

// ---------------------------------------------------------------------------
// CSC mode
// ---------------------------------------------------------------------------

#[test]
fn csc_arrays_time_sorted() {
    let idx = CscIndex::build(&churn_schema(), &churn_wiring()).unwrap();
    let link = LinkDef::link("orders", "customer_id", "customers");
    let adj = &idx.adjacency[&link];
    assert_eq!(*adj.colptr.last().unwrap() as usize, adj.row.len());
    assert_eq!(adj.row.len(), 4);
    for p in 0..adj.colptr.len() - 1 {
        let s = adj.colptr[p] as usize;
        let e = adj.colptr[p + 1] as usize;
        for w in s + 1..e {
            assert!(adj.ts[w] >= adj.ts[w - 1]);
        }
    }
    let c7 = idx.dense["customers"][&EntityId::from("C7")];
    assert_eq!(adj.colptr[c7 + 1] - adj.colptr[c7], 3);
}

#[test]
fn csc_children_bound_and_limit() {
    let idx = CscIndex::build(&churn_schema(), &churn_wiring()).unwrap();
    let link = LinkDef::link("orders", "customer_id", "customers");
    let ids = |kids: Vec<relativedb::Row>| kids.into_iter().map(|k| k.id.to_string()).collect::<Vec<_>>();
    assert_eq!(ids(idx.children(&link, &EntityId::from("C7"), &TemporalBound::at_or_before(t0()), 10)), vec!["O2", "O1"]);
    assert_eq!(ids(idx.children(&link, &EntityId::from("C7"), &TemporalBound::at_or_before(t0()), 1)), vec!["O2"]);
    assert_eq!(ids(idx.children(&link, &EntityId::from("C7"), &TemporalBound::unbounded(), 10)), vec!["O4", "O2", "O1"]);
}

#[test]
fn csc_context_equals_retriever_context() {
    let policy = ContextPolicy { fanouts: Some(vec![8, 8]), max_hops: 2, ..Default::default() };
    let ret = Engine::new(churn_schema(), churn_wiring()).context_policy(policy.clone());
    let csc = Engine::new(churn_schema(), churn_wiring())
        .context_policy(policy)
        .sampler_mode(SamplerMode::Csc)
        .build()
        .unwrap();
    let anchors: [Option<DateTime<Utc>>; 4] =
        [Some(t0()), Some(dt("2026-04-01")), Some(dt("2026-08-01")), None];
    for eid in ["C1", "C7", "C9"] {
        for anchor in anchors {
            let a = ret.assemble_context("customers", &EntityId::from(eid), anchor).unwrap();
            let b = csc.assemble_context("customers", &EntityId::from(eid), anchor).unwrap();
            assert_eq!(a.row_keys(), b.row_keys(), "keys differ for {} {:?}", eid, anchor);
            let ka: Vec<_> = a.rows.iter().map(|r| r.key()).collect();
            let kb: Vec<_> = b.rows.iter().map(|r| r.key()).collect();
            assert_eq!(ka, kb, "order differs for {} {:?}", eid, anchor);
        }
    }
}

#[test]
fn csc_execute_end_to_end() {
    let mut eng = Engine::new(churn_schema(), churn_wiring())
        .sampler_mode(SamplerMode::Csc)
        .build()
        .unwrap();
    let res = eng
        .execute(ExecutionInput::query("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id").anchor_time(t0()))
        .unwrap();
    assert_eq!(res.task_type, TaskType::BinaryClassification);
    let ids: std::collections::HashSet<String> = res.predictions.iter().map(|p| p.id.to_string()).collect();
    assert_eq!(ids, ["C1", "C7", "C9"].iter().map(|s| s.to_string()).collect());
}

// ---------------------------------------------------------------------------
// Model routing
// ---------------------------------------------------------------------------

struct RecordingBackend {
    calls: Arc<Mutex<Vec<(TaskType, String)>>>,
}

impl ModelBackend for RecordingBackend {
    fn score(
        &mut self,
        _query: &ParsedQuery,
        task_type: TaskType,
        contexts: &[relativedb::EntityContext],
        model_uri: &str,
        _config: &ModelConfig,
    ) -> Result<Vec<EntityPrediction>, relativedb::Error> {
        self.calls.lock().unwrap().push((task_type, model_uri.to_string()));
        Ok(contexts.iter().map(|c| EntityPrediction::new(c.entity_id.clone())).collect())
    }
}

#[test]
fn engine_routes_model_uri_by_task_type() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let backend = RecordingBackend { calls: Arc::clone(&calls) };
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(backend));
    let cases = [
        ("PREDICT SUM(orders.qty, 0, 30) FOR EACH customers.customer_id", TaskType::Regression, "hf://stanford-star/rt-j/regression"),
        ("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id", TaskType::BinaryClassification, "hf://stanford-star/rt-j/classification"),
        ("PREDICT SUM(orders.qty, 0, 7, days) FORECAST 4 TIMEFRAMES FOR EACH customers.customer_id", TaskType::Forecasting, "hf://stanford-star/rt-j/regression"),
        ("PREDICT LIST_DISTINCT(orders.qty, 0, 30) RANK TOP 5 FOR EACH customers.customer_id", TaskType::MultilabelRanking, "hf://stanford-star/rt-j/classification"),
    ];
    for (pql, expect_task, expect_uri) in cases {
        let res = eng.execute(ExecutionInput::query(pql).anchor_time(t0())).unwrap();
        assert_eq!(res.task_type, expect_task);
        assert_eq!(res.model_uri, expect_uri);
    }
    let recorded: Vec<String> = calls.lock().unwrap().iter().map(|(_, u)| u.clone()).collect();
    let expected: Vec<String> = cases.iter().map(|(_, _, u)| u.to_string()).collect();
    assert_eq!(recorded, expected);
}

#[test]
fn for_each_without_scanner_raises() {
    // entities-only wiring (no scanners): FOR EACH cannot enumerate.
    let rows = churn_rows();
    let rows = Arc::new(rows);
    let r1 = Arc::clone(&rows);
    let r2 = Arc::clone(&rows);
    let wiring = relativedb::RetrieverWiring::new_wiring()
        .entities("customers", move |t: &str, ids: &[EntityId], _b: &TemporalBound| {
            r1.get(t).map(|rs| rs.iter().filter(|r| ids.contains(&r.id)).cloned().collect()).unwrap_or_default()
        })
        .entities("orders", move |t: &str, ids: &[EntityId], _b: &TemporalBound| {
            r2.get(t).map(|rs| rs.iter().filter(|r| ids.contains(&r.id)).cloned().collect()).unwrap_or_default()
        })
        .default_links(|_l: &LinkDef, _p: &EntityId, _b: &TemporalBound, _n: usize| Vec::new())
        .build();
    let mut eng = Engine::new(churn_schema(), wiring);
    assert!(eng
        .execute(ExecutionInput::query("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id").anchor_time(t0()))
        .is_err());
    // but a pinned id works
    let res = eng
        .execute(ExecutionInput::query("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR customers.customer_id = 'C7'").anchor_time(t0()))
        .unwrap();
    assert_eq!(res.predictions.len(), 1);
}

#[test]
fn model_config_defaults_and_routing() {
    let cfg = ModelConfig::defaults();
    assert_eq!(cfg.classification_model_uri, "hf://stanford-star/rt-j/classification");
    assert_eq!(cfg.regression_model_uri, "hf://stanford-star/rt-j/regression");
    assert_eq!(cfg.embedding_model, "all-MiniLM-L12-v2");
    assert_eq!(cfg.d_text(), 384);
    assert_eq!(cfg.model_uri_for(TaskType::Regression), cfg.regression_model_uri);
    assert_eq!(cfg.model_uri_for(TaskType::Forecasting), cfg.regression_model_uri);
    for t in [TaskType::BinaryClassification, TaskType::MulticlassClassification, TaskType::MultilabelRanking] {
        assert_eq!(cfg.model_uri_for(t), cfg.classification_model_uri);
    }
    // unified uri + embedding mismatch check
    let cfg = ModelConfig::defaults().with_model_uri("file:///models/unified");
    assert_eq!(cfg.model_uri_for(TaskType::Regression), "file:///models/unified");
    assert_eq!(cfg.model_uri_for(TaskType::BinaryClassification), "file:///models/unified");
    assert!(cfg.check_checkpoint_embedding("all-mpnet-base-v2").is_err());
    let relaxed = ModelConfig::defaults().allow_embedding_mismatch(true);
    assert!(relaxed.check_checkpoint_embedding("all-mpnet-base-v2").is_ok());
    assert!(cfg.check_checkpoint_embedding("all-MiniLM-L12-v2").is_ok());
}
