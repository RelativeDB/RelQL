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

use common::{churn_schema, churn_rows, churn_wiring, dt, in_memory_wiring, StubBackend};

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
        .model_backend(Box::new(StubBackend))
        .sampler_mode(SamplerMode::Csc)
        .build()
        .unwrap();
    let res = eng
        .execute(ExecutionInput::query("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id").anchor_time(t0()))
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
        ("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id", TaskType::Regression, "hf://stanford-star/rt-j/regression"),
        ("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id", TaskType::BinaryClassification, "hf://stanford-star/rt-j/classification"),
        ("PREDICT SUM(orders.qty) OVER (7 DAYS FOLLOWING HORIZONS 4) FOR EACH customers.customer_id", TaskType::Forecasting, "hf://stanford-star/rt-j/regression"),
        ("PREDICT LIST_DISTINCT(orders.qty) OVER (30 DAYS FOLLOWING) RANK TOP 5 FOR EACH customers.customer_id", TaskType::MultilabelRanking, "hf://stanford-star/rt-j/classification"),
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
    let mut eng = Engine::new(churn_schema(), wiring).model_backend(Box::new(StubBackend));
    assert!(eng
        .execute(ExecutionInput::query("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id").anchor_time(t0()))
        .is_err());
    // but a pinned id works
    let res = eng
        .execute(ExecutionInput::query("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR customers.customer_id = 'C7'").anchor_time(t0()))
        .unwrap();
    assert_eq!(res.predictions.len(), 1);
}

// ---------------------------------------------------------------------------
// RETURN clause: output shaping + validation
// ---------------------------------------------------------------------------

fn run_c7(pql: &str) -> EntityPrediction {
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(StubBackend));
    let res = eng.execute(ExecutionInput::query(pql).anchor_time(t0())).unwrap();
    assert_eq!(res.predictions.len(), 1);
    res.predictions.into_iter().next().unwrap()
}

fn run_c7_err(pql: &str) -> relativedb::Error {
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(StubBackend));
    eng.execute(ExecutionInput::query(pql).anchor_time(t0())).unwrap_err()
}

#[test]
fn return_class_sets_predicted_class() {
    let p = run_c7(
        "PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR customers.customer_id = 'C7' RETURN CLASS",
    );
    let cls = p.predicted_class.expect("predicted_class set");
    assert!(cls == "true" || cls == "false", "hard label, got {:?}", cls);
}

#[test]
fn return_distribution_two_entries_sum_one() {
    let p = run_c7(
        "PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR customers.customer_id = 'C7' RETURN DISTRIBUTION",
    );
    assert_eq!(p.class_probs.len(), 2);
    let keys: std::collections::HashSet<&str> =
        p.class_probs.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["true", "false"].into_iter().collect());
    let sum: f64 = p.class_probs.iter().map(|(_, v)| v).sum();
    assert!((sum - 1.0).abs() < 1e-9, "class_probs sum to 1, got {}", sum);
}

#[test]
fn return_quantiles_execution_unsupported() {
    // A single-head point model exposes no quantile distribution: execution errors.
    let err = run_c7_err(
        "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR customers.customer_id = 'C7' RETURN QUANTILES (0.1, 0.5, 0.9)",
    );
    let msg = format!("{}", err).to_lowercase();
    assert!(msg.contains("quantiles"), "expected unsupported error, got: {}", err);
}

#[test]
fn return_interval_execution_unsupported() {
    let err = run_c7_err(
        "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR customers.customer_id = 'C7' RETURN INTERVAL 80%",
    );
    let msg = format!("{}", err).to_lowercase();
    assert!(msg.contains("interval"), "expected unsupported error, got: {}", err);
}

#[test]
fn return_default_unchanged() {
    // No RETURN: binary target still reports probability, nothing else.
    let p = run_c7(
        "PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR customers.customer_id = 'C7'",
    );
    assert!(p.probability.is_some());
    assert!(p.predicted_class.is_none());
    assert!(p.class_probs.is_empty());
    assert!(p.quantiles.is_empty());
    assert!(p.interval.is_none());
}

#[test]
fn return_quantiles_on_boolean_target_rejected() {
    let pq = relativedb::parse(
        "PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR customers.customer_id = 'C7' RETURN QUANTILES (0.5)",
    )
    .unwrap();
    assert!(relativedb::validate(&pq, &churn_schema()).is_err());
}

#[test]
fn return_probability_on_regression_target_rejected() {
    let pq = relativedb::parse(
        "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR customers.customer_id = 'C7' RETURN PROBABILITY",
    )
    .unwrap();
    assert!(relativedb::validate(&pq, &churn_schema()).is_err());
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

// ---------------------------------------------------------------------------
// AS OF anchor binding + EXPLAIN (contract EXPLAIN_ASOF_CONTRACT.md)
// ---------------------------------------------------------------------------

const CHURN: &str =
    "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id";

/// Backend that records whether the model was ever asked to score.
struct SpyBackend {
    calls: Arc<Mutex<usize>>,
}
impl ModelBackend for SpyBackend {
    fn score(
        &mut self,
        _query: &ParsedQuery,
        _task_type: TaskType,
        contexts: &[relativedb::EntityContext],
        _model_uri: &str,
        _config: &ModelConfig,
    ) -> Result<Vec<EntityPrediction>, relativedb::Error> {
        *self.calls.lock().unwrap() += 1;
        Ok(contexts.iter().map(|c| EntityPrediction::new(c.entity_id.clone())).collect())
    }
}

fn pred_ids(res: &relativedb::PredictionResult) -> std::collections::HashSet<String> {
    res.predictions.iter().map(|p| p.id.to_string()).collect()
}

fn all_three() -> std::collections::HashSet<String> {
    ["C1", "C7", "C9"].iter().map(|s| s.to_string()).collect()
}

#[test]
fn as_of_date_overrides_anchor_time() {
    // t0 hides O4 (2026-07-05); AS OF 2026-08-01 admits it -> strictly more rows.
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(StubBackend));
    let later = eng
        .explain(ExecutionInput::query(format!("EXPLAIN CONTEXT {} AS OF 2026-08-01", CHURN)).anchor_time(t0()))
        .unwrap();
    let base = eng
        .explain(ExecutionInput::query(format!("EXPLAIN CONTEXT {}", CHURN)).anchor_time(t0()))
        .unwrap();
    let lc = later.context.as_ref().unwrap();
    let bc = base.context.as_ref().unwrap();
    assert!(lc.total_rows > bc.total_rows, "date anchor should admit O4/P3");
    assert_eq!(lc.anchor.unwrap(), dt("2026-08-01"));
    // execution under the date anchor still covers every entity
    let res = eng
        .execute(ExecutionInput::query(format!("{} AS OF 2026-08-01", CHURN)).anchor_time(t0()))
        .unwrap();
    assert_eq!(pred_ids(&res), all_three());
}

#[test]
fn as_of_param_binds_from_params() {
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(StubBackend));
    let res = eng
        .execute(ExecutionInput::query(format!("{} AS OF :t", CHURN)).param("t", dt("2026-08-01")))
        .unwrap();
    assert_eq!(pred_ids(&res), all_three());
}

#[test]
fn as_of_param_missing_raises() {
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(StubBackend));
    let err = eng
        .execute(ExecutionInput::query(format!("{} AS OF :t", CHURN)))
        .unwrap_err();
    assert!(format!("{}", err).contains("t"), "error should name the param: {}", err);
}

#[test]
fn as_of_param_falls_back_to_anchor_time() {
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(StubBackend));
    let res = eng
        .execute(ExecutionInput::query(format!("{} AS OF :t", CHURN)).anchor_time(t0()))
        .unwrap();
    let base = eng.execute(ExecutionInput::query(CHURN).anchor_time(t0())).unwrap();
    let rp: Vec<(String, Option<f64>)> =
        res.predictions.iter().map(|p| (p.id.to_string(), p.probability)).collect();
    let bp: Vec<(String, Option<f64>)> =
        base.predictions.iter().map(|p| (p.id.to_string(), p.probability)).collect();
    assert_eq!(rp, bp);
}

#[test]
fn as_of_now_equals_no_as_of() {
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(StubBackend));
    let now = eng.execute(ExecutionInput::query(format!("{} AS OF NOW", CHURN)).anchor_time(t0())).unwrap();
    let base = eng.execute(ExecutionInput::query(CHURN).anchor_time(t0())).unwrap();
    let np: Vec<Option<f64>> = now.predictions.iter().map(|p| p.probability).collect();
    let bp: Vec<Option<f64>> = base.predictions.iter().map(|p| p.probability).collect();
    assert_eq!(np, bp);
}

#[test]
fn explain_plan_does_not_invoke_model() {
    let calls = Arc::new(Mutex::new(0usize));
    let spy = SpyBackend { calls: Arc::clone(&calls) };
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(spy));
    let res = eng
        .explain(ExecutionInput::query(format!("EXPLAIN PLAN {}", CHURN)).anchor_time(t0()))
        .unwrap();
    assert_eq!(res.mode, "PLAN");
    assert!(res.context.is_none());
    assert!(res.predictions.is_none());
    assert_eq!(*calls.lock().unwrap(), 0); // model never touched
}

#[test]
fn explain_plan_fields() {
    let mut eng = Engine::new(churn_schema(), churn_wiring());
    let res = eng
        .explain(ExecutionInput::query(format!("EXPLAIN PLAN {}", CHURN)).anchor_time(t0()))
        .unwrap();
    let plan = &res.plan;
    assert_eq!(plan.task_type, TaskType::BinaryClassification);
    assert_eq!(plan.output, "probability");
    assert_eq!(plan.entity.table, "customers");
    assert_eq!(plan.entity.pk, "customer_id");
    assert_eq!(plan.entity.selector, "FOR EACH");
    assert_eq!(plan.as_of.source, "execution-anchor");
    assert!(plan.target.contains("COUNT(orders.*)"));
    let tgt: Vec<&relativedb::WindowInfo> =
        plan.windows.iter().filter(|w| w.role == "target").collect();
    assert_eq!(tgt.len(), 1);
    assert_eq!(tgt[0].start, 0.0);
    assert_eq!(tgt[0].end, 90.0);
    assert_eq!(relativedb::TimeUnit::from_keyword("DAYS"), Some(tgt[0].unit));
}

#[test]
fn bare_explain_defaults_to_plan() {
    let calls = Arc::new(Mutex::new(0usize));
    let spy = SpyBackend { calls: Arc::clone(&calls) };
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(spy));
    let res = eng
        .explain(ExecutionInput::query(format!("EXPLAIN {}", CHURN)).anchor_time(t0()))
        .unwrap();
    assert_eq!(res.mode, "PLAN");
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[test]
fn explain_on_non_explain_query_defaults_plan() {
    let mut eng = Engine::new(churn_schema(), churn_wiring());
    let res = eng.explain(ExecutionInput::query(CHURN).anchor_time(t0())).unwrap();
    assert_eq!(res.mode, "PLAN");
    assert!(res.predictions.is_none());
}

#[test]
fn execute_on_explain_query_errors() {
    let mut eng = Engine::new(churn_schema(), churn_wiring());
    let err = eng
        .execute(ExecutionInput::query(format!("EXPLAIN PLAN {}", CHURN)).anchor_time(t0()))
        .unwrap_err();
    assert!(format!("{}", err).to_lowercase().contains("explain"), "got: {}", err);
}

#[test]
fn explain_context_populates_counts_no_predictions() {
    let calls = Arc::new(Mutex::new(0usize));
    let spy = SpyBackend { calls: Arc::clone(&calls) };
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(spy));
    let res = eng
        .explain(ExecutionInput::query(format!("EXPLAIN CONTEXT {}", CHURN)).anchor_time(t0()))
        .unwrap();
    assert_eq!(res.mode, "CONTEXT");
    assert!(res.predictions.is_none());
    assert_eq!(*calls.lock().unwrap(), 0);
    let ctx = res.context.as_ref().unwrap();
    assert_eq!(ctx.entities_covered, 3);
    assert!(ctx.total_rows > 0);
    assert!(ctx.total_cells > 0);
    assert!(ctx.per_table.contains_key("customers"));
}

#[test]
fn explain_analyze_has_predictions() {
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(StubBackend));
    let res = eng
        .explain(ExecutionInput::query(format!("EXPLAIN ANALYZE {}", CHURN)).anchor_time(t0()))
        .unwrap();
    assert_eq!(res.mode, "ANALYZE");
    assert!(res.context.is_some());
    let preds = res.predictions.as_ref().unwrap();
    assert_eq!(pred_ids(preds), all_three());
}

#[test]
fn explain_ablation_warns_not_implemented() {
    let mut eng = Engine::new(churn_schema(), churn_wiring());
    let res = eng
        .explain(
            ExecutionInput::query(format!("EXPLAIN ABLATION {} ABLATE TABLE products", CHURN))
                .anchor_time(t0()),
        )
        .unwrap();
    assert_eq!(res.mode, "ABLATION");
    assert!(res.plan.warnings.iter().any(|w| w.contains("ablation not implemented")));
    assert!(res.plan.ablations.iter().any(|a| a.name == "products"));
}

#[test]
fn render_json_parses() {
    let mut eng = Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(StubBackend));
    let res = eng
        .explain(
            ExecutionInput::query(format!("EXPLAIN ANALYZE FORMAT JSON {}", CHURN))
                .anchor_time(t0()),
        )
        .unwrap();
    let json = res.render();
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(v["mode"], "ANALYZE");
    assert_eq!(v["plan"]["task_type"], "binary_classification");
    assert_eq!(v["predictions"].as_array().unwrap().len(), 3);
    assert_eq!(v["plan"]["as_of"]["source"], "execution-anchor");
}

#[test]
fn render_text_contains_target_and_task() {
    let mut eng = Engine::new(churn_schema(), churn_wiring());
    let res = eng
        .explain(ExecutionInput::query(format!("EXPLAIN PLAN {}", CHURN)).anchor_time(t0()))
        .unwrap();
    let text = res.render();
    assert!(text.contains("binary_classification"));
    assert!(text.contains("COUNT(orders.*)"));
    assert!(text.contains("PLAN"));
}
