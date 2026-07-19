//! End-to-end MULTICLASS + RANKING execution through the native RT-J backend
//! (`RtNativeBackend`), mirroring `CONTRACT.md` §2/§3. Gated on the dylib +
//! classification checkpoint being present (via `RELATIVEDB_RT_LIB` or the
//! sibling `cpp/build`); skips cleanly otherwise. Mirrors the Python/Java
//! multiclass+ranking conformance tests.

mod common;

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use relativedb::native::{load_lib, resolve_model_path, PrecomputedEncoder, RtNativeBackend, D_TEXT};
use relativedb::{Engine, ExecutionInput};

use common::{churn_schema, churn_wiring, dt};

// librt_c's forward runs on a process-global native thread pool that deadlocks
// under concurrent callers; serialize the forward-driving tests in this binary.
static NATIVE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn t0() -> DateTime<Utc> {
    dt("2026-07-01")
}

/// True when the dylib + classification checkpoint are available.
fn native_ready() -> bool {
    if load_lib(None).is_err() {
        return false;
    }
    resolve_model_path("hf://stanford-star/rt-j/classification").is_ok()
}

/// A distinct unit-ish 384-d embedding per string (label `i` peaks at dim `i`).
/// Enough to exercise the cosine decode deterministically without a real MiniLM.
fn label_encoder() -> PrecomputedEncoder {
    let names = ["running shoes", "espresso machine", "yoga mat"];
    let mut table: HashMap<String, Vec<f32>> = HashMap::new();
    for (i, n) in names.iter().enumerate() {
        let mut v = vec![0.05f32; D_TEXT];
        v[i % D_TEXT] = 1.0;
        table.insert((*n).to_string(), v);
    }
    PrecomputedEncoder::new(table)
}

fn engine_with_native() -> Engine {
    let backend = RtNativeBackend::new(Some(churn_schema()), Box::new(label_encoder()));
    Engine::new(churn_schema(), churn_wiring()).model_backend(Box::new(backend))
}

#[test]
fn multiclass_classifies_into_the_scanned_label_domain() {
    let _serial = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !native_ready() {
        eprintln!("SKIP multiclass: librt_c / checkpoint unavailable");
        return;
    }
    let mut eng = engine_with_native();
    // products.name is a TEXT column -> MULTICLASS_CLASSIFICATION. The class
    // domain is the distinct product names scanned under the temporal bound.
    let res = eng
        .execute(
            ExecutionInput::query("PREDICT products.name FOR EACH products.product_id")
                .anchor_time(t0()),
        )
        .expect("execute multiclass");

    assert_eq!(res.task_type, relativedb::TaskType::MulticlassClassification);
    assert_eq!(res.predictions.len(), 3, "one prediction per product");
    let domain = ["espresso machine", "running shoes", "yoga mat"]; // sorted UTF-8

    for p in &res.predictions {
        // class_probs cover the full K-class domain and form a distribution.
        assert_eq!(p.class_probs.len(), domain.len());
        let sum: f64 = p.class_probs.iter().map(|(_, v)| v).sum();
        assert!((sum - 1.0).abs() < 1e-6, "class_probs must sum to 1, got {sum}");
        for (_, v) in &p.class_probs {
            assert!(*v >= 0.0 && *v <= 1.0, "prob out of range: {v}");
        }
        // labels are exactly the sorted scanned domain
        let mut labels: Vec<&str> = p.class_probs.iter().map(|(c, _)| c.as_str()).collect();
        labels.sort();
        assert_eq!(labels, domain);
        // predicted_class is the argmax and is a real label
        let cls = p.predicted_class.as_deref().expect("predicted_class set");
        assert!(domain.contains(&cls), "predicted class {cls:?} not in domain");
        // argmax must be the max-probability class
        let best = p
            .class_probs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(best.0, cls);
    }
}

#[test]
fn ranking_returns_at_most_k_parent_ids() {
    let _serial = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !native_ready() {
        eprintln!("SKIP ranking: librt_c / checkpoint unavailable");
        return;
    }
    let mut eng = engine_with_native();
    // LIST_DISTINCT(orders.product_id) RANK TOP 2 -> MULTILABEL_RANKING. The
    // candidate parents are the distinct products.product_id scanned under the
    // temporal bound; each existence context is scored and the top 2 kept.
    let k = 2usize;
    let res = eng
        .execute(
            ExecutionInput::query(
                "PREDICT LIST_DISTINCT(orders.product_id) OVER (90 DAYS FOLLOWING) \
                 RANK TOP 2 FOR EACH customers.customer_id",
            )
            .anchor_time(t0()),
        )
        .expect("execute ranking");

    assert_eq!(res.task_type, relativedb::TaskType::MultilabelRanking);
    assert_eq!(res.predictions.len(), 3, "one ranked list per customer");
    let parent_ids = ["P1", "P2", "P3"]; // the 3 products in the toy graph

    for p in &res.predictions {
        assert!(p.ranked.len() <= k, "ranked ({}) must be <= k ({k})", p.ranked.len());
        assert!(!p.ranked.is_empty(), "expected a non-empty ranking");
        // ranked entries are stringified parent ids, no duplicates
        let mut seen = std::collections::HashSet::new();
        for id in &p.ranked {
            assert!(parent_ids.contains(&id.as_str()), "unexpected ranked id {id:?}");
            assert!(seen.insert(id.clone()), "duplicate id in ranking: {id:?}");
        }
    }
}
