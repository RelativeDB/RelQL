//! Cross-language ranking-parity test (Rust side).
//!
//! Loads the shared fixture in `benchmarks/xlang_fixture/` — a fixed real
//! MovieLens (ml-latest-small) Top-5 ranking scenario — runs it through the
//! native RT-J backend using the SAME precomputed MiniLM embeddings Python
//! used (`embeddings.tsv`), and asserts the ranking is correct AND
//! non-degenerate.
//!
//! Guards against the two ranking bugs found on 2026-07-18 (no candidate cells
//! emitted; no target token for cell-less entity tables), both of which
//! produced the degenerate candidate-enumeration order `[1, 2, 3, 50, 260]`.
//! See `benchmarks/xlang_fixture/README.md`.
//!
//! Run:
//!   cd rust && RELATIVEDB_RT_LIB=/Users/henneberger/getasterisk/cpp/build/librt_c.dylib \
//!     cargo test --test xlang_parity
//!
//! Skips cleanly (passes without asserting) when the dylib / checkpoint is
//! unavailable, mirroring `tests/native_tasks_tests.rs`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};

use relativedb::native::{load_lib, resolve_model_path, PrecomputedEncoder, RtNativeBackend, D_TEXT};
use relativedb::{
    ContextPolicy, Engine, EntityId, ExecutionInput, LinkDef, RetrieverWiring, Row, SamplerMode,
    Schema, TableDef, TemporalBound, Value, ValueType,
};

// librt_c's forward runs on a process-global native thread pool that deadlocks
// under concurrent callers; serialize forward-driving tests in this binary.
static NATIVE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// True when the dylib + classification checkpoint are both available.
fn native_ready() -> bool {
    load_lib(None).is_ok() && resolve_model_path("hf://stanford-star/rt-j/classification").is_ok()
}

fn fixture_dir() -> String {
    format!("{}/../../benchmarks/xlang_fixture", env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn golden(dir: &str) -> serde_json::Value {
    serde_json::from_str(&read(&format!("{dir}/golden.json"))).expect("parse golden.json")
}

/// String -> [f32; 384] precomputed MiniLM table; missing keys yield a zero
/// vector (exactly `PrecomputedEncoder`'s behaviour).
fn load_embeddings(path: &str) -> HashMap<String, Vec<f32>> {
    let mut table = HashMap::new();
    for line in read(path).lines() {
        if line.is_empty() {
            continue;
        }
        let (key, rest) = line.split_once('\t').expect("embeddings: <string>\\t<floats>");
        let v: Vec<f32> = rest.split_whitespace().map(|x| x.parse::<f32>().unwrap()).collect();
        assert_eq!(v.len(), D_TEXT, "embedding for {key:?} is {} dims", v.len());
        table.insert(key.to_string(), v);
    }
    table
}

fn build_schema() -> Schema {
    Schema::new_schema()
        .table(TableDef::new_table("users").primary_key("user_id").build())
        .table(
            TableDef::new_table("movies")
                .column("title", ValueType::Text)
                .column("genres", ValueType::Text)
                .primary_key("movie_id")
                .build(),
        )
        .table(
            TableDef::new_table("ratings")
                .column("rating", ValueType::Number)
                .column("ts", ValueType::Datetime)
                .primary_key("rating_id")
                .time_column("ts")
                .build(),
        )
        .link(LinkDef::link("ratings", "user_id", "users"))
        .link(LinkDef::link("ratings", "movie_id", "movies"))
        .build()
}

fn build_rows(dir: &str) -> HashMap<String, Vec<Row>> {
    let mut rows: HashMap<String, Vec<Row>> = HashMap::new();
    let mut users: HashSet<i64> = HashSet::new();

    // movies.tsv : movie_id \t title \t genres
    let mut movies = Vec::new();
    for line in read(&format!("{dir}/movies.tsv")).lines() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let mid: i64 = f[0].parse().unwrap();
        movies.push(
            Row::new("movies", EntityId::Int(mid))
                .cell("title", Value::Text(f[1].to_string()))
                .cell("genres", Value::Text(f[2].to_string())),
        );
    }
    rows.insert("movies".to_string(), movies);

    // ratings.tsv : rating_id \t user_id \t movie_id \t rating \t ts_epoch_seconds
    let mut ratings = Vec::new();
    for line in read(&format!("{dir}/ratings.tsv")).lines() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let rid: i64 = f[0].parse().unwrap();
        let uid: i64 = f[1].parse().unwrap();
        let mid: i64 = f[2].parse().unwrap();
        let rating: f64 = f[3].parse().unwrap();
        let ts_epoch: i64 = f[4].parse().unwrap();
        let ts: DateTime<Utc> = DateTime::from_timestamp(ts_epoch, 0).expect("valid epoch");
        users.insert(uid);
        ratings.push(
            Row::new("ratings", EntityId::Int(rid))
                .cell("rating", Value::Number(rating))
                .cell("ts", Value::Datetime(ts))
                .timestamp(ts)
                .parent("user_id", EntityId::Int(uid))
                .parent("movie_id", EntityId::Int(mid)),
        );
    }
    rows.insert("ratings".to_string(), ratings);

    // users have no cells; derived from the ratings history.
    let mut user_ids: Vec<i64> = users.into_iter().collect();
    user_ids.sort();
    rows.insert(
        "users".to_string(),
        user_ids.into_iter().map(|u| Row::new("users", EntityId::Int(u))).collect(),
    );
    rows
}

/// Mirror the reference harness wiring: entities+scanner per table, one
/// default_links (children newest-first honouring bound + limit).
fn build_wiring(rows: HashMap<String, Vec<Row>>) -> RetrieverWiring {
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
                    .filter(|r| bound.admits_row(r))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        kids.sort_by(|a, b| {
            let ka = a.timestamp.map(|t| t.timestamp()).unwrap_or(i64::MIN);
            let kb = b.timestamp.map(|t| t.timestamp()).unwrap_or(i64::MIN);
            kb.cmp(&ka)
        });
        kids.truncate(limit);
        kids
    });

    for t in tables {
        let r_ent = Arc::clone(&rows);
        w = w.entities(t.clone(), move |table: &str, ids: &[EntityId], bound: &TemporalBound| {
            let mut out = Vec::new();
            if let Some(rs) = r_ent.get(table) {
                for id in ids {
                    if let Some(r) = rs.iter().find(|r| &r.id == id) {
                        if bound.admits_row(r) {
                            out.push(r.clone());
                        }
                    }
                }
            }
            out
        });
        let r_sc = Arc::clone(&rows);
        w = w.scanner(t.clone(), move |table: &str, bound: &TemporalBound| {
            r_sc.get(table)
                .map(|rs| rs.iter().filter(|r| bound.admits_row(r)).cloned().collect())
                .unwrap_or_default()
        });
    }
    w.build()
}

fn wide_policy() -> ContextPolicy {
    // benchmarks/harness/datasets.WIDE_POLICY.
    ContextPolicy {
        max_context_cells: 5_000_000,
        bfs_width: 20_000,
        max_hops: 1,
        ..ContextPolicy::default()
    }
}

fn build_engine(dir: &str) -> Engine {
    let schema = build_schema();
    let wiring = build_wiring(build_rows(dir));
    let table = load_embeddings(&format!("{dir}/embeddings.tsv"));
    let encoder = PrecomputedEncoder::new(table);
    let backend = RtNativeBackend::new(Some(schema.clone()), Box::new(encoder));
    Engine::new(schema, wiring)
        .model_backend(Box::new(backend))
        .sampler_mode(SamplerMode::Csc)
        .context_policy(wide_policy())
        .build()
        .expect("build CSC index")
}

/// entity_id -> ranked movie_id list (i64).
fn rank(eng: &mut Engine, query: &str, anchor: DateTime<Utc>, users: &[i64]) -> HashMap<i64, Vec<i64>> {
    let entity_ids: Vec<EntityId> = users.iter().map(|u| EntityId::Int(*u)).collect();
    let res = eng
        .execute(ExecutionInput::query(query).anchor_time(anchor).entity_ids(entity_ids))
        .expect("execute ranking");
    assert_eq!(res.task_type, relativedb::TaskType::MultilabelRanking, "expected ranking task");
    res.predictions
        .iter()
        .map(|p| {
            let uid = match &p.id {
                EntityId::Int(i) => *i,
                other => panic!("non-int user id: {other:?}"),
            };
            let order: Vec<i64> = p.ranked.iter().map(|s| s.parse::<i64>().unwrap()).collect();
            (uid, order)
        })
        .collect()
}

#[test]
fn xlang_ranking_parity() {
    let _serial = NATIVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !native_ready() {
        eprintln!("SKIP xlang_ranking_parity: librt_c / checkpoint unavailable");
        return;
    }

    let dir = fixture_dir();
    let g = golden(&dir);
    let inv = &g["invariants"];

    let query = g["query"].as_str().unwrap().to_string();
    let top_k = g["top_k"].as_u64().unwrap();
    let anchor: DateTime<Utc> = DateTime::from_timestamp(g["anchor_epoch"].as_i64().unwrap(), 0).unwrap();
    let users: Vec<i64> = g["users"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    let degenerate: Vec<i64> = inv["must_not_equal_degenerate_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let expected_top1: HashMap<i64, i64> = inv["expected_top1"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.parse::<i64>().unwrap(), v.as_i64().unwrap()))
        .collect();
    let min_distinct = inv["min_distinct_scores"].as_u64().unwrap() as usize;
    let golden_rust: HashMap<i64, Vec<i64>> = g["per_binding_golden"]["rust"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.parse::<i64>().unwrap(),
                v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect(),
            )
        })
        .collect();
    let mut candidates_sorted: Vec<i64> =
        g["candidate_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    candidates_sorted.sort();

    let mut eng = build_engine(&dir);

    // --- The golden RANK TOP 5 query --------------------------------------
    let top5 = rank(&mut eng, &query, anchor, &users);
    assert_eq!(
        top5.keys().copied().collect::<HashSet<_>>(),
        users.iter().copied().collect::<HashSet<_>>()
    );
    for u in &users {
        let got = &top5[u];
        // (1) not the degenerate candidate-enumeration order
        assert_ne!(got, &degenerate, "user {u} returned the degenerate order — ranking bug");
        // (2) top1 is Silence of the Lambs (strong, tie-free signal)
        assert_eq!(got[0], expected_top1[u], "user {u} top1 wrong: {got:?}");
        // (4) reproduce the captured Rust per-binding golden top-5
        assert_eq!(got, &golden_rust[u], "user {u} top-5 != rust golden");
    }

    // --- (3) candidate discrimination via full RANK TOP 10 ----------------
    // `ranked` exposes only ordered ids (no per-candidate scores), so rank ALL
    // candidates and assert the order is not the sorted candidate-id order
    // (what the degenerate bug produced) and top1 is still 593.
    let full_query = query.replace(&format!("RANK TOP {top_k}"), &format!("RANK TOP {}", candidates_sorted.len()));
    let full = rank(&mut eng, &full_query, anchor, &users);
    for u in &users {
        let order = &full[u];
        let mut order_sorted = order.clone();
        order_sorted.sort();
        assert_eq!(order_sorted, candidates_sorted, "user {u} full ranking is not a candidate permutation");
        assert_ne!(
            order, &candidates_sorted,
            "user {u} full ranking equals the sorted candidate-id order — candidates not discriminated"
        );
        assert_eq!(order[0], expected_top1[u], "user {u} full-rank top1 wrong: {order:?}");
        let distinct: HashSet<i64> = order.iter().copied().collect();
        assert!(distinct.len() >= min_distinct, "user {u} fewer than {min_distinct} distinct candidates");
    }
}
