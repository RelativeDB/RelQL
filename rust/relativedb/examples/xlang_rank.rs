//! Cross-language parity harness: reproduce a MovieLens Top-5 ranking through
//! the native RT-J backend from the Rust binding, on the SAME fixed data +
//! embeddings Python used, and check the top-5 per user matches Python.
//!
//! Run:
//!   RELATIVEDB_RT_LIB=/Users/henneberger/getasterisk/cpp/build/librt_c.dylib \
//!     cargo run --example xlang_rank
//!
//! Shared inputs live under $XLANG_DIR (defaults to the scratchpad path).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};

use relativedb::native::{PrecomputedEncoder, RtNativeBackend, TextEncoder, D_TEXT};
use relativedb::{
    ContextPolicy, Engine, EntityId, ExecutionInput, LinkDef, RetrieverWiring, Row, SamplerMode,
    Schema, TableDef, TemporalBound, ValueType, Value,
};

/// A precomputed encoder that records the unique strings it could not resolve
/// (returning a zero vector for them, exactly like `PrecomputedEncoder`).
struct TrackingEncoder {
    inner: PrecomputedEncoder,
    misses: Arc<Mutex<HashSet<String>>>,
}

impl TrackingEncoder {
    fn new(table: HashMap<String, Vec<f32>>, misses: Arc<Mutex<HashSet<String>>>) -> TrackingEncoder {
        TrackingEncoder { inner: PrecomputedEncoder::new(table), misses }
    }
}

impl TextEncoder for TrackingEncoder {
    fn encode(&self, text: &str) -> Vec<f32> {
        if !self.inner.table.contains_key(text) {
            self.misses.lock().unwrap().insert(text.to_string());
            return vec![0.0; D_TEXT];
        }
        self.inner.encode(text)
    }
}

fn xlang_dir() -> String {
    std::env::var("XLANG_DIR").unwrap_or_else(|_| {
        "/private/tmp/claude-501/-Users-henneberger-getasterisk/\
         9892aecd-aaa6-41c7-bd26-817485487547/scratchpad/xlang"
            .to_string()
    })
}

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

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

/// Mirror the reference harness wiring: entities+scanner for every table, one
/// default_links (child ratings for a parent, newest-first by ts, capped).
fn build_wiring(rows: HashMap<String, Vec<Row>>) -> RetrieverWiring {
    use std::sync::Arc;
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
        // newest-first by ts (rows without ts sort last)
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

/// Minimal extraction of the fields we need from scenario.json.
fn scenario(dir: &str) -> serde_json::Value {
    serde_json::from_str(&read(&format!("{dir}/scenario.json"))).expect("parse scenario.json")
}

fn main() {
    let dir = xlang_dir();
    let sc = scenario(&dir);

    let anchor_epoch = sc["anchor_epoch"].as_i64().unwrap();
    let anchor: DateTime<Utc> = DateTime::from_timestamp(anchor_epoch, 0).unwrap();
    let mut query = sc["query"].as_str().unwrap().to_string();
    let mut top_k = sc["top_k"].as_u64().unwrap();
    // Diagnostic: FULL_RANK=1 widens to TOP 10 to reveal the full candidate order.
    if std::env::var("FULL_RANK").as_deref() == Ok("1") {
        query = query.replace("RANK TOP 5", "RANK TOP 10");
        top_k = 10;
    }
    let users: Vec<i64> = sc["users"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    let title_of: HashMap<String, String> = sc["title_of"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
        .collect();
    let py: HashMap<String, Vec<String>> = sc["python_result"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (k.clone(), v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect())
        })
        .collect();

    println!("query : {query}");
    println!("anchor: {} ({})", anchor_epoch, anchor.to_rfc3339());
    println!("users : {users:?}  top_k={top_k}\n");

    let schema = build_schema();
    let rows = build_rows(&dir);
    let wiring = build_wiring(rows);

    let table = load_embeddings(&format!("{dir}/embeddings.tsv"));
    println!("embeddings loaded: {} strings\n", table.len());
    // The backend takes ownership of the encoder, so share the miss set via Arc.
    let misses_handle: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let encoder = TrackingEncoder::new(table, Arc::clone(&misses_handle));

    let backend = RtNativeBackend::new(Some(schema.clone()), Box::new(encoder));
    // Mirror the Python reference harness: CSC sampler + WIDE context policy
    // (datasets.WIDE_POLICY) so the full per-user history enters context.
    let wide_policy = ContextPolicy {
        max_context_cells: 5_000_000,
        bfs_width: 20_000,
        max_hops: 1,
        ..ContextPolicy::default()
    };
    let mut eng = Engine::new(schema, wiring)
        .model_backend(Box::new(backend))
        .sampler_mode(SamplerMode::Csc)
        .context_policy(wide_policy)
        .build()
        .expect("build CSC index");

    let entity_ids: Vec<EntityId> = users.iter().map(|u| EntityId::Int(*u)).collect();
    let res = eng
        .execute(ExecutionInput::query(query).anchor_time(anchor).entity_ids(entity_ids))
        .expect("execute ranking");

    println!("task_type: {:?}", res.task_type);
    println!("model_uri: {}\n", res.model_uri);

    // Structural parity check: how many rows enter each user's context.
    for u in &users {
        let ctx = eng
            .assemble_context("users", &EntityId::Int(*u), Some(anchor))
            .expect("assemble context");
        let n_ratings = ctx.rows.iter().filter(|r| r.table == "ratings").count();
        println!("context user {u}: {} rows ({} ratings)", ctx.rows.len(), n_ratings);
    }
    println!();

    let mut all_match = true;
    for p in &res.predictions {
        let uid = p.id.to_string();
        let titles: Vec<String> = p
            .ranked
            .iter()
            .map(|mid| title_of.get(mid).cloned().unwrap_or_else(|| format!("<{mid}>")))
            .collect();
        println!("user {}: {}", uid, titles.join(" | "));
        println!("        ids  = {:?}", p.ranked);
        if let Some(expect) = py.get(&uid) {
            let ok = &p.ranked == expect;
            all_match &= ok;
            println!("        python = {:?}  -> {}", expect, if ok { "MATCH" } else { "MISMATCH" });
        }
        println!();
    }

    // Report embedding misses.
    let misses: Vec<String> = misses_handle.lock().unwrap().iter().cloned().collect();
    let bad_semantic: Vec<&String> = misses
        .iter()
        .filter(|m| !m.contains(" of ")) // "<col> of <table>" phrases are system strings
        .collect();
    println!("embedding misses: {} unique", misses.len());
    for m in misses.iter().take(8) {
        println!("   miss: {m:?}");
    }
    if !bad_semantic.is_empty() {
        println!("WARNING: {} non-phrase (title/genre) misses -> BUG: {:?}", bad_semantic.len(), bad_semantic);
    }

    println!("\nOVERALL: {}", if all_match { "ALL USERS MATCH PYTHON" } else { "MISMATCH vs PYTHON" });
    if !all_match {
        std::process::exit(2);
    }
}
