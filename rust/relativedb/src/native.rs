//! Native RT model backend — scores contexts with the golden-verified C++ RT-J
//! engine (`librt_c`) instead of the history baseline.
//!
//! Three layers, mirroring `relativedb.rt_native` (Python) / `com.relativedb.rt`
//! (Java):
//!
//! 1. [`RtLib`] / [`load_lib`] — a `libloading` binding to the C ABI in
//!    `cpp/src/rt_c.h`. The library is lazy-loaded from `RELATIVEDB_RT_LIB`, or the sibling `cpp/build/librt_c.{dylib,so}`; a clear
//!    [`RtError::Unavailable`] is raised when missing.
//! 2. [`TextEncoder`] — the frozen text/schema-phrase embedder (F13/F14). A
//!    [`PrecomputedEncoder`] is provided for tests; a real MiniLM encoder is a
//!    separate concern.
//! 3. [`RtNativeBackend`] — implements [`ModelBackend`]: converts each
//!    assembled context into the RAW PRE-SORT token arrays the engine consumes,
//!    runs one forward pass per batch, and maps the number-head target score
//!    back to the task output (sigmoid for classification, in-context
//!    denormalization for regression/forecasting).

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use libloading::Library;

use crate::engine::{EntityContext, EntityPrediction, ModelBackend};
use crate::engine::ExecutionError;
use crate::evaluate::{eval_bool, eval_value, EvalValue};
use crate::model::ModelConfig;
use crate::pql::ast::{ParsedQuery, TaskType};
use crate::retrieve::Value;
use crate::schema::{Schema, ValueType};
use crate::Error;

pub const D_TEXT: usize = 384;
pub const MAX_F2P: usize = 5;

const SEM_NUMBER: i64 = 0;
const SEM_TEXT: i64 = 1;
const SEM_DATETIME: i64 = 2;
const SEM_BOOLEAN: i64 = 3;

/// Errors from the native binding.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RtError {
    /// librt_c (or a runtime dependency) could not be located/loaded.
    Unavailable(String),
    /// An error reported by the native RT engine.
    Native(String),
}

impl std::fmt::Display for RtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RtError::Unavailable(s) => write!(f, "rt native unavailable: {}", s),
            RtError::Native(s) => write!(f, "rt native error: {}", s),
        }
    }
}
impl std::error::Error for RtError {}

// ---------------------------------------------------------------------------
// libloading binding
// ---------------------------------------------------------------------------

type LoadFn = unsafe extern "C" fn(*const c_char, *mut c_char, usize) -> *mut c_void;
type FreeFn = unsafe extern "C" fn(*mut c_void);
type NumParamsFn = unsafe extern "C" fn(*const c_void) -> i64;
type ForwardFn = unsafe extern "C" fn(
    *const c_void,
    i32,
    i32,
    *const i64, // node
    *const i64, // f2p
    *const i64, // col
    *const i64, // table
    *const u8,  // is_padding
    *const i64, // sem
    *const u8,  // is_target
    *const f32, // number
    *const f32, // datetime
    *const f32, // boolean
    *const f32, // text
    *const f32, // col_name
    i32,        // n_threads
    *mut f32,   // out
    *mut c_char,
    usize,
) -> i32;

/// A loaded `librt_c` with bound signatures (see `cpp/src/rt_c.h`).
pub struct RtLib {
    _lib: Library,
    load: LoadFn,
    free: FreeFn,
    num_params: NumParamsFn,
    forward: ForwardFn,
    pub path: String,
}

impl RtLib {
    pub fn open(path: &str) -> Result<Arc<RtLib>, RtError> {
        unsafe {
            let lib = Library::new(path).map_err(|e| {
                RtError::Unavailable(format!("found {} but could not load it: {}", path, e))
            })?;
            let load: LoadFn = *lib
                .get(b"rt_model_load\0")
                .map_err(|e| RtError::Unavailable(format!("rt_model_load: {}", e)))?;
            let free: FreeFn = *lib
                .get(b"rt_model_free\0")
                .map_err(|e| RtError::Unavailable(format!("rt_model_free: {}", e)))?;
            let num_params: NumParamsFn = *lib
                .get(b"rt_model_num_params\0")
                .map_err(|e| RtError::Unavailable(format!("rt_model_num_params: {}", e)))?;
            let forward: ForwardFn = *lib
                .get(b"rt_forward\0")
                .map_err(|e| RtError::Unavailable(format!("rt_forward: {}", e)))?;
            Ok(Arc::new(RtLib { _lib: lib, load, free, num_params, forward, path: path.to_string() }))
        }
    }

    pub fn load_model(self: &Arc<RtLib>, safetensors_path: &str) -> Result<RtModel, RtError> {
        let cpath = CString::new(safetensors_path).unwrap();
        let mut err = vec![0i8; 512];
        let handle = unsafe {
            (self.load)(cpath.as_ptr(), err.as_mut_ptr() as *mut c_char, err.len())
        };
        if handle.is_null() {
            return Err(RtError::Native(format!(
                "rt_model_load({:?}) failed: {}",
                safetensors_path,
                cstr_message(&err)
            )));
        }
        Ok(RtModel { lib: Arc::clone(self), handle, path: safetensors_path.to_string() })
    }
}

fn cstr_message(buf: &[i8]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// A loaded RT-J checkpoint living in the native engine.
pub struct RtModel {
    lib: Arc<RtLib>,
    handle: *mut c_void,
    pub path: String,
}

impl RtModel {
    pub fn num_params(&self) -> i64 {
        unsafe { (self.lib.num_params)(self.handle) }
    }

    /// Run the forward pass over RAW PRE-SORT arrays (see `rt_c.h`). Returns the
    /// per-batch-row target score `[B]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        b: i32,
        s: i32,
        node_idxs: &[i64],
        f2p: &[i64],
        col_idxs: &[i64],
        table_idxs: &[i64],
        is_padding: &[u8],
        sem_types: &[i64],
        is_target: &[u8],
        number_v: &[f32],
        datetime_v: &[f32],
        boolean_v: &[f32],
        text_v: &[f32],
        col_name_v: &[f32],
        n_threads: i32,
    ) -> Result<Vec<f32>, RtError> {
        let n = (b as usize) * (s as usize);
        assert_eq!(node_idxs.len(), n);
        assert_eq!(f2p.len(), n * MAX_F2P);
        assert_eq!(text_v.len(), n * D_TEXT);
        assert_eq!(col_name_v.len(), n * D_TEXT);
        let mut out = vec![0f32; b as usize];
        let mut err = vec![0i8; 512];
        let rc = unsafe {
            (self.lib.forward)(
                self.handle,
                b,
                s,
                node_idxs.as_ptr(),
                f2p.as_ptr(),
                col_idxs.as_ptr(),
                table_idxs.as_ptr(),
                is_padding.as_ptr(),
                sem_types.as_ptr(),
                is_target.as_ptr(),
                number_v.as_ptr(),
                datetime_v.as_ptr(),
                boolean_v.as_ptr(),
                text_v.as_ptr(),
                col_name_v.as_ptr(),
                n_threads,
                out.as_mut_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(RtError::Native(format!("rt_forward failed ({}): {}", rc, cstr_message(&err))));
        }
        Ok(out)
    }
}

impl Drop for RtModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.lib.free)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

// RtModel wraps a raw handle documented as thread-safe/reentrant in rt_c.h.
unsafe impl Send for RtModel {}
unsafe impl Sync for RtModel {}

// ---------------------------------------------------------------------------
// library discovery
// ---------------------------------------------------------------------------

pub(crate) fn lib_filename() -> &'static str {
    if cfg!(target_os = "macos") {
        "librt_c.dylib"
    } else if cfg!(target_os = "windows") {
        "rt_c.dll"
    } else {
        "librt_c.so"
    }
}

/// Ordered library-discovery candidates shared by every librt_c binding
/// (`RELATIVEDB_RT_LIB` override first, then the sibling `cpp/build` tree).
pub(crate) fn candidate_lib_paths() -> Vec<String> {
    let mut cands = Vec::new();
    for var in ["RELATIVEDB_RT_LIB"] {
        if let Ok(v) = std::env::var(var) {
            if !v.is_empty() {
                cands.push(v);
            }
        }
    }
    let fname = lib_filename();
    // sibling C++ build tree of the monorepo, relative to this crate
    let manifest = env!("CARGO_MANIFEST_DIR");
    cands.push(
        Path::new(manifest)
            .join("..")
            .join("..")
            .join("cpp")
            .join("build")
            .join(fname)
            .to_string_lossy()
            .into_owned(),
    );
    cands.push(format!("../cpp/build/{}", fname));
    cands
}

/// Lazy-load librt_c; returns [`RtError::Unavailable`] listing the searched
/// paths when it cannot be found.
pub fn load_lib(path: Option<&str>) -> Result<Arc<RtLib>, RtError> {
    let candidates: Vec<String> = match path {
        Some(p) => vec![p.to_string()],
        None => candidate_lib_paths(),
    };
    let mut tried = Vec::new();
    for cand in &candidates {
        tried.push(cand.clone());
        if !Path::new(cand).exists() {
            continue;
        }
        return RtLib::open(cand);
    }
    Err(RtError::Unavailable(format!(
        "librt_c was not found (build cpp/ with cmake, or set RELATIVEDB_RT_LIB to the built \
         library). Searched: {}",
        tried.join(", ")
    )))
}

// ---------------------------------------------------------------------------
// checkpoint URI resolution
// ---------------------------------------------------------------------------

fn hf_cache_root() -> PathBuf {
    if let Ok(v) = std::env::var("HF_HOME") {
        return PathBuf::from(v).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache").join("huggingface").join("hub")
}

/// Resolve a checkpoint URI to a local `model.safetensors` path.
///
/// Accepts a filesystem path (file, or directory containing
/// `model.safetensors`), `file://...`, or `hf://org/repo/subdir` (resolved
/// against the local HF cache; no network client).
pub fn resolve_model_path(uri: &str) -> Result<String, RtError> {
    let as_path = |p: &str| -> Option<String> {
        let path = Path::new(p);
        if path.is_file() {
            return Some(p.to_string());
        }
        if path.is_dir() {
            let m = path.join("model.safetensors");
            if m.is_file() {
                return Some(m.to_string_lossy().into_owned());
            }
        }
        None
    };

    if let Some(p) = as_path(uri) {
        return Ok(p);
    }
    if let Some(rest) = uri.strip_prefix("file://") {
        return as_path(rest)
            .ok_or_else(|| RtError::Unavailable(format!("file:// path has no model.safetensors: {:?}", rest)));
    }
    if let Some(rest) = uri.strip_prefix("hf://") {
        let rest = rest.trim_matches('/');
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() < 2 {
            return Err(RtError::Unavailable(format!("malformed hf:// URI: {:?}", uri)));
        }
        let repo_dir = format!("models--{}--{}", parts[0], parts[1]);
        let sub = parts[2..].join("/");
        let snapshots = hf_cache_root().join(&repo_dir).join("snapshots");
        let snap = std::fs::read_dir(&snapshots)
            .ok()
            .and_then(|mut it| it.next())
            .and_then(|e| e.ok())
            .map(|e| e.path())
            .ok_or_else(|| {
                RtError::Unavailable(format!(
                    "hf:// checkpoint not in local cache: {:?} (looked under {})",
                    uri,
                    snapshots.display()
                ))
            })?;
        let mut model = snap;
        if !sub.is_empty() {
            model = model.join(&sub);
        }
        model = model.join("model.safetensors");
        if model.is_file() {
            return Ok(model.to_string_lossy().into_owned());
        }
        return Err(RtError::Unavailable(format!(
            "hf:// checkpoint resolved to {} which does not exist",
            model.display()
        )));
    }
    Err(RtError::Unavailable(format!(
        "cannot resolve model uri {:?} (not a path, not file://, not hf://)",
        uri
    )))
}

// ---------------------------------------------------------------------------
// text embeddings
// ---------------------------------------------------------------------------

/// The frozen text/schema-phrase encoder (F13/F14). Implementations return a
/// `D_TEXT`-dim embedding per string.
pub trait TextEncoder {
    fn encode(&self, text: &str) -> Vec<f32>;
}

/// A precomputed lookup encoder — enough for tests. Unknown strings embed as
/// zeros.
pub struct PrecomputedEncoder {
    pub table: HashMap<String, Vec<f32>>,
    pub dim: usize,
}

impl PrecomputedEncoder {
    pub fn new(table: HashMap<String, Vec<f32>>) -> PrecomputedEncoder {
        PrecomputedEncoder { table, dim: D_TEXT }
    }
    pub fn empty() -> PrecomputedEncoder {
        PrecomputedEncoder { table: HashMap::new(), dim: D_TEXT }
    }
}

impl TextEncoder for PrecomputedEncoder {
    fn encode(&self, text: &str) -> Vec<f32> {
        self.table.get(text).cloned().unwrap_or_else(|| vec![0.0; self.dim])
    }
}

// ---------------------------------------------------------------------------
// context -> RT token batch conversion
// ---------------------------------------------------------------------------

const TASK_TABLE: &str = "task";
const TASK_TIME_COL: &str = "timestamp";
const TASK_LABEL_COL: &str = "label";

#[derive(Clone)]
enum RawVal {
    Num(f64),
    Bool(bool),
    Date(DateTime<Utc>),
    Text(String),
    Mask, // masked target / no value
}

struct Tok {
    node: i64,
    f2p: [i64; MAX_F2P],
    col: (String, String), // (column, table)
    table: String,
    /// schema-declared sem type; the value channel is chosen from `raw` at
    /// collate (bool routes through the number channel, `bool_as_num`).
    #[allow(dead_code)]
    sem: i64,
    is_tgt: bool,
    raw: RawVal,
}

fn days(t: DateTime<Utc>) -> f64 {
    (t.timestamp() as f64 + t.timestamp_subsec_nanos() as f64 / 1e9) / 86_400.0
}

/// A real [`ModelBackend`] over the C++ RT engine.
pub struct RtNativeBackend {
    pub schema: Option<Schema>,
    pub lib_path: Option<String>,
    pub encoder: Box<dyn TextEncoder>,
    pub n_threads: i32,
    pub num_history_windows: usize,
    pub max_seq_len: usize,
    models: HashMap<String, RtModel>,
    lib: Option<Arc<RtLib>>,
}

impl RtNativeBackend {
    pub fn new(schema: Option<Schema>, encoder: Box<dyn TextEncoder>) -> RtNativeBackend {
        RtNativeBackend {
            schema,
            lib_path: None,
            encoder,
            n_threads: 0,
            num_history_windows: 3,
            max_seq_len: 1024,
            models: HashMap::new(),
            lib: None,
        }
    }

    fn model_for(&mut self, model_uri: &str) -> Result<&RtModel, Error> {
        let path = resolve_model_path(model_uri).map_err(Error::from)?;
        if !self.models.contains_key(&path) {
            if self.lib.is_none() {
                self.lib = Some(load_lib(self.lib_path.as_deref()).map_err(Error::from)?);
            }
            let model = self.lib.as_ref().unwrap().load_model(&path).map_err(Error::from)?;
            self.models.insert(path.clone(), model);
        }
        Ok(&self.models[&path])
    }

    fn sem_for_cell(&self, table: &str, col: &str, value: &Value) -> i64 {
        if let Some(schema) = &self.schema {
            if let Some(c) = schema.table(table).and_then(|t| t.column(col)) {
                return match c.value_type {
                    ValueType::Number => SEM_NUMBER,
                    ValueType::Text => SEM_TEXT,
                    ValueType::Datetime => SEM_DATETIME,
                    ValueType::Boolean => SEM_BOOLEAN,
                };
            }
        }
        match value {
            Value::Boolean(_) => SEM_BOOLEAN,
            Value::Number(_) => SEM_NUMBER,
            Value::Datetime(_) => SEM_DATETIME,
            Value::Text(_) => SEM_TEXT,
        }
    }

    fn self_labels(
        &self,
        query: &ParsedQuery,
        task_type: TaskType,
        ctx: &EntityContext,
    ) -> Vec<(DateTime<Utc>, f64)> {
        let window = query.target_aggregations().iter().find_map(|a| a.window);
        let span = match (ctx.anchor, window.and_then(|w| w.span())) {
            (Some(a), Some(s)) => Some((a, s)),
            _ => None,
        };
        let (anchor, span) = match span {
            Some(v) => v,
            None => return Vec::new(),
        };
        let rows_by_table = ctx.rows_by_table();
        let cells = ctx.entity_cells(&query.entity_key.table);
        let mut out = Vec::new();
        for k in 1..=self.num_history_windows as i32 {
            let pa = anchor - span * k;
            let v = if task_type == TaskType::BinaryClassification {
                if eval_bool(&query.target, &rows_by_table, &cells, Some(pa)) { 1.0 } else { 0.0 }
            } else {
                match eval_value(&query.target, &rows_by_table, &cells, Some(pa)) {
                    EvalValue::Num(n) => n,
                    EvalValue::Bool(b) => if b { 1.0 } else { 0.0 },
                    _ => continue,
                }
            };
            out.push((pa, v));
        }
        out
    }

    fn build_sequences(
        &self,
        query: &ParsedQuery,
        task_type: TaskType,
        contexts: &[EntityContext],
    ) -> (Vec<Vec<Tok>>, f64, f64) {
        let entity_table = &query.entity_key.table;
        let mut fk_to_parent: HashMap<String, HashMap<String, String>> = HashMap::new();
        if let Some(schema) = &self.schema {
            for t in &schema.tables {
                let m: HashMap<String, String> = schema
                    .links_from(&t.name)
                    .iter()
                    .map(|l| (l.fk_column.clone(), l.to_table.clone()))
                    .collect();
                fk_to_parent.insert(t.name.clone(), m);
            }
        }

        let mut seqs: Vec<Vec<Tok>> = Vec::new();
        let mut all_labels: Vec<f64> = Vec::new();
        for ctx in contexts {
            let mut seq: Vec<Tok> = Vec::new();
            // node id assignment: rows first, then synthetic task nodes
            let mut node_of: HashMap<(String, String), i64> = HashMap::new();
            let mut next_node = 0i64;
            let node = |key: (String, String), next: &mut i64, map: &mut HashMap<(String, String), i64>| -> i64 {
                *map.entry(key).or_insert_with(|| {
                    let n = *next;
                    *next += 1;
                    n
                })
            };
            // rows claim node ids first (by (table, id-string))
            for r in &ctx.rows {
                let key = (r.table.clone(), r.id.to_string());
                node(key, &mut next_node, &mut node_of);
            }
            // id -> row keys (for schema-less parent linking)
            let mut by_id: HashMap<String, Vec<(String, String)>> = HashMap::new();
            for r in &ctx.rows {
                by_id.entry(r.id.to_string()).or_default().push((r.table.clone(), r.id.to_string()));
            }

            let entity_node = node(
                (entity_table.clone(), ctx.entity_id.to_string()),
                &mut next_node,
                &mut node_of,
            );

            // target task row (masked label)
            let tgt_node = node((TASK_TABLE.into(), "__target__".into()), &mut next_node, &mut node_of);
            if let Some(anchor) = ctx.anchor {
                seq.push(Tok {
                    node: tgt_node,
                    f2p: pad_parents(&[entity_node]),
                    col: (TASK_TIME_COL.into(), TASK_TABLE.into()),
                    table: TASK_TABLE.into(),
                    sem: SEM_DATETIME,
                    is_tgt: false,
                    raw: RawVal::Date(anchor),
                });
            }
            seq.push(Tok {
                node: tgt_node,
                f2p: pad_parents(&[entity_node]),
                col: (TASK_LABEL_COL.into(), TASK_TABLE.into()),
                table: TASK_TABLE.into(),
                sem: SEM_NUMBER,
                is_tgt: true,
                raw: RawVal::Mask,
            });

            // past outcomes (self labels, F65)
            for (ts, label) in self.self_labels(query, task_type, ctx) {
                let hnode = node((TASK_TABLE.into(), ts.to_rfc3339()), &mut next_node, &mut node_of);
                seq.push(Tok {
                    node: hnode,
                    f2p: pad_parents(&[entity_node]),
                    col: (TASK_LABEL_COL.into(), TASK_TABLE.into()),
                    table: TASK_TABLE.into(),
                    sem: SEM_NUMBER,
                    is_tgt: false,
                    raw: RawVal::Num(label),
                });
                seq.push(Tok {
                    node: hnode,
                    f2p: pad_parents(&[entity_node]),
                    col: (TASK_TIME_COL.into(), TASK_TABLE.into()),
                    table: TASK_TABLE.into(),
                    sem: SEM_DATETIME,
                    is_tgt: false,
                    raw: RawVal::Date(ts),
                });
                all_labels.push(label);
            }

            // one token per feature cell
            for r in &ctx.rows {
                let mut parents: Vec<i64> = Vec::new();
                for (fk, pid) in &r.parents {
                    if let Some(ptable) = fk_to_parent.get(&r.table).and_then(|m| m.get(fk)) {
                        let pkey = (ptable.clone(), pid.to_string());
                        if let Some(&pn) = node_of.get(&pkey) {
                            parents.push(pn);
                        }
                        continue;
                    }
                    // no schema: link by unique id match within the context
                    if let Some(cands) = by_id.get(&pid.to_string()) {
                        if cands.len() == 1 {
                            if let Some(&pn) = node_of.get(&cands[0]) {
                                parents.push(pn);
                            }
                        }
                    }
                }
                let rnode = node_of[&(r.table.clone(), r.id.to_string())];
                for (col, v) in &r.cells {
                    if seq.len() >= self.max_seq_len {
                        break;
                    }
                    let sem = self.sem_for_cell(&r.table, col, v);
                    let raw = match v {
                        Value::Number(n) => RawVal::Num(*n),
                        Value::Boolean(b) => RawVal::Bool(*b),
                        Value::Datetime(d) => RawVal::Date(*d),
                        Value::Text(s) => RawVal::Text(s.clone()),
                    };
                    seq.push(Tok {
                        node: rnode,
                        f2p: pad_parents(&parents),
                        col: (col.clone(), r.table.clone()),
                        table: r.table.clone(),
                        sem,
                        is_tgt: false,
                        raw,
                    });
                }
            }
            seqs.push(seq);
        }

        let (label_mu, label_sd) = mean_std(&all_labels);
        (seqs, label_mu, label_sd)
    }

    fn forward(
        &mut self,
        model_uri: &str,
        seqs: &[Vec<Tok>],
        label_mu: f64,
        label_sd: f64,
    ) -> Result<Vec<f32>, Error> {
        // per-(col,table) numeric stats + global datetime stats
        let mut num_vals: HashMap<(String, String), Vec<f64>> = HashMap::new();
        let mut dt_vals: Vec<f64> = Vec::new();
        for seq in seqs {
            for tok in seq {
                if tok.is_tgt {
                    continue;
                }
                match &tok.raw {
                    RawVal::Date(d) => dt_vals.push(days(*d)),
                    RawVal::Num(n) => num_vals.entry(tok.col.clone()).or_default().push(*n),
                    RawVal::Bool(b) => {
                        num_vals.entry(tok.col.clone()).or_default().push(if *b { 1.0 } else { 0.0 })
                    }
                    _ => {}
                }
            }
        }
        let mut stats: HashMap<(String, String), (f64, f64)> = HashMap::new();
        for (k, vals) in &num_vals {
            stats.insert(k.clone(), mean_std(vals));
        }
        stats.insert((TASK_LABEL_COL.into(), TASK_TABLE.into()), (label_mu, label_sd));
        let (dt_mu, dt_sd) = mean_std(&dt_vals);

        let b = seqs.len() as i32;
        let s = seqs.iter().map(|q| q.len()).max().unwrap_or(0).max(1) as i32;
        let n = (b as usize) * (s as usize);
        let mut col_vocab: HashMap<(String, String), i64> = HashMap::new();
        let mut tab_vocab: HashMap<String, i64> = HashMap::new();
        let mut node_idxs = vec![0i64; n];
        let mut f2p = vec![-1i64; n * MAX_F2P];
        let mut col_idxs = vec![0i64; n];
        let mut table_idxs = vec![0i64; n];
        let mut is_padding = vec![1u8; n];
        let mut sem_types = vec![0i64; n];
        let mut is_target = vec![0u8; n];
        let mut number_v = vec![0f32; n];
        let mut datetime_v = vec![0f32; n];
        let boolean_v = vec![0f32; n];
        let mut text_v = vec![0f32; n * D_TEXT];
        let mut col_name_v = vec![0f32; n * D_TEXT];

        for (bi, seq) in seqs.iter().enumerate() {
            for (si, tok) in seq.iter().enumerate() {
                let idx = bi * (s as usize) + si;
                node_idxs[idx] = tok.node;
                for k in 0..MAX_F2P {
                    f2p[idx * MAX_F2P + k] = tok.f2p[k];
                }
                let next_col = col_vocab.len() as i64;
                let cid = *col_vocab.entry(tok.col.clone()).or_insert(next_col);
                col_idxs[idx] = cid;
                let next_tab = tab_vocab.len() as i64;
                let tid = *tab_vocab.entry(tok.table.clone()).or_insert(next_tab);
                table_idxs[idx] = tid;
                is_padding[idx] = 0;
                is_target[idx] = if tok.is_tgt { 1 } else { 0 };

                let phrase = format!("{} of {}", tok.col.0, tok.col.1);
                let emb = self.encoder.encode(&phrase);
                for (k, val) in emb.iter().take(D_TEXT).enumerate() {
                    col_name_v[idx * D_TEXT + k] = *val;
                }

                if tok.is_tgt {
                    number_v[idx] = 0.0;
                    sem_types[idx] = SEM_NUMBER;
                    continue;
                }
                match &tok.raw {
                    RawVal::Text(txt) => {
                        let emb = self.encoder.encode(txt);
                        for (k, val) in emb.iter().take(D_TEXT).enumerate() {
                            text_v[idx * D_TEXT + k] = *val;
                        }
                        sem_types[idx] = SEM_TEXT;
                    }
                    RawVal::Date(d) => {
                        datetime_v[idx] = (((days(*d)) - dt_mu) / dt_sd) as f32;
                        sem_types[idx] = SEM_DATETIME;
                    }
                    RawVal::Num(nv) => {
                        let (mu, sd) = stats.get(&tok.col).copied().unwrap_or((0.0, 1.0));
                        number_v[idx] = ((*nv - mu) / sd) as f32;
                        sem_types[idx] = SEM_NUMBER;
                    }
                    RawVal::Bool(bv) => {
                        let (mu, sd) = stats.get(&tok.col).copied().unwrap_or((0.0, 1.0));
                        let x = if *bv { 1.0 } else { 0.0 };
                        number_v[idx] = ((x - mu) / sd) as f32;
                        sem_types[idx] = SEM_NUMBER;
                    }
                    RawVal::Mask => {
                        number_v[idx] = 0.0;
                        sem_types[idx] = SEM_NUMBER;
                    }
                }
            }
        }

        let n_threads = self.n_threads;
        let model = self.model_for(model_uri)?;
        let scores = model
            .forward(
                b, s, &node_idxs, &f2p, &col_idxs, &table_idxs, &is_padding, &sem_types,
                &is_target, &number_v, &datetime_v, &boolean_v, &text_v, &col_name_v,
                n_threads,
            )
            .map_err(Error::from)?;
        Ok(scores)
    }
}

fn pad_parents(parents: &[i64]) -> [i64; MAX_F2P] {
    let mut out = [-1i64; MAX_F2P];
    for (i, p) in parents.iter().take(MAX_F2P).enumerate() {
        out[i] = *p;
    }
    out
}

fn mean_std(vals: &[f64]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 1.0);
    }
    let mu = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / vals.len() as f64;
    (mu, var.sqrt() + 1e-8)
}

impl ModelBackend for RtNativeBackend {
    fn score(
        &mut self,
        query: &ParsedQuery,
        task_type: TaskType,
        contexts: &[EntityContext],
        model_uri: &str,
        _config: &ModelConfig,
    ) -> Result<Vec<EntityPrediction>, Error> {
        // The single-number score head cannot express a class/label set.
        if matches!(
            task_type,
            TaskType::MulticlassClassification | TaskType::MultilabelRanking
        ) {
            return Err(Error::Execution(ExecutionError(
                "the checkpoint's single score head cannot produce multiclass / ranking output"
                    .into(),
            )));
        }
        if contexts.is_empty() {
            return Ok(Vec::new());
        }
        let (seqs, label_mu, label_sd) = self.build_sequences(query, task_type, contexts);
        let scores = self.forward(model_uri, &seqs, label_mu, label_sd)?;
        let mut preds = Vec::new();
        for (ctx, s) in contexts.iter().zip(scores.iter()) {
            let s = *s as f64;
            let mut p = EntityPrediction::new(ctx.entity_id.clone());
            if task_type == TaskType::BinaryClassification {
                p.probability = Some(1.0 / (1.0 + (-s).exp()));
            } else {
                let v = s * label_sd + label_mu;
                p.value = Some(v);
                if task_type == TaskType::Forecasting {
                    let n = query.num_forecasts.unwrap_or(1).max(1) as usize;
                    p.forecast = vec![v; n];
                }
            }
            preds.push(p);
        }
        Ok(preds)
    }
}
