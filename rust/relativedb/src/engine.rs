//! The execution engine: planning, context assembly, model routing.
//!
//! Two traversal strategies ([`SamplerMode`]):
//!
//! * `Retriever` (default) — pull-per-hop through Entity/Link retrievers.
//! * `Csc` — a materialized in-memory CSC index built from TableScanners
//!   ([`crate::csc`]); refresh with [`Engine::refresh`].
//!
//! Both enforce the temporal bound defensively: every row returned by user code
//! is re-checked and dropped if it is newer than the bound (F24 — a buggy
//! retriever must not leak the future into context).

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};

use crate::csc::CscIndex;
use crate::evaluate::{eval_bool, eval_value, EvalValue};
use crate::model::ModelConfig;
use crate::pql::ast::{ParsedQuery, TargetExpr, TaskType, TimeUnit, Window};
use crate::pql::{parse, validate};
use crate::retrieve::{EntityId, RetrieverWiring, Row, TemporalBound, Value};
use crate::schema::{LinkDef, Schema};
use crate::Error;

/// Raised when execution cannot proceed (e.g. FOR EACH with no enumeration).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExecutionError(pub String);

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "execution error: {}", self.0)
    }
}
impl std::error::Error for ExecutionError {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SamplerMode {
    /// pull-per-hop through retrievers (default)
    Retriever,
    /// materialized in-memory CSC index (scanners)
    Csc,
}

/// Context assembly knobs (storage-agnostic).
///
/// `fanouts` are per-hop child caps; when unset, a uniform
/// `bfs_width` per hop is used (RT geometry). `max_context_cells` is the global
/// cell budget.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ContextPolicy {
    pub max_context_cells: usize,
    pub bfs_width: usize,
    pub fanouts: Option<Vec<usize>>,
    pub max_hops: usize,
    pub cohort_size: usize,
    pub prefer_latest: bool,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        ContextPolicy {
            max_context_cells: 8192,
            bfs_width: 32,
            fanouts: None,
            max_hops: 2,
            cohort_size: 0,
            prefer_latest: true,
        }
    }
}

impl ContextPolicy {
    pub fn fanout_at(&self, hop: usize) -> usize {
        match &self.fanouts {
            Some(f) if !f.is_empty() => f[hop.min(f.len() - 1)],
            _ => self.bfs_width,
        }
    }
    pub fn effective_hops(&self) -> usize {
        match &self.fanouts {
            Some(f) if !f.is_empty() => self.max_hops.min(f.len()),
            _ => self.max_hops,
        }
    }
}

/// Execution input (anchor times, per-entity anchor, id override).
#[derive(Clone, Debug, Default)]
pub struct ExecutionInput {
    pub query: String,
    pub parsed: Option<ParsedQuery>,
    /// "now"; None = unbounded
    pub anchor_time: Option<DateTime<Utc>>,
    /// anchor_time="entity" semantics
    pub per_entity_anchor: bool,
    /// decouple context "now"
    pub context_anchor_time: Option<DateTime<Utc>>,
    /// overrides FOR ... IN (...)
    pub entity_ids: Option<Vec<EntityId>>,
    /// `AS OF :param` bindings (name -> timestamp). Consulted when the query's
    /// `AS OF` clause names a `:param`; empty by default.
    pub params: HashMap<String, DateTime<Utc>>,
}

impl ExecutionInput {
    pub fn query(q: impl Into<String>) -> ExecutionInput {
        ExecutionInput { query: q.into(), ..Default::default() }
    }
    pub fn parsed(pq: ParsedQuery) -> ExecutionInput {
        ExecutionInput { parsed: Some(pq), ..Default::default() }
    }
    pub fn anchor_time(mut self, t: DateTime<Utc>) -> Self {
        self.anchor_time = Some(t);
        self
    }
    pub fn per_entity_anchor(mut self, b: bool) -> Self {
        self.per_entity_anchor = b;
        self
    }
    pub fn context_anchor_time(mut self, t: DateTime<Utc>) -> Self {
        self.context_anchor_time = Some(t);
        self
    }
    pub fn entity_ids(mut self, ids: Vec<EntityId>) -> Self {
        self.entity_ids = Some(ids);
        self
    }
    /// Bind a single `AS OF :name` parameter to a timestamp.
    pub fn param(mut self, name: impl Into<String>, t: DateTime<Utc>) -> Self {
        self.params.insert(name.into(), t);
        self
    }
    /// Replace the whole `AS OF :param` binding map.
    pub fn params(mut self, params: HashMap<String, DateTime<Utc>>) -> Self {
        self.params = params;
        self
    }
}

/// The assembled per-entity context: seed entity row + traversed rows.
#[derive(Clone, Debug)]
pub struct EntityContext {
    pub entity_id: EntityId,
    pub anchor: Option<DateTime<Utc>>,
    pub rows: Vec<Row>,
}

impl EntityContext {
    pub fn row_keys(&self) -> HashSet<(String, EntityId)> {
        self.rows.iter().map(|r| r.key()).collect()
    }
    pub fn cell_count(&self) -> usize {
        self.rows
            .iter()
            .map(|r| r.cells.len() + if r.timestamp.is_some() { 1 } else { 0 })
            .sum()
    }
    pub fn rows_by_table(&self) -> HashMap<String, Vec<Row>> {
        let mut out: HashMap<String, Vec<Row>> = HashMap::new();
        for r in &self.rows {
            out.entry(r.table.clone()).or_default().push(r.clone());
        }
        out
    }
    pub fn entity_cells(&self, entity_table: &str) -> Vec<(String, Value)> {
        for r in &self.rows {
            if r.table == entity_table && r.id == self.entity_id {
                return r.cells.clone();
            }
        }
        Vec::new()
    }
}

/// A single ranked recommendation item (stringified value / FK id).
pub type RankedItem = String;

#[derive(Clone, Debug)]
pub struct EntityPrediction {
    pub id: EntityId,
    pub value: Option<f64>,
    pub probability: Option<f64>,
    pub class_probs: Vec<(String, f64)>,
    pub ranked: Vec<RankedItem>,
    pub forecast: Vec<f64>,
    /// The hard label for `RETURN CLASS`.
    pub predicted_class: Option<String>,
    /// Ordered `(q, value)` pairs for `RETURN QUANTILES`.
    pub quantiles: Vec<(f64, f64)>,
    /// `(lower, upper)` for `RETURN INTERVAL`.
    pub interval: Option<(f64, f64)>,
}

impl EntityPrediction {
    pub fn new(id: EntityId) -> EntityPrediction {
        EntityPrediction {
            id,
            value: None,
            probability: None,
            class_probs: Vec::new(),
            ranked: Vec::new(),
            forecast: Vec::new(),
            predicted_class: None,
            quantiles: Vec::new(),
            interval: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PredictionResult {
    pub task_type: TaskType,
    pub predictions: Vec<EntityPrediction>,
    pub model_uri: String,
}

/// Anything that can score assembled contexts. The built-in
/// [`HistoryBaselineBackend`] is a model-free reference; real backends load the
/// checkpoint at `model_uri` (routed by task type).
pub trait ModelBackend {
    fn score(
        &mut self,
        query: &ParsedQuery,
        task_type: TaskType,
        contexts: &[EntityContext],
        model_uri: &str,
        config: &ModelConfig,
    ) -> Result<Vec<EntityPrediction>, Error>;
}

// ---------------------------------------------------------------------------
// Sampler: the two traversal strategies behind one surface
// ---------------------------------------------------------------------------

enum Sampler<'a> {
    Retriever { wiring: &'a RetrieverWiring },
    Csc(&'a CscIndex),
}

impl<'a> Sampler<'a> {
    fn entities(&self, table: &str, ids: &[EntityId], bound: &TemporalBound) -> Result<Vec<Row>, Error> {
        match self {
            Sampler::Retriever { wiring, .. } => {
                Ok(wiring.entity_retriever(table)?.fetch_by_ids(table, ids, bound))
            }
            Sampler::Csc(idx) => Ok(idx.entities(table, ids, bound)),
        }
    }
    fn children(
        &self,
        link: &LinkDef,
        parent_id: &EntityId,
        bound: &TemporalBound,
        limit: usize,
    ) -> Result<Vec<Row>, Error> {
        match self {
            Sampler::Retriever { wiring, .. } => Ok(wiring
                .link_retriever(&link.from_table)?
                .fetch_children(link, parent_id, bound, limit)),
            Sampler::Csc(idx) => Ok(idx.children(link, parent_id, bound, limit)),
        }
    }
    fn cohort(
        &self,
        table: &str,
        anchor: &EntityId,
        bound: &TemporalBound,
        limit: usize,
    ) -> Vec<EntityId> {
        match self {
            Sampler::Retriever { wiring, .. } => match wiring.cohort_retriever(table) {
                Some(r) => r.fetch_cohort(table, anchor, bound, limit),
                None => Vec::new(),
            },
            Sampler::Csc(idx) => idx.cohort(table, anchor, bound, limit),
        }
    }
    fn all_ids(&self, table: &str) -> Option<Vec<EntityId>> {
        match self {
            Sampler::Retriever { wiring, .. } => {
                if wiring.scanners.contains_key(table) {
                    let s = wiring.table_scanner(table).ok()?;
                    Some(s.scan(table, &TemporalBound::unbounded()).into_iter().map(|r| r.id).collect())
                } else {
                    None
                }
            }
            Sampler::Csc(idx) => Some(idx.all_ids(table)),
        }
    }
}

fn newest_first_key(r: &Row) -> (bool, f64) {
    (
        r.timestamp.is_none(),
        -(r.timestamp.map(|t| t.timestamp() as f64).unwrap_or(0.0)),
    )
}

fn admit(
    rows: Vec<Row>,
    bound: &TemporalBound,
    visited: &mut HashSet<(String, EntityId)>,
    out: &mut Vec<Row>,
) -> Vec<Row> {
    let mut fresh = Vec::new();
    for r in rows {
        if !bound.admits_row(&r) {
            continue; // defensive leakage guard (F24)
        }
        let key = r.key();
        if visited.contains(&key) {
            continue;
        }
        visited.insert(key);
        fresh.push(r.clone());
        out.push(r);
    }
    fresh
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct Engine {
    pub schema: Schema,
    pub wiring: RetrieverWiring,
    pub model_config: ModelConfig,
    pub model_backend: Box<dyn ModelBackend>,
    pub context_policy: ContextPolicy,
    pub sampler_mode: SamplerMode,
    csc_index: Option<CscIndex>,
}

impl Engine {
    pub fn new(schema: Schema, wiring: RetrieverWiring) -> Engine {
        Engine {
            schema,
            wiring,
            model_config: ModelConfig::defaults(),
            model_backend: Box::new(HistoryBaselineBackend::new(3)),
            context_policy: ContextPolicy::default(),
            sampler_mode: SamplerMode::Retriever,
            csc_index: None,
        }
    }

    pub fn model_config(mut self, c: ModelConfig) -> Engine {
        self.model_config = c;
        self
    }
    pub fn model_backend(mut self, b: Box<dyn ModelBackend>) -> Engine {
        self.model_backend = b;
        self
    }
    pub fn context_policy(mut self, p: ContextPolicy) -> Engine {
        self.context_policy = p;
        self
    }
    pub fn sampler_mode(mut self, m: SamplerMode) -> Engine {
        self.sampler_mode = m;
        self
    }

    /// Finalize: build the CSC snapshot if in CSC mode. Call after configuring.
    pub fn build(mut self) -> Result<Engine, Error> {
        if self.sampler_mode == SamplerMode::Csc {
            self.refresh()?;
        }
        Ok(self)
    }

    /// (Re)build the CSC snapshot from the wired TableScanners.
    pub fn refresh(&mut self) -> Result<(), Error> {
        self.csc_index = Some(CscIndex::build(&self.schema, &self.wiring)?);
        Ok(())
    }

    fn sampler(&self) -> Sampler<'_> {
        match self.sampler_mode {
            SamplerMode::Csc => Sampler::Csc(
                self.csc_index
                    .as_ref()
                    .expect("CSC index not built; call build()/refresh() first"),
            ),
            SamplerMode::Retriever => Sampler::Retriever { wiring: &self.wiring },
        }
    }

    // -- context assembly ---------------------------------------------------

    /// The hop loop: seed -> parents (always) -> children (fanout-capped,
    /// newest-first), every row re-checked against the temporal bound.
    pub fn assemble_context(
        &self,
        entity_table: &str,
        entity_id: &EntityId,
        anchor: Option<DateTime<Utc>>,
    ) -> Result<EntityContext, Error> {
        self.assemble_context_with(entity_table, entity_id, anchor, &self.context_policy)
    }

    pub fn assemble_context_with(
        &self,
        entity_table: &str,
        entity_id: &EntityId,
        anchor: Option<DateTime<Utc>>,
        policy: &ContextPolicy,
    ) -> Result<EntityContext, Error> {
        let sampler = self.sampler();
        let bound = match anchor {
            Some(a) => TemporalBound::at_or_before(a),
            None => TemporalBound::unbounded(),
        };
        let mut ctx = EntityContext { entity_id: entity_id.clone(), anchor: bound.as_of, rows: Vec::new() };
        let mut visited: HashSet<(String, EntityId)> = HashSet::new();

        let seed = admit(
            sampler.entities(entity_table, &[entity_id.clone()], &bound)?,
            &bound,
            &mut visited,
            &mut ctx.rows,
        );
        if seed.is_empty() {
            return Ok(ctx);
        }
        let mut frontier: Vec<Row> = seed;

        // optional cohort seeds (similar entities, Tier 1)
        if policy.cohort_size > 0 {
            let cohort_ids = sampler.cohort(entity_table, entity_id, &bound, policy.cohort_size);
            if !cohort_ids.is_empty() {
                let extra = admit(
                    sampler.entities(entity_table, &cohort_ids, &bound)?,
                    &bound,
                    &mut visited,
                    &mut ctx.rows,
                );
                frontier.extend(extra);
            }
        }

        // fk_to_parent: table -> (fk_column -> parent_table)
        let mut fk_to_parent: HashMap<String, HashMap<String, String>> = HashMap::new();
        for t in &self.schema.tables {
            let m: HashMap<String, String> = self
                .schema
                .links_from(&t.name)
                .iter()
                .map(|l| (l.fk_column.clone(), l.to_table.clone()))
                .collect();
            fk_to_parent.insert(t.name.clone(), m);
        }

        for hop in 0..policy.effective_hops() {
            if ctx.cell_count() >= policy.max_context_cells {
                break;
            }
            let fanout = policy.fanout_at(hop);
            let mut next_frontier: Vec<Row> = Vec::new();

            // parents: always followed, batched per table (insertion-ordered)
            let mut wanted_order: Vec<String> = Vec::new();
            let mut wanted: HashMap<String, Vec<EntityId>> = HashMap::new();
            for row in &frontier {
                for (fk, pid) in &row.parents {
                    if let Some(ptable) = fk_to_parent.get(&row.table).and_then(|m| m.get(fk)) {
                        let key = (ptable.clone(), pid.clone());
                        if !visited.contains(&key) {
                            if !wanted.contains_key(ptable) {
                                wanted_order.push(ptable.clone());
                            }
                            wanted.entry(ptable.clone()).or_default().push(pid.clone());
                        }
                    }
                }
            }
            for ptable in &wanted_order {
                let pids = &wanted[ptable];
                let got = admit(
                    sampler.entities(ptable, pids, &bound)?,
                    &bound,
                    &mut visited,
                    &mut ctx.rows,
                );
                next_frontier.extend(got);
            }

            // children: width-bounded, newest-first
            for row in &frontier {
                for link in self.schema.links_to(&row.table) {
                    let mut kids: Vec<Row> = sampler
                        .children(link, &row.id, &bound, fanout)?
                        .into_iter()
                        .filter(|k| bound.admits_row(k))
                        .collect();
                    if policy.prefer_latest {
                        kids.sort_by(|a, b| {
                            let ka = newest_first_key(a);
                            let kb = newest_first_key(b);
                            ka.0.cmp(&kb.0).then(
                                ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal),
                            )
                        });
                    }
                    kids.truncate(fanout);
                    let got = admit(kids, &bound, &mut visited, &mut ctx.rows);
                    next_frontier.extend(got);
                }
                if ctx.cell_count() >= policy.max_context_cells {
                    break;
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }
        Ok(ctx)
    }

    // -- execution ----------------------------------------------------------

    pub fn execute(&mut self, input: ExecutionInput) -> Result<PredictionResult, Error> {
        let pq: ParsedQuery = match input.parsed.clone() {
            Some(p) => p,
            None => parse(&input.query)?,
        };
        validate(&pq, &self.schema)?;
        if pq.explain.is_some() {
            return Err(Error::Execution(ExecutionError(
                "this is an EXPLAIN query — call explain() instead of execute()".into(),
            )));
        }
        let task_type = pq.task_type(Some(&self.schema));
        let model_uri = self.model_config.model_uri_for(task_type).to_string();
        // AS OF: bind the effective anchor before any assembly.
        let eff_input = self.effective_input(&pq, &input)?;
        let contexts = self.assemble_all(&pq, &eff_input)?;
        let preds = self
            .model_backend
            .score(&pq, task_type, &contexts, &model_uri, &self.model_config)?;
        Ok(PredictionResult { task_type, predictions: preds, model_uri })
    }

    /// Resolve, assemble and WHERE-filter the per-entity contexts. `input`'s
    /// `anchor_time` is assumed to already be the effective (AS OF-bound) anchor.
    fn assemble_all(
        &self,
        pq: &ParsedQuery,
        input: &ExecutionInput,
    ) -> Result<Vec<EntityContext>, Error> {
        let entity_table = pq.entity_key.table.clone();
        let ids = self.resolve_entity_ids(pq, input)?;
        let mut contexts: Vec<EntityContext> = Vec::new();
        for eid in ids {
            let anchor = self.anchor_for(&entity_table, &eid, input)?;
            let ctx = self.assemble_context(&entity_table, &eid, anchor)?;
            if let Some(w) = &pq.where_ {
                let ok = eval_bool(
                    w,
                    &ctx.rows_by_table(),
                    &ctx.entity_cells(&entity_table),
                    ctx.anchor,
                );
                if !ok {
                    continue;
                }
            }
            contexts.push(ctx);
        }
        Ok(contexts)
    }

    /// A copy of `input` whose `anchor_time` is the effective AS OF anchor
    /// (see [`Engine::effective_anchor`]).
    fn effective_input(
        &self,
        pq: &ParsedQuery,
        input: &ExecutionInput,
    ) -> Result<ExecutionInput, Error> {
        let eff = self.effective_anchor(pq, input)?;
        let mut out = input.clone();
        out.anchor_time = eff;
        Ok(out)
    }

    /// Resolve the query's `AS OF` clause against `ExecutionInput` (contract
    /// Part A):
    ///
    /// * absent / `AS OF NOW` -> the execution anchor (`input.anchor_time`);
    /// * `AS OF <date>` -> the parsed date (UTC), OVERRIDING `anchor_time`;
    /// * `AS OF :param` -> the value bound in `input.params`, falling back to
    ///   `anchor_time`; a clear error if neither is present.
    fn effective_anchor(
        &self,
        pq: &ParsedQuery,
        input: &ExecutionInput,
    ) -> Result<Option<DateTime<Utc>>, Error> {
        let as_of = match &pq.as_of {
            None => return Ok(input.anchor_time),
            Some(a) => a,
        };
        match as_of.kind.as_str() {
            "now" => Ok(input.anchor_time),
            "date" => {
                let s = as_of.value.as_deref().ok_or_else(|| {
                    Error::Execution(ExecutionError("AS OF <date> is missing its value".into()))
                })?;
                Ok(Some(parse_as_of_date(s)?))
            }
            "param" => {
                let name = as_of.value.as_deref().ok_or_else(|| {
                    Error::Execution(ExecutionError("AS OF :param is missing its name".into()))
                })?;
                if let Some(v) = input.params.get(name) {
                    Ok(Some(*v))
                } else if let Some(t) = input.anchor_time {
                    Ok(Some(t))
                } else {
                    Err(Error::Execution(ExecutionError(format!(
                        "AS OF :{} is unbound: supply it in ExecutionInput.params, or set \
                         anchor_time as a fallback",
                        name
                    ))))
                }
            }
            other => Err(Error::Execution(ExecutionError(format!(
                "unknown AS OF kind {:?}",
                other
            )))),
        }
    }

    /// Convenience: execute a RelQL string with an anchor time.
    pub fn execute_query(
        &mut self,
        query: &str,
        anchor: Option<DateTime<Utc>>,
    ) -> Result<PredictionResult, Error> {
        let mut input = ExecutionInput::query(query);
        input.anchor_time = anchor;
        self.execute(input)
    }

    fn resolve_entity_ids(
        &self,
        pq: &ParsedQuery,
        input: &ExecutionInput,
    ) -> Result<Vec<EntityId>, Error> {
        if let Some(ids) = &input.entity_ids {
            return Ok(ids.clone());
        }
        if !pq.entity_ids.is_empty() {
            return Ok(pq.entity_ids.iter().map(literal_to_entity_id).collect());
        }
        let ids = self.sampler().all_ids(&pq.entity_key.table);
        ids.ok_or_else(|| {
            Error::Execution(ExecutionError(format!(
                "FOR EACH over all {:?} entities needs either explicit entity_ids, a pinned \
                 FOR ... IN (...) selector, or a TableScanner wired for the entity table \
                 (retrievers alone cannot enumerate a table)",
                pq.entity_key.table
            )))
        })
    }

    fn anchor_for(
        &self,
        entity_table: &str,
        entity_id: &EntityId,
        input: &ExecutionInput,
    ) -> Result<Option<DateTime<Utc>>, Error> {
        let anchor = input.context_anchor_time.or(input.anchor_time);
        if input.per_entity_anchor {
            let rows = self.sampler().entities(
                entity_table,
                &[entity_id.clone()],
                &TemporalBound::unbounded(),
            )?;
            if let Some(r) = rows.first() {
                if r.timestamp.is_some() {
                    return Ok(r.timestamp);
                }
            }
        }
        Ok(anchor)
    }
}

// ---------------------------------------------------------------------------
// EXPLAIN
// ---------------------------------------------------------------------------

/// One framed aggregation surfaced in an [`ExplainPlan`].
#[derive(Clone, Debug)]
pub struct WindowInfo {
    pub table: String,
    pub time_column: Option<String>,
    pub start: f64,
    pub end: f64,
    pub unit: TimeUnit,
    pub horizons: i64,
    pub step: Option<f64>,
    /// `target` | `where` | `assuming`.
    pub role: String,
}

/// The seed-entity descriptor in an [`ExplainPlan`].
#[derive(Clone, Debug)]
pub struct EntityPlan {
    pub table: String,
    pub pk: String,
    /// `"FOR EACH"` or the explicit id list rendered as text.
    pub selector: String,
}

/// A declared-but-not-applied `ABLATE` clause.
#[derive(Clone, Debug)]
pub struct AblationPlan {
    pub name: String,
    pub note: String,
}

/// How the effective anchor was bound (contract Part B, `as_of`).
#[derive(Clone, Debug)]
pub struct AsOfPlan {
    /// `query-date` | `query-param` | `execution-anchor`.
    pub source: String,
    pub value: Option<DateTime<Utc>>,
}

/// The parse+validate plan, computed WITHOUT scoring.
#[derive(Clone, Debug)]
pub struct ExplainPlan {
    pub target: String,
    pub task_type: TaskType,
    pub entity: EntityPlan,
    pub output: String,
    pub windows: Vec<WindowInfo>,
    pub where_present: bool,
    pub assuming_present: bool,
    pub as_of: AsOfPlan,
    pub ablations: Vec<AblationPlan>,
    pub warnings: Vec<String>,
}

/// Per-table assembly statistics (CONTEXT / ANALYZE).
#[derive(Clone, Debug, Default)]
pub struct TableStats {
    pub rows: usize,
    pub cells: usize,
    pub min_time: Option<DateTime<Utc>>,
    pub max_time: Option<DateTime<Utc>>,
}

/// The result of assembling context under the effective anchor, without scoring.
#[derive(Clone, Debug)]
pub struct ExplainContext {
    pub anchor: Option<DateTime<Utc>>,
    pub entities_covered: usize,
    pub total_rows: usize,
    pub total_cells: usize,
    pub per_table: BTreeMap<String, TableStats>,
    /// Non-seed rows admitted via a link traversal.
    pub links_traversed: usize,
    /// Rows a scanner surfaced that the temporal bound rejected (best-effort;
    /// only computed for tables with a wired scanner).
    pub rows_rejected: usize,
    /// Schema tables that produced no rows in any entity's context.
    pub tables_unreachable: Vec<String>,
}

/// The full EXPLAIN payload; render with [`ExplainResult::render`].
#[derive(Clone, Debug)]
pub struct ExplainResult {
    /// `PLAN` | `CONTEXT` | `ANALYZE` | `ABLATION`.
    pub mode: String,
    /// `TEXT` | `JSON`.
    pub format: String,
    pub plan: ExplainPlan,
    pub context: Option<ExplainContext>,
    pub predictions: Option<PredictionResult>,
}

impl ExplainResult {
    /// Render per `format`: an indented multi-section TEXT dump, or a stable
    /// snake_case JSON object.
    pub fn render(&self) -> String {
        if self.format.eq_ignore_ascii_case("JSON") {
            self.render_json()
        } else {
            self.render_text()
        }
    }

    fn render_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("EXPLAIN {} (FORMAT {})\n", self.mode, self.format));
        let p = &self.plan;
        s.push_str("PLAN\n");
        s.push_str(&format!("  target:    {}\n", p.target));
        s.push_str(&format!("  task_type: {}\n", task_type_name(p.task_type)));
        s.push_str(&format!(
            "  entity:    {} (pk {}) {}\n",
            p.entity.table, p.entity.pk, p.entity.selector
        ));
        s.push_str(&format!("  output:    {}\n", p.output));
        if p.windows.is_empty() {
            s.push_str("  windows:   (none)\n");
        } else {
            s.push_str("  windows:\n");
            for w in &p.windows {
                s.push_str(&format!(
                    "    - [{}] {}{} ({} .. {} {}, horizons {}{})\n",
                    w.role,
                    w.table,
                    w.time_column
                        .as_ref()
                        .map(|c| format!(".{}", c))
                        .unwrap_or_default(),
                    fmt_f(w.start),
                    fmt_f(w.end),
                    time_unit_name(w.unit),
                    w.horizons,
                    w.step.map(|s| format!(", step {}", fmt_f(s))).unwrap_or_default(),
                ));
            }
        }
        s.push_str(&format!("  where:     {}\n", if p.where_present { "present" } else { "absent" }));
        s.push_str(&format!(
            "  assuming:  {}\n",
            if p.assuming_present { "carried, not applied" } else { "absent" }
        ));
        s.push_str(&format!(
            "  as_of:     source={} value={}\n",
            p.as_of.source,
            p.as_of.value.map(|v| v.to_rfc3339()).unwrap_or_else(|| "unbounded".into())
        ));
        if !p.ablations.is_empty() {
            s.push_str("  ablations:\n");
            for a in &p.ablations {
                s.push_str(&format!("    - {} ({})\n", a.name, a.note));
            }
        }
        if !p.warnings.is_empty() {
            s.push_str("  warnings:\n");
            for w in &p.warnings {
                s.push_str(&format!("    - {}\n", w));
            }
        }
        if let Some(c) = &self.context {
            s.push_str("CONTEXT\n");
            s.push_str(&format!(
                "  anchor:            {}\n",
                c.anchor.map(|v| v.to_rfc3339()).unwrap_or_else(|| "unbounded".into())
            ));
            s.push_str(&format!("  entities_covered:  {}\n", c.entities_covered));
            s.push_str(&format!("  total_rows:        {}\n", c.total_rows));
            s.push_str(&format!("  total_cells:       {}\n", c.total_cells));
            s.push_str(&format!("  links_traversed:   {}\n", c.links_traversed));
            s.push_str(&format!("  rows_rejected:     {}\n", c.rows_rejected));
            if !c.tables_unreachable.is_empty() {
                s.push_str(&format!("  tables_unreachable: {}\n", c.tables_unreachable.join(", ")));
            }
            s.push_str("  per_table:\n");
            for (t, st) in &c.per_table {
                s.push_str(&format!(
                    "    - {}: rows={} cells={} min_time={} max_time={}\n",
                    t,
                    st.rows,
                    st.cells,
                    st.min_time.map(|v| v.to_rfc3339()).unwrap_or_else(|| "-".into()),
                    st.max_time.map(|v| v.to_rfc3339()).unwrap_or_else(|| "-".into()),
                ));
            }
        }
        if let Some(pr) = &self.predictions {
            s.push_str("PREDICTIONS\n");
            s.push_str(&format!("  model_uri:   {}\n", pr.model_uri));
            s.push_str(&format!("  predictions: {}\n", pr.predictions.len()));
        }
        s
    }

    fn render_json(&self) -> String {
        let p = &self.plan;
        let mut out = String::new();
        out.push('{');
        out.push_str(&format!("\"mode\":{},", json_str(&self.mode)));
        out.push_str(&format!("\"format\":{},", json_str(&self.format)));
        // plan
        out.push_str("\"plan\":{");
        out.push_str(&format!("\"target\":{},", json_str(&p.target)));
        out.push_str(&format!("\"task_type\":{},", json_str(task_type_name(p.task_type))));
        out.push_str(&format!(
            "\"entity\":{{\"table\":{},\"pk\":{},\"selector\":{}}},",
            json_str(&p.entity.table),
            json_str(&p.entity.pk),
            json_str(&p.entity.selector)
        ));
        out.push_str(&format!("\"output\":{},", json_str(&p.output)));
        out.push_str("\"windows\":[");
        for (i, w) in p.windows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"table\":{},\"time_column\":{},\"start\":{},\"end\":{},\"unit\":{},\"horizons\":{},\"step\":{},\"role\":{}}}",
                json_str(&w.table),
                w.time_column.as_ref().map(|c| json_str(c)).unwrap_or_else(|| "null".into()),
                json_num(w.start),
                json_num(w.end),
                json_str(time_unit_name(w.unit)),
                w.horizons,
                w.step.map(json_num).unwrap_or_else(|| "null".into()),
                json_str(&w.role),
            ));
        }
        out.push_str("],");
        out.push_str(&format!("\"where_present\":{},", p.where_present));
        out.push_str(&format!("\"assuming_present\":{},", p.assuming_present));
        out.push_str(&format!(
            "\"as_of\":{{\"source\":{},\"value\":{}}},",
            json_str(&p.as_of.source),
            p.as_of.value.map(|v| json_str(&v.to_rfc3339())).unwrap_or_else(|| "null".into())
        ));
        out.push_str("\"ablations\":[");
        for (i, a) in p.ablations.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"table\":{},\"note\":{}}}",
                json_str(&a.name),
                json_str(&a.note)
            ));
        }
        out.push_str("],");
        out.push_str("\"warnings\":[");
        for (i, w) in p.warnings.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&json_str(w));
        }
        out.push_str("]");
        out.push('}'); // plan
        // context
        match &self.context {
            None => out.push_str(",\"context\":null"),
            Some(c) => {
                out.push_str(",\"context\":{");
                out.push_str(&format!(
                    "\"anchor\":{},",
                    c.anchor.map(|v| json_str(&v.to_rfc3339())).unwrap_or_else(|| "null".into())
                ));
                out.push_str(&format!("\"entities_covered\":{},", c.entities_covered));
                out.push_str(&format!("\"total_rows\":{},", c.total_rows));
                out.push_str(&format!("\"total_cells\":{},", c.total_cells));
                out.push_str(&format!("\"links_traversed\":{},", c.links_traversed));
                out.push_str(&format!("\"rows_rejected\":{},", c.rows_rejected));
                out.push_str("\"tables_unreachable\":[");
                for (i, t) in c.tables_unreachable.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&json_str(t));
                }
                out.push_str("],");
                out.push_str("\"tables\":{");
                for (i, (t, st)) in c.per_table.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&format!(
                        "{}:{{\"rows\":{},\"cells\":{},\"min_time\":{},\"max_time\":{}}}",
                        json_str(t),
                        st.rows,
                        st.cells,
                        st.min_time.map(|v| json_str(&v.to_rfc3339())).unwrap_or_else(|| "null".into()),
                        st.max_time.map(|v| json_str(&v.to_rfc3339())).unwrap_or_else(|| "null".into()),
                    ));
                }
                out.push_str("}}"); // per_table, context
            }
        }
        // predictions (array of per-entity results, for cross-language parity)
        match &self.predictions {
            None => out.push_str(",\"predictions\":null"),
            Some(pr) => {
                out.push_str(&format!(",\"model_uri\":{}", json_str(&pr.model_uri)));
                out.push_str(",\"predictions\":[");
                for (i, p) in pr.predictions.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&format!(
                        "{{\"id\":{},\"value\":{},\"probability\":{},\"class\":{},\"ranked\":[{}],\"forecast\":[{}]}}",
                        json_str(&p.id.to_string()),
                        p.value.map(json_num).unwrap_or_else(|| "null".into()),
                        p.probability.map(json_num).unwrap_or_else(|| "null".into()),
                        p.predicted_class.as_ref().map(|c| json_str(c)).unwrap_or_else(|| "null".into()),
                        p.ranked.iter().map(|r| json_str(r)).collect::<Vec<_>>().join(","),
                        p.forecast.iter().map(|f| json_num(*f)).collect::<Vec<_>>().join(","),
                    ));
                }
                out.push(']');
            }
        }
        out.push('}');
        out
    }
}

impl Engine {
    /// Compute an [`ExplainResult`] for `input` (contract Part B). PLAN never
    /// assembles context or scores; CONTEXT assembles but does not score;
    /// ANALYZE assembles and scores; ABLATION returns PLAN with a warning
    /// (ablation is intentionally not implemented). A non-EXPLAIN query is
    /// explained as PLAN.
    pub fn explain(&mut self, input: ExecutionInput) -> Result<ExplainResult, Error> {
        let pq: ParsedQuery = match input.parsed.clone() {
            Some(p) => p,
            None => parse(&input.query)?,
        };
        validate(&pq, &self.schema)?;

        let (mode, format) = match &pq.explain {
            Some(e) => (e.mode.to_uppercase(), e.format.to_uppercase()),
            None => ("PLAN".to_string(), "TEXT".to_string()),
        };
        let eff_input = self.effective_input(&pq, &input)?;
        let mut plan = self.build_plan(&pq, &input, eff_input.anchor_time);

        let mut result = ExplainResult {
            mode: mode.clone(),
            format,
            plan: plan.clone(),
            context: None,
            predictions: None,
        };

        match mode.as_str() {
            "CONTEXT" => {
                let contexts = self.assemble_all(&pq, &eff_input)?;
                result.context = Some(self.context_stats(&contexts, eff_input.anchor_time));
            }
            "ANALYZE" => {
                let task_type = pq.task_type(Some(&self.schema));
                let model_uri = self.model_config.model_uri_for(task_type).to_string();
                let contexts = self.assemble_all(&pq, &eff_input)?;
                result.context = Some(self.context_stats(&contexts, eff_input.anchor_time));
                let preds = self.model_backend.score(
                    &pq,
                    task_type,
                    &contexts,
                    &model_uri,
                    &self.model_config,
                )?;
                result.predictions =
                    Some(PredictionResult { task_type, predictions: preds, model_uri });
            }
            "ABLATION" => {
                plan.warnings.push("ablation not implemented".to_string());
                result.plan = plan;
            }
            // PLAN (and any unknown mode): parse+validate only.
            _ => {}
        }
        Ok(result)
    }

    fn build_plan(
        &self,
        pq: &ParsedQuery,
        input: &ExecutionInput,
        eff_anchor: Option<DateTime<Utc>>,
    ) -> ExplainPlan {
        let task_type = pq.task_type(Some(&self.schema));
        let entity = EntityPlan {
            table: pq.entity_key.table.clone(),
            pk: pq.entity_key.column.clone(),
            selector: self.render_selector(pq, input),
        };
        let output = output_form(pq, task_type);
        let windows = self.collect_windows(pq);
        let as_of = self.plan_as_of(pq, eff_anchor);

        let ablations: Vec<AblationPlan> = pq
            .ablations
            .iter()
            .map(|a| AblationPlan { name: a.name.clone(), note: "declared, not applied".into() })
            .collect();

        let mut warnings = Vec::new();
        if pq.assuming.is_some() {
            warnings.push("ASSUMING is carried but not applied".to_string());
        }
        if !pq.ablations.is_empty() {
            warnings.push("ABLATE clauses are declared but not applied".to_string());
        }

        ExplainPlan {
            target: render_expr(&pq.target),
            task_type,
            entity,
            output,
            windows,
            where_present: pq.where_.is_some(),
            assuming_present: pq.assuming.is_some(),
            as_of,
            ablations,
            warnings,
        }
    }

    fn render_selector(&self, pq: &ParsedQuery, input: &ExecutionInput) -> String {
        if let Some(ids) = &input.entity_ids {
            let list: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
            return format!("IN ({})", list.join(", "));
        }
        if pq.entity_ids.is_empty() {
            "FOR EACH".to_string()
        } else {
            let list: Vec<String> = pq.entity_ids.iter().map(render_lit).collect();
            format!("IN ({})", list.join(", "))
        }
    }

    fn collect_windows(&self, pq: &ParsedQuery) -> Vec<WindowInfo> {
        let mut out = Vec::new();
        let push = |expr: &TargetExpr, role: &str, out: &mut Vec<WindowInfo>| {
            for a in expr.aggregations() {
                if let Some(w) = a.window {
                    let time_column = self
                        .schema
                        .table(&a.column.table)
                        .and_then(|t| t.time_column.clone());
                    out.push(WindowInfo {
                        table: a.column.table.clone(),
                        time_column,
                        start: w.start,
                        end: w.end,
                        unit: w.unit,
                        horizons: w.horizons,
                        step: w.step,
                        role: role.to_string(),
                    });
                }
            }
        };
        push(&pq.target, "target", &mut out);
        if let Some(w) = &pq.where_ {
            push(w, "where", &mut out);
        }
        if let Some(a) = &pq.assuming {
            push(a, "assuming", &mut out);
        }
        out
    }

    fn plan_as_of(&self, pq: &ParsedQuery, eff_anchor: Option<DateTime<Utc>>) -> AsOfPlan {
        let source = match &pq.as_of {
            Some(a) if a.kind == "date" => "query-date",
            Some(a) if a.kind == "param" => "query-param",
            _ => "execution-anchor",
        };
        AsOfPlan { source: source.to_string(), value: eff_anchor }
    }

    fn context_stats(
        &self,
        contexts: &[EntityContext],
        anchor: Option<DateTime<Utc>>,
    ) -> ExplainContext {
        let mut per_table: BTreeMap<String, TableStats> = BTreeMap::new();
        let mut total_rows = 0usize;
        let mut total_cells = 0usize;
        let mut links_traversed = 0usize;

        for ctx in contexts {
            let seed_table = ctx.rows.first().map(|r| r.table.clone());
            for r in &ctx.rows {
                let st = per_table.entry(r.table.clone()).or_default();
                st.rows += 1;
                let cells = r.cells.len() + if r.timestamp.is_some() { 1 } else { 0 };
                st.cells += cells;
                total_rows += 1;
                total_cells += cells;
                if let Some(ts) = r.timestamp {
                    st.min_time = Some(st.min_time.map_or(ts, |m| m.min(ts)));
                    st.max_time = Some(st.max_time.map_or(ts, |m| m.max(ts)));
                }
                // A non-seed row was reached by a link traversal.
                let is_seed =
                    seed_table.as_deref() == Some(r.table.as_str()) && r.id == ctx.entity_id;
                if !is_seed {
                    links_traversed += 1;
                }
            }
        }

        let tables_unreachable: Vec<String> = self
            .schema
            .tables
            .iter()
            .map(|t| t.name.clone())
            .filter(|n| !per_table.contains_key(n))
            .collect();

        let touched: Vec<String> = per_table.keys().cloned().collect();
        let rows_rejected = self.count_rejected_by_bound(&touched, anchor);

        ExplainContext {
            anchor,
            entities_covered: contexts.iter().filter(|c| !c.rows.is_empty()).count(),
            total_rows,
            total_cells,
            per_table,
            links_traversed,
            rows_rejected,
            tables_unreachable,
        }
    }

    /// Best-effort count of rows a wired scanner surfaces that the temporal
    /// bound would reject. Only computed in Retriever mode for tables with a
    /// scanner; returns 0 when the anchor is unbounded or no scanner exists.
    fn count_rejected_by_bound(&self, tables: &[String], anchor: Option<DateTime<Utc>>) -> usize {
        let a = match anchor {
            Some(a) => a,
            None => return 0,
        };
        if self.sampler_mode != SamplerMode::Retriever {
            return 0;
        }
        let mut n = 0usize;
        for t in tables {
            if !self.wiring.scanners.contains_key(t) {
                continue;
            }
            if let Ok(scanner) = self.wiring.table_scanner(t) {
                for r in scanner.scan(t, &TemporalBound::unbounded()) {
                    if let Some(ts) = r.timestamp {
                        if ts > a {
                            n += 1;
                        }
                    }
                }
            }
        }
        n
    }
}

/// Parse an `AS OF <date>` value: `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`, or RFC3339.
fn parse_as_of_date(s: &str) -> Result<DateTime<Utc>, Error> {
    let s = s.trim();
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Ok(dt.and_utc());
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc());
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }
    Err(Error::Execution(ExecutionError(format!(
        "AS OF date {:?} is not parseable (expected YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
        s
    ))))
}

fn task_type_name(t: TaskType) -> &'static str {
    match t {
        TaskType::Regression => "regression",
        TaskType::BinaryClassification => "binary_classification",
        TaskType::MulticlassClassification => "multiclass_classification",
        TaskType::MultilabelRanking => "multilabel_ranking",
        TaskType::Forecasting => "forecasting",
    }
}

fn time_unit_name(u: crate::pql::ast::TimeUnit) -> &'static str {
    use crate::pql::ast::TimeUnit as U;
    match u {
        U::Seconds => "seconds",
        U::Minutes => "minutes",
        U::Hours => "hours",
        U::Days => "days",
        U::Weeks => "weeks",
        U::Months => "months",
        U::Years => "years",
    }
}

fn output_form(pq: &ParsedQuery, task_type: TaskType) -> String {
    if let Some(ret) = &pq.ret {
        return ret.kind.to_lowercase();
    }
    match task_type {
        TaskType::Regression => "value",
        TaskType::BinaryClassification => "probability",
        TaskType::MulticlassClassification => "class",
        TaskType::MultilabelRanking => "ranked",
        TaskType::Forecasting => "value-per-horizon",
    }
    .to_string()
}

fn fmt_f(v: f64) -> String {
    if v.is_infinite() {
        if v > 0.0 { "+inf".into() } else { "-inf".into() }
    } else if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

/// A human-readable normalization of a target/where expression.
fn render_expr(e: &TargetExpr) -> String {
    use crate::pql::ast::CondRhs;
    match e {
        TargetExpr::Aggregation(a) => {
            let mut s = format!("{}({})", a.func.keyword(), a.column);
            if let Some(w) = a.window {
                s.push_str(&format!(
                    " OVER ({} .. {} {})",
                    fmt_f(w.start),
                    fmt_f(w.end),
                    time_unit_name(w.unit)
                ));
            }
            s
        }
        TargetExpr::ColumnRef(c) => format!("{}", c),
        TargetExpr::Condition(c) => {
            let left = render_expr(&c.left);
            let op = render_op(c.op);
            let rhs = match &c.right {
                CondRhs::Empty => String::new(),
                CondRhs::One(l) => render_lit(l),
                CondRhs::List(ls) => {
                    format!("({})", ls.iter().map(render_lit).collect::<Vec<_>>().join(", "))
                }
                CondRhs::Expr(e) => render_expr(e),
            };
            if rhs.is_empty() {
                format!("{} {}", left, op)
            } else {
                format!("{} {} {}", left, op, rhs)
            }
        }
        TargetExpr::LogicalOp(l) => {
            let op = match l.op {
                crate::pql::ast::BoolOp::And => "AND",
                crate::pql::ast::BoolOp::Or => "OR",
            };
            format!("({} {} {})", render_expr(&l.left), op, render_expr(&l.right))
        }
        TargetExpr::Not(e) => format!("NOT ({})", render_expr(e)),
        TargetExpr::Arith(a) => {
            format!("({} {} {})", render_expr(&a.left), a.op, render_expr(&a.right))
        }
        TargetExpr::Func(f) => {
            let args: Vec<String> = f.args.iter().map(render_expr).collect();
            format!("{}({})", f.name, args.join(", "))
        }
        TargetExpr::Case(_) => "CASE ... END".to_string(),
        TargetExpr::Lit(l) => render_lit(l),
    }
}

fn render_op(op: crate::pql::ast::Operator) -> &'static str {
    use crate::pql::ast::Operator as O;
    match op {
        O::Gt => ">",
        O::Lt => "<",
        O::Eq => "=",
        O::Neq => "!=",
        O::Ge => ">=",
        O::Le => "<=",
        O::StartsWith => "STARTS_WITH",
        O::EndsWith => "ENDS_WITH",
        O::Contains => "CONTAINS",
        O::NotContains => "NOT_CONTAINS",
        O::Like => "LIKE",
        O::NotLike => "NOT LIKE",
        O::In => "IN",
        O::NotIn => "NOT IN",
        O::IsNull => "IS NULL",
        O::IsNotNull => "IS NOT NULL",
    }
}

fn render_lit(l: &crate::pql::ast::Literal) -> String {
    use crate::pql::ast::Literal;
    match l {
        Literal::Str(s) => format!("'{}'", s),
        Literal::Num(n) => {
            if n.fract() == 0.0 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        Literal::Bool(b) => format!("{}", b),
        Literal::Date(d) => format!("'{}'", d.to_rfc3339()),
        Literal::Null => "NULL".to_string(),
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_num(v: f64) -> String {
    if v.is_finite() {
        if v.fract() == 0.0 {
            format!("{}", v as i64)
        } else {
            format!("{}", v)
        }
    } else if v > 0.0 {
        "\"+inf\"".to_string()
    } else {
        "\"-inf\"".to_string()
    }
}

fn literal_to_entity_id(l: &crate::pql::ast::Literal) -> EntityId {
    use crate::pql::ast::Literal;
    match l {
        Literal::Num(n) if n.fract() == 0.0 => EntityId::Int(*n as i64),
        Literal::Num(n) => EntityId::Str(format!("{}", n)),
        Literal::Str(s) => EntityId::Str(s.clone()),
        Literal::Bool(b) => EntityId::Str(if *b { "true".into() } else { "false".into() }),
        Literal::Date(d) => EntityId::Str(d.to_rfc3339()),
        Literal::Null => EntityId::Str(String::new()),
    }
}

// ---------------------------------------------------------------------------
// Built-in model-free backend: predicts from the entity's own history
// ---------------------------------------------------------------------------

/// Reference backend, no checkpoint: evaluates the target expression over
/// trailing historical windows of the assembled context ("self labels", F65 —
/// the strongest zero-shot signal). Real model backends implement the same
/// [`ModelBackend`] trait and load `model_uri`.
pub struct HistoryBaselineBackend {
    pub num_history_windows: usize,
}

impl HistoryBaselineBackend {
    pub fn new(num_history_windows: usize) -> HistoryBaselineBackend {
        HistoryBaselineBackend { num_history_windows: num_history_windows.max(1) }
    }

    fn pseudo_anchors(&self, anchor: Option<DateTime<Utc>>, span: Option<chrono::Duration>) -> Vec<Option<DateTime<Utc>>> {
        match (anchor, span) {
            (Some(a), Some(s)) => (1..=self.num_history_windows as i32).map(|k| Some(a - s * k)).collect(),
            _ => vec![anchor],
        }
    }

    /// The per-pseudo-anchor numeric history values of the target expression.
    fn history_values(
        &self,
        query: &ParsedQuery,
        rows_by_table: &HashMap<String, Vec<Row>>,
        cells: &[(String, Value)],
        anchor: Option<DateTime<Utc>>,
        span: Option<chrono::Duration>,
    ) -> Vec<f64> {
        let mut vals = Vec::new();
        for pa in self.pseudo_anchors(anchor, span) {
            let v = eval_value(&query.target, rows_by_table, cells, pa);
            if let Some(n) = v.as_number() {
                vals.push(n);
            }
        }
        vals
    }

    fn history_mean(
        &self,
        query: &ParsedQuery,
        rows_by_table: &HashMap<String, Vec<Row>>,
        cells: &[(String, Value)],
        anchor: Option<DateTime<Utc>>,
        span: Option<chrono::Duration>,
    ) -> Option<f64> {
        let vals = self.history_values(query, rows_by_table, cells, anchor, span);
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    }

    fn history_prob(
        &self,
        query: &ParsedQuery,
        rows_by_table: &HashMap<String, Vec<Row>>,
        cells: &[(String, Value)],
        anchor: Option<DateTime<Utc>>,
        span: Option<chrono::Duration>,
    ) -> f64 {
        let anchors = self.pseudo_anchors(anchor, span);
        let n = anchors.len().max(1);
        let hits = anchors
            .into_iter()
            .filter(|pa| eval_bool(&query.target, rows_by_table, cells, *pa))
            .count();
        hits as f64 / n as f64
    }

    fn latest_value(
        &self,
        query: &ParsedQuery,
        rows_by_table: &HashMap<String, Vec<Row>>,
        cells: &[(String, Value)],
        anchor: Option<DateTime<Utc>>,
        span: Option<chrono::Duration>,
    ) -> EvalValue {
        let pa = self.pseudo_anchors(anchor, span).into_iter().next().flatten();
        let v = eval_value(&query.target, rows_by_table, cells, pa);
        if let EvalValue::List(items) = &v {
            return items.last().map(value_to_eval).unwrap_or(EvalValue::Null);
        }
        v
    }

    fn score_one(
        &self,
        query: &ParsedQuery,
        task_type: TaskType,
        ctx: &EntityContext,
    ) -> EntityPrediction {
        let rows_by_table = ctx.rows_by_table();
        let cells = ctx.entity_cells(&query.entity_key.table);
        let window: Option<Window> =
            query.target_aggregations().iter().find_map(|a| a.window);
        let span = window.and_then(|w| w.span());

        let mut p = match task_type {
            TaskType::Forecasting => {
                let base = self.history_mean(query, &rows_by_table, &cells, ctx.anchor, span);
                let n = query.num_forecasts.unwrap_or(1).max(1) as usize;
                let mut p = EntityPrediction::new(ctx.entity_id.clone());
                p.value = base;
                if let Some(b) = base {
                    p.forecast = vec![b; n];
                }
                p
            }
            TaskType::MultilabelRanking => self.rank(query, &rows_by_table, ctx),
            TaskType::BinaryClassification => {
                let prob = self.history_prob(query, &rows_by_table, &cells, ctx.anchor, span);
                let mut p = EntityPrediction::new(ctx.entity_id.clone());
                p.probability = Some(prob);
                p
            }
            TaskType::MulticlassClassification => {
                let v = self.latest_value(query, &rows_by_table, &cells, ctx.anchor, span);
                let mut p = EntityPrediction::new(ctx.entity_id.clone());
                if !matches!(v, EvalValue::Null) {
                    p.class_probs = vec![(eval_value_to_key(&v), 1.0)];
                }
                p
            }
            TaskType::Regression => {
                let v = self.history_mean(query, &rows_by_table, &cells, ctx.anchor, span);
                let mut p = EntityPrediction::new(ctx.entity_id.clone());
                p.value = v;
                p
            }
        };

        // RETURN shaping: the default (no RETURN) output above is unchanged.
        if let Some(ret) = &query.ret {
            self.apply_return(
                ret,
                query,
                task_type,
                &rows_by_table,
                &cells,
                ctx.anchor,
                span,
                &mut p,
            );
        }
        p
    }

    /// Shape `p` for an explicit `RETURN <kind>`, reusing the same history-window
    /// machinery as the default output (see the RETURN execution contract §3).
    #[allow(clippy::too_many_arguments)]
    fn apply_return(
        &self,
        ret: &crate::pql::ast::ReturnSpec,
        query: &ParsedQuery,
        task_type: TaskType,
        rows_by_table: &HashMap<String, Vec<Row>>,
        cells: &[(String, Value)],
        anchor: Option<DateTime<Utc>>,
        span: Option<chrono::Duration>,
        p: &mut EntityPrediction,
    ) {
        match ret.kind.as_str() {
            "EXPECTED_VALUE" => match task_type {
                TaskType::BinaryClassification => {
                    let prob = self.history_prob(query, rows_by_table, cells, anchor, span);
                    p.value = Some(prob);
                }
                // regression/forecasting: value = history mean (already set for
                // both above; forecasting keeps its forecast series).
                _ => {}
            },
            // PROBABILITY (binary): probability is already the default.
            "PROBABILITY" => {}
            "CLASS" => match task_type {
                TaskType::BinaryClassification => {
                    let prob = p
                        .probability
                        .unwrap_or_else(|| self.history_prob(query, rows_by_table, cells, anchor, span));
                    p.predicted_class =
                        Some(if prob >= 0.5 { "true".into() } else { "false".into() });
                }
                TaskType::MulticlassClassification => {
                    let v = self.latest_value(query, rows_by_table, cells, anchor, span);
                    if !matches!(v, EvalValue::Null) {
                        p.predicted_class = Some(eval_value_to_key(&v));
                    }
                }
                _ => {}
            },
            "DISTRIBUTION" => match task_type {
                TaskType::BinaryClassification => {
                    let prob = p
                        .probability
                        .unwrap_or_else(|| self.history_prob(query, rows_by_table, cells, anchor, span));
                    p.class_probs = vec![("true".into(), prob), ("false".into(), 1.0 - prob)];
                }
                // multiclass: class_probs already holds the distribution.
                _ => {}
            },
            "QUANTILES" => {
                let vals = self.history_values(query, rows_by_table, cells, anchor, span);
                if !vals.is_empty() {
                    p.quantiles = ret
                        .quantiles
                        .iter()
                        .map(|&q| (q, empirical_quantile(&vals, q)))
                        .collect();
                }
            }
            "INTERVAL" => {
                let vals = self.history_values(query, rows_by_table, cells, anchor, span);
                if !vals.is_empty() {
                    let pct = ret.interval.unwrap_or(0) as f64;
                    let lo = (1.0 - pct / 100.0) / 2.0;
                    let hi = 1.0 - lo;
                    p.interval = Some((
                        empirical_quantile(&vals, lo),
                        empirical_quantile(&vals, hi),
                    ));
                }
            }
            // MULTILABEL -> ranking output (already in `ranked`).
            // MULTICLASS -> multiclass class_probs (already set).
            _ => {}
        }
    }

    fn rank(
        &self,
        query: &ParsedQuery,
        rows_by_table: &HashMap<String, Vec<Row>>,
        ctx: &EntityContext,
    ) -> EntityPrediction {
        let agg = match query.target_aggregations().into_iter().next() {
            Some(a) => a,
            None => return EntityPrediction::new(ctx.entity_id.clone()),
        };
        let empty = Vec::new();
        let rows = rows_by_table.get(&agg.column.table).unwrap_or(&empty);
        // FK targets (recommendation pattern) live in Row.parents, never in
        // cells (F17); fall back accordingly.
        let mut order: Vec<String> = Vec::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for r in rows {
            let key = if let Some(v) = r.get_cell(&agg.column.column) {
                Some(value_to_key(v))
            } else {
                r.get_parent(&agg.column.column).map(|p| p.to_string())
            };
            if let Some(k) = key {
                if !counts.contains_key(&k) {
                    order.push(k.clone());
                }
                *counts.entry(k).or_insert(0) += 1;
            }
        }
        // most_common: count desc, ties by first-seen order
        let mut items: Vec<(usize, String)> = order
            .iter()
            .enumerate()
            .map(|(i, k)| (i, k.clone()))
            .collect();
        items.sort_by(|a, b| {
            counts[&b.1]
                .cmp(&counts[&a.1])
                .then(a.0.cmp(&b.0))
        });
        let k = query.top_k.unwrap_or(10).max(0) as usize;
        let ranked = items.into_iter().take(k).map(|(_, key)| key).collect();
        let mut p = EntityPrediction::new(ctx.entity_id.clone());
        p.ranked = ranked;
        p
    }
}

impl ModelBackend for HistoryBaselineBackend {
    fn score(
        &mut self,
        query: &ParsedQuery,
        task_type: TaskType,
        contexts: &[EntityContext],
        _model_uri: &str,
        _config: &ModelConfig,
    ) -> Result<Vec<EntityPrediction>, Error> {
        Ok(contexts.iter().map(|c| self.score_one(query, task_type, c)).collect())
    }
}

/// Empirical quantile of `sample` at `q in [0,1]` using linear interpolation on
/// the sorted sample (numpy.quantile / percentile "linear" semantics).
fn empirical_quantile(sample: &[f64], q: f64) -> f64 {
    debug_assert!(!sample.is_empty());
    let mut xs: Vec<f64> = sample.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = q.clamp(0.0, 1.0);
    let n = xs.len();
    if n == 1 {
        return xs[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    xs[lo] + (xs[hi] - xs[lo]) * frac
}

fn value_to_eval(v: &Value) -> EvalValue {
    match v {
        Value::Number(n) => EvalValue::Num(*n),
        Value::Boolean(b) => EvalValue::Bool(*b),
        Value::Text(s) => EvalValue::Text(s.clone()),
        Value::Datetime(d) => EvalValue::Date(*d),
    }
}

fn value_to_key(v: &Value) -> String {
    match v {
        Value::Number(n) => {
            if n.fract() == 0.0 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        Value::Boolean(b) => format!("{}", b),
        Value::Text(s) => s.clone(),
        Value::Datetime(d) => d.to_rfc3339(),
    }
}

fn eval_value_to_key(v: &EvalValue) -> String {
    match v {
        EvalValue::Num(n) => {
            if n.fract() == 0.0 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        EvalValue::Bool(b) => format!("{}", b),
        EvalValue::Text(s) => s.clone(),
        EvalValue::Date(d) => d.to_rfc3339(),
        EvalValue::Null => "None".into(),
        EvalValue::List(_) => "[...]".into(),
    }
}
