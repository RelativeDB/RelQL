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

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};

use crate::csc::CscIndex;
use crate::evaluate::{eval_bool, eval_value, EvalValue};
use crate::model::ModelConfig;
use crate::pql::ast::{ParsedQuery, TaskType, Window};
use crate::pql::parser::{parse, validate};
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
/// `fanouts` are per-hop child caps (KumoRFM geometry); when unset, a uniform
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
        let task_type = pq.task_type(Some(&self.schema));
        let model_uri = self.model_config.model_uri_for(task_type).to_string();
        let entity_table = pq.entity_key.table.clone();
        let ids = self.resolve_entity_ids(&pq, &input)?;
        let mut contexts: Vec<EntityContext> = Vec::new();
        for eid in ids {
            let anchor = self.anchor_for(&entity_table, &eid, &input)?;
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
        let preds = self
            .model_backend
            .score(&pq, task_type, &contexts, &model_uri, &self.model_config)?;
        Ok(PredictionResult { task_type, predictions: preds, model_uri })
    }

    /// Convenience: execute a PQL string with an anchor time.
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

fn literal_to_entity_id(l: &crate::pql::ast::Literal) -> EntityId {
    use crate::pql::ast::Literal;
    match l {
        Literal::Num(n) if n.fract() == 0.0 => EntityId::Int(*n as i64),
        Literal::Num(n) => EntityId::Str(format!("{}", n)),
        Literal::Str(s) => EntityId::Str(s.clone()),
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

    fn history_mean(
        &self,
        query: &ParsedQuery,
        rows_by_table: &HashMap<String, Vec<Row>>,
        cells: &[(String, Value)],
        anchor: Option<DateTime<Utc>>,
        span: Option<chrono::Duration>,
    ) -> Option<f64> {
        let mut vals = Vec::new();
        for pa in self.pseudo_anchors(anchor, span) {
            let v = eval_value(&query.target, rows_by_table, cells, pa);
            if let Some(n) = v.as_number() {
                vals.push(n);
            }
        }
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

        match task_type {
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
