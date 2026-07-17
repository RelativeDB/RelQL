//! The retriever SPI — the heart of the design.
//!
//! Users implement small traits (blanket-impl'd for closures, so any function
//! of the right shape works). All receive a [`TemporalBound`] — the engine's
//! leakage guard (F24) — which implementations must honor and the engine
//! re-checks defensively.
//!
//! ## SPI shape: synchronous, infallible
//!
//! The Java design specifies `CompletionStage` async retrievers; this Rust peer
//! (like the Python peer) makes them **synchronous** and returning plain `Vec`s.
//! Rationale: async in the SPI would force a runtime choice on every user and
//! colour the whole engine `async` for no benefit in the reference/test paths.
//! A synchronous trait is simpler and directly mirrors Python's callable
//! protocols; a batching/parallel implementation is free to do its own I/O
//! concurrency internally. Fallible variants can be layered by having an
//! implementation buffer errors — the engine itself returns [`crate::Result`]
//! for the parse/validate/wiring/execution errors it owns.
//!
//! Mirrors `dev.relativedb.retrieve` (Java) / `relativedb.retrieve` (Python).

use std::collections::HashMap;
use std::fmt;

use chrono::{DateTime, Utc};

use crate::schema::LinkDef;

/// Opaque row identity. Wraps whatever the user's storage uses.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum EntityId {
    Int(i64),
    Str(String),
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityId::Int(i) => write!(f, "{}", i),
            EntityId::Str(s) => write!(f, "{}", s),
        }
    }
}

impl From<i64> for EntityId {
    fn from(v: i64) -> Self {
        EntityId::Int(v)
    }
}
impl From<&str> for EntityId {
    fn from(v: &str) -> Self {
        EntityId::Str(v.to_string())
    }
}
impl From<String> for EntityId {
    fn from(v: String) -> Self {
        EntityId::Str(v)
    }
}

/// A typed feature cell value. Absence (null) is modelled by omitting the cell.
#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    Number(f64),
    Text(String),
    Datetime(DateTime<Utc>),
    Boolean(bool),
}

impl Value {
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            Value::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_datetime(&self) -> Option<DateTime<Utc>> {
        match self {
            Value::Datetime(d) => Some(*d),
            _ => None,
        }
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Number(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Number(v as f64)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Text(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Text(v)
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Boolean(v)
    }
}
impl From<DateTime<Utc>> for Value {
    fn from(v: DateTime<Utc>) -> Self {
        Value::Datetime(v)
    }
}

/// "Nothing newer than this" — the temporal-leakage guard (F24).
///
/// `as_of == None` means unbounded (static tables without time).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TemporalBound {
    pub as_of: Option<DateTime<Utc>>,
}

impl TemporalBound {
    pub fn at_or_before(t: DateTime<Utc>) -> Self {
        TemporalBound { as_of: Some(t) }
    }
    pub fn unbounded() -> Self {
        TemporalBound { as_of: None }
    }
    pub fn is_unbounded(&self) -> bool {
        self.as_of.is_none()
    }
    /// A row with no timestamp is static and always admitted (inclusive as-of).
    pub fn admits(&self, timestamp: Option<DateTime<Utc>>) -> bool {
        match (self.as_of, timestamp) {
            (Some(bound), Some(ts)) => ts <= bound,
            _ => true,
        }
    }
    pub fn admits_row(&self, row: &Row) -> bool {
        self.admits(row.timestamp)
    }
}

/// One row's typed feature cells.
///
/// IDs and FK values are NOT cells (F17) — links are reported separately via
/// `parents` so the engine can traverse without ever tokenizing identifiers.
/// Missing/null values: simply omit the cell — nulls emit no token.
#[derive(Clone, PartialEq, Debug)]
pub struct Row {
    pub table: String,
    pub id: EntityId,
    /// Insertion-ordered feature cells (order is preserved for tokenization).
    pub cells: Vec<(String, Value)>,
    pub timestamp: Option<DateTime<Utc>>,
    /// FK column -> parent id (F→P edges).
    pub parents: Vec<(String, EntityId)>,
}

impl Row {
    pub fn new(table: impl Into<String>, id: impl Into<EntityId>) -> Row {
        Row {
            table: table.into(),
            id: id.into(),
            cells: Vec::new(),
            timestamp: None,
            parents: Vec::new(),
        }
    }

    pub fn cell(mut self, column: impl Into<String>, value: impl Into<Value>) -> Row {
        self.cells.push((column.into(), value.into()));
        self
    }

    pub fn timestamp(mut self, t: DateTime<Utc>) -> Row {
        self.timestamp = Some(t);
        self
    }

    pub fn parent(mut self, fk_column: impl Into<String>, parent_id: impl Into<EntityId>) -> Row {
        self.parents.push((fk_column.into(), parent_id.into()));
        self
    }

    pub fn get_cell(&self, column: &str) -> Option<&Value> {
        self.cells.iter().find(|(c, _)| c == column).map(|(_, v)| v)
    }

    pub fn get_parent(&self, fk_column: &str) -> Option<&EntityId> {
        self.parents.iter().find(|(c, _)| c == fk_column).map(|(_, v)| v)
    }

    pub fn key(&self) -> (String, EntityId) {
        (self.table.clone(), self.id.clone())
    }
}

/// Batched point lookup: rows of one table by id (DataFetcher analog).
pub trait EntityRetriever {
    fn fetch_by_ids(&self, table: &str, ids: &[EntityId], bound: &TemporalBound) -> Vec<Row>;
}

/// Children of a parent row along one P→F link, newest-first, capped at
/// `limit`. MUST NOT return rows newer than `bound`.
pub trait LinkRetriever {
    fn fetch_children(
        &self,
        link: &LinkDef,
        parent_id: &EntityId,
        bound: &TemporalBound,
        limit: usize,
    ) -> Vec<Row>;
}

/// OPTIONAL: similar/other entity ids of the same table for in-context examples
/// (RT-J Tier 1/2). Without one, context is target-entity-local.
pub trait CohortRetriever {
    fn fetch_cohort(
        &self,
        table: &str,
        anchor: &EntityId,
        bound: &TemporalBound,
        limit: usize,
    ) -> Vec<EntityId>;
}

/// OPTIONAL: stream every row of `table` with time <= bound (any order).
/// Required for [`crate::engine::SamplerMode::Csc`].
pub trait TableScanner {
    fn scan(&self, table: &str, bound: &TemporalBound) -> Vec<Row>;
}

/// Column normalization statistics — training-split-only by contract (F11,F12).
#[derive(Clone, Copy, Debug)]
pub struct ColumnStats {
    pub mean: f64,
    pub std: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct DatetimeStats {
    pub mean: f64,
    pub std: f64,
}

/// Normalization statistics provider (F11/F12). Deliberately a retriever:
/// stats are data-owner knowledge, not engine state.
pub trait StatsProvider {
    fn numeric_stats(&self, table: &str, column: &str) -> ColumnStats;
    fn datetime_stats(&self) -> DatetimeStats;
}

// --- closure blanket impls (Python-protocol ergonomics) --------------------

impl<F> EntityRetriever for F
where
    F: Fn(&str, &[EntityId], &TemporalBound) -> Vec<Row>,
{
    fn fetch_by_ids(&self, table: &str, ids: &[EntityId], bound: &TemporalBound) -> Vec<Row> {
        self(table, ids, bound)
    }
}

impl<F> LinkRetriever for F
where
    F: Fn(&LinkDef, &EntityId, &TemporalBound, usize) -> Vec<Row>,
{
    fn fetch_children(
        &self,
        link: &LinkDef,
        parent_id: &EntityId,
        bound: &TemporalBound,
        limit: usize,
    ) -> Vec<Row> {
        self(link, parent_id, bound, limit)
    }
}

impl<F> CohortRetriever for F
where
    F: Fn(&str, &EntityId, &TemporalBound, usize) -> Vec<EntityId>,
{
    fn fetch_cohort(
        &self,
        table: &str,
        anchor: &EntityId,
        bound: &TemporalBound,
        limit: usize,
    ) -> Vec<EntityId> {
        self(table, anchor, bound, limit)
    }
}

impl<F> TableScanner for F
where
    F: Fn(&str, &TemporalBound) -> Vec<Row>,
{
    fn scan(&self, table: &str, bound: &TemporalBound) -> Vec<Row> {
        self(table, bound)
    }
}

/// Raised when the wiring is missing a required retriever.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WiringError(pub String);

impl fmt::Display for WiringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wiring error: {}", self.0)
    }
}
impl std::error::Error for WiringError {}

/// Wiring: schema element -> implementation. GraphQL RuntimeWiring analog.
#[derive(Default)]
pub struct RetrieverWiring {
    pub entities: HashMap<String, Box<dyn EntityRetriever>>,
    pub links: HashMap<String, Box<dyn LinkRetriever>>,
    pub default_link_retriever: Option<Box<dyn LinkRetriever>>,
    pub cohorts: HashMap<String, Box<dyn CohortRetriever>>,
    pub scanners: HashMap<String, Box<dyn TableScanner>>,
}

impl RetrieverWiring {
    pub fn new_wiring() -> RetrieverWiring {
        RetrieverWiring::default()
    }

    pub fn entity_retriever(&self, table: &str) -> Result<&dyn EntityRetriever, WiringError> {
        self.entities
            .get(table)
            .map(|b| b.as_ref())
            .ok_or_else(|| WiringError(format!("no EntityRetriever wired for table {:?}", table)))
    }

    pub fn link_retriever(&self, from_table: &str) -> Result<&dyn LinkRetriever, WiringError> {
        self.links
            .get(from_table)
            .map(|b| b.as_ref())
            .or(self.default_link_retriever.as_ref().map(|b| b.as_ref()))
            .ok_or_else(|| {
                WiringError(format!(
                    "no LinkRetriever wired for table {:?} and no default_links set",
                    from_table
                ))
            })
    }

    pub fn cohort_retriever(&self, table: &str) -> Option<&dyn CohortRetriever> {
        self.cohorts.get(table).map(|b| b.as_ref())
    }

    pub fn table_scanner(&self, table: &str) -> Result<&dyn TableScanner, WiringError> {
        self.scanners.get(table).map(|b| b.as_ref()).ok_or_else(|| {
            WiringError(format!(
                "no TableScanner wired for table {:?} (required for SamplerMode::Csc)",
                table
            ))
        })
    }

    // -- builder-style mutators (chainable) --------------------------------

    pub fn entities(mut self, table: impl Into<String>, r: impl EntityRetriever + 'static) -> Self {
        self.entities.insert(table.into(), Box::new(r));
        self
    }

    pub fn links(mut self, from_table: impl Into<String>, r: impl LinkRetriever + 'static) -> Self {
        self.links.insert(from_table.into(), Box::new(r));
        self
    }

    pub fn default_links(mut self, r: impl LinkRetriever + 'static) -> Self {
        self.default_link_retriever = Some(Box::new(r));
        self
    }

    pub fn cohort(mut self, table: impl Into<String>, r: impl CohortRetriever + 'static) -> Self {
        self.cohorts.insert(table.into(), Box::new(r));
        self
    }

    pub fn scanner(mut self, table: impl Into<String>, r: impl TableScanner + 'static) -> Self {
        self.scanners.insert(table.into(), Box::new(r));
        self
    }

    pub fn build(self) -> RetrieverWiring {
        self
    }
}
