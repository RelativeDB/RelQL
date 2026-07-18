//! In-memory CSC adjacency over scanner-provided tables.
//!
//! The time-bounded "latest <= anchor" children query — the CSC hot path and
//! the one non-trivial algorithm here — lives once in the C++ layer
//! (`cpp/src/csc.*`, via [`crate::csc_native`]), shared with the Java and Python
//! bindings. This module keeps only the Rust-side bookkeeping: table row
//! storage, the id<->dense-index mapping, and the seed/cohort lookups.
//! `librt_c` is a hard dependency (the same native library the RT-J model and
//! RelQL parser require) — there is no in-language fallback.

use std::collections::HashMap;

use crate::csc_native::NativeCsc;
use crate::native::RtError;
use crate::retrieve::{EntityId, RetrieverWiring, Row, TemporalBound};
use crate::schema::{LinkDef, Schema};

fn epoch(row: &Row) -> f64 {
    // static rows sort first (-inf), so they are admitted under every bound.
    match row.timestamp {
        Some(t) => t.timestamp() as f64 + t.timestamp_subsec_nanos() as f64 / 1e9,
        None => f64::NEG_INFINITY,
    }
}

/// Snapshot index over scanner-provided tables. Rebuild via a new [`build`].
///
/// Per-link adjacency (build + time-bounded children) is delegated to the
/// native `csc_*` implementation; the dense child ids it returns index back
/// into this index's own [`rows`](Self::rows) lists.
pub struct CscIndex {
    pub rows: HashMap<String, Vec<Row>>,
    pub dense: HashMap<String, HashMap<EntityId, usize>>,
    pub adjacency: HashMap<LinkDef, NativeCsc>,
}

impl CscIndex {
    pub fn build(schema: &Schema, wiring: &RetrieverWiring) -> Result<CscIndex, crate::Error> {
        Self::build_bounded(schema, wiring, TemporalBound::unbounded())
    }

    pub fn build_bounded(
        schema: &Schema,
        wiring: &RetrieverWiring,
        bound: TemporalBound,
    ) -> Result<CscIndex, crate::Error> {
        let mut idx = CscIndex {
            rows: HashMap::new(),
            dense: HashMap::new(),
            adjacency: HashMap::new(),
        };
        for table in &schema.tables {
            let scanner = wiring.table_scanner(&table.name)?;
            let rows: Vec<Row> = scanner
                .scan(&table.name, &bound)
                .into_iter()
                .filter(|r| bound.admits_row(r))
                .collect();
            let mut dense = HashMap::new();
            for (i, r) in rows.iter().enumerate() {
                dense.insert(r.id.clone(), i);
            }
            idx.rows.insert(table.name.clone(), rows);
            idx.dense.insert(table.name.clone(), dense);
        }
        for link in &schema.links {
            let adj = idx.build_link(link)?;
            idx.adjacency.insert(link.clone(), adj);
        }
        Ok(idx)
    }

    /// Extract this link's edges `(parent_dense, child_dense, ts)` and hand them
    /// to the native index; the native side sorts, buckets, and binary-searches.
    fn build_link(&self, link: &LinkDef) -> Result<NativeCsc, crate::Error> {
        let empty = Vec::new();
        let children = self.rows.get(&link.from_table).unwrap_or(&empty);
        let empty_dense = HashMap::new();
        let parent_dense = self.dense.get(&link.to_table).unwrap_or(&empty_dense);
        let n_parents = self.rows.get(&link.to_table).map(|v| v.len()).unwrap_or(0);
        let mut edge_parent: Vec<i64> = Vec::new();
        let mut edge_child: Vec<i64> = Vec::new();
        let mut edge_ts: Vec<f64> = Vec::new();
        for (ci, row) in children.iter().enumerate() {
            let pid = match row.get_parent(&link.fk_column) {
                Some(p) => p,
                None => continue,
            };
            let pi = match parent_dense.get(pid) {
                Some(&pi) => pi,
                None => continue, // dangling FK: edge dropped, row still scannable
            };
            edge_parent.push(pi as i64);
            edge_child.push(ci as i64);
            edge_ts.push(epoch(row));
        }
        NativeCsc::new(n_parents as i64, &edge_parent, &edge_child, &edge_ts).map_err(|e| {
            crate::Error::Rt(RtError::Native(format!("csc_build for link {:?}: {}", link, e)))
        })
    }

    pub fn entities(&self, table: &str, ids: &[EntityId], bound: &TemporalBound) -> Vec<Row> {
        let dense = match self.dense.get(table) {
            Some(d) => d,
            None => return Vec::new(),
        };
        let rows = &self.rows[table];
        let mut out = Vec::new();
        for id in ids {
            if let Some(&di) = dense.get(id) {
                if bound.admits_row(&rows[di]) {
                    out.push(rows[di].clone());
                }
            }
        }
        out
    }

    /// Latest `limit` children with time <= bound, newest-first. Delegated to
    /// the native index; the dense ids it returns index back into `rows`.
    pub fn children(
        &self,
        link: &LinkDef,
        parent_id: &EntityId,
        bound: &TemporalBound,
        limit: usize,
    ) -> Vec<Row> {
        let adj = match self.adjacency.get(link) {
            Some(a) => a,
            None => return Vec::new(),
        };
        let pi = match self.dense.get(&link.to_table).and_then(|d| d.get(parent_id)) {
            Some(&pi) => pi,
            None => return Vec::new(),
        };
        let anchor = bound
            .as_of
            .map(|t| t.timestamp() as f64 + t.timestamp_subsec_nanos() as f64 / 1e9)
            .unwrap_or(f64::INFINITY);
        let table_rows = &self.rows[&link.from_table];
        let limit = limit.min(i32::MAX as usize) as i32;
        adj.children(pi as i64, anchor, limit)
            .into_iter()
            .map(|ci| table_rows[ci as usize].clone())
            .collect()
    }

    pub fn all_ids(&self, table: &str) -> Vec<EntityId> {
        self.rows.get(table).map(|rs| rs.iter().map(|r| r.id.clone()).collect()).unwrap_or_default()
    }

    pub fn cohort(
        &self,
        table: &str,
        anchor_id: &EntityId,
        bound: &TemporalBound,
        limit: usize,
    ) -> Vec<EntityId> {
        let mut out = Vec::new();
        if let Some(rows) = self.rows.get(table) {
            for r in rows {
                if &r.id != anchor_id && bound.admits_row(r) {
                    out.push(r.id.clone());
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        out
    }
}
