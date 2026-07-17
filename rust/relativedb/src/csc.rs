//! Materialized in-memory CSC adjacency built from TableScanners.
//!

use std::collections::HashMap;

use crate::retrieve::{EntityId, RetrieverWiring, Row, TemporalBound};
use crate::schema::{LinkDef, Schema};

fn epoch(row: &Row) -> f64 {
    // static rows sort first (-inf), so they are admitted under every bound.
    match row.timestamp {
        Some(t) => t.timestamp() as f64 + t.timestamp_subsec_nanos() as f64 / 1e9,
        None => f64::NEG_INFINITY,
    }
}

/// CSC arrays for one FK link (child `from_table` -> parent `to_table`).
#[derive(Clone, Debug)]
pub struct LinkAdjacency {
    pub link: LinkDef,
    /// indexed by parent dense id (length n_parents + 1)
    pub colptr: Vec<i64>,
    /// child dense ids, per-parent blocks sorted by time asc
    pub row: Vec<i64>,
    /// matching child timestamps (epoch seconds, -inf if none)
    pub ts: Vec<f64>,
}

pub struct CscIndex {
    pub rows: HashMap<String, Vec<Row>>,
    pub dense: HashMap<String, HashMap<EntityId, usize>>,
    pub adjacency: HashMap<LinkDef, LinkAdjacency>,
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
            let adj = idx.build_link(link);
            idx.adjacency.insert(link.clone(), adj);
        }
        Ok(idx)
    }

    fn build_link(&self, link: &LinkDef) -> LinkAdjacency {
        let empty = Vec::new();
        let children = self.rows.get(&link.from_table).unwrap_or(&empty);
        let empty_dense = HashMap::new();
        let parent_dense = self.dense.get(&link.to_table).unwrap_or(&empty_dense);
        let n_parents = self.rows.get(&link.to_table).map(|v| v.len()).unwrap_or(0);
        // (parent dense id, child dense id, child ts)
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for (ci, row) in children.iter().enumerate() {
            let pid = match row.get_parent(&link.fk_column) {
                Some(p) => p,
                None => continue,
            };
            let pi = match parent_dense.get(pid) {
                Some(&pi) => pi,
                None => continue, // dangling FK: edge dropped, row still scannable
            };
            edges.push((pi, ci, epoch(row)));
        }
        // stable sort by (parent, time asc)
        edges.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        });
        let mut colptr = vec![0i64; n_parents + 1];
        let mut rows_arr = Vec::with_capacity(edges.len());
        let mut ts_arr = Vec::with_capacity(edges.len());
        for (pi, ci, t) in &edges {
            colptr[pi + 1] += 1;
            rows_arr.push(*ci as i64);
            ts_arr.push(*t);
        }
        for k in 1..colptr.len() {
            colptr[k] += colptr[k - 1];
        }
        LinkAdjacency { link: link.clone(), colptr, row: rows_arr, ts: ts_arr }
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

    /// Latest `limit` children with time <= bound, newest-first.
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
        let s = adj.colptr[pi] as usize;
        let e = adj.colptr[pi + 1] as usize;
        let anchor = bound.as_of.map(|t| t.timestamp() as f64
            + t.timestamp_subsec_nanos() as f64 / 1e9)
            .unwrap_or(f64::INFINITY);
        // searchsorted(ts[s..e], anchor, side="right") = count of ts <= anchor
        let block = &adj.ts[s..e];
        let hi = block.partition_point(|&x| x <= anchor);
        if limit == 0 {
            return Vec::new();
        }
        let table_rows = &self.rows[&link.from_table];
        let start = s + hi.saturating_sub(limit);
        let picked = &adj.row[start..s + hi];
        picked.iter().rev().map(|&ci| table_rows[ci as usize].clone()).collect()
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
