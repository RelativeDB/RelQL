"""Materialized in-memory CSC adjacency built from TableScanners.

"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import LinkDef, Schema

__all__ = ["CscIndex", "LinkAdjacency"]

_NEG_INF = -math.inf


def _epoch(row: Row) -> float:
    """Row time as float seconds; static rows sort first (-inf), so they are
    admitted under every temporal bound."""
    return row.timestamp.timestamp() if row.timestamp is not None else _NEG_INF


@dataclass(frozen=True)
class LinkAdjacency:
    """CSC arrays for one FK link (child ``from_table`` -> parent ``to_table``).

    ``colptr`` is indexed by parent dense id (length n_parents + 1);
    ``row[colptr[p]:colptr[p+1]]`` are child dense ids sorted by time asc;
    ``ts`` holds the matching child timestamps (epoch seconds, -inf if none).
    """

    link: LinkDef
    colptr: np.ndarray  # int64, shape (n_parents + 1,)
    row: np.ndarray     # int64, shape (n_edges,)
    ts: np.ndarray      # float64, shape (n_edges,)


class CscIndex:
    """Snapshot index over scanner-provided tables. Rebuild via a new build()."""

    def __init__(self) -> None:
        self.rows: dict[str, list[Row]] = {}
        self.dense: dict[str, dict[Any, int]] = {}
        self.adjacency: dict[LinkDef, LinkAdjacency] = {}

    @staticmethod
    def build(schema: Schema, wiring: RetrieverWiring,
              bound: TemporalBound = TemporalBound.unbounded()) -> "CscIndex":
        idx = CscIndex()
        for table in schema.tables:
            scanner = wiring.scanner(table.name)
            rows = [r for r in scanner(table.name, bound) if bound.admits_row(r)]
            idx.rows[table.name] = rows
            idx.dense[table.name] = {r.id: i for i, r in enumerate(rows)}
        for link in schema.links:
            idx.adjacency[link] = idx._build_link(link)
        return idx

    def _build_link(self, link: LinkDef) -> LinkAdjacency:
        children = self.rows.get(link.from_table, [])
        parent_dense = self.dense.get(link.to_table, {})
        n_parents = len(self.rows.get(link.to_table, []))
        if not children or not parent_dense:
            return LinkAdjacency(link,
                                 np.zeros(n_parents + 1, dtype=np.int64),
                                 np.empty(0, dtype=np.int64),
                                 np.empty(0, dtype=np.float64))
        p_idx: list[int] = []
        c_idx: list[int] = []
        c_ts: list[float] = []
        for ci, row in enumerate(children):
            pid = row.parents.get(link.fk_column)
            if pid is None:
                continue
            pi = parent_dense.get(pid)
            if pi is None:
                continue  # dangling FK: edge dropped, row still scannable
            p_idx.append(pi)
            c_idx.append(ci)
            c_ts.append(_epoch(row))
        p = np.asarray(p_idx, dtype=np.int64)
        c = np.asarray(c_idx, dtype=np.int64)
        t = np.asarray(c_ts, dtype=np.float64)
        # sort edges by (parent, time asc) -> per-parent time-sorted buckets
        order = np.lexsort((t, p))
        p, c, t = p[order], c[order], t[order]
        counts = np.bincount(p, minlength=n_parents)
        colptr = np.zeros(n_parents + 1, dtype=np.int64)
        np.cumsum(counts, out=colptr[1:])
        return LinkAdjacency(link, colptr, c, t)

    # -- sampler surface ----------------------------------------------------
    def entities(self, table: str, ids: Sequence[Any],
                 bound: TemporalBound) -> list[Row]:
        dense = self.dense.get(table, {})
        rows = self.rows.get(table, [])
        out: list[Row] = []
        for i in ids:
            di = dense.get(i)
            if di is not None and bound.admits_row(rows[di]):
                out.append(rows[di])
        return out

    def children(self, link: LinkDef, parent_id: Any, bound: TemporalBound,
                 limit: int) -> list[Row]:
        """Latest ``limit`` children with time <= bound, newest-first."""
        adj = self.adjacency.get(link)
        if adj is None:
            return []
        pi = self.dense.get(link.to_table, {}).get(parent_id)
        if pi is None:
            return []
        s, e = int(adj.colptr[pi]), int(adj.colptr[pi + 1])
        anchor = (bound.as_of.timestamp() if bound.as_of is not None
                  else math.inf)
        hi = int(np.searchsorted(adj.ts[s:e], anchor, side="right"))
        picked = adj.row[s:s + hi][-limit:][::-1] if limit > 0 else adj.row[0:0]
        table_rows = self.rows[link.from_table]
        return [table_rows[int(ci)] for ci in picked]

    def all_ids(self, table: str) -> list[Any]:
        return [r.id for r in self.rows.get(table, [])]

    def cohort(self, table: str, anchor_id: Any, bound: TemporalBound,
               limit: int) -> list[Any]:
        """Cheap same-table cohort: first ``limit`` other admitted ids."""
        out: list[Any] = []
        for r in self.rows.get(table, []):
            if r.id != anchor_id and bound.admits_row(r):
                out.append(r.id)
                if len(out) >= limit:
                    break
        return out
