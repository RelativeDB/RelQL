"""In-memory CSC adjacency over scanner-provided tables, pure numpy.

The time-bounded "latest <= anchor" children query — the CSC hot path and the
one non-trivial algorithm here — is a lex-sorted adjacency with a binary
search per query (the former ``cpp/src/csc.*``, ported here when context
creation moved fully into Python). Ties among equal (parent, ts) keep input
edge order — numpy's stable lexsort — so results are byte-for-byte what the
reference produces.
"""
from __future__ import annotations

import math
import warnings
from types import MappingProxyType
from typing import Any, Optional, Sequence

import numpy as np

from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import LinkDef, Schema

__all__ = ["CscIndex", "CscAdjacency"]


class CscAdjacency:
    """Per-link adjacency: build once from edge arrays, then answer many
    time-bounded ``children`` queries.

    Edges are stably sorted by (parent, ts asc); each parent's slice is then
    binary-searched for "latest ≤ anchor", returned newest-first and limited.
    Edges whose parent is out of range (dangling FKs already filtered by the
    caller, but be safe) are dropped, like the reference.
    """

    __slots__ = ("n_parents", "child", "ts", "colptr")

    def __init__(self, n_parents: int, edge_parent: Sequence[int],
                 edge_child: Sequence[int], edge_ts: Sequence[float]):
        self.n_parents = max(0, int(n_parents))
        ep = np.asarray(edge_parent, dtype=np.int64)
        ec = np.asarray(edge_child, dtype=np.int64)
        et = np.asarray(edge_ts, dtype=np.float64)
        keep = (ep >= 0) & (ep < self.n_parents)
        ep, ec, et = ep[keep], ec[keep], et[keep]
        # lexsort's last key is primary; stable, so equal (parent, ts) keep
        # input order — the tie rule the reference relies on.
        order = np.lexsort((et, ep)) if len(ep) else np.zeros(0, np.int64)
        ep, self.child, self.ts = ep[order], ec[order], et[order]
        self.colptr = np.zeros(self.n_parents + 1, dtype=np.int64)
        np.add.at(self.colptr, ep + 1, 1)
        np.cumsum(self.colptr, out=self.colptr)

    def children(self, parent_dense: int, anchor_ts: float,
                 limit: int) -> list[int]:
        """Dense child ids with ts <= anchor, newest-first, at most limit."""
        if limit <= 0 or not (0 <= parent_dense < self.n_parents):
            return []
        s = self.colptr[parent_dense]
        e = self.colptr[parent_dense + 1]
        cnt = int(np.searchsorted(self.ts[s:e], anchor_ts, side="right"))
        take = min(cnt, int(limit))
        return [int(c) for c in self.child[s + cnt - take:s + cnt][::-1]]


def _epoch(row: Row) -> float:
    """Row time as float seconds; static rows sort first (-inf) so they are
    admitted under every temporal bound."""
    return row.timestamp.timestamp() if row.timestamp is not None else -math.inf


class CscIndex:
    """Snapshot index over scanner-provided tables. Rebuild via a new build().

    Per-link adjacency (build + time-bounded children) lives in
    :class:`CscAdjacency`; dense child ids returned by it index back into this
    index's own ``rows`` lists.
    """

    def __init__(self) -> None:
        self.rows: dict[str, list[Row]] = {}
        self.dense: dict[str, dict[Any, int]] = {}
        self.adjacency: dict[LinkDef, CscAdjacency] = {}

    @staticmethod
    def build(schema: Schema, wiring: RetrieverWiring,
              bound: TemporalBound = TemporalBound.unbounded(), *,
              allow_missing_scanners: bool = False) -> "CscIndex":
        idx = CscIndex()
        for table in schema.tables:
            scanner = wiring.scanners.get(table.name)
            if scanner is None:
                if not allow_missing_scanners:
                    scanner = wiring.scanner(table.name)  # raises precise error
                else:
                    idx.rows[table.name] = []
                    idx.dense[table.name] = {}
                    continue
            rows = [Row(r.table, r.id,
                        MappingProxyType(dict(r.cells)), r.timestamp,
                        MappingProxyType({k: tuple(v) if isinstance(v, list)
                                          else v
                                          for k, v in r.parents.items()}))
                    for r in scanner(table.name, bound) if bound.admits_row(r)]
            idx.rows[table.name] = rows
            idx.dense[table.name] = {r.id: i for i, r in enumerate(rows)}
        for link in schema.links:
            idx.adjacency[link] = idx._build_link(link)
        return idx

    def _build_link(self, link: LinkDef) -> CscAdjacency:
        """Extract this link's edges (parent_dense, child_dense, ts); the
        adjacency sorts and buckets them."""
        children = self.rows.get(link.from_table, [])
        parent_dense = self.dense.get(link.to_table, {})
        n_parents = len(self.rows.get(link.to_table, []))
        ep: list[int] = []
        ec: list[int] = []
        et: list[float] = []
        dangling = 0
        candidates = 0
        for ci, row in enumerate(children):
            pid = row.parents.get(link.fk_column)
            if pid is None:
                continue
            for one in (pid if isinstance(pid, (list, tuple)) else (pid,)):
                candidates += 1
                pi = parent_dense.get(one)
                if pi is None:
                    dangling += 1  # edge dropped, row still scannable
                    continue
                ep.append(pi)
                ec.append(ci)
                et.append(_epoch(row))
        if candidates and not ep:
            # A handful of dangling FKs is data; ALL of them dangling is a
            # key-type mismatch (int pk vs str FK after a CSV round-trip)
            # that silently severs the whole link — children() returns
            # nothing, WHERE counts read 0 for everyone.
            warnings.warn(
                f"link {link.from_table}.{link.fk_column} -> "
                f"{link.to_table}: all {candidates} FK values are dangling "
                f"(no matching parent id). Likely a key-type mismatch; the "
                f"link is effectively severed.", UserWarning, stacklevel=4)
        return CscAdjacency(n_parents, ep, ec, et)

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
        anchor = (bound.as_of.timestamp() if bound.as_of is not None
                  else math.inf)
        table_rows = self.rows[link.from_table]
        return [table_rows[ci] for ci in adj.children(pi, anchor, limit)]

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
