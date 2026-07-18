"""In-memory CSC adjacency over scanner-provided tables.

The time-bounded "latest <= anchor" children query — the CSC hot path and the
one non-trivial algorithm here — lives once in the C++ layer (``cpp/src/csc.*``,
via :mod:`relativedb.csc_native`), shared with the Java and Rust bindings.
This module keeps only the Python-side bookkeeping: table row storage, the
id<->dense-index mapping, and the seed/cohort lookups. ``librt_c`` is a hard
dependency (the same native library the RT-J model and PQL parser require).
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from .csc_native import NativeCsc
from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import LinkDef, Schema

__all__ = ["CscIndex"]


def _epoch(row: Row) -> float:
    """Row time as float seconds; static rows sort first (-inf) so they are
    admitted under every temporal bound."""
    return row.timestamp.timestamp() if row.timestamp is not None else -math.inf


class CscIndex:
    """Snapshot index over scanner-provided tables. Rebuild via a new build().

    Per-link adjacency (build + time-bounded children) is delegated to the
    native ``csc_*`` implementation; dense child ids returned by it index back
    into this index's own ``rows`` lists.
    """

    def __init__(self) -> None:
        self.rows: dict[str, list[Row]] = {}
        self.dense: dict[str, dict[Any, int]] = {}
        self.adjacency: dict[LinkDef, NativeCsc] = {}

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

    def _build_link(self, link: LinkDef) -> NativeCsc:
        """Extract this link's edges (parent_dense, child_dense, ts) and hand
        them to the native index; the native side sorts and buckets them."""
        children = self.rows.get(link.from_table, [])
        parent_dense = self.dense.get(link.to_table, {})
        n_parents = len(self.rows.get(link.to_table, []))
        ep: list[int] = []
        ec: list[int] = []
        et: list[float] = []
        for ci, row in enumerate(children):
            pid = row.parents.get(link.fk_column)
            if pid is None:
                continue
            pi = parent_dense.get(pid)
            if pi is None:
                continue  # dangling FK: edge dropped, row still scannable
            ep.append(pi)
            ec.append(ci)
            et.append(_epoch(row))
        return NativeCsc(n_parents, ep, ec, et)

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
