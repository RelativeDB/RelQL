"""Columnar context population: array-backed graph store + traversal.

The engine's only real demand on storage is CONTEXT POPULATION: hand back
the few thousand rows that belong in each prediction's context. The default
path materializes every database row as a Python :class:`~relativedb.Row`
up front, which caps out around a few million rows. This module holds
tables as numpy/pandas columns instead — adjacency as CSR int arrays, cell
counts and timestamps as vectors — and materializes ``Row`` objects lazily,
only for rows a context actually emits. Tens of millions of rows fit in a
few hundred MB.

``ColumnarTraversal`` implements the same shared-context scoring protocol
as :class:`~relativedb.traversal.ReferenceTraversal`'s shared path (walk
tiering with the native RNG-exact sampler, self-label task rows, cohort
targets), but over its own node numbering; it does not reproduce the
reference preprocessor's edge order, so it is a sampling protocol of its
own, not a reference-parity mode. Scores remain official-evaluator scores.

Usage (an "option": pick your context-population backend per Engine):

    store = ColumnarStore(schema, frames, task_frames=..., task_links=...)
    engine = Engine(schema, store.wiring(), model_backend=...,
                    traversal=ColumnarTraversal(store, task_spec_factory=...))
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import Schema
from .task import TaskSpec
from .traversal import TraversalResult, _StdRng, _rand_sample, _U64

__all__ = ["ColumnarStore", "ColumnarTraversal"]


# Microseconds per second. The unit is REQUESTED rather than inferred: the
# original divided by a hardcoded 1e9, correct only for datetime64[ns], and
# pandas >= 3 defaults to microseconds -- which silently placed every timestamp
# 1000x too early, so no temporal bound held and the focal lookup never matched.
# Asking for datetime64[us] explicitly cannot drift with a library default.
_US_PER_SEC = 1e6


def _epoch_seconds(series) -> np.ndarray:
    """A datetime column -> float64 epoch seconds, NaN where missing."""
    values = pd.to_datetime(series, utc=True)
    arr = values.to_numpy(dtype="datetime64[us]")
    out = arr.astype("int64").astype(np.float64) / _US_PER_SEC
    out[np.isnat(arr)] = np.nan
    return out


class _Table:
    """One table's columns plus the derived vectors the traversal needs."""

    def __init__(self, name: str, frame: pd.DataFrame, pkey: Optional[str],
                 time_col: Optional[str], feature_cols: list[str],
                 fk_cols: dict[str, str]):
        self.name = name
        self.frame = frame.reset_index(drop=True)
        self.pkey = pkey
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.fk_cols = fk_cols            # fk column -> parent table
        self.n = len(self.frame)
        self.ids = (self.frame[pkey].to_numpy() if pkey is not None
                    else np.arange(self.n))
        self.id_index = pd.Index(self.ids)
        if time_col is not None and time_col in self.frame:
            self.ts = _epoch_seconds(self.frame[time_col])
        else:
            self.ts = np.full(self.n, np.nan)
        # Non-null feature cell count per row, plus timestamp-as-cell when
        # the time column is also a declared feature (mirrors the default
        # row cell accounting).
        counts = np.zeros(self.n, dtype=np.int32)
        for col in feature_cols:
            counts += self.frame[col].notna().to_numpy(dtype=np.int32)
        self.cell_counts = counts


class ColumnarStore:
    """Array-backed store over a schema's tables (plus optional task tables).

    ``frames`` maps physical table name -> DataFrame. ``task_frames`` maps a
    task table name -> DataFrame whose ``entity_col`` column links it to
    ``entity_table`` (``task_links``), with every remaining column treated
    as a cell. Node ids are assigned per table in sorted-name order,
    physical tables first, then task tables.
    """

    def __init__(self, schema: Schema, frames: dict[str, pd.DataFrame], *,
                 task_frames: Optional[dict[str, pd.DataFrame]] = None,
                 task_links: Optional[dict[str, tuple[str, str, str]]] = None):
        self.schema = schema
        self.tables: dict[str, _Table] = {}
        for tdef in schema.tables:
            frame = frames[tdef.name]
            fk_cols = {l.fk_column: l.to_table
                       for l in schema.links_from(tdef.name)}
            self.tables[tdef.name] = _Table(
                tdef.name, frame, tdef.primary_key, tdef.time_column,
                [c.name for c in tdef.columns if c.name in frame.columns],
                fk_cols)
        self.task_tables: dict[str, _Table] = {}
        self.task_links = dict(task_links or {})
        for name, frame in (task_frames or {}).items():
            entity_table, entity_col, time_col = self.task_links[name]
            cells = [c for c in frame.columns if c != entity_col]
            self.task_tables[name] = _Table(
                name, frame, None, time_col, cells, {entity_col: entity_table})

        # ---- global node numbering --------------------------------------
        self.order = sorted(self.tables) + sorted(self.task_tables)
        self.base: dict[str, int] = {}
        at = 0
        for name in self.order:
            self.base[name] = at
            at += self._table(name).n
        self.n_nodes = at

        # concatenated per-node vectors
        self.node_ts = np.concatenate(
            [self._table(n).ts for n in self.order]) if self.order else \
            np.empty(0)
        self.node_cells = np.concatenate(
            [self._table(n).cell_counts for n in self.order]).astype(np.int32)
        # A timestamp emits one extra cell only where the time column is not
        # already a declared feature column.
        extra = np.concatenate([
            (np.zeros(self._table(n).n, dtype=np.int32)
             if self._table(n).time_col in self._table(n).feature_cols
             else (~np.isnan(self._table(n).ts)).astype(np.int32))
            for n in self.order])
        self.node_cells = self.node_cells + extra
        self.node_table_idx = np.concatenate(
            [np.full(self._table(n).n, i, dtype=np.int32)
             for i, n in enumerate(self.order)])

        self._build_edges()
        self._row_cache: dict[int, Row] = {}
        self._native_graph = None

    # ------------------------------------------------------------------
    def _table(self, name: str) -> _Table:
        return self.tables.get(name) or self.task_tables[name]

    def _resolve_parents(self, table: _Table, fk: str,
                         parent: _Table) -> np.ndarray:
        """fk column values -> parent node positions (-1 where unmatched)."""
        pos = parent.id_index.get_indexer(table.frame[fk])
        isna = table.frame[fk].isna().to_numpy()
        pos = pos.astype(np.int64)
        pos[isna] = -1
        return pos

    def _build_edges(self) -> None:
        """Parent lists (f2p) and child CSR (p2f), both in canonical order:
        children of one parent sort by (has-timestamp, timestamp, position) —
        the same order the row-object path produces."""
        heads: list[np.ndarray] = []      # child node -> parent node edges
        tails: list[np.ndarray] = []
        for name in self.order:
            table = self._table(name)
            child_base = self.base[name]
            for fk, parent_name in table.fk_cols.items():
                parent = self._table(parent_name)
                pos = self._resolve_parents(table, fk, parent)
                ok = pos >= 0
                child_nodes = np.nonzero(ok)[0] + child_base
                parent_nodes = pos[ok] + self.base[parent_name]
                heads.append(parent_nodes)
                tails.append(child_nodes)
        if heads:
            parent_of_edge = np.concatenate(heads)
            child_of_edge = np.concatenate(tails)
        else:
            parent_of_edge = np.empty(0, dtype=np.int64)
            child_of_edge = np.empty(0, dtype=np.int64)

        # p2f CSR: children grouped by parent, ordered by
        # (has_ts, ts, child position) — matching the stable row-path sort.
        child_ts = self.node_ts[child_of_edge]
        has_ts = ~np.isnan(child_ts)
        ts_key = np.where(has_ts, child_ts, -np.inf)
        order = np.lexsort((child_of_edge, ts_key, has_ts, parent_of_edge))
        self.p2f_child = child_of_edge[order].astype(np.int64)
        self.p2f_parent_sorted = parent_of_edge[order]
        counts = np.bincount(parent_of_edge, minlength=self.n_nodes)
        self.p2f_offsets = np.concatenate(
            ([0], np.cumsum(counts))).astype(np.int64)

        # f2p: parents per child, in fk-declaration order.
        forder = np.lexsort((parent_of_edge, child_of_edge))
        self.f2p_parent = parent_of_edge[forder].astype(np.int64)
        fcounts = np.bincount(child_of_edge, minlength=self.n_nodes)
        self.f2p_offsets = np.concatenate(
            ([0], np.cumsum(fcounts))).astype(np.int64)

    def native_graph(self):
        """The native graph, built once. Node ids are this store's numbering,
        and they seed the walk and BFS streams, so the numbering here decides
        the sampling."""
        if self._native_graph is None:
            from .graph_native import NativeGraph
            is_task = np.zeros(self.n_nodes, dtype=np.uint8)
            for name in self.task_tables:
                base = self.base[name]
                is_task[base:base + self._table(name).n] = 1
            self._native_graph = NativeGraph(
                self.node_ts, self.node_cells, self.node_table_idx, is_task,
                self.p2f_parent_sorted, self.p2f_child)
        return self._native_graph

    # ------------------------------------------------------------------
    def children(self, node: int) -> np.ndarray:
        return self.p2f_child[self.p2f_offsets[node]:
                              self.p2f_offsets[node + 1]]

    def parents_of(self, node: int) -> np.ndarray:
        return self.f2p_parent[self.f2p_offsets[node]:
                               self.f2p_offsets[node + 1]]

    def node_of(self, table: str, entity_id: Any) -> Optional[int]:
        t = self._table(table)
        pos = t.id_index.get_indexer([entity_id])[0]
        return None if pos < 0 else self.base[table] + int(pos)

    def table_of(self, node: int) -> str:
        return self.order[self.node_table_idx[node]]

    def row(self, node: int) -> Row:
        """Materialize one node as a Row (cached)."""
        cached = self._row_cache.get(node)
        if cached is not None:
            return cached
        name = self.table_of(node)
        table = self._table(name)
        pos = node - self.base[name]
        rec = table.frame.iloc[pos]
        cells = {}
        for col in table.feature_cols:
            v = rec[col]
            if pd.isna(v) if not isinstance(v, (list, tuple, np.ndarray)) \
                    else False:
                continue
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, pd.Timestamp):
                v = v.to_pydatetime()
            cells[col] = v
        ts = None
        if not math.isnan(self.node_ts[node]):
            ts = pd.Timestamp(self.node_ts[node], unit="s",
                              tz="UTC").to_pydatetime()
        parents = {}
        for fk, parent_name in table.fk_cols.items():
            v = rec[fk]
            if not (pd.isna(v) if not isinstance(v, (list, tuple, np.ndarray))
                    else False):
                parents[fk if name in self.tables else "__entity__"] = (
                    v.item() if isinstance(v, np.generic) else v)
        rid = (table.ids[pos].item()
               if isinstance(table.ids[pos], np.generic) else table.ids[pos])
        row = Row(name, rid, cells, ts, parents)
        self._row_cache[node] = row
        return row

    # ------------------------------------------------------------------
    def wiring(self) -> RetrieverWiring:
        """Thin retrievers over the columnar arrays (entity lookups, child
        expansion, lazy scanners) so the standard Engine wiring contract
        holds without materializing anything up front."""
        store = self

        def entity(table, ids, bound: TemporalBound):
            out = []
            t = store._table(table)
            for pos in t.id_index.get_indexer(list(ids)):
                if pos < 0:
                    continue
                row = store.row(store.base[table] + int(pos))
                if bound.admits_row(row):
                    out.append(row)
            return out

        def links(link, parent_id, bound: TemporalBound, limit):
            pnode = store.node_of(link.to_table, parent_id)
            if pnode is None:
                return []
            kids = [store.row(int(c)) for c in store.children(pnode)
                    if store.table_of(int(c)) == link.from_table]
            kids = [r for r in kids if bound.admits_row(r)]
            kids.sort(key=lambda r: (r.timestamp is None,
                                     -(r.timestamp.timestamp()
                                       if r.timestamp else 0.0)))
            return kids[:limit]

        def make_scanner(table):
            def scan(t, bound: TemporalBound):
                base = store.base[table]
                for pos in range(store._table(table).n):
                    row = store.row(base + pos)
                    if bound.admits_row(row):
                        yield row
            return scan

        builder = RetrieverWiring.new_wiring().default_links(links)
        for name in self.tables:
            builder.entities(name, entity)
            builder.scanner(name, make_scanner(name))
        return builder.build()


class ColumnarTraversal:
    """Shared-context traversal over a :class:`ColumnarStore`.

    Implements the walk-tiered context assembly and the ``cohort_targets``
    contract the engine's shared-context path expects, entirely over the
    store's arrays; only emitted context rows materialize as ``Row``
    objects. Per-entity (non-shared) execution is intentionally
    unsupported — a bare unmasked label row in context would leak the
    answer — so pair this traversal with ``shared_context=True``.
    """

    def __init__(self, store: ColumnarStore,
                 task_spec_factory: Optional[Callable] = None):
        self.store = store
        self.task_spec_factory = task_spec_factory or TaskSpec.from_query
        self._walk_core: Optional[dict] = None
        self._overlay: Optional[dict] = None
        self._focal_lookup: dict[str, dict] = {}
        self._label_ok: dict[str, np.ndarray] = {}

    # ---- store-level cores (built once) --------------------------------
    def _core(self) -> dict:
        if self._walk_core is not None:
            return self._walk_core
        s = self.store
        n = s.n_nodes
        fcounts = np.diff(s.f2p_offsets)
        pcounts = np.diff(s.p2f_offsets)
        counts = fcounts + pcounts
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        total = int(offsets[-1])
        flat = np.empty(total, dtype=np.int32)
        edge_ts = np.empty(total, dtype=np.float64)
        # f2p block per node (parents always admitted)
        child_of_edge = np.repeat(np.arange(n), fcounts)
        within = np.arange(len(s.f2p_parent)) - np.repeat(
            s.f2p_offsets[:-1], fcounts)
        dest = offsets[child_of_edge] + within
        flat[dest] = s.f2p_parent
        edge_ts[dest] = -np.inf
        # p2f block per node (children admitted by timestamp)
        parent_of_edge = np.repeat(np.arange(n), pcounts)
        within = np.arange(len(s.p2f_child)) - np.repeat(
            s.p2f_offsets[:-1], pcounts)
        dest = offsets[parent_of_edge] + fcounts[parent_of_edge] + within
        flat[dest] = s.p2f_child
        child_ts = s.node_ts[s.p2f_child]
        edge_ts[dest] = np.where(np.isnan(child_ts), -np.inf, child_ts)
        self._walk_core = {"offsets": offsets, "flat": flat,
                          "edge_ts": edge_ts, "n_parents": fcounts}
        return self._walk_core

    def _labels_ok(self, task_spec: TaskSpec) -> np.ndarray:
        got = self._label_ok.get(task_spec.table_name)
        if got is not None:
            return got
        s = self.store
        ok = np.zeros(s.n_nodes, dtype=bool)
        t = s._table(task_spec.table_name)
        base = s.base[task_spec.table_name]
        ok[base:base + t.n] = t.frame[
            task_spec.target_column].notna().to_numpy()
        self._label_ok[task_spec.table_name] = ok
        return ok

    def _focal(self, task_spec: TaskSpec) -> dict:
        got = self._focal_lookup.get(task_spec.table_name)
        if got is not None:
            return got
        s = self.store
        t = s._table(task_spec.table_name)
        entity_col = next(iter(t.fk_cols))
        base = s.base[task_spec.table_name]
        lookup = {}
        ents = t.frame[entity_col].to_numpy()
        for pos in range(t.n):
            lookup[(ents[pos].item() if isinstance(ents[pos], np.generic)
                    else ents[pos], float(t.ts[pos]))] = base + pos
        self._focal_lookup[task_spec.table_name] = lookup
        return lookup

    def _overlay_for(self, task_spec: TaskSpec, cutoff_f: float) -> dict:
        cached = self._overlay
        if (cached is not None and cached["cutoff"] == cutoff_f
                and cached["table"] == task_spec.table_name):
            return cached
        core = self._core()
        mask = core["edge_ts"] <= cutoff_f
        kept = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
        offsets = kept[core["offsets"]].astype(np.int32)
        neighbors = np.ascontiguousarray(core["flat"][mask])
        s = self.store
        eligible = (self._labels_ok(task_spec)
                    & (np.isnan(s.node_ts) | (s.node_ts <= cutoff_f)))
        self._overlay = {
            "cutoff": cutoff_f, "table": task_spec.table_name,
            "offsets": offsets, "neighbors": neighbors,
            "eligible": eligible.astype(np.uint8),
        }
        return self._overlay

    # ---- engine contract ------------------------------------------------
    def cohort_targets(self, entity_table, entity_ids, anchor, task_spec,
                       *, history: int):
        s = self.store
        lookup = self._focal(task_spec)
        anchor_f = (anchor.timestamp() if anchor is not None else math.nan)
        targets, inject, extra = [], [], {}
        labels_ok = self._labels_ok(task_spec)
        for eid in entity_ids:
            node = lookup.get((eid, anchor_f))
            if node is None:
                return None
            row = s.row(node)
            targets.append((eid, row.key))
            inject.append(row)
            extra[row.key] = int(node)
            enode = s.node_of(entity_table, eid)
            if enode is not None:
                erow = s.row(enode)
                inject.append(erow)
                extra[erow.key] = int(enode)
                hist_nodes = [int(c) for c in s.children(enode)
                              if s.table_of(int(c)) == task_spec.table_name
                              and labels_ok[int(c)]
                              and s.node_ts[int(c)] < anchor_f]
                hist_nodes.sort(key=lambda c: -s.node_ts[c])
                for c in hist_nodes[:history]:
                    hrow = s.row(c)
                    inject.append(hrow)
                    extra[hrow.key] = c
        return targets, inject, extra


    def traverse(self, schema, graph, entity_table, entity_id, bound, policy,
                 *, query=None) -> TraversalResult:
        """Assemble one entity's context.

        The walk and BFS run in C++ over flat arrays (cpp/src/graph.cpp) and
        hand back the ordered node ids the context emits; only those become
        Row objects. Doing that loop in Python meant materializing the whole
        database first, which is what made the large datasets take minutes.
        """
        if query is None:
            raise RuntimeError("ColumnarTraversal requires a query")
        s = self.store
        task_type = query.task_type(schema)
        task_spec = self.task_spec_factory(query, task_type)
        anchor = bound.as_of
        anchor_f = anchor.timestamp() if anchor is not None else math.nan
        target = self._focal(task_spec).get((entity_id, anchor_f))
        if target is None:
            return TraversalResult()
        cutoff_f = float(s.node_ts[target])
        if math.isnan(cutoff_f):
            cutoff_f = math.inf

        eligible = (self._labels_ok(task_spec)
                    & (np.isnan(s.node_ts) | (s.node_ts <= cutoff_f)))
        eligible = eligible.astype(np.uint8)
        eligible[target] = 0

        # The fallback stage pads a short context from the task table, so the
        # native side needs that table's contiguous node range.
        t = s._table(task_spec.table_name)
        nodes, focal_flags = s.native_graph().assemble(
            int(target), cutoff_f, eligible, policy,
            fallback_base=s.base[task_spec.table_name], fallback_n=t.n,
            max_nodes=max(policy.max_context_cells, 1024))

        rows: list[Row] = []
        node_ids: list[tuple[tuple[str, Any], int]] = []
        focal_keys = set()
        for node, is_focal in zip(nodes, focal_flags):
            node = int(node)
            row = s.row(node)
            if node == target:
                cells_masked = dict(row.cells)
                cells_masked.pop(task_spec.target_column, None)
                row = Row(row.table, row.id, cells_masked, row.timestamp,
                          row.parents)
            rows.append(row)
            node_ids.append((row.key, node))
            if is_focal:
                focal_keys.add(row.key)
        focal_keys = frozenset(focal_keys)
        full = len(rows) > 0 and sum(
            int(s.node_cells[int(n)]) for n in nodes) >= policy.max_context_cells
        return TraversalResult(tuple(rows), focal_keys, 0, full,
                               tuple(node_ids))
