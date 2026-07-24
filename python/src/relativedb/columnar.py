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

from datetime import datetime, timezone

import math
from typing import Any, Callable, Optional

import numpy as np

from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import Schema
from .task import TaskSpec
# The walk and BFS run in C++ now (cpp/src/graph.cpp), so the RNG and
# sampling helpers this module used to need are gone with them.
from .traversal import TraversalResult

__all__ = ["ColumnarStore", "ColumnarTraversal"]


# datetime64 ticks per second, by resolution code. The unit is read off the
# dtype rather than assumed: the original divided by a hardcoded 1e9, right
# only for datetime64[ns], and pandas >= 3 defaults to microseconds -- which
# put every timestamp 1000x too early so no temporal bound held.
_TICKS_PER_SEC = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9,
                  "m": 1.0 / 60, "h": 1.0 / 3600, "D": 1.0 / 86400}


def _column(frame: Any, name: str) -> np.ndarray:
    """One column as a numpy array, from a DataFrame or a dict of arrays.

    Duck-typed on purpose: pandas, polars and pyarrow all expose to_numpy(),
    and a plain dict of numpy arrays needs nothing. Ingest is the binding's
    job, so it accepts whatever the caller already has -- but the store holds
    numpy from here on, and imports no dataframe library.
    """
    col = frame[name]
    to_numpy = getattr(col, "to_numpy", None)
    return np.asarray(to_numpy() if to_numpy is not None else col)


def _columns_of(frame: Any) -> list:
    cols = getattr(frame, "columns", None)
    return list(cols if cols is not None else frame.keys())


def _n_rows(frame: Any) -> int:
    cols = _columns_of(frame)
    return len(_column(frame, cols[0])) if cols else 0


def _epoch_seconds(values) -> np.ndarray:
    """A datetime-ish column -> float64 epoch seconds, NaN where missing."""
    arr = np.asarray(values)
    if arr.dtype == object:
        out = np.empty(len(arr), dtype=np.float64)
        for i, v in enumerate(arr):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[i] = np.nan
            elif isinstance(v, datetime):
                out[i] = v.timestamp()
            elif isinstance(v, np.datetime64):
                out[i] = (np.nan if np.isnat(v) else
                          np.datetime64(v, "us").astype("int64") / 1e6)
            else:
                out[i] = float(v)
        return out
    if not np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype(np.float64)
    unit = np.datetime_data(arr.dtype)[0]
    per_sec = _TICKS_PER_SEC.get(unit)
    if per_sec is None:                        # exotic resolution
        arr = arr.astype("datetime64[us]")
        per_sec = _TICKS_PER_SEC["us"]
    out = arr.astype("int64").astype(np.float64) / per_sec
    out[np.isnat(arr)] = np.nan
    return out


def _is_missing(v: Any) -> bool:
    """Scalar null test without pandas: None, NaN, or NaT."""
    if v is None:
        return True
    if isinstance(v, (list, tuple, np.ndarray)):
        return False
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, np.datetime64):
        return bool(np.isnat(v))
    if isinstance(v, np.floating):
        return bool(np.isnan(v))
    return False


def _notna_mask(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return ~np.isnat(arr)
    if np.issubdtype(arr.dtype, np.floating):
        return ~np.isnan(arr)
    if arr.dtype == object:
        return np.array([not _is_missing(v) for v in arr], dtype=bool)
    return np.ones(len(arr), dtype=bool)


def _scalar(v: Any) -> Any:
    """numpy scalar -> python scalar; datetime64 -> aware datetime."""
    if isinstance(v, np.datetime64):
        secs = np.datetime64(v, "us").astype("int64") / 1e6
        return datetime.fromtimestamp(secs, timezone.utc)
    if isinstance(v, np.generic):
        return v.item()
    return v


class _Table:
    """One table's columns plus the derived vectors the traversal needs."""

    def __init__(self, name: str, frame: Any, pkey: Optional[str],
                 time_col: Optional[str], feature_cols: list,
                 fk_cols: dict):
        self.name = name
        self.pkey = pkey
        self.time_col = time_col
        self.feature_cols = feature_cols
        self.fk_cols = fk_cols            # fk column -> parent table
        self.n = _n_rows(frame)
        present = set(_columns_of(frame))
        wanted = set(feature_cols) | set(fk_cols)
        if pkey is not None:
            wanted.add(pkey)
        if time_col is not None:
            wanted.add(time_col)
        # Columns are pulled into numpy once; the source frame is not kept, so
        # nothing downstream depends on a dataframe library's semantics.
        self.cols: dict = {c: _column(frame, c) for c in wanted if c in present}
        self.ids = (self.cols[pkey] if pkey is not None and pkey in self.cols
                    else np.arange(self.n))
        # Built lazily: only scalar lookups (node_of, the entity retriever)
        # need it, and they are per-query. Link resolution is vectorized
        # instead -- see positions_of.
        self._id_pos: Optional[dict] = None
        self._sorted: Optional[tuple] = None
        if time_col is not None and time_col in self.cols:
            self.ts = _epoch_seconds(self.cols[time_col])
        else:
            self.ts = np.full(self.n, np.nan)
        # Non-null feature cell count per row, plus timestamp-as-cell when
        # the time column is also a declared feature (mirrors the default
        # row cell accounting).
        counts = np.zeros(self.n, dtype=np.int32)
        for col in feature_cols:
            if col in self.cols:
                counts += _notna_mask(self.cols[col]).astype(np.int32)
        self.cell_counts = counts

    @property
    def id_pos(self) -> dict:
        if self._id_pos is None:
            pos: dict = {}
            for i, v in enumerate(self.ids):
                pos.setdefault(_scalar(v), i)
            self._id_pos = pos
        return self._id_pos

    def positions_of(self, values: np.ndarray) -> np.ndarray:
        """Ids -> row positions, -1 where absent. Vectorized: a Python loop
        here is O(edges) and dominated the build (pandas did this with
        Index.get_indexer, and dropping pandas must not drop the vectorization
        with it)."""
        values = np.asarray(values)
        if self._sorted is None:
            order = np.argsort(self.ids, kind="stable")
            self._sorted = (order, np.asarray(self.ids)[order])
        order, sorted_ids = self._sorted
        if len(sorted_ids) == 0:
            return np.full(len(values), -1, dtype=np.int64)
        idx = np.searchsorted(sorted_ids, values)
        idx_clipped = np.clip(idx, 0, len(sorted_ids) - 1)
        hit = sorted_ids[idx_clipped] == values
        return np.where(hit, order[idx_clipped], -1).astype(np.int64)


class ColumnarStore:
    """Array-backed store over a schema's tables (plus optional task tables).

    ``frames`` maps physical table name -> DataFrame. ``task_frames`` maps a
    task table name -> DataFrame whose ``entity_col`` column links it to
    ``entity_table`` (``task_links``), with every remaining column treated
    as a cell. Node ids are assigned per table in sorted-name order,
    physical tables first, then task tables.
    """

    def __init__(self, schema: Schema, frames: dict, *,
                 task_frames: Optional[dict] = None,
                 task_links: Optional[dict[str, tuple[str, str, str]]] = None):
        self.schema = schema
        self.tables: dict[str, _Table] = {}
        for tdef in schema.tables:
            frame = frames[tdef.name]
            fk_cols = {l.fk_column: l.to_table
                       for l in schema.links_from(tdef.name)}
            self.tables[tdef.name] = _Table(
                tdef.name, frame, tdef.primary_key, tdef.time_column,
                [c.name for c in tdef.columns
                 if c.name in set(_columns_of(frame))],
                fk_cols)
        self.task_tables: dict[str, _Table] = {}
        self.task_links = dict(task_links or {})
        for name, frame in (task_frames or {}).items():
            entity_table, entity_col, time_col = self.task_links[name]
            cells = [c for c in _columns_of(frame) if c != entity_col]
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
        if fk not in table.cols:
            return np.full(table.n, -1, dtype=np.int64)
        values = table.cols[fk]
        pos = parent.positions_of(values)
        missing = ~_notna_mask(values)
        if missing.any():
            pos = np.where(missing, -1, pos)
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
        pos = t.id_pos.get(_scalar(entity_id))
        return None if pos is None else self.base[table] + int(pos)

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
        # Index the stored column arrays directly. frame.iloc[pos] built a
        # pandas Series for every materialized row -- on the one path this
        # module exists to make cheap.
        cells = {}
        for col in table.feature_cols:
            if col not in table.cols:
                continue
            v = table.cols[col][pos]
            if _is_missing(v):
                continue
            cells[col] = _scalar(v)
        ts = None
        if not math.isnan(self.node_ts[node]):
            ts = datetime.fromtimestamp(float(self.node_ts[node]),
                                        timezone.utc)
        parents = {}
        for fk, parent_name in table.fk_cols.items():
            if fk not in table.cols:
                continue
            v = table.cols[fk][pos]
            if not _is_missing(v):
                parents[fk if name in self.tables else "__entity__"] = _scalar(v)
        rid = _scalar(table.ids[pos])
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
            for eid in ids:
                pos = t.id_pos.get(_scalar(eid))
                if pos is None:
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
        if task_spec.target_column in t.cols:
            ok[base:base + t.n] = _notna_mask(t.cols[task_spec.target_column])
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
        ents = t.cols[entity_col]
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
