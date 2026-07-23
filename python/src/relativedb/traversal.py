"""Pluggable, temporally-safe relational graph traversal strategies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from .retrieve import Row, TemporalBound
from .schema import Schema
from .evaluate import eval_bool, eval_value
from .relql.ast import ColumnRef, TaskType
from .task import TaskSpec

__all__ = ["GraphAccess", "GraphTraversal", "TraversalResult",
           "BreadthFirstTraversal", "ReferenceTraversal"]


_U32 = 0xFFFF_FFFF
_U64 = 0xFFFF_FFFF_FFFF_FFFF


class _StdRng:
    """rand 0.9.1 StdRng-compatible ChaCha12 stream.

    The reference sampler's observable ordering depends on this exact stream,
    including seed_from_u64's PCG expansion and Canon integer sampling.
    """

    def __init__(self, seed: int):
        state = seed & _U64
        key = []
        for _ in range(8):
            state = (state * 6364136223846793005 + 11634580027462260723) & _U64
            x = (((state >> 18) ^ state) >> 27) & _U32
            rot = state >> 59
            key.append(((x >> rot) | (x << ((-rot) & 31))) & _U32)
        self._key = key
        self._counter = 0
        self._buf: list[int] = []
        self._at = 0

    @staticmethod
    def _rotl(x: int, n: int) -> int:
        return ((x << n) | (x >> (32 - n))) & _U32

    @classmethod
    def _quarter(cls, x: list[int], a: int, b: int, c: int, d: int):
        x[a] = (x[a] + x[b]) & _U32; x[d] ^= x[a]; x[d] = cls._rotl(x[d], 16)
        x[c] = (x[c] + x[d]) & _U32; x[b] ^= x[c]; x[b] = cls._rotl(x[b], 12)
        x[a] = (x[a] + x[b]) & _U32; x[d] ^= x[a]; x[d] = cls._rotl(x[d], 8)
        x[c] = (x[c] + x[d]) & _U32; x[b] ^= x[c]; x[b] = cls._rotl(x[b], 7)

    def _refill(self):
        out = []
        constants = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]
        for block in range(4):
            counter = (self._counter + block) & _U64
            initial = constants + self._key + [counter & _U32, counter >> 32, 0, 0]
            x = initial.copy()
            for _ in range(6):
                self._quarter(x, 0, 4, 8, 12); self._quarter(x, 1, 5, 9, 13)
                self._quarter(x, 2, 6, 10, 14); self._quarter(x, 3, 7, 11, 15)
                self._quarter(x, 0, 5, 10, 15); self._quarter(x, 1, 6, 11, 12)
                self._quarter(x, 2, 7, 8, 13); self._quarter(x, 3, 4, 9, 14)
            out.extend((a + b) & _U32 for a, b in zip(x, initial))
        self._counter = (self._counter + 4) & _U64
        self._buf, self._at = out, 0

    def u32(self) -> int:
        if self._at >= len(self._buf):
            self._refill()
        value = self._buf[self._at]
        self._at += 1
        return value

    def u64(self) -> int:
        return self.u32() | (self.u32() << 32)

    def range(self, stop: int, start: int = 0) -> int:
        if not start < stop:
            raise ValueError("empty RNG range")
        width = stop - start
        # usize delegates to u32 for all graph sizes representable here.
        product = self.u32() * width
        result, low = product >> 32, product & _U32
        if low > ((-width) & _U32):
            new_hi = (self.u32() * width) >> 32
            if low + new_hi > _U32:
                result += 1
        return start + result

    def range_inclusive(self, stop: int) -> int:
        return self.range(stop + 1)

    def uniform_range(self, stop: int) -> int:
        """Sampling from a constructed Uniform(0, stop), as rand's rejection
        index sampler does (distinct from random_range's Canon path)."""
        threshold = ((-stop) & _U32) % stop
        while True:
            product = self.u32() * stop
            high, low = product >> 32, product & _U32
            if low >= threshold:
                return high


def _rand_sample(rng: _StdRng, length: int, amount: int) -> list[int]:
    """rand::seq::index::sample for the u32-sized cases used by contexts."""
    if not 0 <= amount <= length:
        raise ValueError("invalid sample size")
    if amount < 163:
        j = int(length >= 500_000)
        use_inplace = (amount > 11 and length < ([10.0, 70.0 / 9.0][j]
                       + [1.6, 8.0 / 45.0][j] * amount) * amount)
    else:
        j = int(length >= 500_000)
        use_inplace = length < [270.0, 330.0 / 9.0][j] * amount
    if use_inplace:
        indices = list(range(length))
        for i in range(amount):
            k = rng.range(length, i)
            indices[i], indices[k] = indices[k], indices[i]
        return indices[:amount]
    # Floyd is the reference path for the small fallback/BFS samples typical
    # of the 8K-cell evaluator. Rejection is only selected for huge samples.
    if amount < 163:
        indices: list[int] = []
        for j in range(length - amount, length):
            t = rng.range_inclusive(j)
            try:
                pos = indices.index(t)
            except ValueError:
                pass
            else:
                indices[pos] = j
            indices.append(t)
        return indices
    # rand's rejection sampler preserves unique draw order.
    chosen: set[int] = set()
    values: list[int] = []
    while len(values) < amount:
        value = rng.uniform_range(length)
        if value not in chosen:
            chosen.add(value); values.append(value)
    return values


class GraphAccess(Protocol):
    def entities(self, table: str, ids: Sequence[Any],
                 bound: TemporalBound) -> list[Row]: ...
    def children(self, link, parent_id: Any, bound: TemporalBound,
                 limit: int) -> list[Row]: ...
    def cohort(self, table: str, anchor: Any, bound: TemporalBound,
               limit: int) -> list[Any]: ...
    def all_ids(self, table: str) -> Optional[list[Any]]: ...
    def all_rows(self, table: str) -> Optional[list[Row]]: ...


@dataclass(frozen=True)
class TraversalResult:
    rows: tuple[Row, ...] = ()
    focal_row_keys: frozenset[tuple[str, Any]] = frozenset()
    truncated_children: int = 0
    hit_cell_budget: bool = False
    # Stable snapshot-wide node ids. Parents may be referenced even when their
    # cells were not selected into this context, matching the reference graph.
    node_ids: tuple[tuple[tuple[str, Any], int], ...] = ()


class GraphTraversal(Protocol):
    def traverse(self, schema: Schema, graph: GraphAccess, entity_table: str,
                 entity_id: Any, bound: TemporalBound, policy: Any, *,
                 query: Any = None) -> TraversalResult: ...


def _newest_first(row: Row):
    return (row.timestamp is None,
            -(row.timestamp.timestamp() if row.timestamp else 0.0))


class BreadthFirstTraversal:
    """The engine's original cohort-seeded, bounded breadth-first traversal."""

    def traverse(self, schema, graph, entity_table, entity_id, bound, policy,
                 *, query=None) -> TraversalResult:
        rows: list[Row] = []
        visited: set[tuple[str, Any]] = set()
        focal: set[tuple[str, Any]] = set()
        cells = 0
        truncated = 0
        hit_budget = False

        def admit(candidates, is_focal=False):
            nonlocal cells, hit_budget
            fresh = []
            for row in candidates:
                if not bound.admits_row(row) or row.key in visited:
                    continue
                cost = len(row.cells) + (1 if row.timestamp is not None else 0)
                if rows and cells + cost > policy.max_context_cells:
                    hit_budget = True
                    break
                visited.add(row.key)
                rows.append(row)
                cells += cost
                fresh.append((row, is_focal))
                if is_focal:
                    focal.add(row.key)
            return fresh

        frontier = admit(graph.entities(entity_table, [entity_id], bound), True)
        if not frontier:
            return TraversalResult()
        if policy.cohort_size > 0:
            ids = graph.cohort(entity_table, entity_id, bound,
                               policy.cohort_size)
            frontier += admit(graph.entities(entity_table, ids, bound), False)

        fk_parent = {t.name: {l.fk_column: l.to_table
                              for l in schema.links_from(t.name)}
                     for t in schema.tables}
        for hop in range(policy.effective_hops):
            if hit_budget:
                break
            fanout = policy.fanout_at(hop)
            nxt = []
            wanted: dict[tuple[str, bool], list[Any]] = {}
            for row, is_focal in frontier:
                for fk, pid in row.parents.items():
                    table = fk_parent.get(row.table, {}).get(fk)
                    if table is not None and (table, pid) not in visited:
                        wanted.setdefault((table, is_focal), []).append(pid)
            for (table, is_focal), ids in wanted.items():
                nxt += admit(graph.entities(table, ids, bound), is_focal)
            for row, is_focal in frontier:
                for link in schema.links_to(row.table):
                    kids = [r for r in graph.children(
                        link, row.id, bound, fanout + 1) if bound.admits_row(r)]
                    if len(kids) > fanout:
                        truncated += len(kids) - fanout
                    if policy.prefer_latest:
                        kids.sort(key=_newest_first)
                    nxt += admit(kids[:fanout], is_focal)
                    if hit_budget:
                        break
                if hit_budget:
                    break
            frontier = nxt
            if not frontier:
                break
        return TraversalResult(tuple(rows), frozenset(focal), truncated,
                               hit_budget)


class ReferenceTraversal:
    """Reference tiering: target BFS, graph-walk peers, random table fallback."""

    def __init__(self, task_spec_factory=None, task_graph_factory=None):
        # Engine executes timestamp cohorts consecutively. Direct-target rows
        # at one timestamp share the exact same temporally filtered graph, so
        # retain only that most-recent CSR snapshot. A one-entry cache captures
        # the reuse without retaining a full graph for every historical date.
        self._native_graph_key = None
        self._native_graph_value = None
        self.task_spec_factory = task_spec_factory or TaskSpec.from_query
        self.task_graph_factory = task_graph_factory

    def traverse(self, schema, graph, entity_table, entity_id, bound, policy,
                 *, query=None) -> TraversalResult:
        rows_by_table = {t.name: (graph.all_rows(t.name) or [])
                         for t in schema.tables}
        rows = [r for t in schema.tables for r in rows_by_table[t.name]]
        physical_node_ids = {r.key: i for i, r in enumerate(rows)}
        task_node_ids: dict[tuple[str, Any], int] = {}
        reference_p2f_order = None
        reference_f2p_order = None
        sampling_table = entity_table
        target_key = (entity_table, entity_id)
        task_spec = None
        if query is not None:
            task_type = query.task_type(schema)
            task_spec = self.task_spec_factory(query, task_type)
            if not isinstance(task_spec, TaskSpec):
                raise TypeError("task_spec_factory must return a TaskSpec")
            if not task_spec.direct_target:
                sampling_table = task_spec.table_name
                anchor = bound.as_of
                if self.task_graph_factory is not None:
                    supplied = self.task_graph_factory(
                        task_spec, entity_id, anchor)
                    if (not isinstance(supplied, tuple)
                            or len(supplied) not in (3, 4, 5)):
                        raise TypeError(
                            "task_graph_factory must return "
                            "(rows, node_ids, target_key[, p2f_order[, "
                            "f2p_order]])")
                    supplied_rows, supplied_ids, target_key = supplied[:3]
                    if len(supplied) == 4:
                        reference_p2f_order = supplied[3]
                    elif len(supplied) == 5:
                        reference_p2f_order = supplied[3]
                        reference_f2p_order = supplied[4]
                    task_rows = list(supplied_rows)
                    task_node_ids = dict(supplied_ids)
                    if target_key not in task_node_ids:
                        raise RuntimeError(
                            "materialized task graph did not assign the focal "
                            "task row a stable node id")
                else:
                    span = next((a.window.span()
                                 for a in query.target_aggregations
                                 if a.window is not None), None)
                    task_rows = []
                    entity_rows = rows_by_table.get(entity_table, [])
                    task_base = len(physical_node_ids)
                    task_stride = policy.num_history_windows + 1
                    # Self-label history windows are evaluated per entity, but
                    # eval_* aggregates over every row of the aggregated table
                    # it is handed (it expects a per-entity label context).
                    # Scope those tables to rows owned by the entity — via the
                    # FK chain up to the entity table — or every entity's
                    # window would be labeled with the whole cohort's outcomes
                    # (e.g. NOT EXISTS(orders.*) would be 0 for everyone
                    # whenever anyone ordered).
                    agg_tables = {a.column.table
                                  for a in query.target_aggregations
                                  if a.column.table != entity_table}
                    key_index = {r.key: r for r in rows}
                    owners_memo: dict[tuple, frozenset] = {}

                    def owners(row) -> frozenset:
                        got = owners_memo.get(row.key)
                        if got is not None:
                            return got
                        owners_memo[row.key] = frozenset()   # cycle guard
                        out: set = set()
                        for link in schema.links_from(row.table):
                            pid = row.parents.get(link.fk_column)
                            pids = (pid if isinstance(pid, (list, tuple))
                                    else (pid,))
                            for one in pids:
                                if one is None:
                                    continue
                                if link.to_table == entity_table:
                                    out.add(one)
                                else:
                                    parent = key_index.get(
                                        (link.to_table, one))
                                    if parent is not None:
                                        out |= owners(parent)
                        owners_memo[row.key] = frozenset(out)
                        return owners_memo[row.key]

                    for entity_i, entity in enumerate(entity_rows):
                        # Unknown focal target at the requested anchor.
                        if entity.id == entity_id:
                            target_id = (entity.id, anchor, "target")
                            task_rows.append(Row(
                                sampling_table, target_id, {}, timestamp=anchor,
                                parents={"__entity__": entity.id}))
                            target_key = (sampling_table, target_id)
                            task_node_ids[target_key] = (
                                task_base + entity_i * task_stride)
                        if anchor is None or span is None:
                            continue
                        for k in range(1, policy.num_history_windows + 1):
                            ts = anchor - span * k
                            # A historical FOLLOWING label may use outcomes
                            # after its own task timestamp, but never after the
                            # focal prediction anchor.
                            label_cutoff = min(anchor, ts + span)
                            visible = {}
                            for name, table_rows in rows_by_table.items():
                                vis = [r for r in table_rows
                                       if r.timestamp is None
                                       or r.timestamp <= label_cutoff]
                                if name in agg_tables:
                                    vis = [r for r in vis
                                           if entity.id in owners(r)]
                                visible[name] = vis
                            if task_type is TaskType.BINARY_CLASSIFICATION:
                                value = 1.0 if eval_bool(
                                    query.target, visible, entity.cells, ts) else 0.0
                            else:
                                value = eval_value(query.target, visible,
                                                   entity.cells, ts)
                                if isinstance(value, bool):
                                    value = 1.0 if value else 0.0
                                if not isinstance(value, (int, float)):
                                    continue
                            history = Row(
                                sampling_table, (entity.id, ts, k),
                                {task_spec.target_column: value}, timestamp=ts,
                                parents={"__entity__": entity.id})
                            task_rows.append(history)
                            task_node_ids[history.key] = (
                                task_base + entity_i * task_stride + k)
                materialized_by_table: dict[str, list[Row]] = {}
                for task_row in task_rows:
                    materialized_by_table.setdefault(
                        task_row.table, []).append(task_row)
                rows_by_table.update(materialized_by_table)
                rows.extend(task_rows)
        by_key = {r.key: r for r in rows}
        target = by_key.get(target_key)
        if target is None or not bound.admits_row(target):
            return TraversalResult()

        links_from = {t.name: {l.fk_column: l for l in schema.links_from(t.name)}
                      for t in schema.tables}
        p2f: dict[tuple[str, Any], list[Row]] = {}
        p2f_walk: dict[tuple[str, Any], list[Row]] = {}
        isolated_task = bool(
            task_spec is not None and not task_spec.direct_target
            and all(not row.parents
                    for row in rows_by_table.get(sampling_table, ())))
        edge_rows = (rows_by_table[sampling_table] if isolated_task else rows)
        for row in edge_rows:
            if (task_spec is not None and row.table == sampling_table
                    and "__entity__" in row.parents):
                p2f.setdefault((entity_table, row.parents["__entity__"]), []).append(row)
                p2f_walk.setdefault(
                    (entity_table, row.parents["__entity__"]), []).append(row)
            for fk, pid in row.parents.items():
                if fk.startswith("__parent__:"):
                    parent_table = fk.split(":", 1)[1]
                    for one in (pid if isinstance(pid, (list, tuple)) else (pid,)):
                        # Other task tables participate in random walks, but
                        # reference BFS filters them before constructing its
                        # task frontier.
                        p2f_walk.setdefault((parent_table, one), []).append(row)
                    continue
                link = links_from.get(row.table, {}).get(fk)
                if link is not None:
                    for one in (pid if isinstance(pid, (list, tuple)) else (pid,)):
                        p2f.setdefault((link.to_table, one), []).append(row)
                        p2f_walk.setdefault((link.to_table, one), []).append(row)
        # pre.rs sorts every parent-to-foreign adjacency by Option<timestamp>:
        # None first, then ascending event time. BFS samples indices from this
        # ordered list, so insertion/table order is not interchangeable.
        def order_children(parent_key, children):
            if reference_p2f_order is None:
                children.sort(key=lambda row: (
                    row.timestamp is not None,
                    row.timestamp.timestamp()
                    if row.timestamp is not None else 0.0))
                return
            expected = reference_p2f_order.get(parent_key)
            if expected is None:
                if children:
                    raise RuntimeError(
                        f"reference p2f order is missing parent {parent_key!r}")
                return
            rank = {key: index for index, key in enumerate(expected)}
            actual = {row.key for row in children}
            if actual != set(expected):
                missing = set(expected) - actual
                extra = actual - set(expected)
                raise RuntimeError(
                    f"reference p2f order disagrees at {parent_key!r}: "
                    f"missing={sorted(map(str, missing))[:3]}, "
                    f"extra={sorted(map(str, extra))[:3]}")
            children.sort(key=lambda row: rank[row.key])

        for parent_key, children in p2f_walk.items():
            order_children(parent_key, children)
        for parent_key, children in p2f.items():
            if reference_p2f_order is None:
                order_children(parent_key, children)
            else:
                expected = reference_p2f_order.get(parent_key)
                if expected is None:
                    raise RuntimeError(
                        f"reference p2f order is missing parent {parent_key!r}")
                rank = {key: index for index, key in enumerate(expected)}
                unknown = [row.key for row in children if row.key not in rank]
                if unknown:
                    raise RuntimeError(
                        f"reference p2f order is missing children for "
                        f"{parent_key!r}: {unknown[:3]!r}")
                children.sort(key=lambda row: rank[row.key])

        parents_cache: dict[tuple[str, Any], tuple[Row, ...]] = {}

        def parents(row):
            cached = parents_cache.get(row.key)
            if cached is not None:
                return cached
            out = []
            if (row.table == sampling_table and task_spec is not None
                    and not task_spec.direct_target):
                entity = by_key.get((entity_table,
                                     row.parents.get("__entity__")))
                if entity is not None:
                    out.append(entity)
            for fk, pid in row.parents.items():
                if fk.startswith("__parent__:"):
                    parent_table = fk.split(":", 1)[1]
                    for one in (pid if isinstance(pid, (list, tuple)) else (pid,)):
                        parent = by_key.get((parent_table, one))
                        if parent is not None:
                            out.append(parent)
                    continue
                link = links_from.get(row.table, {}).get(fk)
                if link:
                    for one in (pid if isinstance(pid, (list, tuple)) else (pid,)):
                        parent = by_key.get((link.to_table, one))
                        if parent is not None:
                            out.append(parent)
            result = tuple(out)
            if reference_f2p_order is not None:
                expected = reference_f2p_order.get(row.key, ())
                if {parent.key for parent in result} != set(expected):
                    raise RuntimeError(
                        f"reference f2p order disagrees at {row.key!r}")
                rank = {key: index for index, key in enumerate(expected)}
                result = tuple(sorted(result, key=lambda parent: rank[parent.key]))
            parents_cache[row.key] = result
            return result

        def temporally_valid(row, anchor):
            return (row.timestamp is None or anchor.timestamp is None
                    or row.timestamp <= anchor.timestamp)

        target_node_idx = (task_node_ids.get(target.key)
                           if target.key in task_node_ids
                           else physical_node_ids[target.key])
        # Sampler::new_impl first expands the user-facing context seed, then
        # seq_build expands that stored seed again for step zero.
        context_seed = _StdRng(policy.seed).u64()
        step_seed = _StdRng(context_seed).u64()
        walk_rng = _StdRng((step_seed + target_node_idx
                            + 0xD0D0_D0D0_D0D0_D0D0) & _U64)
        bfs_rng = _StdRng((step_seed + target_node_idx
                           + 0xB0B0_B0B0_B0B0_B0B0) & _U64)
        fallback_rng = _StdRng((step_seed + target_node_idx
                                + 0xA5A5_A5A5_A5A5_A5A5) & _U64)

        # A 10k x 20 reference walk revisits the same small neighborhood many
        # thousands of times.  Filtering and allocating that neighbor list on
        # every step dominated end-to-end query execution even though the
        # graph and focal temporal cutoff are invariant for this traversal.
        # Cache the exact ordered list; RNG consumption and resulting walks are
        # unchanged.
        walk_neighbor_cache: dict[tuple[str, Any], tuple[Row, ...]] = {}

        def walk_neighbors(row):
            cached = walk_neighbor_cache.get(row.key)
            if cached is not None:
                return cached
            result = tuple(parents(row)) + tuple(
                r for r in p2f_walk.get(row.key, ())
                if temporally_valid(r, target))
            walk_neighbor_cache[row.key] = result
            return result

        # ReferenceTraversal has one execution contract: the exact native
        # rand-0.9.1 graph walk. Missing/old native libraries are hard errors;
        # never switch algorithms or performance regimes implicitly.
        from .rt_native import load_lib
        native_lib = load_lib()._lib

        graph_identity = getattr(graph, "index", graph)
        graph_key = (id(graph_identity), target.table, target.timestamp)
        cached_graph = (self._native_graph_value
                        if task_spec is not None and task_spec.direct_target
                        and self._native_graph_key == graph_key else None)
        if cached_graph is not None and target.key in cached_graph[1]:
            discovered, position, offsets, neighbors_array, eligible_base = cached_graph
        else:
            discovered = [target]
            position = {target.key: 0}
            neighbor_rows: list[tuple[Row, ...]] = []
            at = 0
            while at < len(discovered):
                nbrs = walk_neighbors(discovered[at])
                neighbor_rows.append(nbrs)
                for neighbor in nbrs:
                    if neighbor.key not in position:
                        position[neighbor.key] = len(discovered)
                        discovered.append(neighbor)
                at += 1
            offsets = np.empty(len(discovered) + 1, dtype=np.int32)
            offsets[0] = 0
            flat: list[int] = []
            for i, nbrs in enumerate(neighbor_rows):
                flat.extend(position[row.key] for row in nbrs)
                offsets[i + 1] = len(flat)
            neighbors_array = np.asarray(flat, dtype=np.int32)
            eligible_base = np.asarray([
                row.table == target.table and temporally_valid(row, target)
                for row in discovered], dtype=np.uint8)
            if task_spec is not None and task_spec.direct_target:
                self._native_graph_key = graph_key
                self._native_graph_value = (
                    discovered, position, offsets, neighbors_array,
                    eligible_base)
        target_position = position[target.key]
        eligible = eligible_base.copy()
        eligible[target_position] = 0
        counts = np.zeros(len(discovered), dtype=np.uint32)
        rc = native_lib.rt_reference_walk_counts(
            len(discovered), offsets, neighbors_array, target_position, eligible,
            (step_seed + target_node_idx
             + 0xD0D0_D0D0_D0D0_D0D0) & _U64,
            policy.num_walks, policy.walk_length, counts)
        if rc:
            raise RuntimeError("native reference walk rejected its graph")
        visits = {row.key: int(count)
                  for row, count in zip(discovered, counts) if count}

        visit_keys = list(visits)
        if visit_keys:
            seeds = np.asarray([
                (step_seed + task_node_ids.get(
                    key, physical_node_ids.get(key))) & _U64
                for key in visit_keys], dtype=np.uint64)
            tie_values = np.empty(len(seeds), dtype=np.uint64)
            rc = native_lib.rt_stdrng_first_u64_batch(
                seeds, len(seeds), tie_values)
            if rc:
                raise RuntimeError("native tie RNG rejected its seeds")
            tie = dict(zip(visit_keys, map(int, tie_values)))
        else:
            tie = {}
        if policy.prefer_latest:
            def peer_key(key):
                row = by_key[key]
                ts = row.timestamp.timestamp() if row.timestamp else -math.inf
                return (-ts, -visits[key], tie[key])
        else:
            def peer_key(key):
                return (-visits[key], tie[key])
        def has_seed_label(row: Row) -> bool:
            if task_spec is None:
                return True
            value = row.cells.get(task_spec.target_column)
            return value is not None and not (
                isinstance(value, float) and math.isnan(value))

        tier1 = [key for key in sorted(visits, key=peer_key)
                 if has_seed_label(by_key[key])]
        visited_depth: dict[tuple[str, Any], int] = {}
        emitted: set[tuple[str, Any]] = set()
        ordered: list[Row] = []
        focal: set[tuple[str, Any]] = set()
        cells = 1  # target cell is emitted separately and first
        full = False
        db_tables = {table.name for table in schema.tables}

        def cell_count(row):
            if (row.table == sampling_table and task_spec is not None
                    and not task_spec.direct_target):
                # Materialized task nodes carry both their target cell and
                # timestamp as schema cells in the reference dataset. The
                # focal target value is unknown here but is still the masked
                # first token and participates in local BFS accounting.
                return (len(row.cells)
                        + (1 if row.timestamp is not None
                           and task_spec.time_column not in row.cells else 0)
                        + (1 if row.key == target.key
                           and task_spec.target_column not in row.cells else 0))
            table = schema.require_table(row.table)
            declared = {c.name for c in table.columns}
            n = sum(1 for c, v in row.cells.items()
                    if c in declared and c != table.primary_key and v is not None)
            n += sum(1 for l in schema.links_from(row.table)
                     if l.feature_type is not None
                     and row.parents.get(l.fk_column) is not None)
            return n

        def extend(seed, is_focal=False):
            nonlocal cells, full
            local_cells = 0
            f2p_stack: list[tuple[int, Row]] = []
            p2f_levels: list[list[Row]] = [[seed]]
            while True:
                if f2p_stack:
                    depth, row = f2p_stack.pop()
                else:
                    depth = next((i for i, level in enumerate(p2f_levels) if level), -1)
                    if depth < 0:
                        return
                    level = p2f_levels[depth]
                    selected = bfs_rng.range(len(level))
                    level[selected], level[-1] = level[-1], level[selected]
                    row = level.pop()
                previous = visited_depth.get(row.key)
                if previous is not None and previous <= depth:
                    continue
                cost = cell_count(row)
                local_cells += cost
                if local_cells >= policy.local_context_cells:
                    return
                visited_depth[row.key] = depth
                if row.key not in emitted:
                    emitted_cost = cost
                    if row.key == target.key:
                        emitted_cost = max(0, emitted_cost - 1)
                    if cells >= policy.max_context_cells:
                        full = True
                        return
                    emitted.add(row.key)
                    ordered.append(row)
                    cells += emitted_cost
                    # The reference fills cell-by-cell and can stop partway
                    # through this row. Keep the row so collation can perform
                    # that exact final-cell truncation.
                    if cells >= policy.max_context_cells:
                        full = True
                    if is_focal:
                        focal.add(row.key)
                for parent in parents(row):
                    f2p_stack.append((depth + 1, parent))
                seed_cutoff = (seed.timestamp if query is not None
                               else (seed.timestamp or bound.as_of))
                valid_kids = [
                    r for r in p2f.get(row.key, [])
                    if r.timestamp is None or (
                        seed_cutoff is not None and r.timestamp <= seed_cutoff)]
                # Reference BFS never subsamples task edges. It keeps task
                # children only when they belong to the seed's task table,
                # then independently samples database children to bfs_width.
                task_kids = [r for r in valid_kids
                             if r.table not in db_tables
                             and r.table == seed.table]
                db_kids = [r for r in valid_kids if r.table in db_tables]
                if len(db_kids) > policy.bfs_width:
                    selected_kids = _rand_sample(
                        bfs_rng, len(db_kids), policy.bfs_width)
                    db_kids = [db_kids[i] for i in selected_kids]
                kids = task_kids + db_kids
                while len(p2f_levels) <= depth + 1:
                    p2f_levels.append([])
                p2f_levels[depth + 1].extend(kids)

        extend(target, True)
        for key in tier1:
            if full:
                break
            extend(by_key[key], False)
        if not full:
            fallback_rows = rows_by_table[sampling_table]
            amount = min(max(policy.max_context_cells - cells, 0),
                         len(fallback_rows))
            fallback = [fallback_rows[i].key for i in
                        _rand_sample(fallback_rng, len(fallback_rows), amount)]
            for key in fallback:
                if full:
                    break
                row = by_key[key]
                if (key == target.key or key in visits
                        or not temporally_valid(row, target)
                        or not has_seed_label(row)):
                    continue
                extend(row, False)
        # Snapshot order is the stable global node identity contract. Virtual
        # task rows follow physical rows in deterministic entity/time order.
        node_id_map = {**physical_node_ids, **task_node_ids}
        node_ids = tuple(node_id_map.items())
        return TraversalResult(tuple(ordered), frozenset(focal), 0, full,
                               node_ids)


class _PolicyView:
    def __init__(self, base, max_cells, cohort_size):
        self._base = base
        self.max_context_cells = max_cells
        self.cohort_size = cohort_size
        self.prefer_latest = base.prefer_latest
    @property
    def effective_hops(self): return self._base.effective_hops
    def fanout_at(self, hop): return self._base.fanout_at(hop)
