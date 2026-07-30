"""Array-backed context assembly, pure Python + numpy.

Port of the former ``cpp/src/graph.cpp`` (removed when context creation moved
fully into Python). The BFS emission's observable output depends on the exact
order the RNG is drawn in — one extra or missing draw shifts every subsequent
choice — so its structure follows the original statement for statement. The
peer-ranking WALK is the exception: it is vectorized over walks on a raw
PCG64 stream (deterministic per seed, not the original per-walk stream) —
one num_walks-wide draw per step, the same protocol as
``traversal._reference_walk_counts`` so the row and columnar paths sample
identical contexts. The fingerprint tests pin the result.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from typing import Optional

import numpy as np

from .traversal import _StdRng, _U64, _rand_sample

__all__ = ["ContextGraph", "ContextTruncated"]


class ContextTruncated(RuntimeError):
    """The emitted-node buffer bound, so part of the context was dropped.

    Nodes and cells are different quantities -- a row whose feature columns are
    all null costs zero cells but still occupies a node slot -- so a buffer
    sized from the cell budget can bind on a real graph. That silently drops
    the tail of a context before the model ever sees it, which is invisible in
    every metric except accuracy. It is an error, never a truncation.
    """


class ContextGraph:
    """A built graph. Construct once per snapshot, assemble many contexts.

    Node numbering is the caller's: ids index the arrays passed in, and they
    seed the walk and BFS streams, so the caller's ordering decides the
    sampling.
    """

    def __init__(self, node_ts, node_cells, node_table, node_is_task,
                 edge_parent, edge_child) -> None:
        self.ts = np.ascontiguousarray(node_ts, dtype=np.float64)
        self.cells = np.ascontiguousarray(node_cells, dtype=np.int32)
        self.table = np.ascontiguousarray(node_table, dtype=np.int32)
        self.is_task = np.ascontiguousarray(node_is_task, dtype=np.uint8)
        ep = np.ascontiguousarray(edge_parent, dtype=np.int64)
        ec = np.ascontiguousarray(edge_child, dtype=np.int64)
        n = self.n_nodes = len(self.ts)
        if not (len(self.cells) == len(self.table) == len(self.is_task) == n):
            raise ValueError("node arrays must have equal length")
        if len(ep) != len(ec):
            raise ValueError("edge arrays must have equal length")

        # CSR both ways. Children of one parent are ordered by
        # (has_ts, ts, node) — timeless first, then ascending timestamp —
        # matching the stable order the row path produces; parents of one
        # child ascend by node id.
        self.f2p_off = np.zeros(n + 1, dtype=np.int64)
        self.p2f_off = np.zeros(n + 1, dtype=np.int64)
        np.add.at(self.f2p_off, ec + 1, 1)
        np.add.at(self.p2f_off, ep + 1, 1)
        np.cumsum(self.f2p_off, out=self.f2p_off)
        np.cumsum(self.p2f_off, out=self.p2f_off)

        # parents of each child: bucket then sort ascending per child
        forder = np.lexsort((ep, ec)) if len(ep) else np.zeros(0, np.int64)
        self.f2p = ep[forder]
        # children of each parent: bucket by parent, order (has_ts, ts, node).
        # NaN sorts unpredictably in lexsort, so encode "timeless" explicitly:
        # has_ts as the second key, -inf standing in for the missing ts.
        if len(ep):
            cts = self.ts[ec]
            has_ts = (~np.isnan(cts)).astype(np.int8)
            tkey = np.where(np.isnan(cts), -np.inf, cts)
            porder = np.lexsort((ec, tkey, has_ts, ep))
            self.p2f = ec[porder]
        else:
            self.p2f = np.zeros(0, np.int64)

        # A per-cutoff prefix cache for the walk overlay: children of one
        # parent are stored timeless-first then ascending, so the ones a
        # cut-off admits are exactly a prefix, computable once per cutoff.
        self._overlay_cutoff: Optional[float] = None
        self._overlay_admitted: Optional[np.ndarray] = None

    def adjacency(self, children: bool = True):
        """The ordered CSR the graph built: (offsets, values)."""
        if children:
            return self.p2f_off.copy(), self.p2f.copy()
        return self.f2p_off.copy(), self.f2p.copy()

    # ------------------------------------------------------------------
    def _admitted_prefix(self, cutoff_ts: float) -> np.ndarray:
        same = (self._overlay_cutoff == cutoff_ts
                or (self._overlay_cutoff is not None
                    and math.isnan(self._overlay_cutoff)
                    and math.isnan(cutoff_ts)))
        if self._overlay_admitted is None or not same:
            # per node: count of children with isnan(ts) or ts <= cutoff.
            # Timeless children sort first, so admitted is a prefix; its
            # length per parent is a vectorized comparison over p2f.
            cts = self.ts[self.p2f]
            ok = np.isnan(cts) | (cts <= cutoff_ts)
            admitted = np.zeros(self.n_nodes, dtype=np.int64)
            # prefix property: count of True values per parent slice equals
            # the partition point.
            csum = np.concatenate(([0], np.cumsum(ok)))
            admitted = (csum[self.p2f_off[1:]] - csum[self.p2f_off[:-1]])
            self._overlay_admitted = admitted
            self._overlay_cutoff = cutoff_ts
        return self._overlay_admitted

    def assemble(self, target: int, cutoff_ts: float, eligible, policy,
                 fallback_base: int = 0, fallback_n: int = 0,
                 max_nodes: int = 1 << 16):
        """Ordered emitted node ids and their focal flags.

        ``eligible`` is a uint8 mask turning on the peer-ranking walk, or None
        for the target's own neighbourhood alone. A context larger than
        ``max_nodes`` raises :class:`ContextTruncated`, never truncates.
        """
        target = int(target)
        if not (0 <= target < self.n_nodes):
            raise RuntimeError("assemble: target out of range")
        ts = self.ts
        cells_of = self.cells
        table_of = self.table
        is_task = self.is_task
        f2p, f2p_off = self.f2p, self.f2p_off
        p2f, p2f_off = self.p2f, self.p2f_off

        context_seed = _StdRng(policy.seed & _U64).u64()
        step_seed = _StdRng(context_seed).u64()
        bfs_rng = _StdRng((step_seed + target + 0xB0B0_B0B0_B0B0_B0B0) & _U64)

        visited_depth: dict[int, int] = {}
        emitted: set[int] = set()
        ordered: list[int] = []
        focal: set[int] = set()
        cells = 1
        state = {"full": False}

        def kids_of(node: int, seed_cut: float) -> list[int]:
            out = []
            for i in range(p2f_off[node], p2f_off[node + 1]):
                c = int(p2f[i])
                t = ts[c]
                if math.isnan(t) or t <= seed_cut:
                    out.append(c)
            return out

        def extend(seed_node: int, is_focal: bool) -> None:
            nonlocal cells
            local_cells = 0
            f2p_stack: list[tuple[int, int]] = []
            levels: list[list[int]] = [[seed_node]]
            seed_cut = ts[seed_node]
            if math.isnan(seed_cut):
                seed_cut = math.inf
            seed_table = table_of[seed_node]
            while True:
                if f2p_stack:
                    depth, node = f2p_stack.pop()
                else:
                    depth = -1
                    for i, level in enumerate(levels):
                        if level:
                            depth = i
                            break
                    if depth < 0:
                        return
                    level = levels[depth]
                    sel = bfs_rng.range(len(level))
                    level[sel], level[-1] = level[-1], level[sel]
                    node = level.pop()
                prev = visited_depth.get(node)
                if prev is not None and prev <= depth:
                    continue
                cost = int(cells_of[node])
                local_cells += cost
                if local_cells >= policy.local_context_cells:
                    return
                visited_depth[node] = depth
                if node not in emitted:
                    emitted_cost = cost
                    if node == target:
                        emitted_cost = max(0, emitted_cost - 1)
                    if cells >= policy.max_context_cells:
                        state["full"] = True
                        return
                    emitted.add(node)
                    ordered.append(node)
                    cells += emitted_cost
                    if cells >= policy.max_context_cells:
                        state["full"] = True
                    if is_focal:
                        focal.add(node)
                for i in range(f2p_off[node], f2p_off[node + 1]):
                    f2p_stack.append((depth + 1, int(f2p[i])))
                valid = kids_of(node, seed_cut)
                # Task rows of the seed's own table are always kept; database
                # children are subject to the fanout cap.
                task_kids: list[int] = []
                db_kids: list[int] = []
                for k in valid:
                    if is_task[k]:
                        if table_of[k] == seed_table:
                            task_kids.append(k)
                    else:
                        db_kids.append(k)
                if len(db_kids) > policy.bfs_width:
                    sel_idx = _rand_sample(bfs_rng, len(db_kids),
                                           policy.bfs_width)
                    db_kids = [db_kids[i] for i in sel_idx]
                while len(levels) <= depth + 1:
                    levels.append([])
                nxt = levels[depth + 1]
                nxt.extend(task_kids)
                nxt.extend(db_kids)

        # ---- tier 1: rank peers by a random walk from the target -----------
        # The walk is over the bidirectional neighbourhood with children cut
        # at the anchor; parents are always admitted. Ranking is by visit
        # count, then recency when prefer_latest, with a per-node RNG draw
        # breaking ties — the same three keys, in the same order, as the row
        # traversal.
        tier1: list[int] = []
        visits_sorted: list[int] = []
        if eligible is not None:
            eligible = np.ascontiguousarray(eligible, dtype=np.uint8)
            admitted = self._admitted_prefix(cutoff_ts)
            # Vectorized walk, one num_walks-wide raw-PCG64 draw per step —
            # the SAME draw protocol as traversal._reference_walk_counts, so
            # the columnar and row paths sample identical contexts. The
            # overlay is "parents, then the children the cut-off admits":
            # children of one parent are stored timeless-first then by
            # ascending timestamp, so the admitted ones are exactly a prefix.
            count_arr = np.zeros(self.n_nodes, dtype=np.int64)
            elig = eligible.astype(bool)
            n_par_of = (f2p_off[1:] - f2p_off[:-1])
            deg_of = n_par_of + admitted
            bg = np.random.PCG64(
                (step_seed + target + 0xD0D0_D0D0_D0D0_D0D0) & _U64)
            cur = np.full(policy.num_walks, target, dtype=np.int64)
            alive = np.ones(policy.num_walks, dtype=bool)
            for _ in range(policy.walk_length):
                if not alive.any():
                    break
                live = cur[alive]
                visited = live[elig[live] & (live != target)]
                if visited.size:
                    np.add.at(count_arr, visited, 1)
                degree = deg_of[live]
                stepping = degree > 0
                if not stepping.any():
                    break
                raw = bg.random_raw(int(stepping.sum()))
                step_from = live[stepping]
                r = (raw % degree[stepping].astype(np.uint64)).astype(np.int64)
                n_par = n_par_of[step_from]
                is_parent = r < n_par
                moved = np.where(
                    is_parent,
                    f2p[np.minimum(f2p_off[step_from] + r,
                                   len(f2p) - 1 if len(f2p) else 0)],
                    p2f[np.minimum(p2f_off[step_from] + (r - n_par),
                                   len(p2f) - 1 if len(p2f) else 0)])
                nxt = live.copy()
                nxt[stepping] = moved
                idx = np.flatnonzero(alive)
                cur[idx] = nxt
                alive[idx[~stepping]] = False
            counts = {int(n): int(c)
                      for n, c in zip(np.flatnonzero(count_arr),
                                      count_arr[count_arr > 0])}
            touched = sorted(counts)
            tie = {n: _StdRng((step_seed + n) & _U64).u64() for n in touched}
            if policy.prefer_latest:
                # Recency is the PRIMARY key when prefer_latest: numpy's
                # lexsort treats its last argument as primary, so ordering
                # counts first was a different ranking, not cosmetic.
                def rank_key(n: int):
                    t = ts[n]
                    t = -math.inf if math.isnan(t) else t
                    return (-t, -counts[n], tie[n])
            else:
                def rank_key(n: int):
                    return (-counts[n], tie[n])
            tier1 = sorted(touched, key=rank_key)
            visits_sorted = touched

        extend(target, True)
        for node in tier1:
            if state["full"]:
                break
            extend(node, False)

        # ---- fallback: pad a short context from the task table -------------
        if not state["full"] and fallback_n > 0:
            fallback_rng = _StdRng(
                (step_seed + target + 0xA5A5_A5A5_A5A5_A5A5) & _U64)
            amount = min(max(policy.max_context_cells - cells, 0),
                         int(fallback_n))
            sel = (_rand_sample(fallback_rng, int(fallback_n), amount)
                   if 0 < amount <= fallback_n else [])
            for pos in sel:
                if state["full"]:
                    break
                node = int(fallback_base) + pos
                if node == target or _in_sorted(visits_sorted, node):
                    continue
                if not (math.isnan(ts[node]) or ts[node] <= cutoff_ts):
                    continue
                if eligible is not None and not eligible[node]:
                    continue
                extend(node, False)

        # A short buffer is REPORTED, never quietly honoured. Emitted nodes
        # are not bounded by the cell budget — a row whose columns are all
        # null costs zero cells and still takes a slot — so a caller sizing
        # max_nodes from max_context_cells undercounts, and truncating here
        # would drop the tail of a context with nothing to show for it.
        if len(ordered) >= max_nodes:
            raise ContextTruncated(
                f"context for node {target} filled the emitted-node buffer "
                f"({len(ordered)} of {max_nodes}); emitted rows are not "
                f"bounded by max_context_cells, since an all-null row costs "
                f"no cells.")
        return (np.asarray(ordered, dtype=np.int64),
                np.asarray([1 if n in focal else 0 for n in ordered],
                           dtype=np.uint8))


def _in_sorted(sorted_list: list[int], value: int) -> bool:
    i = bisect_left(sorted_list, value)
    return i < len(sorted_list) and sorted_list[i] == value
