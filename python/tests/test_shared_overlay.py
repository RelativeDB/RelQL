"""The shared-graph temporal overlay must not depend on anchor order.

`ReferenceTraversal._factory_overlay` recomputes the walk CSR and
eligibility for a cutoff via a vectorized mask over precomputed edges.
Anchor sequences in the wild are usually time-ascending, but nothing may
rely on that: these tests drive the overlay with shuffled, repeated,
descending, and None cutoffs and assert exact equality with a slow
reference rebuild at every step.
"""
from datetime import datetime, timedelta, timezone

import numpy as np

from relativedb import Row
from relativedb.traversal import ReferenceTraversal


T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _fixture_state():
    """A small graph with the shapes the overlay must handle: parent edges
    (always admitted), timestamped children, timestamp-less children, and
    task rows both timestamped and not."""
    entity = Row("e", "E1", {"a": 1.0})
    task_rows = [Row("t", ("E1", k), {"y": float(k)},
                     timestamp=T0 + timedelta(days=10 * k),
                     parents={"__entity__": "E1"})
                 for k in range(4)]
    static_child = Row("t", ("E1", "s"), {"y": 9.0},
                       parents={"__entity__": "E1"})   # no timestamp
    rows = [entity] + task_rows + [static_child]
    node_pos = {r.key: i for i, r in enumerate(rows)}
    by_key = {r.key: r for r in rows}
    p2f_walk = {entity.key: task_rows + [static_child]}
    parent_of = {r.key: (entity,) for r in task_rows + [static_child]}
    parent_of[entity.key] = ()

    state = {
        "rows": rows,
        "by_key": by_key,
        "node_pos": node_pos,
        "p2f_walk": p2f_walk,
        "parents": lambda row: parent_of[row.key],
        "sampling_table": "t",
    }
    return state, rows


def _reference_overlay(state, cutoff):
    """The pre-vectorization per-cutoff rebuild, kept as the oracle."""
    def valid(row):
        return (row.timestamp is None or cutoff is None
                or row.timestamp <= cutoff)

    rows = state["rows"]
    offsets = [0]
    flat = []
    for row in rows:
        for parent in state["parents"](row):
            flat.append(state["node_pos"][parent.key])
        for child in state["p2f_walk"].get(row.key, ()):
            if valid(child):
                flat.append(state["node_pos"][child.key])
        offsets.append(len(flat))
    eligible = [1 if (row.table == state["sampling_table"] and valid(row))
                else 0 for row in rows]
    return (np.asarray(offsets, np.int32), np.asarray(flat, np.int32),
            np.asarray(eligible, np.uint8))


def test_overlay_matches_reference_for_any_anchor_order():
    state, rows = _fixture_state()
    traversal = ReferenceTraversal()
    # Deliberately unsorted: repeats, a regression to an earlier cutoff, an
    # unbounded (None) cutoff, and boundary-equal timestamps.
    cutoffs = [rows[3].timestamp, rows[1].timestamp, None,
               rows[2].timestamp, rows[1].timestamp,
               rows[1].timestamp - timedelta(seconds=1),
               rows[4].timestamp]
    for cutoff in cutoffs:
        # The overlay derives its cutoff from the target row's timestamp.
        target = Row("t", ("E1", "target"), {}, timestamp=cutoff,
                     parents={"__entity__": "E1"})
        state["by_key"][target.key] = target
        traversal._factory_overlay(state, target.key)
        ref_off, ref_flat, ref_elig = _reference_overlay(state, cutoff)
        np.testing.assert_array_equal(state["offsets"], ref_off)
        np.testing.assert_array_equal(state["neighbors"], ref_flat)
        # The synthetic target row is not part of `rows`, so eligibility is
        # compared over the fixture rows only.
        np.testing.assert_array_equal(state["eligible_base"], ref_elig)
        assert state["cutoff"] == cutoff
        del state["by_key"][target.key]


def test_overlay_noop_when_cutoff_unchanged():
    state, rows = _fixture_state()
    traversal = ReferenceTraversal()
    target = Row("t", ("E1", "target"), {}, timestamp=rows[2].timestamp,
                 parents={"__entity__": "E1"})
    state["by_key"][target.key] = target
    traversal._factory_overlay(state, target.key)
    first_neighbors = state["neighbors"]
    traversal._factory_overlay(state, target.key)
    assert state["neighbors"] is first_neighbors   # no rebuild
