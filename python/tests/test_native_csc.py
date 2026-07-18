"""Cross-language conformance: the shared C++ CSC adjacency must agree with a
brute-force reference of python/.../csc.py's children() semantics, node for
node, across randomized graphs. This is the guardrail that lets the CSC hot
path live in one place (C++) without the language bindings diverging.

Skips cleanly if ``librt_c`` hasn't been built.
"""
from __future__ import annotations

import math
import random

import pytest

from relativedb.csc_native import NativeCsc, native_available

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="librt_c not built (run cmake in cpp/); native CSC unavailable")

_NEG_INF = -math.inf


def _ref_children(parent, anchor, limit, ep, ec, et):
    """Brute-force reference matching relativedb.csc.CscIndex.children:
    within the parent's bucket sorted by ts asc, keep ts <= anchor, take the
    last ``limit`` reversed to newest-first. Static rows use ts = -inf."""
    if limit <= 0:
        return []
    bucket = [i for i in range(len(ep)) if ep[i] == parent]
    bucket.sort(key=lambda i: et[i])  # stable sort by ts asc (ties keep order)
    admitted = [i for i in bucket if et[i] <= anchor]
    picked = admitted[-limit:] if limit > 0 else []
    return [ec[i] for i in reversed(picked)]


def _random_graph(rng, n_parents, n_edges):
    ep, ec, et = [], [], []
    for _ in range(n_edges):
        ep.append(rng.randint(0, n_parents - 1))
        ec.append(rng.randint(0, 100_000))
        roll = rng.randint(0, 9)
        if roll == 0:
            et.append(_NEG_INF)              # static row
        elif roll <= 3:
            et.append(float(rng.randint(0, 5)))    # heavy ties
        else:
            et.append(float(rng.randint(0, 1000)))
    return ep, ec, et


@pytest.mark.parametrize("seed,n_parents,n_edges", [
    (1, 1, 20),
    (2, 8, 200),
    (3, 50, 2000),
    (4, 200, 50),     # sparse: many parents with no edges
    (5, 4, 3000),     # dense: heavy ties per parent
])
def test_matches_reference(seed, n_parents, n_edges):
    rng = random.Random(seed)
    ep, ec, et = _random_graph(rng, n_parents, n_edges)
    idx = NativeCsc(n_parents, ep, ec, et)

    anchors = [_NEG_INF, math.inf] + [float(v) for v in range(-2, 8)]
    for _ in range(2000):
        parent = rng.randint(-2, n_parents)         # includes out-of-range
        limit = rng.randint(-1, 8)                   # includes 0 and > bucket
        anchor = rng.choice(anchors + [float(rng.randint(-5, 1005))])
        got = idx.children(parent, anchor, limit)
        want = _ref_children(parent, anchor, limit, ep, ec, et)
        assert got == want, (seed, parent, anchor, limit, got, want)


def test_edge_cases():
    # Empty graph: every query is empty.
    empty = NativeCsc(5, [], [], [])
    assert empty.children(2, 1e9, 4) == []
    # limit 0 and negative -> empty.
    idx = NativeCsc(1, [0, 0], [7, 8], [1.0, 2.0])
    assert idx.children(0, 100.0, 0) == []
    assert idx.children(0, 100.0, -3) == []
    # anchor before all -> empty; static row (-inf) admitted under every bound.
    idx2 = NativeCsc(1, [0, 0], [9, 10], [_NEG_INF, 5.0])
    assert idx2.children(0, -100.0, 4) == [9]        # only the static row
    assert idx2.children(0, 100.0, 4) == [10, 9]     # newest-first
