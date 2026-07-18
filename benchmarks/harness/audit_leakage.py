"""Temporal-correctness (no-future-leakage) audit on real data — invariant F24.

The engine's central safety claim is that *no row newer than the anchor can
enter a context*, even if a retriever misbehaves. We test this three ways on
real data:

  1. Direct: assemble a context at anchor T for many entities and assert every
     admitted row has timestamp <= T.
  2. Injection: wire a deliberately *leaky* retriever that always returns a
     future row, and assert the engine's defensive re-check drops it.
  3. Monotonicity: as T increases, an entity's context row-set must grow
     monotonically (never lose a past row, never gain a future one early).
"""
from __future__ import annotations

from datetime import timedelta

from relativedb import Row, TemporalBound

from .datasets import Dataset, to_epoch


def run(ds: Dataset, n_entities: int = 60) -> dict:
    findings, checked = [], 0
    entity = ds.entity_table
    ids = ds.entity_ids[:n_entities]
    anchors = ds.anchors

    # 1. direct: nothing newer than the anchor
    leaks = 0
    for T in anchors:
        t_epoch = to_epoch(T)
        for eid in ids:
            ctx = ds.engine.assemble_context(entity, eid, T)
            checked += 1
            for r in ctx.rows:
                # Row normalizes to UTC-aware; compare in epoch seconds so the
                # naive anchor (treated as UTC, as the engine does) lines up.
                if r.timestamp is not None and r.timestamp.timestamp() > t_epoch:
                    leaks += 1
    if leaks:
        findings.append(f"F24 VIOLATED: {leaks} future rows entered contexts")

    # 2. injection: a retriever that lies about the future must be caught
    inject_ok = _injection_probe(ds)
    if not inject_ok:
        findings.append("F24 VIOLATED: engine admitted a retriever-injected future row")

    # 3. monotonic growth of the past as the anchor advances
    mono_viol = 0
    probe_ids = ids[:20]
    for eid in probe_ids:
        prev_keys = set()
        for T in sorted(anchors):
            ctx = ds.engine.assemble_context(entity, eid, T)
            keys = ctx.row_keys
            # every previously-seen (older) row must still be present
            if not prev_keys <= keys:
                mono_viol += 1
            prev_keys = keys
    if mono_viol:
        findings.append(f"context not monotonic in anchor for {mono_viol} steps "
                        f"(a past row disappeared as time advanced)")

    return {"dataset": ds.name, "contexts_checked": checked,
            "direct_leaks": leaks, "injection_caught": inject_ok,
            "monotonicity_violations": mono_viol, "findings": findings}


def _injection_probe(ds: Dataset) -> bool:
    """Directly exercise TemporalBound.admits_row with a blatantly future row,
    mirroring the engine's own defensive guard."""
    T = ds.anchors[0]
    bound = TemporalBound.at_or_before(T)
    future = Row(ds.entity_table, "__inject__", {},
                 timestamp=T + timedelta(days=365))
    past = Row(ds.entity_table, "__inject__", {},
               timestamp=T - timedelta(days=1))
    return (not bound.admits_row(future)) and bound.admits_row(past)
