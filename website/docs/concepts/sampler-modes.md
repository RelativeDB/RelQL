---
title: Sampler modes
description: RETRIEVER (pull-per-hop) vs CSC (in-memory index).
---

# Sampler modes

Context assembly walks the graph: seed entity → parents (always followed) →
children (fanout-capped, newest-first) → optional cohort, until the hop limit
or cell budget. Two interchangeable samplers drive this walk — both produce
**identical contexts** (asserted by tests).

## RETRIEVER (default)

Pull-per-hop: the hop loop calls your retrievers for each expansion.

Use when data is **remote, huge, or access-controlled** — nothing is copied,
your retrievers see every access.

## CSC

The engine drains each `TableScanner` once into in-memory
compressed-sparse-column adjacency arrays (time-sorted neighbor lists), then
samples entirely in-process — "latest *w* children ≤ anchor" is one binary
search plus a tail slice.

Use for **latency-sensitive, repeated scoring** over data that fits in
memory. The index is a snapshot; rebuild with `engine.refresh()`.

## Context budgets

`ContextPolicy` supports two geometries:

- per-hop fanouts, e.g. `fanouts(64, 64)`
- a uniform `bfs_width` under a global `max_context_cells` budget

See [Choose a sampler mode](../how-to/choose-sampler-mode) for a decision
guide and benchmark numbers.
