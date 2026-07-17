---
title: Temporal correctness
description: Why relativedb cannot leak the future into a prediction.
---

# Temporal correctness

Temporal leakage — a "future" fact sneaking into the features — is the classic
way predictive systems lie in backtests. relativedb treats leakage prevention
as an **engine guarantee**, not a user discipline.

## The anchor time

Every execution has an anchor time t₀. The prediction target reads the window
*after* t₀; the assembled context may only contain data at or *before* t₀.

## Defense in depth

1. Every retriever call carries a `TemporalBound` — "return nothing newer
   than this". Rows without timestamps (static dimension tables) are always
   admitted.
2. The engine **re-checks every returned row** against the bound and drops
   violations. A buggy or malicious retriever cannot leak the future into
   context. Dedicated tests in all three libraries feed a deliberately broken
   retriever and assert the future row never appears.

## Window direction is validated

Target windows must face the future (non-negative offsets); `WHERE` filter
windows face the past (negative or `-INF` starts). The validator rejects
queries that mix these up.

## Backtesting for free

Because "as of" is an explicit input, evaluating yesterday's model is just
running the same query with yesterday's anchor — no snapshot tables, no
point-in-time joins.
