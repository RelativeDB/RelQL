---
title: Model backends
description: The scoring SPI, the built-in baseline, and checkpoint routing.
---

# Model backends

Scoring is behind a two-method `ModelBackend` SPI. Two implementations ship:

## HistoryBaselineBackend (default)

Model-free: evaluates the query target over the entity's **own trailing
history windows** ("self labels") and aggregates. Transparent, deterministic,
zero artifacts — the whole pipeline runs and tests without a model. Use it for
development, testing, and as a sanity floor for model quality.

## RtNativeBackend

Scores contexts with real **RT-J** checkpoints through the native C++ engine
(`librt_c`). It converts each context into the raw RT token batch — one token
per feature cell, FK links as the node graph, per-column z-scores for numbers,
pinned MiniLM embeddings (384-dim) for text cells and `"<column> of <table>"`
schema phrases — plus a masked *task* row anchored at prediction time, with
the entity's own past outcomes as in-context examples.

Classification logits pass through a sigmoid; regression outputs denormalize
with in-context label statistics.

## Checkpoint routing

`ModelConfig` maps the inferred [task type](/pql/reference/task-types) to a
checkpoint URI:

| Task type | Default URI |
|---|---|
| classification, ranking | `hf://stanford-star/rt-j/classification` |
| regression, forecasting | `hf://stanford-star/rt-j/regression` |
| text embeddings | `all-MiniLM-L12-v2` (pinned, 384-dim) |

`hf://` URIs resolve against the **local** Hugging Face cache only — nothing
downloads implicitly. `file://` and plain paths also work.

## Bring your own

Implement `ModelBackend` to plug in any scorer; the engine hands you assembled
contexts and the routed model URI.
