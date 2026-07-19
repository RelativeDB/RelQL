---
title: Model backends
description: The scoring SPI, the required RT-J backend, and checkpoint routing.
---

# Model backends

Scoring is behind a two-method `ModelBackend` SPI. **A model backend is
required** — the engine has no built-in scorer and raises a clear error if you
execute a query without one. The shipped backend is `RtNativeBackend`, which
runs the RT-J relational foundation model.

## RtNativeBackend

`RtNativeBackend` is the scoring path. It scores contexts with real **RT-J**
checkpoints through the native C++ engine (`librt_c`), so it needs `librt_c`
built and available plus a cached `stanford-star/rt-j` checkpoint.

It converts each context into the raw RT token batch — one token per feature
cell, FK links as the node graph, per-column z-scores for numbers, pinned
MiniLM embeddings (384-dim) for text cells and `"<column> of <table>"` schema
phrases — plus a masked *task* row anchored at prediction time, with the
entity's own past outcomes as in-context examples.

Classification logits pass through a sigmoid; regression outputs denormalize
with in-context label statistics.

### Supported output types

The checkpoint executes **binary classification**, **regression**,
**multiclass classification**, and **ranking**. `RETURN CLASS`, `RETURN
DISTRIBUTION`, `RETURN PROBABILITY`, and `RETURN EXPECTED VALUE` work.
Multiclass reuses the checkpoint's **text head**: the masked target cell is
decoded to a 384-dim embedding and matched by cosine similarity to the class
labels' MiniLM embeddings, yielding a predicted class plus approximate,
uncalibrated class probabilities (a softmax over the cosine scores — the argmax
is reference-exact). Ranking scores each candidate parent ID with the existence
head, sigmoids it, and returns the top *k*. `RETURN QUANTILES` / `RETURN
INTERVAL` are **not** supported — the checkpoint has no variance/quantile head —
and raise a clear error.

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

## Testing device

The engine's own unit tests use a tiny **deterministic stub** backend so the
pipeline can be exercised without loading a checkpoint. It is a test double,
not a shipped or default predictor — do not rely on it to serve real
predictions.
