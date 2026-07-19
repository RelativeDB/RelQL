---
title: Use the native RT-J backend
description: Wire the required RT-J backend — the engine's only scoring path.
---

# Use the native RT-J backend

`RtNativeBackend` is the scoring path: the engine has no model-free default and
raises a clear error if you execute a query without a model backend. This page
sets it up — build `librt_c`, get the checkpoint, and wire the backend.

## 1. Build the C++ engine

```bash
cd cpp
cmake -B build -S . && cmake --build build -j
```

This produces `cpp/build/librt_c.{dylib,so}`. All bindings find it there
automatically; elsewhere, set `RELATIVEDB_RT_LIB=/path/to/librt_c.dylib`.

## 2. Get the checkpoints

Default routing resolves `hf://stanford-star/rt-j/{classification,regression}`
against your **local** Hugging Face cache — nothing downloads implicitly.
`file://` and plain paths work via a custom `ModelConfig`.

## 3. Plug in the backend

**Python** (needs `pip install -e ".[rt]"`):

```python
backend = relativedb.RtNativeBackend(schema=schema)
engine = relativedb.Engine(schema, wiring, model_backend=backend)
result = engine.execute(relativedb.ExecutionInput(query=query, anchor_time=t0))
```

**Java**:

```java
TextEncoder encoder = new PrecomputedEncoder(embeddingTable); // string -> float[384]
try (RtNativeBackend backend = new RtNativeBackend(ModelConfig.defaults(), encoder)) {
    RelativeDbEngine engine = RelativeDbEngine.newEngine(schema, wiring)
        .modelBackend(backend).build();
}
```

**Rust**:

```rust
let engine = Engine::new(schema, wiring)
    .model_backend(Box::new(RtNativeBackend::new(...)));
```

## What to expect

- Classification returns probabilities (sigmoid over logits); regression
  returns denormalized values.
- Text cells require MiniLM embeddings: Python computes them with
  sentence-transformers; Java and Rust take a `TextEncoder` (a precomputed
  table works for closed vocabularies).
- Multiclass classification executes via the checkpoint's text head (the masked
  target cell is decoded to a 384-d embedding and matched by cosine to the class
  labels' MiniLM embeddings), returning a predicted class plus approximate,
  uncalibrated class probabilities.
- Ranking (`RANK TOP k`) executes via per-candidate existence scoring: candidate
  parent IDs are each scored with the existence head, sigmoided, and the top *k*
  returned.
- `RETURN QUANTILES`/`INTERVAL` remain unsupported (no variance/quantile head in
  the checkpoint) and raise a clear error.
- A missing library or checkpoint raises a clear, actionable error — nothing
  fails silently.
