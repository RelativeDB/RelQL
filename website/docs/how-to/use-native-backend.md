---
title: Use the native RT-J backend
description: Swap the history baseline for the real relational transformer.
---

# Use the native RT-J backend

Goal: score predictions with the real RT-J model instead of the built-in
history baseline.

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
backend = relativedb.RtNativeBackend(schema=ds.schema)
df = ds.predict(query, anchor_time=t0, model_backend=backend)
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
- Multiclass and ranking currently fall back to the history baseline (the C
  ABI exposes a single score head).
- A missing library or checkpoint raises a clear, actionable error — nothing
  fails silently.
