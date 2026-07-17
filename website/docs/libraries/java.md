---
title: Java
description: The relativedb Java library.
---

# Java library

Gradle modules under group `dev.relativedb`; requires Java 17+.

| Module | Contents |
|---|---|
| `relativedb-core` | Schema builder, retriever SPI, ANTLR-based PQL parser + validation, context assembly (both sampler modes), model SPI |
| `relativedb-rt` | Optional JNA binding to the native RT-J engine: `RtNativeBackend implements ModelBackend` |

```bash
cd java
./gradlew test
```

## API shape

```java
RelativeDbSchema schema = RelativeDbSchema.newSchema()... .build();
RetrieverWiring wiring  = RetrieverWiring.newWiring()... .build();

RelativeDbEngine engine = RelativeDbEngine.newEngine(schema, wiring)
    .samplerMode(SamplerMode.CSC)        // optional
    .modelBackend(backend)               // optional; default = history baseline
    .build();

PredictionResult r = engine.execute(ExecutionInput.newInput()
    .query("PREDICT ...")
    .anchorTime(t0)
    .entityIds(ids)                      // omit + FOR EACH → TableScanner enumerates
    .build()).toCompletableFuture().join();
```

Key packages: `dev.relativedb.schema`, `.retrieve`, `.query` (entry point
`Pql.parse` / `Pql.validate`), `.engine`, `.model`, `.rt`.

## Async retriever SPI

Retrievers return `CompletionStage` — fan out to remote services without
blocking the engine. Every call carries a `TemporalBound`; the engine
re-checks all returned rows.

## Native backend

`RtNativeBackend` loads `librt_c` lazily: system property `relativedb.rt.lib`
→ env `RELATIVEDB_RT_LIB` → sibling `cpp/build/` → loader path. `hf://`
checkpoint URIs resolve from the local HF cache (override root with
`relativedb.rt.hf.cache` / `RELATIVEDB_RT_HF_CACHE`). Text embeddings come
through the `TextEncoder` SPI (`PrecomputedEncoder` for closed vocabularies).

A golden-forward test replays `cpp/testdata/*.bin` and matches the
PyTorch-verified scores; it auto-skips when the native library is absent.
