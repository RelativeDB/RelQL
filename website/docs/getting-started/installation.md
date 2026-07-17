---
title: Installation
description: Install the Python, Java, or Rust library.
---

# Installation

Pick the library for your stack. All three are peers — same engine, same
semantics.

## Python

Requires Python 3.10+. Core depends only on numpy.

```bash
cd python
pip install -e ".[pandas]"
```

Extras: `[pandas]` for the DataFrame convenience layer, `[rt]` for the native
RT-J backend (sentence-transformers + huggingface_hub), `[dev]` for pytest.

## Java

Requires Java 17+. Gradle modules under group `dev.relativedb`:

- `relativedb-core` — engine, schema, PQL parser, retriever SPI
- `relativedb-rt` — optional JNA binding to the native RT-J engine

```bash
cd java
./gradlew build
```

## Rust

Cargo workspace; the crate depends only on `chrono` and `libloading`.

```bash
cd rust
cargo build
```

## Native model engine (optional)

The RT-J model backend needs the C++ library `librt_c`:

```bash
cd cpp
cmake -B build -S . && cmake --build build -j
```

All libraries auto-discover `cpp/build/librt_c.{dylib,so}`, or set
`RELATIVEDB_RT_LIB`. Everything else works without it — the default backend is
model-free.
