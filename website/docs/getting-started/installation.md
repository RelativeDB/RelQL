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
pip install relationdb
```

The core depends only on numpy. Extras: `[rt]` for the native RT-J backend
(sentence-transformers + huggingface_hub), `[dev]` for pytest. Pandas and
storage clients belong to your application; relationdb provides no bundled
connectors. The distribution is `relationdb`, while the Python import remains
`import relativedb`.

## Java

Requires Java 17+. Maven publications under group `com.relativedb`:

- `relationdb` — engine, schema, RelQL parser, retriever SPI
- `relationdb-rt` — optional JNA binding to the native RT-J engine

```kotlin
dependencies {
    implementation("com.relativedb:relationdb:0.1.0")
    // implementation("com.relativedb:relationdb-rt:0.1.0")
}
```

## Rust

The crates.io distribution is `relationdb`; the established Rust crate API is
`relativedb`. It depends only on `chrono` and `libloading`.

```bash
cargo add relationdb
```

These registry coordinates are prepared but will not resolve until the first
release is published. See [Releasing the libraries](../contributing/releases)
for the manual dry-run workflow and registry setup.

## Native model engine (required for scoring)

Scoring requires a model backend — there is no model-free default. The shipped
backend, `RtNativeBackend`, runs the RT-J relational model through the C++
library `librt_c`:

```bash
cd cpp
cmake -B build -S . && cmake --build build -j
```

All libraries auto-discover `cpp/build/librt_c.{dylib,so}`, or set
`RELATIVEDB_RT_LIB`. Parsing and validation work without it, but executing a
query needs `librt_c` plus a cached `stanford-star/rt-j` checkpoint.
