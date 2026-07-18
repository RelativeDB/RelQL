---
title: Rust
description: The relativedb Rust crate.
---

# Rust library

crates.io package `relationdb` (crate API `relativedb`, edition 2021); depends
only on `chrono` and `libloading`.

```bash
cargo add relationdb
```

## API shape

```rust
use relativedb::{Engine, ExecutionInput, RetrieverWiring, Schema, /* ... */};

let schema = Schema::new_schema()... .build();
let wiring = RetrieverWiring::new_wiring()
    .entities("customers", /* closure */)
    .default_links(/* closure: newest-first, honors bound + limit */)
    .scanner("customers", /* closure: enables FOR EACH + CSC */)
    .build();

let mut engine = Engine::new(schema, wiring);
let result = engine.execute(
    ExecutionInput::query("PREDICT ...").anchor_time(t0))?;
```

Modules: `schema`, `retrieve`, `pql` (hand-written recursive-descent parser,
no ANTLR runtime), `engine`, `model`, `native`, `csc`. Errors surface through
`relativedb::Result` / `relativedb::Error`.

## Design decision: synchronous SPI

Where Java's SPI is async, the Rust (and Python) SPI is synchronous and
infallible — traits return plain `Vec`s. An async SPI would force a runtime
choice on every user and color the whole engine `async`; batching retrievers
can run their own I/O concurrency internally.

## Native backend

`native::RtNativeBackend` binds `librt_c` via `libloading`, discovered from
`RELATIVEDB_RT_LIB` or the sibling `cpp/build/`. The golden gate:

```bash
RELATIVEDB_RT_LIB=../cpp/build/librt_c.dylib \
  cargo test --test golden_tests -- --nocapture
```

The shared 44-query RelQL corpus lives in this crate
(`tests/data/examples.pql`) and is exercised by all three languages.
