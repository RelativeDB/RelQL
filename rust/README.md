# relativedb (Rust)

Predictive queries (**PQL**) over your own data — the Rust peer of the Java
(`com.relativedb.*`) and Python (`relativedb`) libraries. Same concepts, same
behavior, idiomatic Rust.

The crates.io package is named `relationdb`; its established crate API remains
`relativedb`:

```bash
cargo add relationdb
```

GraphQL-style execution: the engine owns the query language, planning, context
assembly, and model routing — **all data access goes through user-defined
retrievers**. No bundled database connectors. The same PQL query runs against a
JDBC service, a REST microservice, a feature store, or an in-memory test double
— only the wiring changes.

## Modules

| Module | What it holds |
|---|---|
| `schema` | `Schema`, `TableDef`, `ColumnDef`, `LinkDef`, `ValueType`; builder + validation (links resolve, link targets need PKs, PK/FK columns may not be feature columns — the **F17** invariant). |
| `retrieve` | The SPI traits `EntityRetriever` / `LinkRetriever` / `CohortRetriever` / `TableScanner` / `StatsProvider`; `Row` (typed cells, timestamp, parent FK edges — IDs never become cells), `TemporalBound` (inclusive as-of; static rows always admitted), `RetrieverWiring`. |
| `pql` | A hand-written recursive-descent parser for [`Pql.g4`](../../rt/grammar/Pql.g4) (no ANTLR runtime), a typed AST, schema-bound `validate`, and task-type inference. |
| `engine` | `Engine`, `ExecutionInput`, `ContextPolicy`, `SamplerMode`, `PredictionResult`; the real hop-loop context assembler and the in-memory CSC sampler (`csc`). |
| `model` | `ModelConfig` with the default RT-J URIs + task-type routing. |
| `native` | The shared native backend binding the golden-verified C++ RT-J engine via its C ABI (`librt_c`), plus a precomputed-embeddings `TextEncoder`. |

## Quickstart — the churn example (design §6)

```rust
use relativedb::{
    Engine, EntityId, ExecutionInput, LinkDef, RetrieverWiring, Row, Schema,
    TableDef, TemporalBound, ValueType,
};
use chrono::{NaiveDate, TimeZone, Utc};

fn day(s: &str) -> chrono::DateTime<Utc> {
    Utc.from_utc_datetime(&NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .unwrap().and_hms_opt(0, 0, 0).unwrap())
}

// 1. Declare the graph — shape only, no URLs or credentials.
let schema = Schema::new_schema()
    .table(TableDef::new_table("customers")
        .column("age", ValueType::Number)
        .column("signup_date", ValueType::Datetime)
        .primary_key("customer_id").build())
    .table(TableDef::new_table("orders")
        .column("qty", ValueType::Number)
        .column("order_date", ValueType::Datetime)
        .primary_key("order_id")
        .time_column("order_date").build())
    .link(LinkDef::link("orders", "customer_id", "customers"))
    .build();

// 2. Wire retrievers (closures work directly). Here: an in-memory double.
let customers = vec![
    Row::new("customers", "C1").cell("age", 34.0).cell("signup_date", day("2026-02-10")),
    Row::new("customers", "C7").cell("age", 52.0).cell("signup_date", day("2026-01-20")),
];
let orders = vec![
    Row::new("orders", "O1").cell("qty", 1.0).cell("order_date", day("2026-03-10"))
        .timestamp(day("2026-03-10")).parent("customer_id", "C7"),
    Row::new("orders", "O2").cell("qty", 2.0).cell("order_date", day("2026-05-02"))
        .timestamp(day("2026-05-02")).parent("customer_id", "C7"),
];

let cust = customers.clone();
let ords_c = orders.clone();
let ords_l = orders.clone();
let cust_scan = customers.clone();

let wiring = RetrieverWiring::new_wiring()
    .entities("customers", move |_t: &str, ids: &[EntityId], _b: &TemporalBound| {
        cust.iter().filter(|r| ids.contains(&r.id)).cloned().collect()
    })
    .entities("orders", move |_t: &str, ids: &[EntityId], _b: &TemporalBound| {
        ords_c.iter().filter(|r| ids.contains(&r.id)).cloned().collect()
    })
    // newest-first children, honoring the temporal bound the engine passes in
    .default_links(move |link: &LinkDef, pid: &EntityId, b: &TemporalBound, limit: usize| {
        let mut kids: Vec<Row> = ords_l.iter()
            .filter(|r| r.get_parent(&link.fk_column) == Some(pid) && b.admits_row(r))
            .cloned().collect();
        kids.sort_by(|a, x| x.timestamp.cmp(&a.timestamp));
        kids.truncate(limit);
        kids
    })
    // a TableScanner lets `FOR EACH` enumerate the entity table
    .scanner("customers", move |_t: &str, _b: &TemporalBound| cust_scan.clone())
    .build();

// 3. Build the engine (defaults route clf queries to rt-j/classification,
//    regression/forecasting to rt-j/regression, MiniLM-L12-v2 embeddings).
let mut engine = Engine::new(schema, wiring);

// 4. Run a churn query as of a fixed anchor time.
let result = engine.execute(
    ExecutionInput::query(
        "PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id",
    )
    .anchor_time(day("2026-07-01")),
).unwrap();

assert_eq!(result.task_type, relativedb::TaskType::BinaryClassification);
for p in &result.predictions {
    println!("{} churn probability = {:?}", p.id, p.probability);
}
```

The default backend is a model-free `HistoryBaselineBackend` (predicts from the
entity's own trailing history — "self labels", F65). Swap in the real RT-J
checkpoint with `.model_backend(Box::new(RtNativeBackend::new(...)))`.

## Traversal modes

Both modes produce **identical contexts** (verified in the tests); pick by data
locality:

```rust
use relativedb::SamplerMode;
// pull-per-hop through the retrievers (default) — data is remote/huge/gated
Engine::new(schema, wiring); // SamplerMode::Retriever

// or a materialized in-memory CSC index built from the TableScanners —
// latency-sensitive, repeated queries over data that fits in memory
Engine::new(schema, wiring).sampler_mode(SamplerMode::Csc).build()?;
```

`ContextPolicy` carries both budget geometries: RT-style `max_context_cells` +
`bfs_width`, and KumoRFM-style per-hop `fanouts(...)`. Every returned row is
re-checked against the `TemporalBound` in both modes, so a buggy retriever can
never leak the future into context (F24).

## Native RT-J backend

`native` binds the golden-verified C++ engine (`librt_c`) through `libloading`,
lazily discovered from `RELATIVEDB_RT_LIB` or the sibling
`../cpp/build/librt_c.{dylib,so}`. `hf://…` checkpoint URIs resolve against the
local Hugging Face cache (no HF client); `file://` and plain paths work too.
Classification scores are logits (sigmoid → probability); regression scores are
normalized (denormalized with the in-context label stats).

## Design decision: synchronous SPI

The Java design specifies `CompletionStage` (async) retrievers. This Rust peer
(like the Python peer) makes the SPI **synchronous** and infallible — traits
return plain `Vec`s. An async SPI would force a runtime choice on every user and
colour the whole engine `async` for no benefit in the reference paths; a
batching/parallel implementation is free to run its own I/O concurrency
internally. The engine's own parse / validate / wiring / execution errors surface
through `relativedb::Result` / `relativedb::Error`.

## Tests

```bash
cargo test                                   # parser (44-query corpus) + engine (leakage, CSC≡retriever)
RELATIVEDB_RT_LIB=../cpp/build/librt_c.dylib \
  cargo test --test golden_tests -- --nocapture   # golden gate vs the C++ engine
```

The golden test feeds the raw PRE-sort arrays from `cpp/testdata/*.bin` through
the native binding and asserts both checkpoints' target scores within 2e-3 of
the PyTorch-verified reference.
