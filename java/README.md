# relativedb (Java)

A GraphQL-style engine for PQL predictive queries. The engine owns the query
language, planning, context assembly, and model invocation — but **never
touches a database**: all data access goes through user-implemented
**retrievers** wired to a declared schema.

## Modules

| Module | What it is |
|---|---|
| `com.relativedb:relationdb` | Schema builder, retriever SPI, ANTLR-based PQL parser + semantic validation, pure-Java context assembly (RETRIEVER and CSC sampler modes), model SPI |
| `com.relativedb:relationdb-rt` | Optional JNA binding to the golden-verified C++ RT inference engine (`librt_c`): `RtNativeBackend implements ModelBackend`, scoring with the real Relational Transformer |

Requires Java 17+. Build and test:

```sh
./gradlew test
```

## Quickstart

Declare the graph's *shape* (no URLs, no credentials), wire retrievers over
whatever storage you have, and execute PQL:

```java
import com.relativedb.schema.*;
import com.relativedb.retrieve.*;
import com.relativedb.engine.*;
import static com.relativedb.schema.ValueType.*;

RelativeDbSchema schema = RelativeDbSchema.newSchema()
    .table(TableDef.newTable("customers").column("age", NUMBER)
        .column("signup_date", DATETIME).primaryKey("customer_id").build())
    .table(TableDef.newTable("orders").column("qty", NUMBER)
        .column("order_date", DATETIME).primaryKey("order_id")
        .timeColumn("order_date").build())
    .link(LinkDef.link("orders", "customer_id", "customers"))
    .build();

RetrieverWiring wiring = RetrieverWiring.newWiring()
    .entities("customers", (table, ids, bound) -> customerDao.byIds(ids))
    .entities("orders",    (table, ids, bound) -> orderDao.byIds(ids, bound))
    .defaultLinks((link, parent, bound, limit) ->
        orderDao.recentByCustomer(parent, bound.asOf().orElse(Instant.MAX), limit))
    .build();

RelativeDbEngine engine = RelativeDbEngine.newEngine(schema, wiring)
    .modelBackend(myBackend)   // any ModelBackend; without one, ModelConfig.defaults()
    .build();                  //   routes clf → hf://stanford-star/rt-j/classification,
                               //   reg/forecast → hf://stanford-star/rt-j/regression,
                               //   embeddings all-MiniLM-L12-v2 (shared by both)

PredictionResult churn = engine.execute(ExecutionInput.newInput()
    .query("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id")
    .anchorTime(Instant.parse("2026-07-01T00:00:00Z"))
    .entityIds(List.of(42L))
    .build()).toCompletableFuture().join();
```

The same query runs against JDBC, a REST microservice, a feature store, or a
CSV-backed test double — only the wiring changes.

## The retriever contract

Three small async interfaces mirror the three sampling primitives:

- `EntityRetriever` — batched point lookup (seed rows, parents)
- `LinkRetriever` — children of a parent along one FK link, newest-first,
  capped at the engine's fanout
- `CohortRetriever` (optional) — similar entities for in-context examples

plus `TableScanner` (optional bulk streaming, enables CSC mode) and
`StatsProvider` (training-split normalization stats). Every call carries a
`TemporalBound` ("nothing newer than this") — and the engine **re-checks every
returned row** and drops violations, so a buggy retriever cannot leak the
future into context.

`Row` carries typed cells (`NUMBER | TEXT | DATETIME | BOOLEAN`), an optional
timestamp, and parent *edges*. IDs and FK values are never cells — there is no
way to hand the engine an identifier as a feature.

## Sampler modes

```java
RelativeDbEngine.newEngine(schema, wiring)
    .samplerMode(SamplerMode.RETRIEVER)   // default: pull-per-hop via retrievers
    .samplerMode(SamplerMode.CSC)         // in-memory CSC index from TableScanners
```

- **RETRIEVER** — the hop loop calls your retrievers per expansion. Right when
  data is remote, huge, or access-controlled.
- **CSC** — `TableScanner` streams are drained once into per-link
  `colptr`/`row` adjacency arrays with time-sorted neighbor lists; "latest w
  children ≤ anchor" is a binary search + tail slice. Right for
  latency-sensitive, repeated scoring over data that fits in memory.
  Rebuild the snapshot with `engine.refresh()`.

Context budgets support both geometries: a global cell budget with uniform
width (`maxContextCells` + `bfsWidth`), or per-hop caps (`fanouts(64, 64)`).

## PQL

The full grammar lives in `relativedb-core/src/main/antlr/Pql.g4`. Examples:

```sql
PREDICT SUM(orders.qty, 0, 30) FOR EACH customers.customer_id
PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR users.user_id IN (42, 123)
PREDICT LIST_DISTINCT(transactions.article_id, 0, 30) RANK TOP 12 FOR EACH customers.customer_id
PREDICT SUM(usage.count, 0, 1, days) FORECAST 28 TIMEFRAMES FOR EACH accounts.account_id
```

`Pql.parse(query)` gives the typed AST; `Pql.validate(query, schema)` binds it
against the schema (unknown tables/columns, type mismatches, window signs,
LIST_DISTINCT ⇒ CLASSIFY|RANK, static/temporal mixing) and infers the
`TaskType`, which drives model routing in `ModelConfig`.

## Native model backend (relativedb-rt)

`RtNativeBackend` scores `TokenBatch`es with the real RT-J model through the
golden-verified C++ inference engine (`cpp/build/librt_c.dylib`, C ABI in
`cpp/src/rt_c.h`):

```java
TextEncoder encoder = new PrecomputedEncoder(embeddingTable); // string -> float[384]
try (RtNativeBackend backend = new RtNativeBackend(ModelConfig.defaults(), encoder)) {
    RelativeDbEngine engine = RelativeDbEngine.newEngine(schema, wiring)
        .modelBackend(backend).build();
    // ...
}
```

- **Library loading** is lazy: system property `relativedb.rt.lib` → env
  `RELATIVEDB_RT_LIB` → relative lookup of `cpp/build/librt_c.dylib` (works from
  the repo root or the `java/` tree) → loader path. A missing library fails
  with build/override instructions.
- **Checkpoint routing** follows `ModelConfig.modelUriFor(TaskType)`:
  classification-family tasks → the classification checkpoint (logits; the
  backend applies a sigmoid to fill `probability`), regression/forecasting →
  the regression checkpoint (NORMALIZED values; you denormalize with
  train-split stats). URIs may be `file://`, plain paths, or `hf://org/repo/
  subdir` — the latter resolves against the *local* Hugging Face cache only
  (no downloading; override the cache root with `-Drelativedb.rt.hf.cache` or
  `RELATIVEDB_RT_HF_CACHE`).
- **Text embeddings**: the engine needs 384-dim all-MiniLM-L12-v2 vectors for
  cell text and column names. The `TextEncoder` SPI supplies them;
  `PrecomputedEncoder` covers closed vocabularies, and a real MiniLM encoder
  is a deliberately separate concern.
- **Golden gate**: `RtGoldenForwardTest` replays `cpp/testdata/*.bin` through
  the binding and matches the PyTorch-verified scores for both checkpoints
  (skipped automatically where the dylib/checkpoints are absent).
