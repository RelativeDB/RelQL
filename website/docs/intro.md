---
id: intro
title: The relativedb engine
slug: /
description: The complete engine guide — install, concepts, how-to, and the language libraries, in one page.
---

# What is relativedb?

relativedb answers questions about the **future** of your relational data. You
declare the shape of your tables and links, wire small **retriever** callbacks
over your existing storage, and write a predictive query:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers
```

That's 90-day churn for every customer — no feature engineering, no training
pipeline, and no temporal leakage by construction.

## How it fits together

1. **RelQL** — a SQL-flavored query language for predictions. Parsed and
   validated against your declared schema. See the [RelQL docs](/relql/).
2. **Retrievers** — the engine never touches your database. All data access
   goes through callbacks you implement, GraphQL-style. See
   [Retrievers](#retrievers).
3. **Temporal context assembly** — the engine hops your relational graph to
   build a per-entity context, and guarantees nothing newer than the anchor
   time enters it. See [Temporal correctness](#temporal-correctness).
4. **Model backends** — contexts are scored by a required, pluggable backend.
   The shipped one is `RtNativeBackend`, running **RT-J**, a relational
   transformer foundation model that predicts in-context. There is no model-free
   default. See [Model backends](#model-backends).

## Three peer libraries

The engine is implemented natively in [Python](#python-library),
[Java](#java-library), and [Rust](#rust-library) — same concepts, same
behavior, idiomatic APIs. A shared [C++ inference engine](#c-inference-engine-rtcpp)
serves the RT-J model to all three.

## How to read this page

This is the whole engine guide in one document: install, concepts, how-to
guides, and the per-language APIs.

- [Installation](#installation) and [Quickstart](#quickstart) — first
  prediction in minutes.
- Concepts — [architecture](#architecture), [retrievers](#retrievers),
  [temporal correctness](#temporal-correctness),
  [sampler modes](#sampler-modes), [model backends](#model-backends).
- How-to — [churn](#predict-churn), [forecasting](#forecast-demand),
  [ranking](#rank-recommendations),
  [custom retrievers](#wire-custom-retrievers),
  [the native backend](#use-the-native-rt-j-backend).
- Libraries — [Python](#python-library), [Java](#java-library),
  [Rust](#rust-library), [C++](#c-inference-engine-rtcpp).

The query language itself is documented separately, in
[the RelQL language reference](/relql/).


## Installation

Pick the library for your stack. All three are peers — same engine, same
semantics.

### Python

Requires Python 3.10+. Core depends only on numpy.

```bash
pip install relationdb
```

The core depends only on numpy. Extras: `[rt]` for the native RT-J backend
(sentence-transformers + huggingface_hub), `[dev]` for pytest. Pandas and
storage clients belong to your application; relationdb provides no bundled
connectors. The distribution is `relationdb`, while the Python import remains
`import relativedb`.

### Java

Requires Java 17+. Maven publications under group `com.relativedb`:

- `relationdb` — engine, schema, RelQL parser, retriever SPI
- `relationdb-rt` — optional JNA binding to the native RT-J engine

```kotlin
dependencies {
    implementation("com.relativedb:relationdb:0.1.0")
    // implementation("com.relativedb:relationdb-rt:0.1.0")
}
```

### Rust

The crates.io distribution is `relationdb`; the established Rust crate API is
`relativedb`. It depends only on `chrono` and `libloading`.

```bash
cargo add relationdb
```

These registry coordinates are prepared but will not resolve until the first
release is published. See [Releasing the libraries](#releasing-the-libraries)
for the manual dry-run workflow and registry setup.

### Native model engine (required for scoring)

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


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Quickstart

Predict 90-day churn for every customer, as of July 1.

<Tabs groupId="lang">
<TabItem value="python" label="Python">

Declare the schema and wire callbacks over your own storage. If your data is
in pandas, pandas remains an application dependency:

```python
import pandas as pd
from relativedb import Engine, ExecutionInput, RetrieverWiring, Row, RtNativeBackend

customer_rows = [Row("customers", r.customer_id, {"age": float(r.age)})
                 for r in customers.itertuples()]
by_id = {row.id: row for row in customer_rows}

def fetch_customers(table, ids, bound):
    return [by_id[row_id] for row_id in ids if row_id in by_id]

wiring = (RetrieverWiring.new_wiring()
    .entities("customers", fetch_customers)
    .entities("orders", fetch_orders)
    .default_links(fetch_order_children)
    .scanner("customers", scan_customers)
    .build())

# Scoring requires a model backend. RtNativeBackend runs the RT-J relational
# model; it needs librt_c and a cached stanford-star/rt-j checkpoint.
engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema))
result = engine.execute(ExecutionInput(
    query="PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers",
    anchor_time=pd.Timestamp("2026-07-01").to_pydatetime()))
```

</TabItem>
<TabItem value="java" label="Java">

Declare the schema, wire retrievers over your DAOs, execute:

```java
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

// Scoring requires a model backend; RtNativeBackend runs the RT-J model.
RelativeDbEngine engine = RelativeDbEngine.newEngine(schema, wiring)
    .modelBackend(new RtNativeBackend(schema))
    .build();

PredictionResult churn = engine.execute(ExecutionInput.newInput()
    .query("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers")
    .anchorTime(Instant.parse("2026-07-01T00:00:00Z"))
    .build()).toCompletableFuture().join();
```

</TabItem>
<TabItem value="rust" label="Rust">

Closures implement the retriever traits directly:

```rust
let schema = Schema::new_schema()
    .table(TableDef::new_table("customers")
        .column("age", ValueType::Number)
        .column("signup_date", ValueType::Datetime)
        .primary_key("customer_id").build())
    .table(TableDef::new_table("orders")
        .column("qty", ValueType::Number)
        .column("order_date", ValueType::Datetime)
        .primary_key("order_id").time_column("order_date").build())
    .link(LinkDef::link("orders", "customer_id", "customers"))
    .build();

let wiring = RetrieverWiring::new_wiring()
    .entities("customers", entity_lookup)
    .default_links(newest_first_children)
    .scanner("customers", customer_scan)   // enables whole-table FROM
    .build();

// Scoring requires a model backend; RtNativeBackend runs the RT-J model.
let mut engine = Engine::new(schema, wiring)
    .model_backend(RtNativeBackend::new(&schema)?);
let result = engine.execute(
    ExecutionInput::query(
        "PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers")
    .anchor_time(anchor))?;
```

</TabItem>
</Tabs>

:::info A model backend is required
Scoring runs through the RT-J relational foundation model via `RtNativeBackend`
— there is no model-free default, and the engine errors without a backend. See
[Use the native backend](#use-the-native-rt-j-backend) for building `librt_c`
and fetching the checkpoint. The checkpoint executes binary classification,
regression, multiclass classification (a predicted class plus approximate
probabilities, via the text head's cosine match to the class labels' MiniLM
embeddings), and ranking (top-k via per-candidate existence scoring). `RETURN
QUANTILES`/`INTERVAL` are still not supported.
:::

### Next steps

- Learn the query language: [RelQL tutorial](/relql/#relql-tutorial)
- Understand the pipeline: [Architecture](#architecture)
- Connect real storage: [Wire custom retrievers](#wire-custom-retrievers)


## Architecture

Every query runs through the same four stages, in every language:

```
 RelQL string
    │  parse       → typed AST (syntax errors here)
    ▼
 ParsedQuery
    │  validate    → bind names/types/windows against the Schema; infer TaskType
    ▼
 ValidatedQuery
    │  assemble    → hop loop through YOUR retrievers, bounded by anchor time
    ▼
 per-entity contexts
    │  score       → ModelBackend routed by TaskType
    ▼
 PredictionResult  → one prediction per entity (value and/or probability)
```

### Inputs

- **Query** — a RelQL string. See the [RelQL docs](/relql/).
- **Anchor time** — the "as of" instant t₀. Context may only contain data at
  or before it; the prediction concerns the window after it.
- **Entities** — `FROM` over a table, either enumerated whole (via a
  `TableScanner`) or narrowed by a primary-key predicate in `WHERE`, whose ids
  are supplied as a bind parameter (`WHERE t.pk IN :ids`).

### The schema carries shape only

Tables, typed columns (`NUMBER | TEXT | DATETIME | BOOLEAN`), primary keys,
per-table time columns, and FK links. No URLs, no credentials.

Validation enforces that links resolve and that link targets have primary
keys. FK columns are graph edges, not values.

A primary key is identity by default. When the key itself carries meaning — a
SKU, an ISBN, an airport code — declare it as a column as well, exactly as you
would a `time_column`, and it is emitted as a feature cell too:

```python
TableDef.new_table("users").primary_key("user_id")          # identity only
TableDef.new_table("products")                              # ...and a feature
    .column("stock_code", ValueType.TEXT).primary_key("stock_code")
```

Leave synthetic keys out: autoincrement ids track insertion order, so feeding
one to the model invites it to read the id as a tenure proxy that will not
survive a new id range.

### Execution is GraphQL-style

The engine owns the language, planning, context assembly, and model routing.
It never connects to a database: all data access goes through
[retrievers](#retrievers) you implement. The same query runs against JDBC, a
REST service, a feature store, or an in-memory test double — only the wiring
changes.


## Retrievers

The engine never touches a database. It asks **your** code for rows through
five small interfaces (Java: async interfaces; Python: plain callables; Rust:
traits, usually closures):

| Interface | Signature (conceptual) | Role |
|---|---|---|
| `EntityRetriever` | `(table, ids, bound) → rows` | Batched point lookup: seed rows, parents |
| `LinkRetriever` | `(link, parent_id, bound, limit) → rows` | Children along one FK link, newest-first |
| `CohortRetriever` *(optional)* | `(table, anchor, bound, limit) → ids` | Similar entities for in-context examples |
| `TableScanner` *(optional)* | `(table, bound) → row stream` | Bulk streaming; enables whole-table `FROM` and CSC mode |
| `StatsProvider` *(optional)* | — | Normalization statistics |

### Rows

A `Row` carries typed cells, an optional timestamp, and **parent edges**
(`{fk_column: parent_id}`). FK values are not cells — they surface as edges.
The primary key surfaces as identity, and additionally as a cell when the
schema declares it as a column.

:::caution
A row whose table declares no feature columns emits no tokens, and a
token-less row that others link through is a dead end — nothing below it can
reach the prediction, and every entity scores alike. The engine raises
`ContextConnectivityWarning` when it detects this. Give the table a feature
column, or declare its primary key as one.
:::

### Wiring

A `RetrieverWiring` binds retrievers to tables and links, with a
`default_links` catch-all. It is validated against the schema when the engine
is built, so a missing retriever fails fast, not mid-query.

### Contract essentials

- Return **nothing newer** than the `TemporalBound` you are given. (The
  engine re-checks anyway — see
  [Temporal correctness](#temporal-correctness).)
- `LinkRetriever` returns children **newest-first**, capped at `limit`.
- Retrievers own their I/O: batching, caching, auth, and concurrency are
  yours. In Java the SPI is async (`CompletionStage`); Python and Rust are
  deliberately synchronous.


## Temporal correctness

Temporal leakage — a "future" fact sneaking into the features — is the classic
way predictive systems lie in backtests. relativedb treats leakage prevention
as an **engine guarantee**, not a user discipline.

### The anchor time

Every execution has an anchor time t₀. The prediction target reads the window
*after* t₀; the assembled context may only contain data at or *before* t₀.

### Defense in depth

1. Every retriever call carries a `TemporalBound` — "return nothing newer
   than this". Rows without timestamps (static dimension tables) are always
   admitted.
2. The engine **re-checks every returned row** against the bound and drops
   violations. A buggy or malicious retriever cannot leak the future into
   context. Dedicated tests in all three libraries feed a deliberately broken
   retriever and assert the future row never appears.

### Window direction is validated

Target windows must face the future (non-negative offsets); `WHERE` filter
windows face the past (negative or `-INF` starts). The validator rejects
queries that mix these up.

### Backtesting for free

Because "as of" is an explicit input, evaluating yesterday's model is just
running the same query with yesterday's anchor — no snapshot tables, no
point-in-time joins.


## Sampler modes

Context assembly walks the graph: seed entity → parents (always followed) →
children (fanout-capped, newest-first) → optional cohort, until the hop limit
or cell budget. Two interchangeable samplers drive this walk — both produce
**identical contexts** (asserted by tests).

### RETRIEVER (default)

Pull-per-hop: the hop loop calls your retrievers for each expansion.

Use when data is **remote, huge, or access-controlled** — nothing is copied,
your retrievers see every access.

### CSC

The engine drains each `TableScanner` once into in-memory
compressed-sparse-column adjacency arrays (time-sorted neighbor lists), then
samples entirely in-process — "latest *w* children ≤ anchor" is one binary
search plus a tail slice.

Use for **latency-sensitive, repeated scoring** over data that fits in
memory. The index is a snapshot; rebuild with `engine.refresh()`.

### Context budgets

`ContextPolicy` supports two geometries:

- per-hop fanouts, e.g. `fanouts(64, 64)`
- a uniform `bfs_width` under a global `max_context_cells` budget

See [Choose a sampler mode](#choose-a-sampler-mode) for a decision
guide and benchmark numbers.


## Model backends

Scoring is behind a two-method `ModelBackend` SPI. **A model backend is
required** — the engine has no built-in scorer and raises a clear error if you
execute a query without one. The shipped backend is `RtNativeBackend`, which
runs the RT-J relational foundation model.

### RtNativeBackend

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

#### Supported output types

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

### Checkpoint routing

`ModelConfig` maps the inferred [task type](/relql/#task-types) to a
checkpoint URI:

| Task type | Default URI |
|---|---|
| classification, ranking | `hf://stanford-star/rt-j/classification` |
| regression, forecasting | `hf://stanford-star/rt-j/regression` |
| text embeddings | `all-MiniLM-L12-v2` (pinned, 384-dim) |

`hf://` URIs resolve against the **local** Hugging Face cache only — nothing
downloads implicitly. `file://` and plain paths also work.

### Bring your own

Implement `ModelBackend` to plug in any scorer; the engine hands you assembled
contexts and the routed model URI.

### Testing device

The engine's own unit tests use a tiny **deterministic stub** backend so the
pipeline can be exercised without loading a checkpoint. It is a test double,
not a shipped or default predictor — do not rely on it to serve real
predictions.


## Relational transformers

relativedb's default real model is **RT-J** (`stanford-star/rt-j`), a
relational transformer — a foundation model for relational data.

### The idea

A transformer normally attends over a sequence of word tokens. A relational
transformer attends over a **small subgraph of your database**: each token is
one cell (a feature value of some row), and attention is masked along the
structure that relates them:

- **column attention** — cells of the same column, across rows
- **feature attention** — cells of the same row and its FK parents
- **neighbor attention** — cells of a row's FK children

There are no positional encodings — structure is carried entirely by the
masks. The model reads the entity, its neighborhood, and a handful of labeled
in-context examples (including the entity's own past outcomes), and predicts
the masked target cell in a single forward pass.

### Why this beats the alternatives

**vs. classical tabular ML (GBDTs on feature tables).** No hand-built
features, no per-task training, no train/serve skew, and no temporal-leakage
bugs hiding in feature SQL. Change the question, change the query string.

**vs. graph neural networks.** GNNs are also structure-aware but are trained
per task and per schema. A relational transformer is pretrained across many
schemas and predicts **in-context** — the relational analogue of prompting an
LLM instead of fine-tuning one.

**vs. LLMs on serialized rows.** Flattening tables into text throws away
types, keys, and time, and burns context on formatting. The relational
transformer consumes typed cells and real graph structure directly.

### In relativedb

The engine assembles the temporally-bounded context; the model scores it.
Checkpoints are routed per task type (classification vs regression), text
cells embed with a pinned MiniLM encoder, and inference runs in a
[dependency-light C++ engine](#c-inference-engine-rtcpp) verified against the PyTorch
reference. Scoring requires this model backend (`RtNativeBackend`) — there is
no model-free default; the engine's own tests use a deterministic stub.


## Predict churn

Goal: for every *currently active* customer, the probability of no activity in
the next 30 days.

### 1. Wire your tables

```python
import pandas as pd
import relativedb

# Declare `schema`, translate DataFrame records to `relativedb.Row`, and wire
# entity/link/scanner callbacks. The complete example connector is linked below.
wiring = relativedb.RetrieverWiring.new_wiring()...build()
engine = relativedb.Engine(schema, wiring)
```

### 2. Write the query

Define churn in the target; restrict the population to active users in
`WHERE` (past-facing window):

```sql
PREDICT NOT EXISTS(events.*) OVER (30 DAYS FOLLOWING)
FROM users
WHERE EXISTS(events.*) OVER (90 DAYS PRECEDING)
```

The target aggregates the *next* 30 days (`30 DAYS FOLLOWING`); the `WHERE`
looks *back* 90 days (`90 DAYS PRECEDING`). `EXISTS(events.*)` is the boolean
existence check — shorthand for the older `COUNT(events.*) > 0` idiom.

### 3. Score as of today

```python
result = engine.execute(relativedb.ExecutionInput(
    query=query, anchor_time=pd.Timestamp.utcnow().to_pydatetime()))
df = pd.DataFrame({"entity_id": [p.id for p in result.predictions],
                   "probability": [p.probability for p in result.predictions]})
df.sort_values("probability", ascending=False).head(20)
```

Each row is `entity_id, probability`. Users inactive for 90+ days are excluded
by the `WHERE` clause — they already churned.

### Variations

- Different definition: `NOT EXISTS(orders.*) OVER (60 DAYS FOLLOWING)` (no
  purchase) or `SUM(usage.minutes) OVER (30 DAYS FOLLOWING) < 10` (low usage).
- Backtest: rerun with a past `anchor_time` and compare against what actually
  happened — the engine guarantees the context is point-in-time correct.
- Real model: construct the engine with
  `model_backend=relativedb.RtNativeBackend(schema=schema)`
  ([guide](#use-the-native-rt-j-backend)).

A complete self-checking version lives at `examples/industry/growth_churn.py`;
its `pandas_connector.py` shows the application-owned connector.


## Forecast demand

Goal: weekly unit sales per store for the next four weeks.

### The query

A window with a `HORIZONS N` clause repeats the frame N times, back to back,
and asks the engine for one value per horizon. A multi-horizon window *implies*
forecasting — there is no separate clause:

```sql
PREDICT SUM(sales.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)
FROM stores
```

Horizon 1 covers days (0, 7], horizon 2 covers (7, 14], and so on. With no
`STEP`, each horizon steps forward by the frame width (7 days here), so the
four horizons tile the next 28 days without overlap.

### Run it

```python
result = engine.execute(ExecutionInput(query=query, anchor_time=t0))
forecasts = {p.id: p.forecast for p in result.predictions}
```

Here `engine` uses the schema and application-owned retrievers wired over your
store and sales data. The result has one prediction per store with four
horizons. A window whose `HORIZONS > 1` routes to the regression checkpoint.

### Overlapping horizons with STEP

`STEP` sets how far each horizon advances; it defaults to the frame width. Make
`STEP` smaller than the frame to get overlapping, rolling windows — for
example a 30-day trailing demand projection re-issued every 7 days, six times:

```sql
PREDICT SUM(sales.qty) OVER (30 DAYS FOLLOWING HORIZONS 6 STEP 7 DAYS)
FROM stores
```

Each of the six horizons still spans 30 days, but their start points are only
7 days apart, so consecutive horizons overlap.

### Notes

- The base frame can be any unit: `SUM(usage.count) OVER (1 DAY FOLLOWING
  HORIZONS 28)` gives a daily 4-week forecast.
- Pin the prediction time with `AS OF` and pick an output shape with `RETURN`,
  e.g. `... AS OF :prediction_time RETURN EXPECTED VALUE`.
- Backtest by moving `anchor_time` (or `AS OF`) into the past; the engine
  guarantees each horizon only saw data available at that anchor.
- A complete self-checking version lives at
  `examples/industry/bizops_demand_forecast.py`.


## Rank recommendations

Goal: for each customer, the top 3 products they are most likely to order in
the next 30 days ("buy it again").

### The query

`LIST_DISTINCT` predicts a set of linked IDs; `RANK TOP K` turns it into a
ranking task:

```sql
PREDICT LIST_DISTINCT(orders.product_id) OVER (30 DAYS FOLLOWING RANK TOP 3)
FROM customers
```

### Run it

```python
result = engine.execute(ExecutionInput(query=query, anchor_time=t0))
rankings = {p.id: p.ranked for p in result.predictions}
```

Here `engine` is already wired to your application-owned retrievers. The
result contains a ranked list of product IDs per customer. Note
`orders.product_id` is an FK — the ranking works over graph edges
(`Row.parents`), never over ID feature values.

### Notes

- `K` bounds the returned list, not the candidate set.
- Use `CLASSIFY` instead of `RANK TOP K` for a multilabel-style yes/no per
  item.
- A complete self-checking version (habitual staple ranked #1 per customer)
  lives at `examples/industry/pzn_buy_it_again.py`.


## Wire custom retrievers

Goal: run RelQL against data the engine cannot see — behind a DAO, a REST
service, or a feature store.

### 1. Declare the schema (shape only)

```python
from relativedb import Schema, TableDef, LinkDef, ValueType

schema = (Schema.new_schema()
    .table(TableDef.new_table("customers")
        .column("age", ValueType.NUMBER)
        .primary_key("customer_id").build())
    .table(TableDef.new_table("orders")
        .column("qty", ValueType.NUMBER)
        .column("order_date", ValueType.DATETIME)
        .primary_key("order_id").time_column("order_date").build())
    .link(LinkDef("orders", "customer_id", "customers"))
    .build())
```

### 2. Implement the two required retrievers

```python
from relativedb import RetrieverWiring, Row

wiring = (RetrieverWiring.new_wiring()
    # batched point lookup
    .entities("customers", lambda table, ids, bound: customer_dao.by_ids(ids))
    .entities("orders",    lambda table, ids, bound: order_dao.by_ids(ids, bound))
    # children of a parent along an FK link: newest-first, ≤ limit, ≤ bound
    .default_links(lambda link, parent_id, bound, limit:
                   order_dao.recent_by_customer(parent_id, bound.as_of, limit))
    .build())
```

Retriever rules:

- Respect the `bound` — return nothing newer. (The engine re-checks and drops
  violations, but pushing the bound into your query is faster.)
- Return `Row` objects: typed cells + timestamp + parent edges. Never put IDs
  in cells.
- `default_links` must be newest-first and honor `limit` — push
  `ORDER BY ts DESC LIMIT n` into your store.

### 3. Optional: enable whole-table `FROM` and CSC

`FROM table` needs a `TableScanner` to enumerate the population:

```python
    .scanner("customers", lambda table, bound: customer_dao.scan_all(bound))
```

Scanners also unlock [CSC mode](#sampler-modes) for in-memory
sampling.

### 4. Build and run

```python
from relativedb import Engine, ExecutionInput

engine = Engine(schema, wiring)   # wiring validated here — missing pieces fail fast
result = engine.execute(ExecutionInput(query=..., anchor_time=...))
```

In Java the same SPI is async (`CompletionStage`) — see the
[Java library](#java-library).


## Choose a sampler mode

Both modes produce identical contexts. Choose by data locality.

| | RETRIEVER (default) | CSC |
|---|---|---|
| Data location | stays in your store | copied into an in-memory index |
| Freshness | live, per query | snapshot (`engine.refresh()`) |
| Requires | Entity + Link retrievers | `TableScanner` per table |
| Best for | remote, huge, or access-controlled data | repeated low-latency scoring |

### Switching

```python
from relativedb import Engine, ExecutionInput, SamplerMode
engine = Engine(schema, wiring, sampler_mode=SamplerMode.CSC)
result = engine.execute(ExecutionInput(query=query, anchor_time=t0))
```

```java
RelativeDbEngine.newEngine(schema, wiring).samplerMode(SamplerMode.CSC).build();
```

### What CSC buys you

On a synthetic churn workload (10,000 customers, 200,000 orders, history
baseline, M-series laptop), scoring every customer:

| Approach | Time | Throughput |
|---|---|---|
| relativedb, CSC sampler | 0.66 s | ~15,000 entities/s |
| naive per-entity pandas loop | 57.4 s | ~174 entities/s |

(Reproduce with `examples/bench_naive_vs_csc.py`.) The CSC index turns each
"latest *w* children ≤ anchor" expansion into a binary search plus a tail
slice, and its build cost is paid once per snapshot.

### Rule of thumb

Start with RETRIEVER. Move to CSC when the same engine scores many queries or
large populations and the data fits in memory.


## Use the native RT-J backend

`RtNativeBackend` is the scoring path: the engine has no model-free default and
raises a clear error if you execute a query without a model backend. This page
sets it up — build `librt_c`, get the checkpoint, and wire the backend.

### 1. Build the C++ engine

```bash
cd cpp
cmake -B build -S . && cmake --build build -j
```

This produces `cpp/build/librt_c.{dylib,so}`. All bindings find it there
automatically; elsewhere, set `RELATIVEDB_RT_LIB=/path/to/librt_c.dylib`.

### 2. Get the checkpoints

Default routing resolves `hf://stanford-star/rt-j/{classification,regression}`
against your **local** Hugging Face cache — nothing downloads implicitly.
`file://` and plain paths work via a custom `ModelConfig`.

### 3. Plug in the backend

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

### What to expect

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


## Python library

PyPI distribution `relationdb` (imported as `relativedb`). Python 3.10+; core
depends only on numpy.

```bash
pip install relationdb    # extras: [rt], [dev]
```

### Bring your own connector

The package deliberately has no pandas adapter, database client, or schema
inference. Applications translate their records to `Row` and wire retriever
callbacks. Pandas can still be used entirely in application code:

```python
import pandas as pd
from relativedb import Engine, ExecutionInput, RetrieverWiring, Row

rows = [Row("customers", r.customer_id, {"age": float(r.age)})
        for r in customers.itertuples()]
# Implement entity/link/scanner callbacks over `rows` and your other frames.
wiring = RetrieverWiring.new_wiring()...build()
result = Engine(schema, wiring).execute(ExecutionInput(query=query, anchor_time=t0))
df = pd.DataFrame({"entity_id": [p.id for p in result.predictions]})
```

The repository’s `examples/industry/pandas_connector.py` is a complete,
application-owned reference connector—not part of the installed package.

### The core API

- **Schema** — `Schema.new_schema().table(TableDef.new_table(...)...).link(LinkDef(...)).build()`
- **Wiring** — `RetrieverWiring.new_wiring().entities(...).default_links(...).scanner(...).build()`;
  retrievers are plain callables (`typing.Protocol`)
- **Engine** — `Engine(schema, wiring, model_backend=...)`;
  `engine.execute(ExecutionInput(query=..., anchor_time=..., params=...))`;
  `params` binds the query's `:name` placeholders — the anchor, the cohort
  (`WHERE t.pk IN :ids`), and any other parameterized literal
- **RelQL** — `relativedb.parse(q)`, `relativedb.validate(pq, schema)`,
  `pq.task_type()`
- **Backends** — a model backend is required; `RtNativeBackend(schema=...)`
  runs RT-J. There is no model-free default

Errors are specific: `RelqlSyntaxError`, `RelqlValidationError`, `SchemaError`,
`WiringError`, `ExecutionError`, `RtNativeUnavailableError`.

### Tests

```bash
.venv/bin/python -m pytest
```

Covers the shared 67-query RelQL corpus (+20 rejections), the temporal-leakage
guard, CSC ≡ retriever equivalence, model routing, and the retriever→churn
path end to end.


## Java library

Maven publications under group `com.relativedb`; requires Java 17+.

| Module | Contents |
|---|---|
| `relationdb` | Schema builder, retriever SPI, RelQL parser (native `librt_c`) + validation, context assembly (both sampler modes), model SPI |
| `relationdb-rt` | Optional JNA binding to the native RT-J engine: `RtNativeBackend implements ModelBackend` |

```kotlin
dependencies {
    implementation("com.relativedb:relationdb:0.1.0")
    // implementation("com.relativedb:relationdb-rt:0.1.0")
}
```

### API shape

```java
RelativeDbSchema schema = RelativeDbSchema.newSchema()... .build();
RetrieverWiring wiring  = RetrieverWiring.newWiring()... .build();

RelativeDbEngine engine = RelativeDbEngine.newEngine(schema, wiring)
    .samplerMode(SamplerMode.CSC)        // optional
    .modelBackend(new RtNativeBackend(schema))   // required; runs RT-J
    .build();

PredictionResult r = engine.execute(ExecutionInput.newInput()
    .query("PREDICT ... FROM customers WHERE customers.customer_id IN :ids")
    .anchorTime(t0)
    .param("ids", ids)                   // omit the pk predicate + FROM
    .build()).toCompletableFuture().join();   //   → TableScanner enumerates
```

Key packages: `com.relativedb.schema`, `.retrieve`, `.query` (entry point
`Relql.parse` / `Relql.validate`), `.engine`, `.model`, `.rt`.

### Async retriever SPI

Retrievers return `CompletionStage` — fan out to remote services without
blocking the engine. Every call carries a `TemporalBound`; the engine
re-checks all returned rows.

### Native backend

`RtNativeBackend` loads `librt_c` lazily: system property `relativedb.rt.lib`
→ env `RELATIVEDB_RT_LIB` → sibling `cpp/build/` → loader path. `hf://`
checkpoint URIs resolve from the local HF cache (override root with
`relativedb.rt.hf.cache` / `RELATIVEDB_RT_HF_CACHE`). Text embeddings come
through the `TextEncoder` SPI (`PrecomputedEncoder` for closed vocabularies).

A golden-forward test replays `cpp/testdata/*.bin` and matches the
PyTorch-verified scores; it auto-skips when the native library is absent.


## Rust library

crates.io package `relationdb` (crate API `relativedb`, edition 2021); depends
only on `chrono` and `libloading`.

```bash
cargo add relationdb
```

### API shape

```rust
use relativedb::{Engine, ExecutionInput, RetrieverWiring, Schema, /* ... */};

let schema = Schema::new_schema()... .build();
let wiring = RetrieverWiring::new_wiring()
    .entities("customers", /* closure */)
    .default_links(/* closure: newest-first, honors bound + limit */)
    .scanner("customers", /* closure: enables whole-table FROM + CSC */)
    .build();

let mut engine = Engine::new(schema, wiring);
let result = engine.execute(
    ExecutionInput::query("PREDICT ...").anchor_time(t0))?;
```

Modules: `schema`, `retrieve`, `relql` (decodes the shared native `librt_c`
parser's JSON AST), `engine`, `model`, `native`, `csc`. Errors surface through
`relativedb::Result` / `relativedb::Error`.

### Design decision: synchronous SPI

Where Java's SPI is async, the Rust (and Python) SPI is synchronous and
infallible — traits return plain `Vec`s. An async SPI would force a runtime
choice on every user and color the whole engine `async`; batching retrievers
can run their own I/O concurrency internally.

### Native backend

`native::RtNativeBackend` binds `librt_c` via `libloading`, discovered from
`RELATIVEDB_RT_LIB` or the sibling `cpp/build/`. The golden gate:

```bash
RELATIVEDB_RT_LIB=../cpp/build/librt_c.dylib \
  cargo test --test golden_tests -- --nocapture
```

The shared 67-query RelQL corpus lives in this crate
(`tests/data/examples.relql`) and is exercised by all three languages.


## C++ inference engine (rt.cpp)

A dependency-light C++20 implementation of the RT-J forward pass — ~700 lines,
no torch, no Python at inference. It backs the `RtNativeBackend` in all three
libraries through one C ABI (`librt_c`).

### What it implements

- 12 blocks of column / feature / neighbor **masked attention** + SwiGLU FFN,
  pre-RMSNorm residuals; no positional encodings — structure is carried by the
  masks
- Faithful attention details: per-head QK-RMSNorm, log(kv-count) query
  scaling, sigmoid output gating, score scale `1/head_dim`
- Per-sem-type value encoders, number-head decoding, built-in safetensors
  loader (bf16 → fp32)

### Performance design

Idioms from llama.cpp / vLLM on Apple Accelerate: stacked-QKV GEMM panels over
the whole batch, grouped masked attention that never materializes S×S,
persistent thread pool, zero allocation inside the block loop.

### Build and verify

```bash
cd cpp
cmake -B build -S . && cmake --build build -j

./build/rt_test testdata <path>/classification/model.safetensors  # golden gate
./build/rt_bench <testdata> <model.safetensors>                   # batching + speed + memory
```

Targets: `rt` (static lib), **`librt_c`** (shared, the C ABI in `src/rt_c.h`),
`rt_test`, `rt_bench`.

The golden test replays a batch dumped from the PyTorch reference and matches
final scores to ~3–4 decimals — remaining differences are fp32 op-ordering
drift. The Java, Python, and Rust bindings each re-run this gate through
their own FFI layer.


## Releasing the libraries

The manual **Release libraries** GitHub Actions workflow builds and verifies
all three distributions. It does not publish by default. Run it with
`publish` left off to test the complete packaging path and retain the Python,
Java, and Rust artifacts on the workflow run.

The public distribution coordinates are:

| Ecosystem | Distribution |
|---|---|
| PyPI | `relationdb` (import `relativedb`) |
| Maven Central | `com.relativedb:relationdb` and `com.relativedb:relationdb-rt` |
| crates.io | `relationdb` (crate API `relativedb`) |

### One-time registry setup

Before the first publish:

1. Create protected GitHub environments named `pypi`, `crates-io`, and
   `maven-central`, ideally with required reviewers.
2. Register `.github/workflows/release-libraries.yml` as a PyPI trusted
   publisher for the `relationdb` project and the `pypi` environment. No PyPI
   API token is needed.
3. Add a crates.io token as the `CARGO_REGISTRY_TOKEN` environment secret on
   `crates-io`.
4. Verify ownership of the `com.relativedb` namespace in the Sonatype Central
   Portal. Add `MAVEN_CENTRAL_USERNAME`, `MAVEN_CENTRAL_PASSWORD`,
   `MAVEN_SIGNING_KEY`, and `MAVEN_SIGNING_PASSWORD` as `maven-central`
   environment secrets. The signing key is the ASCII-armored private key.

### Publishing

Update the matching version in `python/pyproject.toml`,
`rust/relativedb/Cargo.toml`, and `java/build.gradle`, then run tests locally.
Dispatch **Release libraries** with `publish` off first. Inspect its three
uploaded artifacts. When they are ready, dispatch the same commit again with
`publish` enabled and approve each protected environment.

Publishing is manual-only: pushing a branch or tag cannot trigger it.
