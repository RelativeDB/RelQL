# relativedb

**Predictive queries over your own relational data.**

relativedb is a predictive-query engine: you declare the *shape* of your
relational data (tables, keys, links), wire small **retriever** callbacks over
whatever storage you already have, and ask questions about the **future** in
**RelQL** — a SQL-flavored Predictive Query Language:

```sql
PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0
FOR EACH customers.customer_id
```

*"For every customer, what is the probability they place zero orders in the
next 90 days?"* — i.e. 90-day churn, as one line, with no feature engineering,
no training pipeline, and no possibility of accidentally leaking the future
into the features.

The engine parses and validates the query, walks your relational graph through
your retrievers to assemble a **temporally-bounded context** per entity, and
scores it with a pluggable model backend — by default a transparent
history-based baseline, and optionally **RT-J** (the Stanford Relational
Transformer successor, `stanford-star/rt-j`), a relational foundation model
that predicts in-context, served by a golden-verified native C++ inference
engine.

The same engine is implemented three times, as true peers with identical
concepts and behavior:

| Library | Directory | Package / artifact | Notes |
|---|---|---|---|
| **Python** | [`python/`](python/) | `relationdb` (PyPI; import `relativedb`) | Storage-neutral retriever SPI; applications own every connector |
| **Java** | [`java/`](java/) | `com.relativedb:relationdb`, `relationdb-rt` (Maven Central) | ANTLR-based parser; async (`CompletionStage`) retriever SPI |
| **Rust** | [`rust/`](rust/) | `relationdb` (crates.io; crate API `relativedb`) | Hand-written parser; deliberately synchronous SPI |

plus a shared native model runtime:

| Component | Directory | What it is |
|---|---|---|
| **rt.cpp** | [`cpp/`](cpp/) | ~700-line dependency-light C++20 implementation of the RT-J forward pass (`librt_c`), golden-verified against the PyTorch reference; all three libraries bind to it |
| **Examples** | [`examples/industry/`](examples/industry/) | Self-checking, runnable industry scenarios (churn, fraud, demand forecast, personalization) |

---

## Table of contents

- [Why relativedb](#why-relativedb)
- [Use cases](#use-cases)
- [How it works](#how-it-works)
  - [Execution model](#execution-model)
  - [The retriever contract](#the-retriever-contract)
  - [Temporal correctness](#temporal-correctness)
  - [Sampler modes: RETRIEVER vs CSC](#sampler-modes-retriever-vs-csc)
  - [Model backends and routing](#model-backends-and-routing)
- [RelQL — the Predictive Query Language](#relql--the-predictive-query-language)
  - [Clause structure](#clause-structure)
  - [Aggregations and time windows](#aggregations-and-time-windows)
  - [Conditions and operators](#conditions-and-operators)
  - [Task-type inference](#task-type-inference)
  - [Example gallery](#example-gallery)
- [The Python library](#the-python-library)
- [The Java library](#the-java-library)
- [The Rust library](#the-rust-library)
- [rt.cpp — the native inference engine](#rtcpp--the-native-inference-engine)
- [Runnable industry examples](#runnable-industry-examples)
- [Configuration reference](#configuration-reference)
- [Design invariants](#design-invariants)
- [Repository layout](#repository-layout)
- [License](#license)

---

## Why relativedb

Most business-critical ML questions are **predictive queries over relational
data**: will this customer churn, will this transaction charge back, how many
units will this store sell next week, which products will this shopper buy
again. The classical path to answering them is long and fragile:

1. hand-craft point-in-time-correct feature tables (the single most common
   source of silent bugs — *temporal leakage*),
2. build and maintain a training pipeline per question,
3. train, deploy, and monitor a model per question,
4. repeat from scratch when the question changes.

relativedb collapses that into a query. Three ideas make this work:

**1. The question is a query, not a pipeline.** RelQL expresses the target
("count of orders in the next 90 days is zero"), the population ("for each
customer"), and any filters or assumptions — declaratively. Change the
question, change the string.

**2. The engine never touches your database.** Execution is GraphQL-style: the
engine owns the query language, planning, temporal context assembly, and model
routing, and **all data access goes through user-defined retrievers**. There
are no bundled connectors, no credentials, no SQL generation. The same query
runs against JDBC, a REST microservice, a feature store, pandas DataFrames, or
an in-memory test double — only the wiring changes. Your access-control,
caching, and batching logic stays yours.

**3. The model is a foundation model that predicts in-context.** Instead of
training a model per question, the assembled per-entity context (the entity,
its neighbors across FK links, its own historical outcomes) is scored by a
relational transformer in a single forward pass — the relational analogue of
prompting an LLM. A model-free history baseline is built in, so the whole
pipeline runs (and is testable) with zero model artifacts.

## Use cases

Everything expressible as *"predict an aggregate/attribute of linked future (or
missing) data, per entity, as of a point in time"*:

| Domain | Question | RelQL sketch |
|---|---|---|
| **Growth / subscription** | Which active users will churn? | `PREDICT NOT EXISTS(events.*) OVER (30 DAYS FOLLOWING) FOR EACH users.user_id WHERE EXISTS(events.*) OVER (90 DAYS PRECEDING)` |
| **Payments / fraud** | Which accounts will incur a chargeback? | `PREDICT EXISTS(chargebacks.*) OVER (60 DAYS FOLLOWING) FOR EACH accounts.account_id` |
| **Retail / bizops** | Units sold per store, next 4 weeks, weekly? | `PREDICT SUM(sales.qty) OVER (7 DAYS FOLLOWING HORIZONS 4) FOR EACH stores.store_id` |
| **Personalization** | Which products will a shopper buy again? | `PREDICT LIST_DISTINCT(orders.product_id) OVER (30 DAYS FOLLOWING) RANK TOP 3 FOR EACH customers.customer_id` |
| **Revenue** | Customer spend over the next quarter (LTV slice)? | `PREDICT SUM(transactions.price) OVER (90 DAYS FOLLOWING) FOR EACH customers.customer_id` |
| **Credit / risk** | Will this loan avoid denial? | `PREDICT LAST(loan.status) OVER (30 DAYS FOLLOWING) NOT LIKE '%DENIED' FOR EACH loan.id` |
| **Data quality** | Is this static attribute missing/predictable? | `PREDICT articles.description IS NULL FOR EACH articles.id` |
| **What-if** | Would this user churn *if* they were on premium? | `PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id = 42 ASSUMING users.plan = 'premium'` |

Four of these are implemented end-to-end (with planted signal and assertions)
in [`examples/industry/`](examples/industry/).

## How it works

### Execution model

Every execution follows the same four stages, in every language:

```
 RelQL string
    │  parse            (typed AST; syntax errors here)
    ▼
 ParsedQuery
    │  validate         (bind names/types/windows against the declared Schema;
    ▼                    infer the TaskType)
 ValidatedQuery
    │  assemble         (hop loop: seed entity → parents → children → cohort,
    ▼                    through YOUR retrievers, bounded by the anchor time)
 per-entity contexts
    │  score            (ModelBackend routed by TaskType via ModelConfig)
    ▼
 PredictionResult      (one prediction per entity: value and/or probability)
```

The inputs are: the **query**, an **anchor time** (the "as of" instant t₀ —
predictions concern the window after it, context may only contain data at or
before it), and either an explicit list of **entity IDs** or `FOR EACH` over
the whole entity table (enumerated via a `TableScanner`).

The **schema** you declare carries shape only — table names, typed columns
(`NUMBER | TEXT | DATETIME | BOOLEAN`), primary keys, per-table time columns,
and FK links (`orders.customer_id → customers`). No URLs, no credentials.
Schema validation enforces that links resolve, link targets have primary keys,
and — importantly — that **PK/FK columns are never feature columns**: IDs are
graph edges, not values (invariant F17). There is no way to hand the model an
identifier as a feature.

### The retriever contract

Five small interfaces (Java: async interfaces; Python: `typing.Protocol`
callables; Rust: traits, implemented by closures) mirror the sampling
primitives:

- **`EntityRetriever(table, ids, bound) → rows`** — batched point lookup
  (seed rows, parents).
- **`LinkRetriever(link, parent_id, bound, limit) → rows`** — children of one
  parent along one FK link, newest-first, capped at the engine's fanout.
- **`CohortRetriever(table, anchor, bound, limit) → ids`** *(optional)* —
  similar entities used to seed in-context examples.
- **`TableScanner(table, bound) → row stream`** *(optional)* — bulk streaming;
  required for CSC mode and for enumerating `FOR EACH` populations.
- **`StatsProvider`** *(optional)* — training-split normalization statistics.

A `Row` carries typed cells, an optional timestamp, and **parent edges**
(`{fk_column: parent_id}`). A `RetrieverWiring` binds retrievers to tables and
links (with a `default_links` catch-all), and is validated against the schema
at engine construction.

### Temporal correctness

Temporal leakage — a "future" fact sneaking into the features — is the classic
way predictive systems lie to you in backtests. relativedb treats it as an
engine-level guarantee, not a user discipline:

- Every retriever call carries a **`TemporalBound`** — "return nothing newer
  than this". Rows with no timestamp (static dimension tables) are always
  admitted.
- The engine **re-checks every returned row** against the bound and drops
  violations — defense in depth (invariant F24). A buggy or malicious
  retriever cannot leak the future into context. This is covered by dedicated
  tests in all three libraries.
- Target windows read the *future* relative to the anchor; `WHERE`-filter
  windows read the *past* (negative or `-INF` starts). The validator enforces
  the signs.

### Sampler modes: RETRIEVER vs CSC

Both modes produce **identical contexts** (asserted by tests); choose by data
locality:

- **`RETRIEVER`** (default) — pull-per-hop: the hop loop calls your retrievers
  for each expansion. Right when data is remote, huge, or access-controlled.
- **`CSC`** — the engine drains each `TableScanner` once into in-memory
  compressed-sparse-column adjacency arrays (`colptr`/`row`, neighbor lists
  time-sorted), then samples multi-hop context in-process: "latest *w*
  children ≤ anchor" is one binary search plus a tail slice. Right for
  latency-sensitive, repeated scoring over data that fits in memory. The index
  is a snapshot; rebuild with `engine.refresh()`.

The hop loop itself is shared: seed → parents (always followed) → children
(fanout-capped, newest-first) → cohort (optional), stopping at the hop limit
or the context budget. `ContextPolicy` supports both budget geometries:
per-hop `fanouts(64, 64)` (KumoRFM-style) or a uniform `bfs_width` under a
global `max_context_cells` budget (RT-style).

### Model backends and routing

The model is behind a two-method **`ModelBackend`** SPI. Out of the box:

- **`HistoryBaselineBackend`** (default) — model-free: evaluates the target
  over the entity's own trailing history windows ("self labels", invariant
  F65). Transparent, deterministic, zero artifacts — the entire pipeline runs
  and tests without any model.
- **`RtNativeBackend`** — scores contexts with the real **RT-J** checkpoints
  through the C++ engine (`librt_c`). It converts each assembled context into
  the raw RT token batch — one token per feature cell, FK links as the node
  graph, per-column z-scores for numbers/booleans, MiniLM
  (`all-MiniLM-L12-v2`, 384-dim, pinned) embeddings for text cells and
  `"<column> of <table>"` schema phrases — plus a synthetic masked *task* row
  anchored at prediction time, with the entity's own past outcomes as
  in-context examples. Classification logits pass through a sigmoid;
  regression outputs denormalize with in-context label statistics.

**`ModelConfig`** routes checkpoints by inferred task type:

| TaskType | Default model URI |
|---|---|
| binary / multiclass classification, ranking | `hf://stanford-star/rt-j/classification` |
| regression, forecasting | `hf://stanford-star/rt-j/regression` |
| text embeddings (both) | `all-MiniLM-L12-v2` (384-dim, pinned) |

`hf://` URIs resolve against the **local** Hugging Face cache (no implicit
downloading); `file://` and plain paths work too.

---

## RelQL — the Predictive Query Language

RelQL derives from Kumo/KumoRFM's predictive query language. The canonical
parser is single-sourced in C++ ([`cpp/src/pql.hpp`](cpp/src/pql.hpp) /
[`cpp/src/pql.cpp`](cpp/src/pql.cpp)) and emits a JSON AST that the Python, Rust,
and Java bindings decode, all verified against a shared query corpus
([`rust/relativedb/tests/data/examples.pql`](rust/relativedb/tests/data/examples.pql))
plus malformed-query rejection cases.

### Clause structure

Clause order is significant:

```sql
[EXPLAIN [PLAN|CONTEXT|ANALYZE|ABLATION] [FORMAT TEXT|JSON]]  -- optional: inspect the plan
PREDICT   <target> [CLASSIFY | RANK TOP <k>]  -- required: what to predict
FOR [EACH] <entity_table>.<pkey>              -- required: the population
          [= <literal> | IN (<list>)]         --   ...or explicit entities
[WHERE     <condition>]                       -- optional: filter (past-facing windows)
[ASSUMING  <temporal_condition>]              -- optional: counterfactual assumption
[AS OF     <anchor>]                          -- optional: bind NOW (:param | DATE | NOW)
[ABLATE TABLE <name>]                         -- optional: drop a table from context (repeatable)
[RETURN    <output>]                          -- optional: request the output object
[WINDOW    <name> AS (<window_spec>)]         -- optional: reusable frame (repeatable)
```

The trailing clauses (`WHERE`, `ASSUMING`, `AS OF`, `ABLATE TABLE`, `RETURN`,
`WINDOW`) may appear in any order; each at most once except `WINDOW`, which is
repeatable. Keywords are case-insensitive; common words like `count` are *soft*
keywords and remain usable as column names (`usage.count` parses).

The **target** is a value expression: a static column reference
(`customers.age`, `articles.description IS NULL`), an aggregation over linked
rows in a temporal frame, or an arithmetic/functional combination of these
(`+ - * /`, parens, `CASE WHEN ... THEN ... ELSE ... END`, `COALESCE`, `NULLIF`,
`ABS/LOG/EXP/LEAST/GREATEST`, `TRUE/FALSE`, column-to-column comparisons),
optionally compared against a literal.

There is no `FORECAST` clause: a target whose window carries `HORIZONS N`
(below) makes the query forecasting.

### Aggregations and time windows

```
AGG( table.column | table.* [WHERE <row filter>] ) [ OVER (<window_spec>) | OVER <name> ]
```

- **Functions**: `SUM, AVG, MIN, MAX, COUNT, COUNT_DISTINCT, LIST_DISTINCT,
  FIRST, LAST, EXISTS`. `EXISTS(t.*)` / `NOT EXISTS(t.*)` is a boolean existence
  test (a cleaner spelling of `COUNT(t.*) > 0`).
- **Temporal frames** are attached with a trailing `OVER (...)` clause:

  ```
  window_spec := frame [HORIZONS <positive-int> [STEP <duration>]]
  frame       := RANGE BETWEEN bound AND bound
               | <duration> PRECEDING          -- shorthand: RANGE BETWEEN <dur> PRECEDING AND NOW
               | <duration> FOLLOWING          -- shorthand: RANGE BETWEEN NOW AND <dur> FOLLOWING
               | UNBOUNDED PRECEDING           -- shorthand: all history up to NOW
  bound       := NOW | <duration> PRECEDING | <duration> FOLLOWING
               | UNBOUNDED PRECEDING | UNBOUNDED FOLLOWING
  duration    := <positive-number> <unit>
  ```

- **Units**: `SECOND(S), MINUTE(S), HOUR(S), DAY(S), WEEK(S), MONTH(S), YEAR(S)`,
  singular or plural, case-insensitive. Frame membership is `(lower, upper]` —
  start-excluded, end-included, relative to the anchor `NOW`.
- **Target** frames must be future-facing (`FOLLOWING`); **filter** frames (in
  `WHERE`) are past-facing (`PRECEDING`, `UNBOUNDED PRECEDING`).
- **Inline row filters**: `COUNT(t.* WHERE t.amount > 10) OVER (30 DAYS FOLLOWING)`.
- **Multi-horizon** (forecasting): `SUM(sales.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)`
  evaluates 4 shifted copies of the frame; `STEP` sets the distance between
  horizon starts (defaults to the frame width) and enables overlapping horizons.
- **Named frames**: declare once with a trailing `WINDOW name AS (<window_spec>)`
  and reference with `OVER name`; this also guarantees alignment when one target
  combines two framed expressions.
- `LIST_DISTINCT` targets take a ranking directive: `RANK TOP K` (ranking) or
  `CLASSIFY` (multilabel-style classification).

### Output intent

`RETURN` requests the desired predictive object; the validator rejects an output
incompatible with the target. Options: `EXPECTED VALUE`, `PROBABILITY`, `CLASS`,
`DISTRIBUTION`, `QUANTILES (0.10, 0.50, 0.90)`, `INTERVAL 90%`, `MULTILABEL`,
`MULTICLASS`. When omitted, the output is inferred from the task type.

`AS OF <anchor>` binds `NOW` (the prediction anchor) explicitly — a `:param`, a
`DATE`, or `NOW`; otherwise the execution input's anchor time binds it.

### Explaining a query

`EXPLAIN` is a prefix. Bare `EXPLAIN` (or `EXPLAIN PLAN`) reports the static
parse/binding/plan and does **not** invoke the model; `EXPLAIN CONTEXT`
assembles context without scoring; `EXPLAIN ANALYZE` runs the query and reports
actual behavior; `EXPLAIN ABLATION` compares ablation variants. `FORMAT TEXT`
(default) is for people, `FORMAT JSON` is a machine-readable schema.

### Conditions and operators

Comparisons `= == != > >= < <=`; boolean composition `AND OR NOT`; membership
`IN (…) / NOT IN (…)`; null tests `IS [NOT] NULL`; string predicates `LIKE`
(SQL `%` wildcards), `CONTAINS`, `STARTS WITH`, `ENDS WITH`. Literals are
numbers, quoted strings, and booleans.

`WHERE` filters the **population** (using entity attributes and past-facing
aggregations). `ASSUMING` states a counterfactual (parsed, validated, and
carried on the query; injecting it into assembled context is an open design
question upstream and not yet applied).

### Task-type inference

The validator infers a `TaskType` from the target's shape; the task type
drives model routing and the output form (value vs probability vs ranked
list):

| Target shape | TaskType |
|---|---|
| bare aggregation (`SUM(...)`, `COUNT(...)`) | regression |
| boolean / compared-to-a-literal target (`COUNT(...) = 0`, `EXISTS(...)`) | binary classification |
| `FIRST`/`LAST`/static categorical column | multiclass classification |
| `LIST_DISTINCT(...) RANK TOP K` | ranking |
| any target whose window has `HORIZONS N` (N > 1) | forecasting |

### Example gallery

Real queries from the shared test corpus:

```sql
PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id

PREDICT COUNT(transactions.*) OVER (30 DAYS FOLLOWING) = 0
FOR EACH customers.customer_id
WHERE COUNT(transactions.*) OVER (90 DAYS PRECEDING) > 0

PREDICT LIST_DISTINCT(transactions.article_id) OVER (30 DAYS FOLLOWING) RANK TOP 12
FOR EACH customers.customer_id

PREDICT SUM(usage.count) OVER (1 DAY FOLLOWING HORIZONS 28)
FOR EACH accounts.account_id

PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id IN (42, 123)

PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id = 42
ASSUMING users.plan = 'premium'

PREDICT LAST(loan.status) OVER (30 DAYS FOLLOWING) NOT LIKE '%DENIED' FOR EACH loan.id

PREDICT articles.description IS NULL FOR EACH articles.id

PREDICT movie.title STARTS WITH 'The' FOR EACH movie.id

PREDICT SUM(transactions.value) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING) > 100
FOR EACH customers.customer_id
WHERE customers.location NOT IN ('ALASKA', 'HAWAII')

-- A future-window forecast, anchored explicitly, returning quantiles:
PREDICT SUM(sales.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)
FOR EACH stores.store_id
AS OF :prediction_time
RETURN QUANTILES (0.10, 0.50, 0.90)

-- Reusable named frame combining two aggregations:
PREDICT SUM(orders.revenue) OVER w - SUM(orders.cost) OVER w
FOR EACH customers.customer_id
WINDOW w AS (30 DAYS FOLLOWING)
```

---

## The Python library

Full docs: [`python/README.md`](python/README.md). Requires Python 3.10+; core
depends only on numpy.

```bash
cd python
pip install -e "."    # extras: [rt] (native backend), [dev] (pytest)
```

### Quickstart: wire your own data

The library has no pandas adapter or bundled connectors. Declare the schema,
translate records into `Row` objects, and wire retrieval callbacks over the
storage your application owns:

```python
import pandas as pd  # application dependency, not a relationdb dependency
from relativedb import Engine, ExecutionInput, RetrieverWiring, Row

customer_rows = [Row("customers", r.customer_id, {"age": float(r.age)})
                 for r in customers.itertuples()]
order_rows = [Row("orders", r.order_id, {"qty": float(r.qty),
                  "order_date": r.order_date.to_pydatetime()},
                  timestamp=r.order_date.to_pydatetime(),
                  parents={"customer_id": r.customer_id})
              for r in orders.itertuples()]
# Build entity/link/scanner callbacks over these rows (or query your DAO).
wiring = RetrieverWiring.new_wiring()...build()
result = Engine(schema, wiring).execute(ExecutionInput(query=query, anchor_time=t0))
df = pd.DataFrame({"entity_id": [p.id for p in result.predictions],
                   "probability": [p.probability for p in result.predictions]})
```

An order dated after the anchor can never enter context — the engine re-checks
every row against the temporal bound even if a retriever misbehaves.

### Schema and retriever API

Explicit schema, and retrievers as plain callables over any storage:

```python
from relativedb import (Schema, TableDef, LinkDef, ValueType,
                        RetrieverWiring, Engine, ExecutionInput)

schema = (Schema.new_schema()
    .table(TableDef.new_table("customers")
        .column("age", ValueType.NUMBER)
        .column("signup_date", ValueType.DATETIME)
        .primary_key("customer_id").build())
    .table(TableDef.new_table("orders")
        .column("qty", ValueType.NUMBER)
        .column("order_date", ValueType.DATETIME)
        .primary_key("order_id").time_column("order_date").build())
    .link(LinkDef("orders", "customer_id", "customers"))
    .build())

wiring = (RetrieverWiring.new_wiring()
    .entities("customers", lambda table, ids, bound: customer_dao.by_ids(ids))
    .entities("orders",    lambda table, ids, bound: order_dao.by_ids(ids, bound))
    .default_links(lambda link, parent_id, bound, limit:
                   order_dao.recent_by_customer(parent_id, bound.as_of, limit))
    .build())

engine = Engine(schema, wiring)
result = engine.execute(ExecutionInput(
    query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR customers.customer_id = 'C7'",
    anchor_time=t0))
```

Parse and validate independently of execution:

```python
pq = relativedb.parse("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id")
pq.task_type()                    # TaskType.REGRESSION
relativedb.validate(pq, schema)   # binds names/types/windows against the schema
```

### Native RT-J backend

```python
backend = relativedb.RtNativeBackend(schema=schema)
result = Engine(schema, wiring, model_backend=backend).execute(
    ExecutionInput(query=query, anchor_time=t0))
```

Needs `pip install -e ".[rt]"` (sentence-transformers + huggingface_hub) and
the built C++ library — found via `RELATIVEDB_RT_LIB` or the sibling
`cpp/build/librt_c.dylib`; a clear `RtNativeUnavailableError` is raised
otherwise. Multiclass and ranking fall back to the history baseline (the C ABI
exposes a single score head).

### Testing

```bash
.venv/bin/python -m pytest
```

Covers the full RelQL corpus and rejections, the temporal-leakage guard,
CSC ≡ retriever context equivalence, model-URI routing, and explicit
retriever wiring end to end.

---

## The Java library

Full docs: [`java/README.md`](java/README.md). Requires Java 17+. Gradle
Maven publications under group `com.relativedb`:

- **`relationdb`** — schema builder, retriever SPI, ANTLR-based RelQL
  parser + semantic validation, context assembly (both sampler modes), model
  SPI.
- **`relationdb-rt`** — optional JNA binding to `librt_c`:
  `RtNativeBackend implements ModelBackend`.

```bash
cd java
./gradlew test
```

### Quickstart

The retriever SPI is **async** (`CompletionStage`) — retrievers can fan out to
remote services without blocking the engine:

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
    .modelBackend(myBackend)      // omit for the history baseline
    .build();

PredictionResult churn = engine.execute(ExecutionInput.newInput()
    .query("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id")
    .anchorTime(Instant.parse("2026-07-01T00:00:00Z"))
    .entityIds(List.of(42L))
    .build()).toCompletableFuture().join();
```

Sampler mode and context budgets are builder options
(`.samplerMode(SamplerMode.CSC)`, `ContextPolicy` with `maxContextCells` +
`bfsWidth` or `fanouts(64, 64)`).

`Pql.parse(query)` returns the typed AST; `Pql.validate(query, schema)` binds
it and infers the `TaskType`.

### Native model backend

```java
TextEncoder encoder = new PrecomputedEncoder(embeddingTable); // string -> float[384]
try (RtNativeBackend backend = new RtNativeBackend(ModelConfig.defaults(), encoder)) {
    RelativeDbEngine engine = RelativeDbEngine.newEngine(schema, wiring)
        .modelBackend(backend).build();
}
```

Library discovery is lazy: system property `relativedb.rt.lib` → env
`RELATIVEDB_RT_LIB` → the sibling `cpp/build/librt_c.dylib` → the loader path.
The golden-forward test replays `cpp/testdata/*.bin` through the binding and
matches the PyTorch-verified scores (auto-skipped when the dylib/checkpoints
are absent).

---

## The Rust library

Full docs: [`rust/README.md`](rust/README.md). Cargo workspace with the
`relativedb` crate (edition 2021; depends on `chrono` + `libloading` only).

```bash
cd rust
cargo test
```

### Quickstart

Closures implement the retriever traits directly:

```rust
use relativedb::{
    Engine, EntityId, ExecutionInput, LinkDef, RetrieverWiring, Row, Schema,
    TableDef, TemporalBound, ValueType,
};

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
    .entities("customers", move |_t: &str, ids: &[EntityId], _b: &TemporalBound| {
        customers.iter().filter(|r| ids.contains(&r.id)).cloned().collect()
    })
    .default_links(move |link: &LinkDef, pid: &EntityId, b: &TemporalBound, limit: usize| {
        let mut kids: Vec<Row> = orders.iter()
            .filter(|r| r.get_parent(&link.fk_column) == Some(pid) && b.admits_row(r))
            .cloned().collect();
        kids.sort_by(|a, x| x.timestamp.cmp(&a.timestamp));
        kids.truncate(limit);
        kids
    })
    .scanner("customers", move |_t: &str, _b: &TemporalBound| cust_scan.clone())
    .build();

let mut engine = Engine::new(schema, wiring);
let result = engine.execute(
    ExecutionInput::query(
        "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id")
    .anchor_time(anchor),
)?;

assert_eq!(result.task_type, relativedb::TaskType::BinaryClassification);
for p in &result.predictions {
    println!("{} churn probability = {:?}", p.id, p.probability);
}
```

**Design decision — synchronous SPI.** Where Java's SPI is async, the Rust
(and Python) SPI is synchronous and infallible: traits return plain `Vec`s. An
async SPI would force a runtime choice on every user and color the whole
engine `async` for no benefit in the reference paths; batching retrievers are
free to run their own I/O concurrency internally.

The `native` module binds `librt_c` via `libloading` (`RtNativeBackend`),
discovered from `RELATIVEDB_RT_LIB` or the sibling
`cpp/build/librt_c.{dylib,so}`. The golden gate:

```bash
RELATIVEDB_RT_LIB=../cpp/build/librt_c.dylib \
  cargo test --test golden_tests -- --nocapture
```

---

## rt.cpp — the native inference engine

Full docs: [`cpp/README.md`](cpp/README.md).

A dependency-light C++20 implementation of the RT-J forward pass — ~700 lines,
no torch, no Python at inference. It faithfully ports the reference model: 12
blocks of column/feature/neighbor **masked attention** (structure carried by
masks, no positional encodings) with SwiGLU FFNs, per-head QK-RMSNorm,
log(kv-count) query scaling, sigmoid output gating, per-sem-type value
encoders, and a built-in safetensors (bf16 → fp32) loader. Optimized with
llama.cpp/vllm idioms on Apple Accelerate: stacked-QKV GEMM panels, grouped
masked attention that never materializes S×S, persistent thread pool,
zero allocation inside the block loop.

```bash
cd cpp
cmake -B build -S . && cmake --build build -j
./build/rt_test testdata <path>/classification/model.safetensors   # golden gate
./build/rt_bench                                                   # benchmark
```

Builds the `rt` static library, the **`librt_c` shared library** (the C ABI in
`src/rt_c.h` that all three language bindings load), `rt_test`, and
`rt_bench`. The golden test replays a batch dumped from the PyTorch reference
and matches final scores to ~3–4 decimals (fp32 op-ordering drift only).

---

## Runnable industry examples

[`examples/industry/`](examples/industry/) — each example generates synthetic
data with a **planted signal**, runs a real RelQL query through the full
pipeline, and **asserts** the predictions recover the signal:

| Example | Industry | Pattern |
|---|---|---|
| `growth_churn.py` | Subscription / streaming | binary churn with an activity `WHERE` filter |
| `fraud_chargeback.py` | Payments | rare-event risk scoring (all 8 planted abusers recovered) |
| `bizops_demand_forecast.py` | Retail | multi-horizon `OVER (... HORIZONS N)` |
| `pzn_buy_it_again.py` | Grocery / personalization | `LIST_DISTINCT … RANK TOP K` ranking |

```bash
cd examples/industry
../../python/.venv/bin/python growth_churn.py
```

A Java counterpart of the churn example lives at
[`java/relativedb-core/src/test/java/com/relativedb/GrowthChurnExampleTest.java`](java/relativedb-core/src/test/java/com/relativedb/GrowthChurnExampleTest.java).

---

## Configuration reference

| Variable | Applies to | Meaning |
|---|---|---|
| `RELATIVEDB_RT_LIB` | Python, Java, Rust | Path to the built `librt_c` shared library (otherwise the sibling `cpp/build/` is probed) |
| `relativedb.rt.lib` (system property) | Java | Same, takes precedence over the env var |
| `RELATIVEDB_RT_HF_CACHE` / `relativedb.rt.hf.cache` | Java | Override the local Hugging Face cache root used to resolve `hf://` checkpoint URIs |

Checkpoint URIs accept `hf://org/repo/subdir` (resolved from the local HF
cache only — nothing downloads implicitly), `file://`, and plain paths.

---

## Design invariants

Recurring invariant numbers you will see referenced in code and docs:

- **F17** — PK/FK columns are graph edges, never feature cells. The schema
  builder rejects ID-typed feature columns; `Row` has no way to carry an ID as
  a value.
- **F24** — the engine re-validates every retriever-returned row against the
  `TemporalBound` and drops violations; temporal safety does not depend on
  retriever correctness.
- **F65** — "self labels": the entity's own past target outcomes are computed
  over trailing windows and included as in-context examples (and power the
  model-free baseline).
- **F13/F14** — text cells and `"<column> of <table>"` schema phrases embed
  with the pinned MiniLM encoder.
- **F52** — booleans route through the number head (`bool_as_num`).

---

## Repository layout

```
relativedb/
├── python/                 # Python library (pip package `relativedb`)
│   ├── src/relativedb/     #   schema, retrieve, pql/, engine, csc, model,
│   │                       #   engine, retriever SPI, rt_native
│   └── tests/
├── java/                   # Java library (Gradle, group com.relativedb)
│   ├── relativedb-core/    #   engine + ANTLR grammar (src/main/antlr/Pql.g4)
│   └── relativedb-rt/      #   JNA binding to librt_c
├── rust/                   # Cargo workspace
│   └── relativedb/         #   the `relativedb` crate (+ shared RelQL corpus in tests/data/)
├── cpp/                    # rt.cpp — native RT-J inference (librt_c)
└── examples/
    └── industry/           # self-checking end-to-end scenarios
```

Test entry points: `./gradlew test` (java/), `python -m pytest` (python/),
`cargo test` (rust/), `./build/rt_test` (cpp/).

---

## License

Apache-2.0 (declared by each package manifest).
