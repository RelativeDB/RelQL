# What is RelativeDB/RelQL?
RelativeDB is an optimized implementation of Relational Transformers (2026), surfaced as RelQL, a predictive query language for relational data. You declare the shape of your
relational data (tables, keys, links), wire small retriever callbacks over
whatever storage you already have, and ask questions about the **future**:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FOR EACH customers.customer_id
```

*"For every customer, what is the probability they place zero orders in the
next 90 days?"*.

## The model
The work is based on these papers:

| Resource | Description | Date      |
| --- | --- |-----------|
| [stanford-star/relational-transformer](https://github.com/stanford-star/relational-transformer) | RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions — code, in progress | Jul 2026  |
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | Paper (arXiv:2510.06377) | Oct, 2025 |

# Docs

Read the [RelQL book](https://relql.com/docs/).

# Appetizer

```sql
# Per active customer, probability of zero orders in the next 90 days.
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FOR EACH customers.customer_id
WHERE EXISTS(orders.*) OVER (90 DAYS PRECEDING)

# Expected spend per customer over the next quarter.
PREDICT SUM(transactions.price) OVER (90 DAYS FOLLOWING)
FOR EACH customers.customer_id

# The 12 articles each customer is most likely to buy next.
PREDICT LIST_DISTINCT(transactions.article_id) OVER (30 DAYS FOLLOWING)
RANK TOP 12
FOR EACH customers.customer_id

# Units sold per store, one value for each of the next 4 weeks.
PREDICT SUM(sales.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)
FOR EACH stores.store_id

# Will spend in the 15–45 day window exceed $100?
PREDICT SUM(transactions.value) OVER w > 100
FOR EACH customers.customer_id
WHERE customers.location NOT IN ('ALASKA', 'HAWAII')
WINDOW w AS (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING)

# Auto-label a GitHub issue: predict its label from title, body, and history.
PREDICT issues.label FOR EACH issues.id
WHERE issues.label IS NULL

-- What-if: would customer 42 churn if we moved them to the premium plan?
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) 
FOR EACH customers.customer_id
WHERE customers.customer_id = 42
ASSUMING customers.plan = 'premium'

-- Anchored forecast with uncertainty: units sold per store, weekly, as of a
-- given time, returning 3 quantiles per horizon instead of a point estimate.
PREDICT SUM(sales.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)
FOR EACH stores.store_id
AS OF :prediction_time
RETURN QUANTILES (0.10, 0.50, 0.90)

-- Named window reused across two aggregations: predicted gross margin per customer.
PREDICT SUM(orders.revenue) OVER w - SUM(orders.cost) OVER w
FOR EACH customers.customer_id
WINDOW w AS (30 DAYS FOLLOWING)
```

---

## The Python library

```bash
pip install relativedb
```

## Quickstart: 90-day churn from your own DataFrames

The "will customer C7 churn?" scenario: three linked tables, prediction time
t0 = 2026-07-01.

```python
import pandas as pd
import relativedb

customers = pd.DataFrame({
    "customer_id": ["C1", "C7", "C9"],
    "age": [34, 52, 27],
    "signup_date": pd.to_datetime(["2026-02-10", "2026-01-20", "2026-03-05"]),
})
products = pd.DataFrame({
    "product_id": ["P1", "P2", "P3"],
    "price": [25.0, 90.0, 35.0],
    "name": ["running shoes", "espresso machine", "yoga mat"],
})
orders = pd.DataFrame({
    "order_id": ["O1", "O2", "O3", "O4"],
    "customer_id": ["C7", "C7", "C1", "C7"],
    "product_id": ["P2", "P1", "P3", "P3"],
    "qty": [1, 2, 1, 1],
    "order_date": pd.to_datetime(
        ["2026-03-10", "2026-05-02", "2026-06-20", "2026-07-05"]),
})

schema = (relativedb.Schema.new_schema()
    .table(relativedb.TableDef.new_table("customers")
        .column("age", relativedb.ValueType.NUMBER)
        .column("signup_date", relativedb.ValueType.DATETIME)
        .primary_key("customer_id").build())
    .table(relativedb.TableDef.new_table("orders")
        .column("qty", relativedb.ValueType.NUMBER)
        .column("order_date", relativedb.ValueType.DATETIME)
        .primary_key("order_id").time_column("order_date").build())
    .link(relativedb.LinkDef("orders", "customer_id", "customers")).build())

# Your connector translates DataFrame records into RelativeDB.Row objects.
# See examples/industry/pandas_connector.py for a complete implementation.
wiring = wire_my_dataframes(schema, {"customers": customers, "orders": orders})
# Scoring requires a model backend. The RT-J relational foundation model
# (RtNativeBackend) needs librt_c and a cached stanford-star/rt-j checkpoint.
engine = relativedb.Engine(schema, wiring,
    model_backend=relativedb.RtNativeBackend(schema=schema))
result = engine.execute(relativedb.ExecutionInput(
    query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id",
    anchor_time=pd.Timestamp("2026-07-01").to_pydatetime()))
df = pd.DataFrame({"entity_id": [p.id for p in result.predictions],
                   "probability": [p.probability for p in result.predictions]})
```


```bash
cd python
pip install -e "."    # extras: [rt] (native backend), [dev] (pytest)
```

### Quickstart: wire your own data

The library has no pandas adapter or bundled connectors. Declare the schema,
translate records into `Row` objects, and wire retrieval callbacks over the
storage your application owns:

```python
import pandas as pd  # application dependency, not a RelativeDB dependency
from relativedb import Engine, ExecutionInput, RetrieverWiring, Row, RtNativeBackend

customer_rows = [Row("customers", r.customer_id, {"age": float(r.age)})
                 for r in customers.itertuples()]
order_rows = [Row("orders", r.order_id, {"qty": float(r.qty),
                  "order_date": r.order_date.to_pydatetime()},
                  timestamp=r.order_date.to_pydatetime(),
                  parents={"customer_id": r.customer_id})
              for r in orders.itertuples()]
# Build entity/link/scanner callbacks over these rows (or query your DAO).
wiring = RetrieverWiring.new_wiring()...build()
# A model backend is required; RtNativeBackend runs the RT-J relational model.
engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema))
result = engine.execute(ExecutionInput(query=query, anchor_time=t0))
df = pd.DataFrame({"entity_id": [p.id for p in result.predictions],
                   "probability": [p.probability for p in result.predictions]})
```

An order dated after the anchor can never enter context — the engine re-checks
every row against the temporal bound even if a retriever misbehaves.

### Schema and retriever API

Explicit schema, and retrievers as plain callables over any storage:

```python
from relativedb import (Schema, TableDef, LinkDef, ValueType,
                        RetrieverWiring, Engine, ExecutionInput, RtNativeBackend)

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

engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema))
result = engine.execute(ExecutionInput(
    query="PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FOR EACH customers.customer_id",
    entity_ids=["C7"],   # score just this cohort; omit to score the whole table
    anchor_time=t0))
```

Parse and validate independently of execution:

```python
pq = relativedb.parse("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id")
pq.task_type()                    # TaskType.REGRESSION
relativedb.validate(pq, schema)   # binds names/types/windows against the schema
```

## The Java library

```xml
<!-- Maven -->
<dependency>
  <groupId>com.relativedb</groupId>
  <artifactId>relativedb</artifactId>
  <version>0.1.0</version>
</dependency>
```

```groovy
// Gradle
implementation("com.relativedb:relativedb:0.1.0")
```

### Quickstart

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
    .modelBackend(new RtNativeBackend(schema))   // required; the RT-J relational model
    .build();

PredictionResult churn = engine.execute(ExecutionInput.newInput()
    .query("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FOR EACH customers.customer_id")
    .anchorTime(Instant.parse("2026-07-01T00:00:00Z"))
    .entityIds(List.of(42L))
    .build()).toCompletableFuture().join();
```

## The Rust library

```bash
cargo add relativedb
```

```toml
# Cargo.toml
[dependencies]
relativedb = "0.1.0"
```

### Quickstart

```rust
use relativedb::{
    Engine, EntityId, ExecutionInput, LinkDef, RetrieverWiring, Row, RtNativeBackend,
    Schema, TableDef, TemporalBound, ValueType,
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

// Scoring requires a model backend; RtNativeBackend runs the RT-J relational model.
let mut engine = Engine::new(schema, wiring).model_backend(RtNativeBackend::new(&schema)?);
let result = engine.execute(
    ExecutionInput::query(
        "PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FOR EACH customers.customer_id")
    .anchor_time(anchor),
)?;

assert_eq!(result.task_type, relativedb::TaskType::BinaryClassification);
for p in &result.predictions {
    println!("{} churn probability = {:?}", p.id, p.probability);
}
```

## Performance — CPU vs Metal (MPS)

The native engine (`librt_c`) runs on the **CPU** (Apple Accelerate / AMX) and,
on Apple Silicon, on the **GPU** via a Metal/MPS backend (`MPSMatrixMultiplication`
projections + custom Metal attention/FFN kernels). Both produce **numerically
identical** output (`max|Δ| = 0`) and pass the batch-isolation check. Select with
`rt_bench --device cpu|mps` (or `RT_DEVICE` in the C ABI).

Measured on an Apple Silicon laptop, RT-J classification checkpoint, fp32:

**Batch scaling** (context S=16, the per-entity scoring path) — MPS parallelizes
across the batch and saturates ~3.2–3.5× CPU throughput:

| Batch | CPU ms/fwd | MPS ms/fwd | MPS speedup | CPU ms/entity | MPS ms/entity |
|---:|---:|---:|:---:|---:|---:|
| 1    | 16.4   | 7.2   | **2.3×** | 16.4 | 7.2 |
| 20   | 64.2   | 19.7  | **3.3×** | 3.2  | 1.0 |
| 80   | 220.7  | 62.6  | **3.5×** | 2.8  | 0.8 |
| 160  | 431.4  | 122.6 | **3.5×** | 2.7  | 0.8 |
| 640  | 1581.6 | 485.8 | **3.3×** | 2.5  | 0.8 |
| 1280 | 3138.3 | 982.0 | **3.2×** | 2.5  | 0.8 |

**Context length** (single sequence, B=1) — RT has no positional encodings and no
fixed context cap; the reference runs context up to 8192. MPS's edge shrinks as `S`
grows; **beyond S ≈ 1–2k the two land within measurement noise** and trade places
run-to-run (only a few iterations at these sizes) — no consistent winner for a lone
long sequence:

| B × S | CPU ms/fwd | MPS ms/fwd | faster |
|---|---:|---:|:---:|
| 1 × 256  | 58.6   | 41.8   | **MPS 1.4×** |
| 1 × 1024 | 197.2  | 158.0  | **MPS 1.25×** |
| 1 × 2048 | 387.8  | 405.9  | CPU 1.05× (~tie) |
| 1 × 4096 | 834.8  | 1031.6 | **CPU 1.24×** |
| 1 × 8192 | 2320.0 | 2189.9 | MPS 1.06× (~tie) |

**Batched** — MPS's clear win lives in the short-context, high-batch regime;
batching restores its lead only while sequences stay moderate:

| B × S | CPU ms/fwd | MPS ms/fwd | MPS speedup |
|---|---:|---:|:---:|
| 16 × 256  | 747.6  | 270.7  | **2.8×** |
| 8 × 1024  | 1389.6 | 625.7  | **2.2×** |
| 32 × 1024 | 5233.9 | 2795.0 | **1.9×** |
| 8 × 4096  | 6104.9 | 4928.4 | **1.2×** |
| 4 × 8192  | 7365.3 | 8024.0 | 0.92× (CPU) |

Peak throughput: MPS ~21,000 tok/s (short seq) vs CPU ~6,500 tok/s — but both fall
to ~3,500–4,000 tok/s at S=8192 (attention-bound), where they converge. MPS uses
less memory at the largest shapes. **Rule of thumb:** MPS for short-context batched
scoring (2.3–3.5×); at long single sequences the two are roughly on par. Reproduce
with `cpp/build/rt_bench <testdata> <ckpt> --device {cpu,mps}` (a few % run-to-run
variance; raise iteration counts for the long-context rows).

## Design invariants

Recurring invariant numbers you will see referenced in code and docs:

- **F17** — PK/FK columns are graph edges, never feature cells. The schema
  builder rejects ID-typed feature columns; `Row` has no way to carry an ID as
  a value.
- **F24** — the engine re-validates every retriever-returned row against the
  `TemporalBound` and drops violations; temporal safety does not depend on
  retriever correctness.
- **F65** — "self labels": the entity's own past target outcomes are computed
  over trailing windows and included as in-context examples for the model.
- **F13/F14** — text cells and `"<column> of <table>"` schema phrases embed
  with the pinned MiniLM encoder.
- **F52** — booleans route through the number head (`bool_as_num`).

## License

Apache-2.0 (declared by each package manifest).
