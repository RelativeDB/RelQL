<p align="center">
  <img src="website/static/img/logo.svg" alt="relativedb logo" width="120" />
</p>

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
Relational Transformers work by pretraining a 22m parameter model on relational data for prediction and classifications tasks. This method has been shown to scale, and remarkably, shows the emergence of zero-shot ability on novel tasks.

| Resource | Description | Date      |
| --- | --- |-----------|
| [stanford-star/relational-transformer](https://github.com/stanford-star/relational-transformer) | RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions — code, in progress | Jul 2026  |
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | Paper (arXiv:2510.06377) | Oct, 2025 |

### Checkpoints

int8/int4 run low-precision matmuls — int8×int8 integer dot products on CPU,
packed weights streamed straight into the GPU kernels — so the weights are
never expanded to fp32. Pick by size vs. accuracy:

| Checkpoint | On-disk | Latency | Throughput | Accuracy | Download |
| --- | --- | --- | --- | --- | --- |
| fp32 | 171 MB | 317 ms | 6.5k tok/s | reference | — |
| int8 | 88 MB | 453 ms | 4.5k tok/s | ±0.01 | [rt-j-int8](https://huggingface.co/RelativeDB/rt-j-int8) |
| int4 | 64 MB | 464 ms | 4.4k tok/s | ±0.15 | [rt-j-int4](https://huggingface.co/RelativeDB/rt-j-int4) |
| fp16 | 172 MB | 483 ms | 4.2k tok/s | identical | [rt-j-fp16](https://huggingface.co/RelativeDB/rt-j-fp16) |

<sub>Apple M3 Pro, Metal/MPS, single entity at 2048-token context. Latency =
ms/forward; throughput = tokens/s. Accuracy = target-score deviation vs. the
fp32 golden batch (sign and ranking preserved for every format). fp32 leads on
GPU because it uses Apple's tuned `MPSMatrixMultiplication`; on CPU the formats
are within ~5%. Full sweep:
[`cpp/README.md`](cpp/README.md#benchmarks-rt_bench-apple-m3-pro).</sub>

# Docs

Read the [RelQL book](https://relql.com/docs/).

# Appetizer

```sql
# Auto-label a GitHub issue: predict its label from title, body, and history.
PREDICT issues.label FOR EACH issues.id
WHERE issues.label IS NULL

-- Would customer 42 churn if we moved them to the premium plan?
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) 
FOR EACH customers.customer_id
WHERE customers.customer_id = 42
ASSUMING customers.plan = 'premium'

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

# Predicted gross margin per customer.
PREDICT SUM(orders.revenue) OVER w - SUM(orders.cost) OVER w
FOR EACH customers.customer_id
WINDOW w AS (30 DAYS FOLLOWING)
```
---

## The Python library

```bash
pip install relativedb
```

<details>
<summary><b>Quickstart: 90-day churn from your own DataFrames</b></summary>

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

</details>

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

<details>
<summary><b>Quickstart</b></summary>

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

</details>

## The Rust library

```bash
cargo add relativedb
```

```toml
# Cargo.toml
[dependencies]
relativedb = "0.1.0"
```

<details>
<summary><b>Quickstart</b></summary>

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

</details>

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
