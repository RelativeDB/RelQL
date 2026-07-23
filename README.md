<p align="center">
  <img src="website/static/img/logo.svg" alt="relativedb logo" width="120" />
</p>

# What is RelativeDB/RelQL?
RelativeDB is an optimized implementation of Relational Transformers, exposed through RelQL, a query language for predicting what happens next:

```sql
PREDICT NOT EXISTS(orders.*)
FROM customers
```

*"For every customer, what is the probability they don't place an order"*.

RelativeDB works best with many tables, 10-100 tables, without needing any feature engineering. Subgraphs are discovered automatically, though you can ablate them to find what features really matter. Since RelativeDB uses a pretrained model, it will work in environments with very little data.

## The Model
RelativeDB is based on these papers:

| Resource | Description | Date      |
| --- | --- |-----------|
| [stanford-star/relational-transformer](https://github.com/stanford-star/relational-transformer) | RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions — code, in progress | Jul 2026  |
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | Paper (arXiv:2510.06377) | Oct, 2025 |

# Appetizer

```sql
# Auto-label a GitHub issue: predict its label from title, body, and history.
PREDICT issues.label
WHERE issues.label IS NULL

# Would customer 42 churn if we moved them to the premium plan?
PREDICT NOT EXISTS(orders.*)
FROM customers c
WHERE c.customer_id = 42
ASSUMING c.plan = 'premium'

# Expected spend per customer over the next quarter.
PREDICT SUM(transactions.price) OVER (90 DAYS FOLLOWING)
FROM customers

# The 12 articles each customer is most likely to buy next.
PREDICT ARRAY_AGG(transactions.article_id) OVER (30 DAYS FOLLOWING RANK TOP 12)
FROM customers

# Units sold per store, one value for each of the next 4 weeks.
PREDICT SUM(sales.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)
FROM stores

# Will spend in the 15–45 day window exceed $100?
PREDICT SUM(transactions.value) OVER w > 100
FROM customers c
WHERE c.location NOT IN ('ALASKA', 'HAWAII')
WINDOW w AS (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING)

# Predicted gross margin per customer.
PREDICT SUM(orders.revenue) OVER w - SUM(orders.cost) OVER w
FROM customers
WINDOW w AS (30 DAYS FOLLOWING)
```

# Docs

Read the [RelQL book](https://relql.com/docs/).

# Checkpoints

| Checkpoint | On-disk | Latency | Throughput | Accuracy | Download |
| --- | --- | --- | --- | --- | --- |
| fp32 | 342 MB | 317 ms | 6.5k tok/s | reference | — |
| fp16 | 172 MB | 483 ms | 4.2k tok/s | identical | [rt-j-fp16](https://huggingface.co/RelativeDB/rt-j-fp16) |
| int8 | 88 MB | 453 ms | 4.5k tok/s | ±0.01 | [rt-j-int8](https://huggingface.co/RelativeDB/rt-j-int8) |
| int4 | 64 MB | 464 ms | 4.4k tok/s | ±0.15 | [rt-j-int4](https://huggingface.co/RelativeDB/rt-j-int4) |

# Benchmark snapshot

Tested two Formula 1 predictions: whether a driver finishes in the top three,
and the driver's finishing position:

| Task metric and split | Reference RT-J | XGBoost | RelativeDB zero-shot | RelativeDB fine-tuned |
|---|---:|---:|---:|---:|
| Top-three finish, test AUROC ↑ (128 cells) | **0.711160** | 0.681974 | 0.710650 | — |
| Top-three finish, test AUROC ↑ (8,192 cells) | MPS failed | 0.892310 | **0.913213** | — |
| Top-three finish, validation AUROC ↑ (8,192 cells) | — | — | 0.872373 | **0.889627** |

# The Python library

```bash
pip install relativedb
```

<details>
<summary><b>Quickstart: 90-day churn, copy-paste runnable</b></summary>

Retrievers are plain callables over your own storage — here an in-memory
dict stands in for your database. The first run downloads the pretrained
checkpoint and text encoder from Hugging Face.

```python
from datetime import datetime, timezone
from relativedb import (Schema, TableDef, LinkDef, ValueType, Row,
                        RetrieverWiring, Engine, ExecutionInput,
                        RtNativeBackend)

dt = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

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

ROWS = {
    "customers": [
        Row("customers", "C1", {"age": 34.0, "signup_date": dt("2026-02-10")}),
        Row("customers", "C7", {"age": 52.0, "signup_date": dt("2026-01-20")}),
        Row("customers", "C9", {"age": 27.0, "signup_date": dt("2026-03-05")}),
    ],
    "orders": [
        Row("orders", "O1", {"qty": 1.0, "order_date": dt("2026-03-10")},
            timestamp=dt("2026-03-10"), parents={"customer_id": "C7"}),
        Row("orders", "O2", {"qty": 2.0, "order_date": dt("2026-05-02")},
            timestamp=dt("2026-05-02"), parents={"customer_id": "C7"}),
        Row("orders", "O3", {"qty": 1.0, "order_date": dt("2026-06-20")},
            timestamp=dt("2026-06-20"), parents={"customer_id": "C1"}),
    ],
}
BY_ID = {t: {r.id: r for r in rs} for t, rs in ROWS.items()}

def entity(table, ids, bound):
    rows = (BY_ID[table].get(i) for i in ids)
    return [r for r in rows if r is not None and bound.admits_row(r)]

def links(link, parent_id, bound, limit):
    kids = [r for r in ROWS[link.from_table]
            if r.parents.get(link.fk_column) == parent_id
            and bound.admits_row(r)]
    kids.sort(key=lambda r: (r.timestamp is None,
                             -(r.timestamp.timestamp() if r.timestamp else 0)))
    return kids[:limit]

def make_scanner(table):
    def scan(t, bound):
        return (r for r in ROWS[table] if bound.admits_row(r))
    return scan

wiring = RetrieverWiring.new_wiring().default_links(links)
for t in ROWS:
    wiring.entities(t, entity)
    wiring.scanner(t, make_scanner(t))
wiring = wiring.build()

engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema))
result = engine.execute(ExecutionInput(
    query="PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers",
    anchor_time=dt("2026-07-01")))
for p in result.predictions:
    print(f"customer {p.id}: P(churn in 90 days) = {p.probability:.3f}")
```

</details>

# Notes
- The multi-label head requires tuning (e.g. RANK TOP 10)
