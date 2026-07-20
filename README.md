<p align="center">
  <img src="website/static/img/logo.svg" alt="relativedb logo" width="120" />
</p>

# What is RelativeDB/RelQL?
We have dozens to hundreds of tables of data in our ecosystem. Lets say we want to predict customer churn? We could spend a great deal of time looking for all of the features we need (recent transactions, complaint calls, refunds issued, etc). Or we can have a graph search across our tables build a small subgraph, and use a pretrained model that is trained to figure it out. This is the real advantage of relativedb, we look through your whole graph of data to find what is interesting, and use a zero-shot model to make predictions. 

So this scales with more data. Meaning you can fine tune all of your data, and any other data you get your hands on, to improve the model.

Try not to look at what the model does now, look at what it could do in the future. This technique can produce comparable results to things like xgboost, but isn't trained on things like multiclass identification so the results are junk without fine tuning.

RelativeDB is an optimized implementation of Relational Transformers (2026), surfaced as RelQL, a predictive query language for relational data. You declare the shape of your
relational data (tables, keys, links), wire small retriever callbacks over
whatever storage you already have, and ask questions about the **future**:

```sql
PREDICT NOT EXISTS(orders.*)
FROM customers
```

*"For every customer, what is the probability they don't place an order"*.

RelativeDB works best for large graphs, 10-100 tables. Unlike xgboost, it works well with text since all text values get encoded into latent space with the rest of the features.

## The model
Relational Transformers work by utilizing a pretrained 22m parameter model from over 650 databases of relational data for prediction and classifications tasks. This method has been shown to scale, and remarkably, shows the emergence of zero-shot ability on novel tasks.

| Resource | Description | Date      |
| --- | --- |-----------|
| [stanford-star/relational-transformer](https://github.com/stanford-star/relational-transformer) | RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions — code, in progress | Jul 2026  |
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | Paper (arXiv:2510.06377) | Oct, 2025 |

# Appetizer

```sql
# Auto-label a GitHub issue: predict its label from title, body, and history.
PREDICT label
FROM issues
WHERE label IS NULL

# or simply
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

### Checkpoints

| Checkpoint | On-disk | Latency | Throughput | Accuracy | Download |
| --- | --- | --- | --- | --- | --- |
| fp32 | 171 MB | 317 ms | 6.5k tok/s | reference | — |
| fp16 | 172 MB | 483 ms | 4.2k tok/s | identical | [rt-j-fp16](https://huggingface.co/RelativeDB/rt-j-fp16) |
| int8 | 88 MB | 453 ms | 4.5k tok/s | ±0.01 | [rt-j-int8](https://huggingface.co/RelativeDB/rt-j-int8) |
| int4 | 64 MB | 464 ms | 4.4k tok/s | ±0.15 | [rt-j-int4](https://huggingface.co/RelativeDB/rt-j-int4) |

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
    query="PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers "
          "WHERE customers.customer_id IN :ids",
    params={"ids": ["C7"]},   # the cohort; drop the WHERE to score every customer
    anchor_time=t0))
```

</details>
