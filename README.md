<p align="center">
  <img src="website/static/img/logo.svg" alt="relativedb logo" width="120" />
</p>

# What is RelativeDB/RelQL?
RelativeDB predicts over the shape of your data.

Given a customer, account, transaction, or any other entity, it follows the relationships across your tables and builds a local graph around it. Orders stay orders. Returns stay returns. Support calls, refunds, text, and account history keep their meaning and their connections.

That graph is passed directly to a relational model, so adding more tables gives it more to work with instead of creating another feature-engineering project.

RelativeDB is an optimized implementation of Relational Transformers, exposed through RelQL, a query language for predicting what happens next.

```sql
PREDICT NOT EXISTS(orders.*)
FROM customers
```

*"For every customer, what is the probability they don't place an order"*.

RelativeDB works best for large graphs, 10-100 tables. Unlike xgboost, it works well with text since all text values get encoded into latent space with the rest of the features.

## The model

RelativeDB runs the current RT-J checkpoint family, an approximately 86-million-parameter relational model. It can make zero-shot classification and regression predictions over a new database without fitting a task-specific model first.

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

### Fine-tuning

RelativeDB can fine-tune the complete RT-J model for a binary classification or regression task on Apple Silicon. Training runs in C++ with Metal/MPS primitives and does not use Torch. It updates the transformer, input encoders, normalization parameters, and output decoder together.

Fine-tuning uses the reference sampler and keeps training, validation, and test data separate. The released zero-shot model is the starting checkpoint and the validation score is the promotion gate. A checkpoint is kept only when it improves on the best validated model. If validation gets worse, the trainer restores the best checkpoint, lowers the learning rate, and continues. Model weights and AdamW state are saved together so a stopped run can resume at a validation boundary.

The physical batch is allowed to shrink for the available memory, but gradient accumulation keeps the requested effective batch unchanged. An 8,192-cell context is much more expensive than the reference script's usual 1,024-cell fine-tuning context, so long runs on an M3 can take hours.

See [evaluation/README.md](evaluation/README.md#native-full-checkpoint-training) for the training command and recovery workflow. Frozen-head fitting is a separate adapter feature for multiclass and ranking tasks; it is not reported as full-model fine-tuning.

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
