<p align="center">
  <img src="https://raw.githubusercontent.com/RelativeDB/RelQL/main/website/static/img/logo.svg" alt="relativedb logo" width="120" />
</p>

# RelativeDB / RelQL

RelativeDB is an optimized implementation of Relational Transformers, exposed
through RelQL, a query language for predicting what happens next:

```sql
PREDICT NOT EXISTS(orders.*)
FROM customers
```

*"For every customer, what is the probability they don't place an order."*

RelativeDB works best with many tables (10–100) and needs no feature
engineering. Subgraphs are discovered automatically, though you can ablate
them to find what features really matter. Because it uses a pretrained model,
it works in environments with very little data.

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
```

## Install

```bash
pip install relativedb                 # pure Python: parse, plan, assemble
pip install relativedb[engine]         # + the local native engine (librt_c)
```

Python 3.10 or newer. The base package is pure Python (numpy only): RelQL
parsing, planning, and context creation all run client-side, and the model is
reached through a **scoring backend** — either a cloud backend URL or the
optional in-process engine:

```python
engine = Engine(schema, wiring, model_backend="https://scoring.example.com")
# or, with relativedb-engine installed:
from relativedb_engine import RtNativeBackend
engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema))  # relativedb-engine
```

`relativedb-engine` wheels bundle `librt_c` (the C++ RT-J engine with its
native MiniLM text encoder — no torch, no Python embedding) for macOS
(universal2, 13.0+; Accelerate and Metal) and manylinux x86_64 / aarch64.
Windows is not supported. On any other platform build `cpp/` with CMake and
point `RELATIVEDB_RT_LIB` at the built `librt_c`. The cloud backend is the
same engine behind HTTP: `cpp/build/rt_serve --port 8500`.

## Quickstart: 90-day churn from your own DataFrames

A sketch — `customer_dao`, `order_dao` and `t0` stand in for your storage and
your anchor time. A copy-paste runnable version with an in-memory database is
in the [repository README](https://github.com/RelativeDB/RelQL#the-python-library).

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

engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema))  # relativedb-engine
result = engine.execute(ExecutionInput(
    query="PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers "
          "WHERE customers.customer_id IN :ids",
    params={"ids": ["C7"]},   # the cohort; drop the WHERE to score every customer
    anchor_time=t0))
```

## Checkpoints

Model checkpoints resolve through the Hugging Face cache on first use. Set
`RELATIVEDB_RT_QUANTIZED` to `f16`, `q8`, or `q4` to trade footprint for
precision:

| Checkpoint | On-disk | Accuracy | Download |
| --- | --- | --- | --- |
| fp32 | 342 MB | reference | — |
| fp16 | 172 MB | identical | [rt-j-fp16](https://huggingface.co/RelativeDB/rt-j-fp16) |
| int8 | 88 MB | ±0.01 | [rt-j-int8](https://huggingface.co/RelativeDB/rt-j-int8) |
| int4 | 64 MB | ±0.15 | [rt-j-int4](https://huggingface.co/RelativeDB/rt-j-int4) |

## The model

RelativeDB is based on:

- [stanford-star/relational-transformer](https://github.com/stanford-star/relational-transformer) — RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions
- [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) (arXiv:2510.06377)

## Development

```bash
pip install -e ".[dev]"

pytest -m "not integration"   # unit tier: no checkpoint, no network (<1s)
pytest -m integration         # native kernels + the real rt-j checkpoint
pytest                        # everything
```

Both tiers run from this directory or from the repository root. The unit
tier is pure Python — no native library, no checkpoint, no network. The
integration tier lives with the engine package (`python-engine/tests`) and
resolves `hf://RelativeDB/rt-j-fp16/…` through the Hugging Face cache (~326 MB
fp32, plus ~128 MB for the pinned MiniLM text encoder).

Set `RELATIVEDB_REQUIRE_NATIVE=1` to make a missing library or an
unresolvable checkpoint a hard failure instead of a skip. CI sets it on the
integration job, so a cold or broken model cache turns the build red rather
than reporting "0 tests ran, all green".

Coverage:

```bash
pytest --cov=relativedb --cov-report=xml:coverage-python.xml --cov-report=term
```

## Docs

Read the [RelQL book](https://relql.com/docs/).

## License

Apache-2.0.
