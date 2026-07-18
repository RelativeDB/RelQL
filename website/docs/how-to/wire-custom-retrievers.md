---
title: Wire custom retrievers
description: Connect the engine to your own storage — a DAO, REST service, or feature store.
---

# Wire custom retrievers

Goal: run RelQL against data the engine cannot see — behind a DAO, a REST
service, or a feature store.

## 1. Declare the schema (shape only)

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

## 2. Implement the two required retrievers

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

## 3. Optional: enable `FOR EACH` and CSC

`FOR EACH table.pk` needs a `TableScanner` to enumerate the population:

```python
    .scanner("customers", lambda table, bound: customer_dao.scan_all(bound))
```

Scanners also unlock [CSC mode](../concepts/sampler-modes) for in-memory
sampling.

## 4. Build and run

```python
from relativedb import Engine, ExecutionInput

engine = Engine(schema, wiring)   # wiring validated here — missing pieces fail fast
result = engine.execute(ExecutionInput(query=..., anchor_time=...))
```

In Java the same SPI is async (`CompletionStage`) — see the
[Java library](../libraries/java).
