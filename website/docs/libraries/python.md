---
title: Python
description: The relativedb Python library.
---

# Python library

Package `relativedb`. Python 3.10+; core depends only on numpy.

```bash
cd python
pip install -e ".[pandas]"    # extras: [pandas], [rt], [dev]
```

## The pandas layer

`from_dataframes` infers a schema from your frames (PKs from `*_id` naming,
value types from dtypes, time columns from datetime columns) and wires
in-memory retrievers:

```python
ds = relativedb.from_dataframes(
    {"customers": customers, "orders": orders},
    links=[("orders", "customer_id", "customers")])

df = ds.predict(query, anchor_time=t0)        # DataFrame: entity_id, probability/value
```

Overrides: `primary_keys={...}`, `time_columns={...}`. Sampler mode and
context knobs pass through `.predict(...)`.

## The core API

- **Schema** — `Schema.new_schema().table(TableDef.new_table(...)...).link(LinkDef(...)).build()`
- **Wiring** — `RetrieverWiring.new_wiring().entities(...).default_links(...).scanner(...).build()`;
  retrievers are plain callables (`typing.Protocol`)
- **Engine** — `Engine(schema, wiring)`;
  `engine.execute(ExecutionInput(query=..., anchor_time=..., entity_ids=...))`
- **PQL** — `relativedb.parse(q)`, `relativedb.validate(pq, schema)`,
  `pq.task_type()`
- **Backends** — `HistoryBaselineBackend` (default),
  `RtNativeBackend(schema=...)` for RT-J

Errors are specific: `PqlSyntaxError`, `PqlValidationError`, `SchemaError`,
`WiringError`, `ExecutionError`, `RtNativeUnavailableError`.

## Tests

```bash
.venv/bin/python -m pytest
```

Covers the shared 44-query PQL corpus (+20 rejections), the temporal-leakage
guard, CSC ≡ retriever equivalence, model routing, and the DataFrames→churn
path end to end.
