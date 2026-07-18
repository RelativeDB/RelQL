---
title: Python
description: The relativedb Python library.
---

# Python library

PyPI distribution `relationdb` (imported as `relativedb`). Python 3.10+; core
depends only on numpy.

```bash
pip install relationdb    # extras: [rt], [dev]
```

## Bring your own connector

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

## The core API

- **Schema** — `Schema.new_schema().table(TableDef.new_table(...)...).link(LinkDef(...)).build()`
- **Wiring** — `RetrieverWiring.new_wiring().entities(...).default_links(...).scanner(...).build()`;
  retrievers are plain callables (`typing.Protocol`)
- **Engine** — `Engine(schema, wiring)`;
  `engine.execute(ExecutionInput(query=..., anchor_time=..., entity_ids=...))`
- **RelQL** — `relativedb.parse(q)`, `relativedb.validate(pq, schema)`,
  `pq.task_type()`
- **Backends** — `HistoryBaselineBackend` (default),
  `RtNativeBackend(schema=...)` for RT-J

Errors are specific: `PqlSyntaxError`, `PqlValidationError`, `SchemaError`,
`WiringError`, `ExecutionError`, `RtNativeUnavailableError`.

## Tests

```bash
.venv/bin/python -m pytest
```

Covers the shared 44-query RelQL corpus (+20 rejections), the temporal-leakage
guard, CSC ≡ retriever equivalence, model routing, and the retriever→churn
path end to end.
