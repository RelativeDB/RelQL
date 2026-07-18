# relativedb

Predictive queries (PQL) over **your own data**. GraphQL-style execution: the
engine owns the query language, planning, temporal context assembly, and model
routing — all data access goes through **user-defined retrievers** wired to a
declared schema. There are **no bundled database connectors**: bring
DataFrames, a DAO, a REST client, anything.

```bash
pip install relationdb
```

Requires Python 3.10+ and depends only on numpy. The distribution is named
`relationdb`; the established Python import remains `relativedb`. There is no
pandas adapter or bundled storage connector.

For a source checkout, use `pip install -e .` from this directory.

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

# Your connector translates DataFrame records into relationdb.Row objects.
# See examples/industry/pandas_connector.py for a complete implementation.
wiring = wire_my_dataframes(schema, {"customers": customers, "orders": orders})
result = relativedb.Engine(schema, wiring).execute(relativedb.ExecutionInput(
    query="PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id",
    anchor_time=pd.Timestamp("2026-07-01").to_pydatetime()))
df = pd.DataFrame({"entity_id": [p.id for p in result.predictions],
                   "probability": [p.probability for p in result.predictions]})
```

The schema and connector explicitly keep FK and PK columns out of feature
cells—they are graph edges (the F17 invariant). Order O4
(2026-07-05) is **after** the anchor and can never enter context: the engine
re-checks every row against the temporal bound even if a retriever misbehaves.

## The retriever SPI

```python
from relativedb import (Schema, TableDef, ColumnDef, LinkDef, ValueType,
                      RetrieverWiring, Row, TemporalBound,
                      Engine, ExecutionInput, ContextPolicy, SamplerMode)

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

engine = Engine(schema, wiring)   # ModelConfig.defaults():
                                  #   classification -> hf://stanford-star/rt-j/classification
                                  #   regression/forecasting -> hf://stanford-star/rt-j/regression
                                  #   embeddings: all-MiniLM-L12-v2 (384-d, pinned)
result = engine.execute(ExecutionInput(
    query="PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR customers.customer_id = 'C7'",
    anchor_time=pd.Timestamp("2026-07-01").to_pydatetime()))
```

Retrievers are plain callables (`typing.Protocol`s):

- `EntityRetriever(table, ids, bound) -> list[Row]` — batched point lookup
- `LinkRetriever(link, parent_id, bound, limit) -> list[Row]` — children,
  newest-first, never newer than `bound`
- `CohortRetriever(table, anchor, bound, limit) -> list[id]` — optional
  similar-entity seeds
- `TableScanner(table, bound) -> iterable[Row]` — optional bulk stream,
  required for CSC mode and for enumerating `FOR EACH` entities

`Row` carries typed cells, an optional timestamp, and parent edges
(`{fk_column: parent_id}`) — IDs are never feature values.

## Sampler modes

- `SamplerMode.RETRIEVER` (default): pull-per-hop through your retrievers;
  right when data is remote, huge, or access-controlled.
- `SamplerMode.CSC`: the engine drains each `TableScanner` once into numpy
  `colptr`/`row` adjacency arrays (neighbor lists time-sorted), then samples
  multi-hop context in-process — one binary search + tail slice per node.
  Snapshot semantics; rebuild with `engine.refresh()`.

Context knobs: `ContextPolicy(fanouts=(64, 64), max_hops=2,
max_context_cells=8192, cohort_size=..., prefer_latest=True)` — per-hop
fanouts (KumoRFM geometry) or a uniform `bfs_width` with a global cell budget
(RT geometry). Parents are always followed; children are width-bounded and
newest-first; `MONTHS` windows use a 30-day approximation.

## PQL

The full grammar (`grammar/Pql.g4` upstream) is supported by a hand-written
recursive-descent parser — aggregations with windows and inline filters,
`FORECAST N TIMEFRAMES`, `RANK TOP K` / `CLASSIFY`, `WHERE` / `ASSUMING`,
`IN` / `LIKE` / `STARTS WITH` / `IS NULL`, `-INF` bounds, soft keywords
(`usage.count`), case-insensitive keywords, comments.

```python
pq = relativedb.parse("PREDICT SUM(orders.qty, 0, 30) FOR EACH customers.customer_id")
pq.task_type()                    # TaskType.REGRESSION
relativedb.validate(pq, schema)     # bind names/types/windows against the schema
```

Task inference routes the model: bare aggregation → regression; aggregation
vs literal → binary classification; `FIRST`/`LAST`/static categorical →
multiclass; `LIST_DISTINCT` (+ `RANK TOP K`) → ranking; `FORECAST` →
forecasting.

## Model backends

`Engine` ships with a model-free `HistoryBaselineBackend` (evaluates the
target over the entity's own trailing history windows — the "self labels"
signal). Real checkpoint backends implement the two-method `ModelBackend`
protocol and receive the routed `model_uri` from `ModelConfig`. `ASSUMING`
clauses are parsed and carried on the query but not yet applied to context
(counterfactual injection is an open design question upstream).

## Native RT backend (optional)

`relativedb.RtNativeBackend` scores contexts with the golden-verified C++ RT-J
inference engine (`cpp/`, exposed as the `librt_c` shared library) instead of
the history baseline. It converts each assembled context into the raw RT
token batch — one token per feature cell, FK links as the node graph (F17),
per-column z-scores for numbers/booleans and a global datetime stat
(F11/F12), `bool_as_num` label routing (F52), MiniLM (`all-MiniLM-L12-v2`)
embeddings for text cells and `"<column> of <table>"` schema phrases
(F13/F14) — plus a synthetic masked `task` row anchored at prediction time
with the entity's own past outcomes as in-context self labels (F65).
Classification logits pass through a sigmoid to yield probabilities;
regression scores denormalize with the in-context label stats. Checkpoints
route per `ModelConfig` (`hf://` URIs resolve via huggingface_hub,
cache-first; local paths work too).

```python
backend = relativedb.RtNativeBackend(schema=schema)
result = relativedb.Engine(schema, wiring, model_backend=backend).execute(
    relativedb.ExecutionInput(query=query, anchor_time=t0))
```

Needs `pip install -e ".[rt]"` (sentence-transformers + huggingface_hub) and
the built library — found via `RELATIVEDB_RT_LIB` or the sibling
`cpp/build/librt_c.dylib`; a clear `RtNativeUnavailableError` is raised
otherwise. Multiclass classification and ranking fall back to the history
baseline (the C ABI exposes a single score head).

## Tests

```
.venv/bin/python -m pytest
```

Covers the full 44-query PQL corpus (plus 20 malformed rejections), the
temporal-leakage guard (a future row never enters context, even from a buggy
retriever), CSC/retriever context equivalence, model-URI routing, and the
explicit retriever-to-churn-prediction path end to end.
