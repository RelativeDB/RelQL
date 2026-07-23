---
id: intro
title: The relativedb engine
slug: /
description: "The complete engine guide: install, concepts, how-to, and the Python library, in one page."
---

# What is relativedb?

relativedb answers questions about the **future** of your relational data. You
declare the shape of your tables and links, wire small **retriever** callbacks
over your existing storage, and write a predictive query:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers
```

## How it fits together

1. **RelQL.** A SQL-flavored query language for predictions. Parsed and
   validated against your declared schema. See the [RelQL docs](/relql/).
2. **Retrievers.** The engine never touches your database. All data access
   goes through callbacks you implement, GraphQL-style. See
   [Retrievers](#retrievers).
3. **Temporal context assembly.** The engine hops your relational graph to
   build a per-entity context, and guarantees nothing newer than the anchor
   time enters it. See [Temporal correctness](#temporal-correctness).

## Installation

### Python

Requires Python 3.10+.

```bash
pip install relativedb
```

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Quickstart

Predict 90-day churn for every customer, as of July 1.

<Tabs groupId="lang">
<TabItem value="python" label="Python">

Declare the schema and wire callbacks over your own storage. If your data is
in pandas, pandas remains an application dependency:

```python
import pandas as pd
from relativedb import Engine, ExecutionInput, RetrieverWiring, Row, RtNativeBackend

customer_rows = [Row("customers", r.customer_id, {"age": float(r.age)})
                 for r in customers.itertuples()]
by_id = {row.id: row for row in customer_rows}

def fetch_customers(table, ids, bound):
    return [by_id[row_id] for row_id in ids if row_id in by_id]

wiring = (RetrieverWiring.new_wiring()
    .entities("customers", fetch_customers)
    .entities("orders", fetch_orders)
    .default_links(fetch_order_children)
    .scanner("customers", scan_customers)
    .build())

# Scoring requires a model backend. RtNativeBackend runs the RT-J relational
# model; it needs librt_c and a cached stanford-star/rt-j checkpoint.
engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema))
result = engine.execute(ExecutionInput(
    query="PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers",
    anchor_time=pd.Timestamp("2026-07-01").to_pydatetime()))
```

</TabItem>
</Tabs>


### Next steps

- Learn the query language: [RelQL tutorial](/relql/#relql-tutorial)
- Wire your own storage: [Retrievers](#retrievers)
- Fit a ranking adapter on your history: [Fit a multiclass/ranking adapter](#fit-a-multiclassranking-adapter)


## Retrievers

The engine never touches a database. It asks **your** code for rows through
four small interfaces, each a plain Python callable:

| Interface | Signature (conceptual) | Role |
|---|---|---|
| `EntityRetriever` | `(table, ids, bound) → rows` | Batched point lookup: seed rows, parents |
| `LinkRetriever` | `(link, parent_id, bound, limit) → rows` | Children along one FK link, newest-first |
| `CohortRetriever` *(optional)* | `(table, anchor, bound, limit) → ids` | Similar entities for in-context examples |
| `TableScanner` *(optional)* | `(table, bound) → row stream` | Bulk streaming. Enables whole-table `FROM` and CSC mode |

(Normalization statistics are not a retriever: pass a `ColumnStats` to the
model backend.)

### Rows

A `Row` carries typed cells, an optional timestamp, and **parent edges**
(`{fk_column: parent_id}`). FK values are not cells. They surface as edges.
The primary key surfaces as identity, and additionally as a cell when the
schema declares it as a column.

:::caution
A row whose table declares no feature columns emits no tokens. A token-less row
that others link through is a dead end: nothing below it reaches the prediction,
and every entity scores alike. The engine emits a `ContextConnectivityWarning`
when it detects this. Give the table a feature column, or declare its primary
key as one.
:::

### Keys as features

A primary key is identity by default. When the key carries meaning (a stock
code, an ISBN, an airport code), declare it as a column too, the same way you
declare a `time_column`. The engine then emits it as a feature cell:

```python
TableDef.new_table("users").primary_key("user_id")           # identity only
(TableDef.new_table("products")                              # ...and a feature
    .column("stock_code", ValueType.TEXT).primary_key("stock_code"))
```

Leave synthetic keys out. Autoincrement IDs track insertion order, so the model
reads one as a proxy for tenure, a signal that breaks on a new ID range.

### Wiring

A `RetrieverWiring` binds retrievers to tables and links, with a
`default_links` catch-all. It is validated against the schema when the engine
is built, so a missing retriever fails at startup, before any query runs.

## Temporal correctness

Temporal leakage (a "future" fact sneaking into the features) is the classic
way predictive systems lie in backtests. relativedb makes leakage prevention
an **engine guarantee** rather than something you have to remember.

### The anchor time

Every execution has an anchor time t₀. The prediction target reads the window
*after* t₀. The assembled context may only contain data at or *before* t₀.

### Defense in depth

1. Every retriever call carries a `TemporalBound`: "return nothing newer
   than this". Rows without timestamps (static dimension tables) are always
   admitted.
2. The engine **re-checks every returned row** against the bound and drops
   violations. A buggy or malicious retriever cannot leak the future into
   context. Dedicated tests in all three libraries feed a deliberately broken
   retriever and assert the future row never appears.

### Window direction is validated

Target windows must face the future (non-negative offsets). `WHERE` filter
windows face the past (negative or `-INF` starts). The validator rejects
queries that mix these up.

### Backtesting for free

Because "as of" is an explicit input, evaluating yesterday's model is just
running the same query with yesterday's anchor. No snapshot tables, no
point-in-time joins.

## Sampler modes

Context assembly walks the graph: seed entity → parents (always followed) →
children (fanout-capped, newest-first) → optional cohort, until the hop limit
or cell budget. Two interchangeable samplers drive this walk, and both produce
**identical contexts** (asserted by tests).

### RETRIEVER (default)

Pull-per-hop: the hop loop calls your retrievers for each expansion.

Use when data is **remote, huge, or access-controlled**. Nothing is copied, and
your retrievers see every access.

### CSC

The engine drains each `TableScanner` once into in-memory
compressed-sparse-column adjacency arrays (time-sorted neighbor lists), then
samples entirely in-process, so "latest *w* children ≤ anchor" is one binary
search plus a tail slice.

Use for **latency-sensitive, repeated scoring** over data that fits in
memory. The index is a **snapshot**, built once when the engine is
constructed and immutable thereafter. To pick up changed data, construct
a new engine.

### Context budgets

`ContextPolicy` supports two geometries:

- per-hop fanouts, e.g. `fanouts=(64, 64)`
- a uniform `bfs_width` (default 32) under a global `max_context_cells`
  budget (default 2048)

All of it is configurable from the Python API. The model backend has a
matching token-sequence cap, `max_seq_len` (default 2048; the reference
evaluation runs at 8192):

```python
from relativedb import ContextPolicy, Engine, RtNativeBackend

engine = Engine(schema, wiring,
    context_policy=ContextPolicy(max_context_cells=8192, bfs_width=64),
    model_backend=RtNativeBackend(schema=schema, max_seq_len=8192))
```

Larger budgets admit more history per entity and cost latency; keep
`max_seq_len` at or above `max_context_cells` so assembled context is not
truncated at the model boundary.

See [Choose a sampler mode](#choose-a-sampler-mode) for a decision guide.

#### Supported output types

The checkpoint executes **binary classification**, **regression**,
**multiclass classification**, and **ranking**. `RETURN CLASS`, `RETURN
DISTRIBUTION`, `RETURN PROBABILITY`, and `RETURN EXPECTED VALUE` work.
Multiclass reuses the checkpoint's **text head**. It decodes the masked target
cell to a 384-dim embedding, then matches that against the class labels' MiniLM
embeddings by cosine similarity. You get a predicted class and approximate class
probabilities: the argmax is reference-exact, while the probabilities are only a
softmax over cosine scores. Ranking scores each candidate parent ID with the existence
head, sigmoids it, and returns the top *k*. `RETURN QUANTILES` and
`RETURN INTERVAL` are **not part of the language**: the model exposes a single
point estimate, not a distribution, so they could never execute. A query using
them is rejected at parse time with a message naming them.

## Choose a sampler mode

Both modes produce identical contexts. Choose by data locality.

| | RETRIEVER (default) | CSC |
|---|---|---|
| Data location | stays in your store | copied into an in-memory index |
| Freshness | live, per query | snapshot, fixed at construction |
| Requires | Entity + Link retrievers | `TableScanner` per table |
| Best for | remote, huge, or access-controlled data | repeated low-latency scoring |

### Switching

```python
from relativedb import Engine, ExecutionInput, SamplerMode
engine = Engine(schema, wiring, sampler_mode=SamplerMode.CSC)
result = engine.execute(ExecutionInput(query=query, anchor_time=t0))
```

### What CSC buys you

On a synthetic churn workload (10,000 customers, 200,000 orders, history
baseline, M-series laptop), scoring every customer:

| Approach | Time | Throughput |
|---|---|---|
| relativedb, CSC sampler | 0.66 s | ~15,000 entities/s |
| naive per-entity pandas loop | 57.4 s | ~174 entities/s |

The CSC index turns each
"latest *w* children ≤ anchor" expansion into a binary search plus a tail
slice, and its build cost is paid once per snapshot.

### Rule of thumb

Start with RETRIEVER. Move to CSC when the same engine scores many queries or
large populations and the data fits in memory.


## Fit a multiclass/ranking adapter

The released checkpoint is zero-shot. For multiclass or multilabel ranking,
you can fit a small task head over the **frozen** transformer. Nothing in the transformer
changes, so the engine encodes each example once into its target-cell state and
fits a ~2 KB adapter quickly.

This is not full-model fine-tuning. `fit_head` covers **multiclass and
ranking** only; scalar binary/regression tasks reject the frozen adapter and go
through `Engine.finetune()` instead, which trains the complete backbone
natively (requires Apple MPS).

```python
head = engine.fit_head(
    query=Q,
    anchors=[t - timedelta(days=d) for d in (150, 120, 90, 60)],  # past cut-offs
    params={"ids": cohort},
    epochs=300, learning_rate=1e-2)

head          # <FineTunedHead ranking on 2760 examples loss 4.10->3.65>
head.save("head.safetensors")

tuned = Engine(schema, wiring, model_backend=RtNativeBackend(
    schema=schema, wiring=wiring, head="head.safetensors"))
```

### Labels come from the query

At each anchor the context is bounded exactly as it would be at prediction
time, while the **label** is read from the target's own window *after* it. The
query therefore defines its own supervision and nothing needs labelling by hand.

Pass `labels={(entity_id, anchor): value}` to override. For ranking, that is a
`{candidate_id: relevance}` mapping.

:::caution
Choose training anchors strictly **before** the anchor you evaluate at, or the
head learns from the future it is meant to predict.
:::

### What to expect

- The head replaces the checkpoint's zero-shot head only for the task it was
  trained on.
- Fitting requires **Metal**. Inference on a trained head is plain CPU, so an
  adapter trained on a Mac serves anywhere.
- Ranking groups with no relevant candidate in the window carry no signal and
  are skipped, and the count is reported, since listwise loss needs a positive
  per group.
- Judge the result on **held-out** anchors, never on training loss. A falling
  loss with flat held-out quality means the head is data-limited. Adding epochs
  will not help. Add anchors or entities.
