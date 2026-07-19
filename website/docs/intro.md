---
id: intro
title: The relativedb engine
slug: /
description: The complete engine guide — install, concepts, how-to, and the language libraries, in one page.
---

# What is relativedb?

relativedb answers questions about the **future** of your relational data. You
declare the shape of your tables and links, wire small **retriever** callbacks
over your existing storage, and write a predictive query:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers
```

That's 90-day churn for every customer — no feature engineering, no training
pipeline, and no temporal leakage by construction.

## How it fits together

1. **RelQL** — a SQL-flavored query language for predictions. Parsed and
   validated against your declared schema. See the [RelQL docs](/relql/).
2. **Retrievers** — the engine never touches your database. All data access
   goes through callbacks you implement, GraphQL-style. See
   [Retrievers](#retrievers).
3. **Temporal context assembly** — the engine hops your relational graph to
   build a per-entity context, and guarantees nothing newer than the anchor
   time enters it. See [Temporal correctness](#temporal-correctness).
4. **Model backends** — contexts are scored by a required, pluggable backend.
   The shipped one is `RtNativeBackend`, running **RT-J**, a relational
   transformer foundation model that predicts in-context. There is no
   model-free default. See
   [Use the native RT-J backend](#use-the-native-rt-j-backend).


## Installation

### Python

Requires Python 3.10+.

```bash
pip install relationdb
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
- Score with the shipped model: [Use the native RT-J backend](#use-the-native-rt-j-backend)
- Connect real storage: [Wire custom retrievers](#wire-custom-retrievers)


## Retrievers

The engine never touches a database. It asks **your** code for rows through
five small interfaces (Java: async interfaces; Python: plain callables; Rust:
traits, usually closures):

| Interface | Signature (conceptual) | Role |
|---|---|---|
| `EntityRetriever` | `(table, ids, bound) → rows` | Batched point lookup: seed rows, parents |
| `LinkRetriever` | `(link, parent_id, bound, limit) → rows` | Children along one FK link, newest-first |
| `CohortRetriever` *(optional)* | `(table, anchor, bound, limit) → ids` | Similar entities for in-context examples |
| `TableScanner` *(optional)* | `(table, bound) → row stream` | Bulk streaming; enables whole-table `FROM` and CSC mode |
| `StatsProvider` *(optional)* | — | Normalization statistics |

### Rows

A `Row` carries typed cells, an optional timestamp, and **parent edges**
(`{fk_column: parent_id}`). FK values are not cells — they surface as edges.
The primary key surfaces as identity, and additionally as a cell when the
schema declares it as a column.

:::caution
A row whose table declares no feature columns emits no tokens, and a
token-less row that others link through is a dead end — nothing below it can
reach the prediction, and every entity scores alike. The engine raises
`ContextConnectivityWarning` when it detects this. Give the table a feature
column, or declare its primary key as one.
:::

### Keys as features

A primary key is identity by default. When the key itself carries meaning — a
SKU, an ISBN, an airport code — declare it as a column as well, exactly as you
would a `time_column`, and it is emitted as a feature cell too:

```python
TableDef.new_table("users").primary_key("user_id")          # identity only
TableDef.new_table("products")                              # ...and a feature
    .column("stock_code", ValueType.TEXT).primary_key("stock_code")
```

Leave synthetic keys out: autoincrement ids track insertion order, so feeding
one to the model invites it to read the id as a tenure proxy that will not
survive a new id range.

### Wiring

A `RetrieverWiring` binds retrievers to tables and links, with a
`default_links` catch-all. It is validated against the schema when the engine
is built, so a missing retriever fails fast, not mid-query.

## Temporal correctness

Temporal leakage — a "future" fact sneaking into the features — is the classic
way predictive systems lie in backtests. relativedb treats leakage prevention
as an **engine guarantee**, not a user discipline.

### The anchor time

Every execution has an anchor time t₀. The prediction target reads the window
*after* t₀; the assembled context may only contain data at or *before* t₀.

### Defense in depth

1. Every retriever call carries a `TemporalBound` — "return nothing newer
   than this". Rows without timestamps (static dimension tables) are always
   admitted.
2. The engine **re-checks every returned row** against the bound and drops
   violations. A buggy or malicious retriever cannot leak the future into
   context. Dedicated tests in all three libraries feed a deliberately broken
   retriever and assert the future row never appears.

### Window direction is validated

Target windows must face the future (non-negative offsets); `WHERE` filter
windows face the past (negative or `-INF` starts). The validator rejects
queries that mix these up.

### Backtesting for free

Because "as of" is an explicit input, evaluating yesterday's model is just
running the same query with yesterday's anchor — no snapshot tables, no
point-in-time joins.

## Sampler modes

Context assembly walks the graph: seed entity → parents (always followed) →
children (fanout-capped, newest-first) → optional cohort, until the hop limit
or cell budget. Two interchangeable samplers drive this walk — both produce
**identical contexts** (asserted by tests).

### RETRIEVER (default)

Pull-per-hop: the hop loop calls your retrievers for each expansion.

Use when data is **remote, huge, or access-controlled** — nothing is copied,
your retrievers see every access.

### CSC

The engine drains each `TableScanner` once into in-memory
compressed-sparse-column adjacency arrays (time-sorted neighbor lists), then
samples entirely in-process — "latest *w* children ≤ anchor" is one binary
search plus a tail slice.

Use for **latency-sensitive, repeated scoring** over data that fits in
memory. The index is a **snapshot**, built once when the engine is
constructed and immutable thereafter — to pick up changed data, construct
a new engine.

### Context budgets

`ContextPolicy` supports two geometries:

- per-hop fanouts, e.g. `fanouts(64, 64)`
- a uniform `bfs_width` under a global `max_context_cells` budget

See [Choose a sampler mode](#choose-a-sampler-mode) for a decision
guide and benchmark numbers.

#### Supported output types

The checkpoint executes **binary classification**, **regression**,
**multiclass classification**, and **ranking**. `RETURN CLASS`, `RETURN
DISTRIBUTION`, `RETURN PROBABILITY`, and `RETURN EXPECTED VALUE` work.
Multiclass reuses the checkpoint's **text head**: the masked target cell is
decoded to a 384-dim embedding and matched by cosine similarity to the class
labels' MiniLM embeddings, yielding a predicted class plus approximate,
uncalibrated class probabilities (a softmax over the cosine scores — the argmax
is reference-exact). Ranking scores each candidate parent ID with the existence
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

(Reproduce with `examples/bench_naive_vs_csc.py`.) The CSC index turns each
"latest *w* children ≤ anchor" expansion into a binary search plus a tail
slice, and its build cost is paid once per snapshot.

### Rule of thumb

Start with RETRIEVER. Move to CSC when the same engine scores many queries or
large populations and the data fits in memory.


## Fine-tune a task head

The released checkpoint is zero-shot. When you have history to learn from, train
a small task head over the **frozen** backbone: the transformer is never
updated, so each example is encoded once into its target-cell state and fitting
a ~2 KB adapter is fast.

```python
head = engine.finetune(
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

Pass `labels={(entity_id, anchor): value}` to override — for ranking, a
`{candidate_id: relevance}` mapping.

:::caution
Choose training anchors strictly **before** the anchor you evaluate at, or the
head learns from the future it is meant to predict.
:::

### What to expect

- Works for all four task types; the head replaces the checkpoint's zero-shot
  head only for the task it was trained on.
- Fitting requires **Metal**. Inference on a trained head is plain CPU, so an
  adapter trained on a Mac serves anywhere.
- Ranking groups with no relevant candidate in the window carry no signal and
  are skipped — the count is reported, since listwise loss needs a positive per
  group.
- Judge the result on **held-out** anchors, never on training loss. A falling
  loss with flat held-out quality means the head is data-limited: add anchors or
  entities, not epochs.
