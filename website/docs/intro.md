---
id: intro
title: What is relativedb?
slug: /
description: A predictive-query engine that runs RelQL over your own relational data.
---

# What is relativedb?

relativedb answers questions about the **future** of your relational data. You
declare the shape of your tables and links, wire small **retriever** callbacks
over your existing storage, and write a predictive query:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FOR EACH customers.customer_id
```

That's 90-day churn for every customer — no feature engineering, no training
pipeline, and no temporal leakage by construction.

## How it fits together

1. **RelQL** — a SQL-flavored query language for predictions. Parsed and
   validated against your declared schema. See the [RelQL docs](/pql/).
2. **Retrievers** — the engine never touches your database. All data access
   goes through callbacks you implement, GraphQL-style. See
   [Retrievers](concepts/retrievers).
3. **Temporal context assembly** — the engine hops your relational graph to
   build a per-entity context, and guarantees nothing newer than the anchor
   time enters it. See [Temporal correctness](concepts/temporal-correctness).
4. **Model backends** — contexts are scored by a required, pluggable backend.
   The shipped one is `RtNativeBackend`, running **RT-J**, a relational
   transformer foundation model that predicts in-context. There is no model-free
   default. See [Model backends](concepts/model-backends).

## Three peer libraries

The engine is implemented natively in [Python](libraries/python),
[Java](libraries/java), and [Rust](libraries/rust) — same concepts, same
behavior, idiomatic APIs. A shared [C++ inference engine](libraries/cpp)
serves the RT-J model to all three.

## Where to go next

- [Installation](getting-started/installation) and
  [Quickstart](getting-started/quickstart) — first prediction in minutes.
- [RelQL tutorial](/pql/tutorial) — learn the language step by step.
- [How-to guides](how-to/predict-churn) — churn, ranking, forecasting,
  custom retrievers, the native model backend.
