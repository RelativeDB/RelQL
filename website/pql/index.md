---
id: index
title: RelQL overview
slug: /
description: The Predictive Query Language — SQL for questions about the future.
---

# RelQL — the Predictive Query Language

RelQL expresses predictions the way SQL expresses lookups. One statement names a
**target** (what to predict), a **population** (who to predict it for), and an
**anchor-relative time window**:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FOR EACH customers.customer_id
```

*For every customer, will they place zero orders in the 90 days after the
anchor time?*

## Why a language?

- **Declarative** — changing the question means changing the string, not a
  pipeline.
- **Validated** — every query is bound against your schema before execution:
  unknown names, type mismatches, and backwards time windows are rejected up
  front.
- **Self-routing** — the query's shape determines the
  [task type](reference/task-types) (classification, regression, ranking,
  forecasting), which selects the model checkpoint and output form.

RelQL derives from Kumo/KumoRFM's predictive query language. One grammar, three
verified implementations (ANTLR in Java; recursive descent in Python and
Rust), all tested against a shared 44-query corpus.

## Learn it

- New to RelQL? Start with the [tutorial](tutorial) — it builds a query up
  clause by clause.
- Looking something up? See the [reference](reference/query-structure).
- Want patterns to copy? See the [cookbook](cookbook).
