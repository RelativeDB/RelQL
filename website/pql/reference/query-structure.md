---
title: Query structure
description: Clauses, keywords, and identifiers.
---

# Query structure

Clause order is significant:

```sql
PREDICT   <target>                       -- required: what to predict
[FORECAST <N> TIMEFRAMES]                -- optional: N successive windows
FOR [EACH] <entity_table>.<pkey>         -- required: the population
          [= <literal> | IN (<list>)]    --   ...or explicit entities
[WHERE     <condition>]                  -- optional: entity filter (past-facing)
[ASSUMING  <temporal_condition>]         -- optional: counterfactual
```

## Clauses

- **`PREDICT <target>`** — a static column reference
  (`customers.age`, `articles.description IS NULL`) or an
  [aggregation](aggregations) over linked rows, optionally compared to a
  literal.
- **`FORECAST N TIMEFRAMES`** — repeats the target window N times back to
  back; makes the task [forecasting](task-types).
- **`FOR EACH table.pk`** — predict for every entity (requires a
  `TableScanner`). `FOR table.pk = <lit>` / `IN (...)` selects explicit
  entities.
- **`WHERE <condition>`** — filters the population using static attributes
  and past-facing aggregations. See [Conditions](conditions).
- **`ASSUMING <condition>`** — a counterfactual assumption, parsed and
  validated and carried on the query (not yet applied to context assembly).

## Lexical rules

- Keywords are **case-insensitive**: `PREDICT`, `FORECAST`, `TIMEFRAMES`,
  `FOR`, `EACH`, `WHERE`, `ASSUMING`, `CLASSIFY`, `RANK`, `TOP`.
- Aggregation and condition words (`COUNT`, `SUM`, `AND`, `LIKE`, ...) are
  **soft keywords** — still usable as column names (`usage.count` parses).
- Column references are always qualified: `table.column`; `table.*` counts
  rows.
- Literals: numbers, `'quoted strings'`, booleans. `-INF` marks an unbounded
  window start.
- Comments are supported.
