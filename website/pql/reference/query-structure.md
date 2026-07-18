---
title: Query structure
description: Clauses, keywords, and identifiers.
---

# Query structure

Clause order is significant:

```sql
[EXPLAIN [PLAN|CONTEXT|ANALYZE|ABLATION] [FORMAT TEXT|JSON]]  -- optional: inspect, don't (necessarily) run
PREDICT   <target> [CLASSIFY | RANK TOP <k>]   -- required: what to predict
FOR [EACH] <entity_table>.<pkey>               -- required: the population
          [= <literal> | IN (<list>)]          --   ...or explicit entities
[WHERE     <condition>]                        -- optional: entity filter (past-facing)
[ASSUMING  <condition>]                        -- optional: counterfactual
[AS OF     <anchor>]                           -- optional: bind the anchor time
[ABLATE TABLE <name>]                          -- optional: drop a table from context
[RETURN    <return_spec>]                       -- optional: choose the output form
[WINDOW    <name> AS (<window_spec>)]           -- optional, repeatable: named frames
```

The **trailing clauses** — `WHERE`, `ASSUMING`, `AS OF`, `ABLATE TABLE`,
`RETURN`, `WINDOW` — may appear in any order after `FOR`. Each may appear at
most once, except `WINDOW`, which repeats (one per named frame).

There is no `FORECAST N TIMEFRAMES` clause. To forecast, give the target's
window multiple horizons (`... OVER (7 DAYS FOLLOWING HORIZONS 4)`); a
multi-horizon window *implies* [forecasting](task-types). See
[Aggregations & windows](aggregations).

## Clauses

- **`PREDICT <target>`** — a static column reference
  (`customers.age`, `articles.description IS NULL`), an
  [aggregation](aggregations) over linked rows in an `OVER` frame, or a
  richer expression (arithmetic, `CASE WHEN … END`, `COALESCE`, column-to-column
  comparison), optionally compared to a literal. `CLASSIFY` and `RANK TOP k`
  are target directives (see [task types](task-types)).
- **`FOR EACH table.pk`** — predict for every entity (requires a
  `TableScanner`). `FOR table.pk = <lit>` / `IN (...)` selects explicit
  entities.
- **`WHERE <condition>`** — filters the population using static attributes
  and past-facing aggregations. See [Conditions](conditions).
- **`ASSUMING <condition>`** — a counterfactual assumption, parsed and
  validated and carried on the query (not yet applied to context assembly).
- **`AS OF <anchor>`** — binds the anchor time (the instant `NOW` and every
  frame are measured from). The anchor is a parameter (`:prediction_time`), a
  `DATE` literal (`2026-07-01`), or `NOW`.
- **`ABLATE TABLE <name>`** — removes a table from assembled context, for
  measuring its contribution.
- **`RETURN <return_spec>`** — selects the output form (see below).
- **`WINDOW <name> AS (<window_spec>)`** — declares a reusable named frame,
  referenced elsewhere as `OVER <name>`. Declared exactly once; referencing an
  undeclared name is an error. See [Aggregations & windows](aggregations).

## RETURN — output form

`RETURN` overrides the default output implied by the task type:

```
EXPECTED VALUE | PROBABILITY | CLASS | DISTRIBUTION
| QUANTILES (<num>, ...) | INTERVAL <int> [%] | MULTILABEL | MULTICLASS
```

```sql
PREDICT SUM(payments.amount) OVER (30 DAYS FOLLOWING)
FOR EACH customers.customer_id
AS OF :t
RETURN INTERVAL 90%
```

## EXPLAIN — inspect without running

An `EXPLAIN` prefix asks the engine to describe what it *would* do:

```
EXPLAIN [PLAN | CONTEXT | ANALYZE | ABLATION] [FORMAT TEXT | JSON]
```

Bare `EXPLAIN` means `EXPLAIN PLAN` and does **not** invoke the model.
`CONTEXT` shows the assembled context, `ANALYZE` and `ABLATION` run the model
and report timing / table contributions.

```sql
EXPLAIN PLAN FORMAT TEXT
PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING)
FOR EACH customers.customer_id
ABLATE TABLE support_tickets
RETURN PROBABILITY
```

## Lexical rules

- Keywords are **case-insensitive**: `PREDICT`, `OVER`, `FOLLOWING`,
  `PRECEDING`, `RANGE`, `BETWEEN`, `HORIZONS`, `STEP`, `WINDOW`, `AS OF`,
  `RETURN`, `EXPLAIN`, `FOR`, `EACH`, `WHERE`, `ASSUMING`, `CLASSIFY`, `RANK`,
  `TOP`.
- Aggregation and condition words (`COUNT`, `SUM`, `AND`, `LIKE`, ...) are
  **soft keywords** — still usable as column names (`usage.count` parses).
- Column references are always qualified: `table.column`; `table.*` counts
  rows.
- Literals: numbers, `'quoted strings'`, booleans, `DATE`s (`2026-07-01`).
  Frame bounds use `UNBOUNDED PRECEDING` for all history.
- Comments are supported.
