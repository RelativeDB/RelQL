---
title: Aggregations & windows
description: Aggregation functions and anchor-relative time windows.
---

# Aggregations and time windows

```
AGG( table.column | table.* [WHERE <row filter>] ) OVER ( <window_spec> )
```

An aggregation names a column (or `table.*`), an optional inline row filter,
and a **frame** introduced by `OVER`. The frame — not positional offsets —
carries the time window.

## Functions

`SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `COUNT_DISTINCT`, `LIST_DISTINCT`,
`FIRST`, `LAST`, `EXISTS`, `NOT EXISTS`.

- `COUNT(table.*)` counts rows.
- `FIRST` / `LAST` pick a value by row time — useful for status columns.
- `LIST_DISTINCT` predicts a set of values (usually FK IDs) and takes a
  directive: `RANK TOP K` (ranking) or `CLASSIFY`.
- `EXISTS(table.*)` / `NOT EXISTS(table.*)` is a boolean existence test — true
  when any matching row falls in the frame. It reads more directly than the
  `COUNT(...) > 0` idiom (which is still valid): `EXISTS(orders.*) OVER (90 DAYS
  PRECEDING)`.

## The OVER frame

A frame is measured relative to the anchor time (`NOW`); membership is
start-**exclusive**, end-**inclusive**. Direction comes from `PRECEDING`
(past) / `FOLLOWING` (future), and durations are always **positive**.

```
window_spec := frame [HORIZONS <positive-int> [STEP <positive-duration>]]

frame := RANGE BETWEEN <bound> AND <bound>
       | <positive-duration> PRECEDING        -- shorthand: (NOW - dur, NOW]
       | <positive-duration> FOLLOWING        -- shorthand: (NOW, NOW + dur]
       | UNBOUNDED PRECEDING                   -- all history up to NOW

bound := NOW
       | <positive-duration> PRECEDING
       | <positive-duration> FOLLOWING
       | UNBOUNDED PRECEDING
       | UNBOUNDED FOLLOWING

duration := <positive-number> <unit>
```

The single-bound forms are **shorthand** for a frame with one endpoint at
`NOW`. Use the full `RANGE BETWEEN` form when neither endpoint is `NOW`:

```sql
COUNT(orders.*) OVER (30 DAYS FOLLOWING)      -- (NOW, NOW+30d]  : the next 30 days
COUNT(orders.*) OVER (90 DAYS PRECEDING)      -- (NOW-90d, NOW]  : the last 90 days
COUNT(orders.*) OVER (UNBOUNDED PRECEDING)    -- all history up to NOW
SUM(sales.qty)  OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING)
                                              -- a future window not starting now
```

- Units: `SECONDS`, `MINUTES`, `HOURS`, `DAYS`, `WEEKS`, `MONTHS`, `YEARS`
  (singular or plural, case-insensitive; a month is a 30-day approximation).
- **Target** frames face the future (`FOLLOWING`). **Filter** frames (inside
  `WHERE`) face the past (`PRECEDING` / `UNBOUNDED PRECEDING`). The validator
  enforces both directions.

## Multiple horizons (forecasting)

Append `HORIZONS N` to repeat the frame N times back to back — a multi-horizon
window *is* a forecast (there is no separate `FORECAST` clause). `STEP`
optionally sets the stride between horizons; it defaults to the frame width, so
give a smaller `STEP` for overlapping horizons:

```sql
SUM(usage.count) OVER (1 DAY FOLLOWING HORIZONS 28)          -- 28 daily steps
SUM(sales.qty)   OVER (30 DAYS FOLLOWING HORIZONS 6 STEP 7 DAYS)  -- overlapping
```

## Named windows

Declare a frame once with a trailing `WINDOW` clause and reference it by name
as `OVER <name>` — handy when several aggregations share one frame:

```sql
PREDICT SUM(orders.revenue) OVER w - SUM(orders.cost) OVER w
FOR EACH customers.customer_id
WINDOW w AS (30 DAYS FOLLOWING)
```

A window name is declared exactly once and accepts every frame form, including
`HORIZONS` / `STEP`. Referencing an undeclared name is an error.

## Inline row filters

Filter the rows being aggregated (distinct from `WHERE`, which filters
entities):

```sql
COUNT(transactions.* WHERE transactions.amount > 10) OVER (30 DAYS FOLLOWING)
```
