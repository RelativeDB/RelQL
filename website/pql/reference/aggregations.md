---
title: Aggregations & windows
description: Aggregation functions and anchor-relative time windows.
---

# Aggregations and time windows

```
AGG( table.column | table.* [WHERE <row filter>], start, end [, unit] )
```

## Functions

`SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `COUNT_DISTINCT`, `LIST_DISTINCT`,
`FIRST`, `LAST`.

- `COUNT(table.*)` counts rows.
- `FIRST` / `LAST` pick a value by row time — useful for status columns.
- `LIST_DISTINCT` predicts a set of values (usually FK IDs) and takes a
  directive: `RANK TOP K` (ranking) or `CLASSIFY`.

## Windows

Offsets are relative to the anchor time t₀; `start` is **excluded**, `end` is
**included**:

```sql
COUNT(orders.*, 0, 30, days)      -- (t₀, t₀+30d]   : the next 30 days
COUNT(orders.*, -90, 0, days)     -- (t₀-90d, t₀]   : the last 90 days
COUNT(orders.*, -INF, 0)          -- all history up to t₀
SUM(sales.qty, 15, 45, days)      -- a future window not starting now
```

- Units: `SECONDS`, `MINUTES`, `HOURS`, `DAYS`, `WEEKS`, `MONTHS` (a month is
  a 30-day approximation). Omitted unit = days.
- **Target** windows must face the future (non-negative offsets).
- **Filter** windows (inside `WHERE`) face the past — negative or `-INF`
  starts. The validator enforces both directions.

## Inline row filters

Filter the rows being aggregated (distinct from `WHERE`, which filters
entities):

```sql
COUNT(transactions.* WHERE transactions.amount > 10, 0, 30, days)
```
