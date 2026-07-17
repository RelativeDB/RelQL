---
title: Predict churn
description: Score churn risk for active customers from DataFrames.
---

# Predict churn

Goal: for every *currently active* customer, the probability of no activity in
the next 30 days.

## 1. Load your tables

```python
import pandas as pd
import relativedb

ds = relativedb.from_dataframes(
    {"users": users, "events": events},
    links=[("events", "user_id", "users")])
```

## 2. Write the query

Define churn in the target; restrict the population to active users in
`WHERE` (past-facing window):

```sql
PREDICT COUNT(events.*, 0, 30, days) = 0
FOR EACH users.user_id
WHERE COUNT(events.*, -90, 0, days) > 0
```

## 3. Score as of today

```python
df = ds.predict(query, anchor_time=pd.Timestamp.utcnow().normalize())
df.sort_values("probability", ascending=False).head(20)
```

Each row is `entity_id, probability`. Users inactive for 90+ days are excluded
by the `WHERE` clause — they already churned.

## Variations

- Different definition: `COUNT(orders.*, 0, 60, days) = 0` (no purchase) or
  `SUM(usage.minutes, 0, 30, days) < 10` (low usage).
- Backtest: rerun with a past `anchor_time` and compare against what actually
  happened — the engine guarantees the context is point-in-time correct.
- Real model: pass `model_backend=relativedb.RtNativeBackend(schema=ds.schema)`
  ([guide](use-native-backend)).

A complete self-checking version lives at `examples/industry/growth_churn.py`.
