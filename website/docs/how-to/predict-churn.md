---
title: Predict churn
description: Score churn risk for active customers through your own retrievers.
---

# Predict churn

Goal: for every *currently active* customer, the probability of no activity in
the next 30 days.

## 1. Wire your tables

```python
import pandas as pd
import relativedb

# Declare `schema`, translate DataFrame records to `relativedb.Row`, and wire
# entity/link/scanner callbacks. The complete example connector is linked below.
wiring = relativedb.RetrieverWiring.new_wiring()...build()
engine = relativedb.Engine(schema, wiring)
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
result = engine.execute(relativedb.ExecutionInput(
    query=query, anchor_time=pd.Timestamp.utcnow().to_pydatetime()))
df = pd.DataFrame({"entity_id": [p.id for p in result.predictions],
                   "probability": [p.probability for p in result.predictions]})
df.sort_values("probability", ascending=False).head(20)
```

Each row is `entity_id, probability`. Users inactive for 90+ days are excluded
by the `WHERE` clause — they already churned.

## Variations

- Different definition: `COUNT(orders.*, 0, 60, days) = 0` (no purchase) or
  `SUM(usage.minutes, 0, 30, days) < 10` (low usage).
- Backtest: rerun with a past `anchor_time` and compare against what actually
  happened — the engine guarantees the context is point-in-time correct.
- Real model: construct the engine with
  `model_backend=relativedb.RtNativeBackend(schema=schema)`
  ([guide](use-native-backend)).

A complete self-checking version lives at `examples/industry/growth_churn.py`;
its `pandas_connector.py` shows the application-owned connector.
