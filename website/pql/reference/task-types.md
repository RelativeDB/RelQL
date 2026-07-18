---
title: Task types
description: How the target's shape selects the task and model.
---

# Task types

The validator infers a task type from the target's shape. The task type
selects the model checkpoint and the output form — you never declare it.

| Target shape | Task type | Output |
|---|---|---|
| bare aggregation — `SUM(...)`, `COUNT(...)` | regression | value |
| aggregation vs literal — `COUNT(...) = 0` | binary classification | probability |
| `EXISTS(...)` / `NOT EXISTS(...)` (boolean target) | binary classification | probability |
| `FIRST` / `LAST` / static categorical column | multiclass classification | class + probabilities |
| `LIST_DISTINCT(...) RANK TOP K` | ranking | ranked ID list |
| any target whose window has `HORIZONS > 1` | forecasting | value per horizon |

## Model routing

`ModelConfig` maps task types to checkpoints — by default the classification
family routes to `hf://stanford-star/rt-j/classification` and
regression/forecasting to `hf://stanford-star/rt-j/regression`.

## Checking a query

Every library exposes the inference without executing:

```python
pq = relativedb.parse("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id")
pq.task_type()    # TaskType.REGRESSION
```
