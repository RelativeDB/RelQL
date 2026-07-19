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

The output column above is the *logical* form each task produces. The RT-J
backend (`RtNativeBackend`) is the only scoring path. It executes **binary
classification**, **regression**, **multiclass classification**, and
**ranking**. Multiclass reuses the checkpoint's text head: the masked target
cell is predicted as a 384-d embedding and matched by cosine similarity to the
class labels' `all-MiniLM-L12-v2` embeddings, returning a predicted class plus
approximate class probabilities (a softmax over the cosine scores — the argmax
is the reference-exact class, the probabilities are an uncalibrated
approximation, not a trained softmax head). Ranking scores each candidate parent
ID with the existence head and returns the top *k*. `RETURN QUANTILES`/`INTERVAL`
remain unsupported — the checkpoint has no variance/quantile head — and raise a
clear error (see [Model backends](/docs/concepts/model-backends)).

## Checking a query

Every library exposes the inference without executing:

```python
pq = relativedb.parse("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id")
pq.task_type()    # TaskType.REGRESSION
```
