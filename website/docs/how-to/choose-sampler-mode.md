---
title: Choose a sampler mode
description: RETRIEVER vs CSC, with numbers.
---

# Choose a sampler mode

Both modes produce identical contexts. Choose by data locality.

| | RETRIEVER (default) | CSC |
|---|---|---|
| Data location | stays in your store | copied into an in-memory index |
| Freshness | live, per query | snapshot (`engine.refresh()`) |
| Requires | Entity + Link retrievers | `TableScanner` per table |
| Best for | remote, huge, or access-controlled data | repeated low-latency scoring |

## Switching

```python
from relativedb import SamplerMode
df = ds.predict(query, anchor_time=t0, sampler_mode=SamplerMode.CSC)
```

```java
RelativeDbEngine.newEngine(schema, wiring).samplerMode(SamplerMode.CSC).build();
```

## What CSC buys you

On a synthetic churn workload (10,000 customers, 200,000 orders, history
baseline, M-series laptop), scoring every customer:

| Approach | Time | Throughput |
|---|---|---|
| relativedb, CSC sampler | 0.66 s | ~15,000 entities/s |
| naive per-entity pandas loop | 57.4 s | ~174 entities/s |

(Reproduce with `examples/bench_naive_vs_csc.py`.) The CSC index turns each
"latest *w* children ≤ anchor" expansion into a binary search plus a tail
slice, and its build cost is paid once per snapshot.

## Rule of thumb

Start with RETRIEVER. Move to CSC when the same engine scores many queries or
large populations and the data fits in memory.
