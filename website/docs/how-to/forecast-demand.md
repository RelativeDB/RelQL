---
title: Forecast demand
description: Multi-horizon forecasts with FORECAST N TIMEFRAMES.
---

# Forecast demand

Goal: weekly unit sales per store for the next four weeks.

## The query

`FORECAST N TIMEFRAMES` repeats the target window N times, back to back:

```sql
PREDICT SUM(sales.qty, 0, 7, days) FORECAST 4 TIMEFRAMES
FOR EACH stores.store_id
```

Timeframe 1 covers days (0, 7], timeframe 2 covers (7, 14], and so on.

## Run it

```python
result = engine.execute(ExecutionInput(query=query, anchor_time=t0))
forecasts = {p.id: p.forecast for p in result.predictions}
```

Here `engine` uses the schema and application-owned retrievers wired over your
store and sales data. The result has one prediction per store with four
timeframes. Forecasting routes to the regression checkpoint.

## Notes

- The base window can be any unit: `SUM(usage.count, 0, 1, days) FORECAST 28
  TIMEFRAMES` gives a daily 4-week forecast.
- Backtest by moving `anchor_time` into the past; the engine guarantees each
  forecast only saw data available at that anchor.
- A complete self-checking version lives at
  `examples/industry/bizops_demand_forecast.py`.
