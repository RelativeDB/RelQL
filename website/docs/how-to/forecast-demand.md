---
title: Forecast demand
description: Multi-horizon forecasts with OVER (... HORIZONS N) windows.
---

# Forecast demand

Goal: weekly unit sales per store for the next four weeks.

## The query

A window with a `HORIZONS N` clause repeats the frame N times, back to back,
and asks the engine for one value per horizon. A multi-horizon window *implies*
forecasting — there is no separate clause:

```sql
PREDICT SUM(sales.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)
FOR EACH stores.store_id
```

Horizon 1 covers days (0, 7], horizon 2 covers (7, 14], and so on. With no
`STEP`, each horizon steps forward by the frame width (7 days here), so the
four horizons tile the next 28 days without overlap.

## Run it

```python
result = engine.execute(ExecutionInput(query=query, anchor_time=t0))
forecasts = {p.id: p.forecast for p in result.predictions}
```

Here `engine` uses the schema and application-owned retrievers wired over your
store and sales data. The result has one prediction per store with four
horizons. A window whose `HORIZONS > 1` routes to the regression checkpoint.

## Overlapping horizons with STEP

`STEP` sets how far each horizon advances; it defaults to the frame width. Make
`STEP` smaller than the frame to get overlapping, rolling windows — for
example a 30-day trailing demand projection re-issued every 7 days, six times:

```sql
PREDICT SUM(sales.qty) OVER (30 DAYS FOLLOWING HORIZONS 6 STEP 7 DAYS)
FOR EACH stores.store_id
```

Each of the six horizons still spans 30 days, but their start points are only
7 days apart, so consecutive horizons overlap.

## Notes

- The base frame can be any unit: `SUM(usage.count) OVER (1 DAY FOLLOWING
  HORIZONS 28)` gives a daily 4-week forecast.
- Pin the prediction time with `AS OF` and pick an output shape with `RETURN`,
  e.g. `... AS OF :prediction_time RETURN EXPECTED VALUE`.
- Backtest by moving `anchor_time` (or `AS OF`) into the past; the engine
  guarantees each horizon only saw data available at that anchor.
- A complete self-checking version lives at
  `examples/industry/bizops_demand_forecast.py`.
