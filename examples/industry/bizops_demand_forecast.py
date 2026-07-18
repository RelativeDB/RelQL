"""BizOps: multi-horizon demand forecasting (retail).

predict weekly units
sold per store for the next 4 weeks.

    PREDICT SUM(sales.qty, 0, 7, days)
    FORECAST 4 TIMEFRAMES
    FOR EACH stores.store_id

FORECAST N TIMEFRAMES yields N values, each spaced by the aggregation window
(7 days x 4 = a 28-day outlook). Planted signal: a flagship store sells ~5x
the volume of an outlet store.
"""
import numpy as np
import pandas as pd
from pandas_connector import predictions_frame, wire_pandas_frames
from relativedb import Engine, ExecutionInput, LinkDef, Schema, TableDef, ValueType

rng = np.random.default_rng(23)
ANCHOR = pd.Timestamp("2026-07-01")

stores = pd.DataFrame({
    "store_id": ["S_FLAGSHIP", "S_MALL", "S_OUTLET"],
    "sqft": [12000.0, 6000.0, 2000.0],
    "city": ["NYC", "Austin", "Reno"],
})
DAILY = {"S_FLAGSHIP": 50, "S_MALL": 20, "S_OUTLET": 10}

rows = []
for sid, lam in DAILY.items():
    for d in range(1, 181):                       # 6 months of daily history
        ts = ANCHOR - pd.Timedelta(days=d)
        weekend = 1.5 if ts.dayofweek >= 5 else 1.0
        rows.append((f"X{len(rows):05d}", sid, ts,
                     float(rng.poisson(lam * weekend))))
sales = pd.DataFrame(rows, columns=["sale_id", "store_id", "ts", "qty"])

schema = (Schema.new_schema()
          .table(TableDef.new_table("stores")
                 .column("sqft", ValueType.NUMBER)
                 .column("city", ValueType.TEXT)
                 .primary_key("store_id").build())
          .table(TableDef.new_table("sales")
                 .column("ts", ValueType.DATETIME)
                 .column("qty", ValueType.NUMBER)
                 .primary_key("sale_id").time_column("ts").build())
          .link(LinkDef("sales", "store_id", "stores")).build())
wiring = wire_pandas_frames(schema, {"stores": stores, "sales": sales})
result = Engine(schema, wiring).execute(ExecutionInput(
    query="PREDICT SUM(sales.qty, 0, 7, days) FORECAST 4 TIMEFRAMES "
          "FOR EACH stores.store_id",
    anchor_time=ANCHOR.to_pydatetime()))
df = predictions_frame(result)

print(df.to_string())

# --- checks ---------------------------------------------------------------
by = {r.entity_id: r for r in df.itertuples()}
assert all(len(by[s].forecast) == 4 for s in DAILY), "4 timeframes per store"
flag = np.mean(by["S_FLAGSHIP"].forecast)
outlet = np.mean(by["S_OUTLET"].forecast)
print(f"mean weekly forecast — flagship: {flag:.0f}   outlet: {outlet:.0f}")
assert flag > 3 * outlet, "flagship must forecast well above the outlet"
expected_flag_week = 50 * (5 + 1.5 * 2)          # weekday + weekend uplift
assert 0.5 * expected_flag_week < flag < 1.5 * expected_flag_week, \
    "forecast should be in the plausible weekly range"
print("OK bizops_demand_forecast")
