"""Growth: subscription/inactivity churn (streaming service).

Predict users unlikely to be
active in the next 30 days, restricted to users who WERE active in the last
90 (no point re-engaging users who already left).

    PREDICT COUNT(events.*, 0, 30, days) = 0
    FOR EACH users.user_id
    WHERE COUNT(events.*, -90, 0, days) > 0

Synthetic data plants the signal: "engaged" users stream weekly right up to
the anchor; "fading" users stopped ~6 weeks ago.
"""
import numpy as np
import pandas as pd
from pandas_connector import predictions_frame, wire_pandas_frames
from relativedb import Engine, ExecutionInput, LinkDef, Schema, TableDef, ValueType

rng = np.random.default_rng(7)
ANCHOR = pd.Timestamp("2026-07-01")

n_users = 40
users = pd.DataFrame({
    "user_id": [f"U{i:03d}" for i in range(n_users)],
    "age": rng.integers(18, 70, n_users),
    "plan": rng.choice(["basic", "standard", "premium"], n_users),
    "signup_date": pd.to_datetime("2025-06-01")
    + pd.to_timedelta(rng.integers(0, 300, n_users), unit="D"),
})

# First half engaged (stream weekly until the anchor), second half fading
# (last activity ~45 days before the anchor), last 5 fully churned long ago
# (filtered out by the WHERE clause).
rows = []
for i in range(n_users):
    if i < 20:                      # engaged
        last, cadence, n = 2, 7, 26
    elif i < 35:                    # fading
        last, cadence, n = 45, 7, 12
    else:                           # long gone (inactive > 90 days)
        last, cadence, n = 120, 7, 8
    for k in range(n):
        ts = ANCHOR - pd.Timedelta(days=last + k * cadence + int(rng.integers(0, 3)))
        rows.append((f"E{len(rows):05d}", users.user_id[i], ts,
                     float(rng.integers(10, 120))))
events = pd.DataFrame(rows, columns=["event_id", "user_id", "ts", "minutes"])

schema = (Schema.new_schema()
          .table(TableDef.new_table("users")
                 .column("age", ValueType.NUMBER)
                 .column("plan", ValueType.TEXT)
                 .column("signup_date", ValueType.DATETIME)
                 .primary_key("user_id").build())
          .table(TableDef.new_table("events")
                 .column("ts", ValueType.DATETIME)
                 .column("minutes", ValueType.NUMBER)
                 .primary_key("event_id").time_column("ts").build())
          .link(LinkDef("events", "user_id", "users")).build())
wiring = wire_pandas_frames(schema, {"users": users, "events": events})
result = Engine(schema, wiring).execute(ExecutionInput(
    query="PREDICT COUNT(events.*, 0, 30, days) = 0 "
          "FOR EACH users.user_id "
          "WHERE COUNT(events.*, -90, 0, days) > 0",
    anchor_time=ANCHOR.to_pydatetime()))
df = predictions_frame(result)

df = df.sort_values("probability", ascending=False).reset_index(drop=True)
print(df.head(8).to_string())
print(f"...{len(df)} users scored (long-inactive users excluded by WHERE)")

# --- checks ---------------------------------------------------------------
assert len(df) == 35, "WHERE should keep only users active in the last 90d"
engaged = df[df.entity_id.str[1:].astype(int) < 20].probability.mean()
fading = df[df.entity_id.str[1:].astype(int).between(20, 34)].probability.mean()
print(f"mean churn risk — engaged: {engaged:.2f}   fading: {fading:.2f}")
assert fading > engaged + 0.2, "fading users must score clearly riskier"
print("OK growth_churn")
