"""Personalization: buy-it-again recommendations (grocery).

rank the products each
customer is most likely to purchase in the next 30 days. Uses the canonical
Kumo recommendation pattern — LIST_DISTINCT over a *foreign key* with
RANK TOP K (FK values reach the ranker through Row.parents, never as cells).

    PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 3
    FOR EACH customers.customer_id

Planted signal: each customer has a habitual staple (bought weekly), a
secondary item (bought monthly), and one-off purchases.
"""
import numpy as np
import pandas as pd
from pandas_connector import predictions_frame, wire_pandas_frames
from relativedb import Engine, ExecutionInput, LinkDef, Schema, TableDef, ValueType

rng = np.random.default_rng(42)
ANCHOR = pd.Timestamp("2026-07-01")

products = pd.DataFrame({
    "product_id": ["P_COFFEE", "P_MILK", "P_BREAD", "P_CHEESE", "P_WINE", "P_SOAP"],
    "price": [12.0, 3.5, 4.0, 8.0, 15.0, 5.0],
    "category": ["pantry", "dairy", "bakery", "dairy", "alcohol", "household"],
})

staples = {"C_ALICE": "P_COFFEE", "C_BOB": "P_MILK", "C_CARA": "P_BREAD"}
secondary = {"C_ALICE": "P_MILK", "C_BOB": "P_BREAD", "C_CARA": "P_CHEESE"}
customers = pd.DataFrame({
    "customer_id": list(staples),
    "member_since": pd.to_datetime(["2024-05-01", "2025-01-15", "2025-11-02"]),
})

rows = []
for cid in staples:
    for w in range(16):                                   # weekly staple
        ts = ANCHOR - pd.Timedelta(days=3 + 7 * w + int(rng.integers(0, 2)))
        rows.append((f"O{len(rows):05d}", cid, staples[cid], ts, 1.0))
    for m in range(4):                                    # monthly secondary
        ts = ANCHOR - pd.Timedelta(days=10 + 30 * m)
        rows.append((f"O{len(rows):05d}", cid, secondary[cid], ts, 1.0))
    oneoff = rng.choice(products.product_id, 2, replace=False)   # noise
    for p in oneoff:
        ts = ANCHOR - pd.Timedelta(days=int(rng.integers(40, 170)))
        rows.append((f"O{len(rows):05d}", cid, p, ts, 1.0))
orders = pd.DataFrame(rows, columns=["order_id", "customer_id", "product_id",
                                     "ts", "qty"])

schema = (Schema.new_schema()
          .table(TableDef.new_table("customers")
                 .column("member_since", ValueType.DATETIME)
                 .primary_key("customer_id").build())
          .table(TableDef.new_table("products")
                 .column("price", ValueType.NUMBER)
                 .column("category", ValueType.TEXT)
                 .primary_key("product_id").build())
          .table(TableDef.new_table("orders")
                 .column("ts", ValueType.DATETIME)
                 .column("qty", ValueType.NUMBER)
                 .primary_key("order_id").time_column("ts").build())
          .link(LinkDef("orders", "customer_id", "customers"))
          .link(LinkDef("orders", "product_id", "products")).build())
wiring = wire_pandas_frames(schema, {
    "customers": customers, "products": products, "orders": orders,
})
result = Engine(schema, wiring).execute(ExecutionInput(
    query="PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 3 "
          "FOR EACH customers.customer_id",
    anchor_time=ANCHOR.to_pydatetime()))
df = predictions_frame(result)

print(df.to_string())

# --- checks ---------------------------------------------------------------
ranked = {r.entity_id: list(r.ranked) for r in df.itertuples()}
for cid, staple in staples.items():
    assert ranked[cid], f"{cid}: empty recommendation list"
    assert ranked[cid][0] == staple, \
        f"{cid}: habitual staple {staple} must rank first, got {ranked[cid]}"
    assert secondary[cid] in ranked[cid][:3], \
        f"{cid}: secondary item should appear in top 3"
print("OK pzn_buy_it_again")
