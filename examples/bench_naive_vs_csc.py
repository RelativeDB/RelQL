"""Benchmark: relativedb (CSC sampler) vs a naive per-entity pandas loop.

Both compute the same 90-day churn scores (history-baseline) for every
customer on a synthetic dataset. The naive loop re-filters the orders frame
per customer per trailing window — the way a hand-rolled script would.

Run:  python/.venv/bin/python examples/bench_naive_vs_csc.py [n_customers]
"""
import sys
import time

import numpy as np
import pandas as pd

from industry.pandas_connector import predictions_frame, wire_pandas_frames
from relativedb import (Engine, ExecutionInput, LinkDef, SamplerMode, Schema,
                        TableDef, ValueType)

N_CUSTOMERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
ORDERS_PER_CUSTOMER = 20
ANCHOR = pd.Timestamp("2026-07-01")

rng = np.random.default_rng(7)

customers = pd.DataFrame({
    "customer_id": [f"C{i}" for i in range(N_CUSTOMERS)],
    "age": rng.integers(18, 80, N_CUSTOMERS),
    "signup_date": ANCHOR - pd.to_timedelta(rng.integers(100, 1000, N_CUSTOMERS), unit="D"),
})
n_orders = N_CUSTOMERS * ORDERS_PER_CUSTOMER
orders = pd.DataFrame({
    "order_id": [f"O{i}" for i in range(n_orders)],
    "customer_id": rng.choice(customers.customer_id, n_orders),
    "qty": rng.integers(1, 5, n_orders),
    "order_date": ANCHOR - pd.to_timedelta(rng.integers(0, 720, n_orders), unit="D"),
})

QUERY = "PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id"

# --- relationdb, with an application-owned pandas connector -----------------
schema = (Schema.new_schema()
          .table(TableDef.new_table("customers")
                 .column("age", ValueType.NUMBER)
                 .column("signup_date", ValueType.DATETIME)
                 .primary_key("customer_id").build())
          .table(TableDef.new_table("orders")
                 .column("qty", ValueType.NUMBER)
                 .column("order_date", ValueType.DATETIME)
                 .primary_key("order_id").time_column("order_date").build())
          .link(LinkDef("orders", "customer_id", "customers")).build())
wiring = wire_pandas_frames(schema, {"customers": customers, "orders": orders})
t0 = time.perf_counter()
result = Engine(schema, wiring, sampler_mode=SamplerMode.CSC).execute(
    ExecutionInput(query=QUERY, anchor_time=ANCHOR.to_pydatetime()))
df = predictions_frame(result)
t_relativedb = time.perf_counter() - t0

# --- naive pandas loop -------------------------------------------------------
# Same semantics as the history baseline: for each customer, evaluate the
# target over trailing 90-day windows of their own history and average.
t0 = time.perf_counter()
naive = {}
window = pd.Timedelta(days=90)
for cid in customers.customer_id:
    hist = orders[(orders.customer_id == cid) & (orders.order_date <= ANCHOR)]
    outcomes = []
    end = ANCHOR
    for _ in range(4):  # 4 trailing self-label windows
        start = end - window
        outcomes.append(int(((hist.order_date > start) & (hist.order_date <= end)).sum() == 0))
        end = start
    naive[cid] = float(np.mean(outcomes))
t_naive = time.perf_counter() - t0

print(f"entities              : {N_CUSTOMERS:,} customers, {n_orders:,} orders")
print(f"relativedb (CSC)      : {t_relativedb:8.2f} s   ({N_CUSTOMERS / t_relativedb:,.0f} entities/s)")
print(f"naive pandas loop     : {t_naive:8.2f} s   ({N_CUSTOMERS / t_naive:,.0f} entities/s)")
print(f"speedup               : {t_naive / t_relativedb:6.1f}x")
