"""Fraud: chargeback-risk scoring (payments platform).

score accounts by the
risk of a chargeback in the next 60 days, using transaction volume and prior
chargeback history in the relational graph.

    PREDICT COUNT(chargebacks.*, 0, 60, days) > 0
    FOR EACH accounts.account_id

Planted signal: "abuser" accounts have a history of periodic chargebacks
alongside bursty high-value transactions; "clean" accounts transact normally
and have none.
"""
import numpy as np
import pandas as pd
import relativedb

rng = np.random.default_rng(11)
ANCHOR = pd.Timestamp("2026-07-01")

n_acct = 30
accounts = pd.DataFrame({
    "account_id": [f"A{i:03d}" for i in range(n_acct)],
    "country": rng.choice(["US", "DE", "BR"], n_acct),
    "created_at": pd.to_datetime("2025-01-01")
    + pd.to_timedelta(rng.integers(0, 400, n_acct), unit="D"),
})

txn_rows, cb_rows = [], []
for i in range(n_acct):
    abuser = i < 8                                   # A000–A007 planted risky
    n_txn = int(rng.integers(20, 40))
    for _ in range(n_txn):
        ts = ANCHOR - pd.Timedelta(days=int(rng.integers(1, 180)))
        amount = float(rng.gamma(2.0, 120.0 if abuser else 30.0))
        txn_rows.append((f"T{len(txn_rows):05d}", accounts.account_id[i], ts, amount))
    if abuser:                                       # a chargeback every ~6 weeks
        for k in range(4):
            ts = ANCHOR - pd.Timedelta(days=20 + 42 * k + int(rng.integers(0, 10)))
            cb_rows.append((f"C{len(cb_rows):04d}", accounts.account_id[i], ts,
                            float(rng.gamma(2.0, 150.0))))

transactions = pd.DataFrame(txn_rows, columns=["txn_id", "account_id", "ts", "amount"])
chargebacks = pd.DataFrame(cb_rows, columns=["cb_id", "account_id", "ts", "amount"])

ds = relativedb.from_dataframes(
    {"accounts": accounts, "transactions": transactions, "chargebacks": chargebacks},
    links=[("transactions", "account_id", "accounts"),
           ("chargebacks", "account_id", "accounts")])

df = ds.predict(
    "PREDICT COUNT(chargebacks.*, 0, 60, days) > 0 FOR EACH accounts.account_id",
    anchor_time=ANCHOR)

df = df.sort_values("probability", ascending=False).reset_index(drop=True)
print(df.head(10).to_string())

# --- checks ---------------------------------------------------------------
top8 = set(df.head(8).entity_id)
planted = {f"A{i:03d}" for i in range(8)}
overlap = len(top8 & planted)
print(f"planted abusers recovered in top-8: {overlap}/8")
assert overlap >= 6, "risk ranking must surface the planted abuser accounts"
clean_mean = df[~df.entity_id.isin(planted)].probability.mean()
assert clean_mean < 0.1, "clean accounts should score near zero"
print("OK fraud_chargeback")
