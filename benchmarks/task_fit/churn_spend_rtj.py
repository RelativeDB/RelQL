"""Online Retail II — the tasks RT-J was actually trained for:
  (1) 30-day churn      : binary classification, measured by AUC vs a recency baseline
  (2) 30-day spend      : regression, measured by Spearman vs a persistence baseline
Ground truth = real future purchases. Temporal-correct (retrievers honor the bound)."""
import sys
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from harness import datasets
from relativedb import (Engine, ExecutionInput, RtNativeBackend, SamplerMode,
                        Schema, TableDef, LinkDef, ValueType, ContextPolicy)

ANCHOR = pd.Timestamp("2011-06-01")
FWD, BACK = 30, 90
N = 300

csv = datasets.CORPUS / "online_retail" / "online_retail_II.csv"
df = pd.read_csv(csv, parse_dates=["InvoiceDate"])
df = df.rename(columns={"Customer ID": "customer_id", "Invoice": "invoice",
                        "StockCode": "stock_code", "Quantity": "quantity",
                        "Price": "price", "InvoiceDate": "ts"})
df = df[df["customer_id"].notna() & (df["quantity"] > 0) & (df["price"] > 0)]
df = df[~df["invoice"].astype(str).str.startswith("C")]
df["customer_id"] = df["customer_id"].astype("int64")
df["amount"] = df["quantity"] * df["price"]
df["line_id"] = np.arange(len(df))

t0, tb, tf = ANCHOR, ANCHOR - pd.Timedelta(days=BACK), ANCHOR + pd.Timedelta(days=FWD)
active = df[(df.ts > tb) & (df.ts <= t0)].groupby("customer_id")
active_ids = list(active.groups)
rng = np.random.default_rng(0)
sample = sorted(rng.choice(active_ids, size=min(N, len(active_ids)), replace=False).tolist())

fut = df[(df.ts > t0) & (df.ts <= tf)]
fut_spend = fut.groupby("customer_id")["amount"].sum()
fut_buyers = set(fut.customer_id.unique())
last_purchase = df[df.ts <= t0].groupby("customer_id")["ts"].max()
prior_spend = (df[(df.ts > t0 - pd.Timedelta(days=FWD)) & (df.ts <= t0)]
               .groupby("customer_id")["amount"].sum())

# --- schema: customers + purchases (no products needed) ------------------
schema = (Schema.new_schema()
          .table(TableDef.new_table("customers").primary_key("customer_id").build())
          .table(TableDef.new_table("purchases")
                 .column("quantity", ValueType.NUMBER)
                 .column("amount", ValueType.NUMBER)
                 .column("ts", ValueType.DATETIME)
                 .primary_key("line_id").time_column("ts").build())
          .link(LinkDef("purchases", "customer_id", "customers")).build())
pur = df[df.customer_id.isin(set(sample))]
frames = {"customers": pd.DataFrame({"customer_id": sample}),
          "purchases": pur[["line_id", "customer_id", "quantity", "amount", "ts"]]}
wiring = datasets._wire(schema, frames)
policy = ContextPolicy(max_context_cells=20_000_000, bfs_width=16384, max_hops=1)
engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema),
                sampler_mode=SamplerMode.CSC, context_policy=policy)

def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(s); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0: return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])

def pval(p):
    for at in ("value", "prediction", "expected_value"):
        if getattr(p, at, None) is not None: return float(getattr(p, at))
    return float(getattr(p, "probability"))

# ================= (1) CHURN — binary classification =====================
print(f"anchor {ANCHOR.date()} | {len(sample)} active customers | "
      f"churn = no purchase in next {FWD}d\n")
res = engine.execute(ExecutionInput(
    query=f"PREDICT NOT EXISTS(purchases.*) OVER ({FWD} DAYS FOLLOWING) "
          f"FROM customers WHERE customers.customer_id IN :ids "
          f"WHERE EXISTS(purchases.*) OVER ({BACK} DAYS PRECEDING)",
    params={"ids": sample}, anchor_time=ANCHOR.to_pydatetime()))
ids = [p.id for p in res.predictions]
churn_prob = np.array([p.probability for p in res.predictions])
churned = np.array([0 if i in fut_buyers else 1 for i in ids])   # 1 = churned
# recency baseline: more days since last purchase -> more likely churn
recency = np.array([(t0 - last_purchase.get(i, tb)).days for i in ids], float)
print("=== (1) CHURN  (binary classification) ===")
print(f"  base churn rate      : {churned.mean():.1%}  ({churned.sum()}/{len(churned)})")
print(f"  RT-J model     AUC   : {auc(churn_prob, churned):.3f}")
print(f"  recency baseline AUC : {auc(recency, churned):.3f}")

# ================= (2) SPEND — regression ================================
res2 = engine.execute(ExecutionInput(
    query=f"PREDICT SUM(purchases.amount) OVER ({FWD} DAYS FOLLOWING) "
          f"FROM customers WHERE customers.customer_id IN :ids",
    params={"ids": sample}, anchor_time=ANCHOR.to_pydatetime()))
ids2 = [p.id for p in res2.predictions]
pred_spend = np.array([pval(p) for p in res2.predictions])
true_spend = np.array([float(fut_spend.get(i, 0.0)) for i in ids2])
base_spend = np.array([float(prior_spend.get(i, 0.0)) for i in ids2])  # persistence
print("\n=== (2) SPEND next 30d  (regression) ===")
print(f"  mean actual spend        : £{true_spend.mean():.0f}")
print(f"  RT-J model     Spearman  : {spearman(pred_spend, true_spend):.3f} | "
      f"MAE £{np.abs(pred_spend - true_spend).mean():.0f}")
print(f"  persistence baseline     : {spearman(base_spend, true_spend):.3f} | "
      f"MAE £{np.abs(base_spend - true_spend).mean():.0f}")
print("\ndone.")
