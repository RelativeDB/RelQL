"""Give RT-J the SAME 11 RFM features the GBDT used — as cells on the customers
table (computed as-of the anchor) — and re-measure churn AUC / spend Spearman.
Question: does hand-engineering features into the schema help the zero-shot model?"""
import sys
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from harness import datasets
from relativedb import (Engine, ExecutionInput, RtNativeBackend, SamplerMode,
                        Schema, TableDef, LinkDef, ValueType, ContextPolicy)

FWD, BACK = 30, 90
ANCHOR = pd.Timestamp("2011-06-01")

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

# same 300 test customers as before (seed 0)
active = sorted(df[(df.ts > ANCHOR - pd.Timedelta(days=BACK)) & (df.ts <= ANCHOR)]
                .customer_id.unique())
sample = sorted(np.random.default_rng(0).choice(active, size=min(300, len(active)),
                                                 replace=False).tolist())

# ---- RFM features as-of the anchor (descriptive names -> the model embeds them) ---
past = df[df.ts <= ANCHOR]
w90 = past[past.ts > ANCHOR - pd.Timedelta(days=BACK)]
w30 = past[past.ts > ANCHOR - pd.Timedelta(days=FWD)]
gA, g9 = past.groupby("customer_id"), w90.groupby("customer_id")
last, first = gA["ts"].max(), gA["ts"].min()
inv_all, mon_all = gA["invoice"].nunique(), gA["amount"].sum()
inv90, mon90, lines90, qty90 = (g9["invoice"].nunique(), g9["amount"].sum(),
                                g9.size(), g9["quantity"].sum())
mon30, dprod = w30.groupby("customer_id")["amount"].sum(), gA["stock_code"].nunique()
rows = []
for c in sample:
    ia = int(inv_all.get(c, 0))
    rows.append({"customer_id": c,
        "days_since_last_purchase": float((ANCHOR - last.get(c, ANCHOR)).days),
        "invoices_last_90d": float(inv90.get(c, 0)),
        "invoices_total": float(ia),
        "spend_last_30d": float(mon30.get(c, 0.0)),
        "spend_last_90d": float(mon90.get(c, 0.0)),
        "spend_total": float(mon_all.get(c, 0.0)),
        "tenure_days": float((ANCHOR - first.get(c, ANCHOR)).days),
        "lines_last_90d": float(lines90.get(c, 0)),
        "distinct_products": float(dprod.get(c, 0)),
        "avg_invoice_value": float(mon_all.get(c, 0.0) / ia) if ia else 0.0,
        "qty_last_90d": float(qty90.get(c, 0))})
customers = pd.DataFrame(rows)
RFM = [c for c in customers.columns if c != "customer_id"]

# ---- schema: customers now CARRY the RFM feature cells ------------------
ct = TableDef.new_table("customers")
for f in RFM:
    ct = ct.column(f, ValueType.NUMBER)
schema = (Schema.new_schema()
          .table(ct.primary_key("customer_id").build())
          .table(TableDef.new_table("purchases")
                 .column("quantity", ValueType.NUMBER).column("amount", ValueType.NUMBER)
                 .column("ts", ValueType.DATETIME)
                 .primary_key("line_id").time_column("ts").build())
          .link(LinkDef("purchases", "customer_id", "customers")).build())
pur = df[df.customer_id.isin(set(sample))]
frames = {"customers": customers,
          "purchases": pur[["line_id", "customer_id", "quantity", "amount", "ts"]]}
wiring = datasets._wire(schema, frames)
policy = ContextPolicy(max_context_cells=20_000_000, bfs_width=8192, max_hops=1)
engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema),
                sampler_mode=SamplerMode.CSC, context_policy=policy)

fut = df[(df.ts > ANCHOR) & (df.ts <= ANCHOR + pd.Timedelta(days=FWD))]
buyers = set(fut.customer_id.unique())
fut_spend = fut.groupby("customer_id")["amount"].sum()

def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    order = np.argsort(s); r = np.empty_like(order, float); r[order] = np.arange(1, len(s)+1)
    npos, nneg = y.sum(), (1-y).sum()
    return (r[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg)
def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])
def pval(p):
    for at in ("value", "prediction", "expected_value"):
        if getattr(p, at, None) is not None: return float(getattr(p, at))
    return float(getattr(p, "probability"))

print(f"RT-J + {len(RFM)} RFM feature cells | {len(sample)} customers @ {ANCHOR.date()}\n")
r1 = engine.execute(ExecutionInput(
    query=f"PREDICT NOT EXISTS(purchases.*) OVER ({FWD} DAYS FOLLOWING) "
          f"FOR EACH customers.customer_id WHERE EXISTS(purchases.*) OVER ({BACK} DAYS PRECEDING)",
    entity_ids=sample, anchor_time=ANCHOR.to_pydatetime()))
ids = [p.id for p in r1.predictions]
churn_p = np.array([p.probability for p in r1.predictions])
churned = np.array([0 if i in buyers else 1 for i in ids])
a = auc(churn_p, churned)

r2 = engine.execute(ExecutionInput(
    query=f"PREDICT SUM(purchases.amount) OVER ({FWD} DAYS FOLLOWING) FOR EACH customers.customer_id",
    entity_ids=sample, anchor_time=ANCHOR.to_pydatetime()))
ids2 = [p.id for p in r2.predictions]
ps = np.array([pval(p) for p in r2.predictions]); ts = np.array([float(fut_spend.get(i,0.0)) for i in ids2])
sp, mae = spearman(ps, ts), float(np.abs(ps-ts).mean())

print("             CHURN AUC   SPEND Spearman / MAE")
print(f"  RT-J bare :   0.607        0.291 / £287")
print(f"  RT-J +RFM :   {a:.3f}        {sp:.3f} / £{mae:.0f}")
print(f"  GBDT      :   0.757        0.431 / £219")
print(f"\n  gain from RFM cells: AUC {a-0.607:+.3f} | Spearman {sp-0.291:+.3f}")
print(f"  remaining gap to GBDT: AUC {0.757-a:+.3f} | Spearman {0.431-sp:+.3f}")
print("done.")
