"""Online Retail II 'buy-it-again' ranking, scored by RT-J, measured against
real future purchases (recall@k / hit-rate) vs a popularity baseline."""
import sys
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from harness import datasets
from relativedb import (Engine, ExecutionInput, RtNativeBackend, SamplerMode,
                        Schema, TableDef, LinkDef, ValueType, ContextPolicy)

ANCHOR = pd.Timestamp("2011-06-01")
WINDOW_DAYS = 30
TOP_PRODUCTS = 300          # candidate universe (bounds cost)
N_CUSTOMERS = 20
K = 20

# ---- load + clean (mirror datasets.online_retail) -----------------------
csv = datasets.CORPUS / "online_retail" / "online_retail_II.csv"
df = pd.read_csv(csv, parse_dates=["InvoiceDate"])
df = df.rename(columns={"Customer ID": "customer_id", "Invoice": "invoice",
                        "StockCode": "stock_code", "Quantity": "quantity",
                        "Price": "price", "InvoiceDate": "ts",
                        "Description": "description"})
df = df[df["customer_id"].notna() & (df["quantity"] > 0) & (df["price"] > 0)]
df = df[~df["invoice"].astype(str).str.startswith("C")]
df["customer_id"] = df["customer_id"].astype("int64")
df["stock_code"] = df["stock_code"].astype(str)
df["amount"] = df["quantity"] * df["price"]
df["line_id"] = np.arange(len(df))

# candidate products = the TOP_PRODUCTS most frequent stock codes
top_codes = df["stock_code"].value_counts().index[:TOP_PRODUCTS].tolist()
top_set = set(top_codes)
desc_of = (df[df.stock_code.isin(top_set)].groupby("stock_code")["description"]
           .agg(lambda s: s.value_counts().index[0]))
products = pd.DataFrame({"stock_code": top_codes,
                         "description": [str(desc_of.get(c, c)) for c in top_codes]})

past = df[df["ts"] <= ANCHOR]
future = df[(df["ts"] > ANCHOR) & (df["ts"] <= ANCHOR + pd.Timedelta(days=WINDOW_DAYS))]
# ground truth per customer: future purchases that are IN the candidate set
future_items = (future[future.stock_code.isin(top_set)]
                .groupby("customer_id")["stock_code"].agg(set))
past_counts = past.groupby("customer_id")["stock_code"].count()

# customers with real history AND a real, in-candidate future to predict
elig = [c for c in future_items.index
        if past_counts.get(c, 0) >= 5 and len(future_items[c]) >= 1]
elig = sorted(elig, key=lambda c: -past_counts.get(c, 0))[:N_CUSTOMERS]
print(f"scoring {len(elig)} customers | {len(products)} candidate products | "
      f"anchor {ANCHOR.date()} | +{WINDOW_DAYS}d window | k={K}")

# ---- schema with a real products table (stock_code as FK) ---------------
schema = (Schema.new_schema()
          .table(TableDef.new_table("customers").primary_key("customer_id").build())
          .table(TableDef.new_table("products")
                 .column("description", ValueType.TEXT)
                 .primary_key("stock_code").build())
          .table(TableDef.new_table("purchases")
                 .column("quantity", ValueType.NUMBER)
                 .column("amount", ValueType.NUMBER)
                 .column("ts", ValueType.DATETIME)
                 .primary_key("line_id").time_column("ts").build())
          .link(LinkDef("purchases", "customer_id", "customers"))
          .link(LinkDef("purchases", "stock_code", "products")).build())

# keep purchases whose product is a candidate (so FK edges resolve)
pur = df[df.stock_code.isin(top_set)]
customers = pd.DataFrame({"customer_id": sorted(set(elig))})
# only load history/purchases for scored customers (bounds memory + cost)
pur = pur[pur.customer_id.isin(set(elig))]
frames = {"customers": customers, "products": products,
          "purchases": pur[["line_id", "customer_id", "stock_code",
                            "quantity", "amount", "ts"]]}
wiring = datasets._wire(schema, frames)
policy = ContextPolicy(max_context_cells=5_000_000, bfs_width=512, max_hops=1)
engine = Engine(schema, wiring,
                model_backend=RtNativeBackend(schema=schema, wiring=wiring),
                sampler_mode=SamplerMode.CSC, context_policy=policy)

# ---- score ranking ------------------------------------------------------
res = engine.execute(ExecutionInput(
    query=f"PREDICT LIST_DISTINCT(purchases.stock_code) OVER ({WINDOW_DAYS} DAYS "
          f"FOLLOWING) RANK TOP {K} FOR EACH customers.customer_id",
    entity_ids=elig, anchor_time=ANCHOR.to_pydatetime()))

pop_topk = top_codes[:K]                      # popularity baseline

def recall_hit(pred, truth):
    if not truth:
        return None, None
    inter = len(set(pred) & truth)
    return inter / len(truth), 1.0 if inter > 0 else 0.0

m_rec, m_hit, p_rec, p_hit, rep = [], [], [], [], []
examples = []
by_id = {p.id: p for p in res.predictions}
for c in elig:
    truth = future_items[c]
    pred = list(by_id[c].ranked) if c in by_id else []
    r, h = recall_hit(pred, truth)
    pr, ph = recall_hit(pop_topk, truth)
    if r is None:
        continue
    m_rec.append(r); m_hit.append(h); p_rec.append(pr); p_hit.append(ph)
    past_items = set(past[past.customer_id == c]["stock_code"]) & top_set
    rep.append(len(truth & past_items) / len(truth))     # repeat-purchase rate
    if len(examples) < 4:
        examples.append((c, pred, truth, past_items))

print(f"\n=== RT-J ranking vs popularity baseline (recall@{K}, hit-rate@{K}) ===")
print(f"  RT-J model : recall {np.mean(m_rec):.3f} | hit-rate {np.mean(m_hit):.3f}")
print(f"  popularity : recall {np.mean(p_rec):.3f} | hit-rate {np.mean(p_hit):.3f}")
print(f"  (avg future basket is {np.mean([len(future_items[c]) for c in elig]):.1f} "
      f"in-candidate items; {np.mean(rep):.0%} were repeat purchases)")

print("\n=== example customers (predicted top-8 vs what they actually bought) ===")
short = lambda code: str(desc_of.get(code, code))[:34]
for c, pred, truth, past_items in examples:
    hits = [p for p in pred if p in truth]
    print(f"\ncustomer {c}: recall {len(set(pred)&truth)}/{len(truth)}, "
          f"{len(hits)} correct in top-{K}")
    print("  predicted top-8: " + " | ".join(short(p) for p in pred[:8]))
    print("  ✓ correctly predicted: " +
          (", ".join(short(h) for h in hits[:6]) if hits else "(none)"))
print("\ndone.")
