"""GBDT (sklearn HistGradientBoosting — XGBoost-family) with hand-built RFM
features vs RT-J zero-shot, SAME churn/spend tasks, SAME test customers.
GBDT is SUPERVISED (trains on earlier monthly cohorts); RT-J saw no training."""
import sys
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from harness import datasets
import xgboost as xgb
from sklearn.metrics import roc_auc_score

FWD, BACK = 30, 90
TEST_ANCHOR = pd.Timestamp("2011-06-01")
TRAIN_ANCHORS = [pd.Timestamp(d) for d in
                 ["2010-12-01", "2011-01-01", "2011-02-01", "2011-03-01", "2011-04-01"]]

csv = datasets.CORPUS / "online_retail" / "online_retail_II.csv"
df = pd.read_csv(csv, parse_dates=["InvoiceDate"])
df = df.rename(columns={"Customer ID": "customer_id", "Invoice": "invoice",
                        "StockCode": "stock_code", "Quantity": "quantity",
                        "Price": "price", "InvoiceDate": "ts"})
df = df[df["customer_id"].notna() & (df["quantity"] > 0) & (df["price"] > 0)]
df = df[~df["invoice"].astype(str).str.startswith("C")]
df["customer_id"] = df["customer_id"].astype("int64")
df["amount"] = df["quantity"] * df["price"]

FEATS = ["recency", "freq_inv_90", "freq_inv_all", "mon_30", "mon_90", "mon_all",
         "tenure", "n_lines_90", "distinct_prod", "avg_inv", "qty_90"]

def build(anchor, ids=None):
    past = df[df.ts <= anchor]
    w90 = past[past.ts > anchor - pd.Timedelta(days=BACK)]
    w30 = past[past.ts > anchor - pd.Timedelta(days=FWD)]
    if ids is None:
        ids = sorted(w90.customer_id.unique())           # active in prior 90d
    g_all = past.groupby("customer_id")
    g90 = w90.groupby("customer_id")
    rows = []
    last = g_all["ts"].max(); first = g_all["ts"].min()
    inv_all = g_all["invoice"].nunique(); mon_all = g_all["amount"].sum()
    inv90 = g90["invoice"].nunique(); mon90 = g90["amount"].sum()
    lines90 = g90.size(); qty90 = g90["quantity"].sum()
    mon30 = w30.groupby("customer_id")["amount"].sum()
    dp = g_all["stock_code"].nunique()
    for c in ids:
        ia = inv_all.get(c, 0)
        rows.append([
            (anchor - last.get(c, anchor - pd.Timedelta(days=BACK))).days,
            inv90.get(c, 0), ia, mon30.get(c, 0.0), mon90.get(c, 0.0), mon_all.get(c, 0.0),
            (anchor - first.get(c, anchor)).days, lines90.get(c, 0), dp.get(c, 0),
            (mon_all.get(c, 0.0) / ia) if ia else 0.0, qty90.get(c, 0)])
    X = pd.DataFrame(rows, columns=FEATS, index=ids)
    fut = df[(df.ts > anchor) & (df.ts <= anchor + pd.Timedelta(days=FWD))]
    fut_spend = fut.groupby("customer_id")["amount"].sum()
    buyers = set(fut.customer_id.unique())
    churn = pd.Series([0 if c in buyers else 1 for c in ids], index=ids)
    spend = pd.Series([float(fut_spend.get(c, 0.0)) for c in ids], index=ids)
    return X, churn, spend

# --- training pool (earlier cohorts) ------------------------------------
Xtr, ytr_churn, ytr_spend = [], [], []
for a in TRAIN_ANCHORS:
    X, ch, sp = build(a)
    Xtr.append(X); ytr_churn.append(ch); ytr_spend.append(sp)
Xtr = pd.concat(Xtr); ytr_churn = pd.concat(ytr_churn); ytr_spend = pd.concat(ytr_spend)

# --- test set: the SAME 300 customers RT-J used (seed 0) ----------------
active_test = sorted(df[(df.ts > TEST_ANCHOR - pd.Timedelta(days=BACK)) &
                        (df.ts <= TEST_ANCHOR)].customer_id.unique())
rng = np.random.default_rng(0)
test_ids = sorted(rng.choice(active_test, size=min(300, len(active_test)),
                             replace=False).tolist())
Xte, yte_churn, yte_spend = build(TEST_ANCHOR, ids=test_ids)
print(f"train {len(Xtr)} cust-anchor rows ({len(TRAIN_ANCHORS)} cohorts) | "
      f"test {len(Xte)} customers @ {TEST_ANCHOR.date()}\n")

def spearman(a, b):
    ra = pd.Series(np.asarray(a, float)).rank(); rb = pd.Series(np.asarray(b, float)).rank()
    return float(np.corrcoef(ra, rb)[0, 1])

# --- churn (classification) ---------------------------------------------
clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=0)
clf.fit(Xtr, ytr_churn)
p_churn = clf.predict_proba(Xte)[:, 1]
gbdt_auc = roc_auc_score(yte_churn, p_churn)

# --- spend (regression, log1p target) -----------------------------------
reg = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.8, random_state=0)
reg.fit(Xtr, np.log1p(ytr_spend))
pred_spend = np.expm1(reg.predict(Xte))
gbdt_sp = spearman(pred_spend, yte_spend.values)
gbdt_mae = float(np.abs(pred_spend - yte_spend.values).mean())

RTJ = {"auc": 0.607, "sp": 0.291, "mae": 287.0}
print(f"[xgboost {xgb.__version__}]")
print("=== CHURN (binary) — AUC ===")
print(f"  XGBoost (supervised, RFM feats): {gbdt_auc:.3f}")
print(f"  RT-J (zero-shot)               : {RTJ['auc']:.3f}")
print(f"  diff (XGB - RT-J)              : {gbdt_auc - RTJ['auc']:+.3f}")
print("\n=== SPEND (regression) — Spearman / MAE ===")
print(f"  XGBoost : Spearman {gbdt_sp:.3f} | MAE £{gbdt_mae:.0f}")
print(f"  RT-J    : Spearman {RTJ['sp']:.3f} | MAE £{RTJ['mae']:.0f}")
print(f"  diff    : Spearman {gbdt_sp - RTJ['sp']:+.3f} | MAE £{gbdt_mae - RTJ['mae']:+.0f}")
imp = pd.Series(clf.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\ntop-5 churn features (XGBoost gain):")
for f, v in imp.head(5).items():
    print(f"    {f:24} {v:.3f}")
print("done.")
