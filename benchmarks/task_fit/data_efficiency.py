"""Where RT-J earns its keep: data efficiency. XGBoost (supervised, RFM feats)
AUC as a function of #labeled training examples, vs RT-J's zero-shot 0.607.
How many labels is the zero-shot foundation model worth?"""
import sys, pathlib
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from harness import datasets
import xgboost as xgb
from sklearn.metrics import roc_auc_score

FWD, BACK = 30, 90
TEST_ANCHOR = pd.Timestamp("2011-06-01")
TRAIN_ANCHORS = [pd.Timestamp(d) for d in
                 ["2010-12-01","2011-01-01","2011-02-01","2011-03-01","2011-04-01"]]
RTJ_CHURN_AUC = 0.607   # measured earlier, zero-shot

csv = datasets.CORPUS / "online_retail" / "online_retail_II.csv"
df = pd.read_csv(csv, parse_dates=["InvoiceDate"])
df = df.rename(columns={"Customer ID":"customer_id","Invoice":"invoice","StockCode":"stock_code",
                        "Quantity":"quantity","Price":"price","InvoiceDate":"ts"})
df = df[df.customer_id.notna() & (df.quantity>0) & (df.price>0)]
df = df[~df.invoice.astype(str).str.startswith("C")]
df["customer_id"]=df.customer_id.astype("int64"); df["amount"]=df.quantity*df.price
FEATS=["recency","freq_inv_90","freq_inv_all","mon_30","mon_90","mon_all","tenure",
       "n_lines_90","distinct_prod","avg_inv","qty_90"]

def build(anchor, ids=None):
    past=df[df.ts<=anchor]; w90=past[past.ts>anchor-pd.Timedelta(days=BACK)]
    w30=past[past.ts>anchor-pd.Timedelta(days=FWD)]
    if ids is None: ids=sorted(w90.customer_id.unique())
    gA,g9=past.groupby("customer_id"),w90.groupby("customer_id")
    last,first=gA.ts.max(),gA.ts.min(); ia=gA.invoice.nunique(); ma=gA.amount.sum()
    i9,m9,l9,q9=g9.invoice.nunique(),g9.amount.sum(),g9.size(),g9.quantity.sum()
    m3=w30.groupby("customer_id").amount.sum(); dp=gA.stock_code.nunique()
    rows=[[ (anchor-last.get(c,anchor)).days,i9.get(c,0),ia.get(c,0),m3.get(c,0.0),m9.get(c,0.0),
            ma.get(c,0.0),(anchor-first.get(c,anchor)).days,l9.get(c,0),dp.get(c,0),
            (ma.get(c,0.0)/ia.get(c,1)) if ia.get(c,0) else 0.0,q9.get(c,0)] for c in ids]
    X=pd.DataFrame(rows,columns=FEATS,index=ids)
    fut=df[(df.ts>anchor)&(df.ts<=anchor+pd.Timedelta(days=FWD))]; buyers=set(fut.customer_id.unique())
    churn=pd.Series([0 if c in buyers else 1 for c in ids],index=ids)
    return X,churn

Xtr,ytr=[],[]
for a in TRAIN_ANCHORS:
    X,c=build(a); Xtr.append(X); ytr.append(c)
Xtr,ytr=pd.concat(Xtr).reset_index(drop=True),pd.concat(ytr).reset_index(drop=True)
active=sorted(df[(df.ts>TEST_ANCHOR-pd.Timedelta(days=BACK))&(df.ts<=TEST_ANCHOR)].customer_id.unique())
test_ids=sorted(np.random.default_rng(0).choice(active,size=300,replace=False).tolist())
Xte,yte=build(TEST_ANCHOR,ids=test_ids)

def fit_auc(n, seed):
    idx=np.random.default_rng(seed).choice(len(Xtr),size=n,replace=False)
    Xs,ys=Xtr.iloc[idx],ytr.iloc[idx]
    if ys.nunique()<2: return np.nan
    m=xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=4,
                        subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",random_state=seed)
    m.fit(Xs,ys); return roc_auc_score(yte,m.predict_proba(Xte)[:,1])

print(f"XGBoost churn AUC vs #labels  (RT-J zero-shot = {RTJ_CHURN_AUC})\n")
print(f"  {'#labels':>8}   {'XGBoost AUC':>12}   vs RT-J")
sizes=[3,5,8,12,18,25,40,80]
prev_below=True; crossover=None
for n in sizes:
    aucs=[fit_auc(n,s) for s in range(25)]           # 8 seeds, average out subsample noise
    mean=np.nanmean(aucs)
    flag = "below" if mean<RTJ_CHURN_AUC else "ABOVE"
    print(f"  {n:>8}   {mean:>12.3f}   {flag}")
    if prev_below and mean>=RTJ_CHURN_AUC and crossover is None:
        crossover=n
    prev_below = mean<RTJ_CHURN_AUC
print(f"\n  → XGBoost first matches RT-J's zero-shot AUC around ~{crossover} labeled examples.")
print(f"    i.e. RT-J delivers, with zero labels + zero features, what a tuned GBDT needs")
print(f"    ~{crossover} labeled examples (and 11 hand features) to reach.")
print("done.")
