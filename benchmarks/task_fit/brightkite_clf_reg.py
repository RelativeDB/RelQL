"""Brightkite check-ins (SNAP) — a DIFFERENT domain (mobility/social).
Same task shapes as retail: churn (binary) + activity-count (regression).
RT-J zero-shot vs naive baselines vs supervised XGBoost."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from harness import datasets
from relativedb import (Engine, ExecutionInput, RtNativeBackend, SamplerMode,
                        Schema, TableDef, LinkDef, ValueType, ContextPolicy)
import xgboost as xgb
from sklearn.metrics import roc_auc_score

FWD, BACK = 30, 90
TEST_ANCHOR = pd.Timestamp("2009-06-01")
TRAIN_ANCHORS = [pd.Timestamp(d) for d in ["2009-01-01","2009-02-01","2009-03-01","2009-04-01"]]
N_TEST, N_TRAIN_PER = 300, 1500

gz = datasets.CORPUS / "brightkite.txt.gz"
df = pd.read_csv(gz, sep="\t", compression="gzip", header=None,
                 names=["user_id","ts","lat","lon","location_id"], dtype={"location_id":str})
df = df[df.location_id.notna() & (df.location_id!="")]
df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True).dt.tz_localize(None)
df = df[df.ts.notna()].copy()
df["checkin_id"] = np.arange(len(df))

def feats(anchor, ids):
    past = df[df.ts <= anchor]
    w90 = past[past.ts > anchor - pd.Timedelta(days=BACK)]
    w30 = past[past.ts > anchor - pd.Timedelta(days=FWD)]
    gA, g9 = past.groupby("user_id"), w90.groupby("user_id")
    last, first = gA.ts.max(), gA.ts.min()
    fall, f90, f30 = gA.size(), g9.size(), w30.groupby("user_id").size()
    loc_all, loc90 = gA.location_id.nunique(), g9.location_id.nunique()
    days90 = g9.ts.apply(lambda s: s.dt.normalize().nunique()) if len(w90) else pd.Series(dtype=float)
    rows = []
    for u in ids:
        rows.append([(anchor - last.get(u, anchor)).days, f90.get(u,0), f30.get(u,0),
                     fall.get(u,0), (anchor - first.get(u, anchor)).days,
                     loc_all.get(u,0), loc90.get(u,0), days90.get(u,0)])
    X = pd.DataFrame(rows, columns=["recency","freq90","freq30","freq_all","tenure",
                                    "loc_all","loc90","active_days90"], index=ids)
    fut = df[(df.ts > anchor) & (df.ts <= anchor + pd.Timedelta(days=FWD))]
    cnt = fut.groupby("user_id").size()
    churn = pd.Series([0 if cnt.get(u,0)>0 else 1 for u in ids], index=ids)
    count = pd.Series([float(cnt.get(u,0)) for u in ids], index=ids)
    return X, churn, count

rng = np.random.default_rng(0)
def active(anchor):
    return sorted(df[(df.ts > anchor-pd.Timedelta(days=BACK)) & (df.ts<=anchor)].user_id.unique())
test_ids = sorted(rng.choice(active(TEST_ANCHOR), size=N_TEST, replace=False).tolist())

# ---- XGBoost: train on earlier anchors ---------------------------------
Xtr, ytr_c, ytr_n = [], [], []
for a in TRAIN_ANCHORS:
    ids = sorted(rng.choice(active(a), size=min(N_TRAIN_PER, len(active(a))), replace=False).tolist())
    X, c, n = feats(a, ids); Xtr.append(X); ytr_c.append(c); ytr_n.append(n)
Xtr, ytr_c, ytr_n = pd.concat(Xtr), pd.concat(ytr_c), pd.concat(ytr_n)
Xte, yte_c, yte_n = feats(TEST_ANCHOR, test_ids)

clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=0)
clf.fit(Xtr, ytr_c); xgb_auc = roc_auc_score(yte_c, clf.predict_proba(Xte)[:,1])
reg = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.8, random_state=0)
reg.fit(Xtr, np.log1p(ytr_n)); xgb_pred = np.expm1(reg.predict(Xte))

def spear(a,b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0,1])
def auc(s,y):
    s,y=np.asarray(s,float),np.asarray(y,int); o=np.argsort(s); r=np.empty_like(o,float); r[o]=np.arange(1,len(s)+1)
    return (r[y==1].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(1-y).sum())

# ---- RT-J zero-shot on the same 300 test users -------------------------
schema = (Schema.new_schema()
          .table(TableDef.new_table("users").primary_key("user_id").build())
          .table(TableDef.new_table("checkins").column("location_id", ValueType.TEXT)
                 .column("ts", ValueType.DATETIME).primary_key("checkin_id").time_column("ts").build())
          .link(LinkDef("checkins","user_id","users")).build())
ck = df[df.user_id.isin(set(test_ids))]
frames = {"users": pd.DataFrame({"user_id": test_ids}),
          "checkins": ck[["checkin_id","user_id","location_id","ts"]]}
wiring = datasets._wire(schema, frames)
engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema),
                sampler_mode=SamplerMode.CSC,
                context_policy=ContextPolicy(max_context_cells=20_000_000, bfs_width=4096, max_hops=1))

r1 = engine.execute(ExecutionInput(
    query=f"PREDICT NOT EXISTS(checkins.*) OVER ({FWD} DAYS FOLLOWING) "
          f"FOR EACH users.user_id WHERE EXISTS(checkins.*) OVER ({BACK} DAYS PRECEDING)",
    entity_ids=test_ids, anchor_time=TEST_ANCHOR.to_pydatetime()))
ids = [p.id for p in r1.predictions]
rtj_auc = auc(np.array([p.probability for p in r1.predictions]),
              np.array([int(yte_c[i]) for i in ids]))
recency_auc = auc(Xte.loc[ids,"recency"].values, np.array([int(yte_c[i]) for i in ids]))

r2 = engine.execute(ExecutionInput(
    query=f"PREDICT COUNT(checkins.*) OVER ({FWD} DAYS FOLLOWING) FOR EACH users.user_id",
    entity_ids=test_ids, anchor_time=TEST_ANCHOR.to_pydatetime()))
ids2 = [p.id for p in r2.predictions]
def pv(p):
    for at in ("value","prediction","expected_value"):
        if getattr(p,at,None) is not None: return float(getattr(p,at))
    return float(getattr(p,"probability"))
rtj_pred = np.array([pv(p) for p in r2.predictions]); true_n = np.array([float(yte_n[i]) for i in ids2])
persist = Xte.loc[ids2,"freq30"].values

print(f"BRIGHTKITE check-ins | anchor {TEST_ANCHOR.date()} | {len(test_ids)} users | +{FWD}d\n")
print("=== CHURN (binary) — AUC ===")
print(f"  base churn rate : {np.mean([int(yte_c[i]) for i in ids]):.1%}")
print(f"  RT-J zero-shot  : {rtj_auc:.3f}")
print(f"  recency baseline: {recency_auc:.3f}")
print(f"  XGBoost (superv): {xgb_auc:.3f}")
print("\n=== ACTIVITY COUNT next 30d (regression) — Spearman / MAE ===")
print(f"  RT-J zero-shot  : {spear(rtj_pred,true_n):.3f} | MAE {np.abs(rtj_pred-true_n).mean():.1f}")
print(f"  persistence base: {spear(persist,true_n):.3f} | MAE {np.abs(persist-true_n).mean():.1f}")
print(f"  XGBoost (superv): {spear(xgb_pred,true_n):.3f} | MAE {np.abs(xgb_pred-true_n).mean():.1f}")
print("done.")
