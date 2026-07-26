"""XGBoost on the same synthesized WTI ladder, against the same held-out days.

Same rows, same split, same metrics as `wti/predict.py`, so the three
approaches are directly comparable:

    martingale   normal centred on the prior close, sigma = trailing vol
    RT-J         frozen relational transformer, zero-shot
    xgboost      gradient boosting on the row's own features

The features are deliberately raw — the model gets the same columns the
relational database exposed, plus the one derived term the martingale is
built from (distance in units of volatility), so we can see whether anything
beyond that carries signal.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import xgboost as xgb

DATA = Path(__file__).resolve().parent.parent / "data" / "wti"
DOW = {d: i for i, d in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri"))}


def rows():
    days = {d["day"]: d for d in json.loads((DATA / "days.json").read_text())}
    strikes = json.loads((DATA / "strikes.json").read_text())
    X, y, meta = [], [], []
    for s in strikes:
        d = days[s["day"]]
        vol = d["vol_30d"] / 100
        sigma = d["prior_close"] * vol
        z = (s["threshold"] - d["prior_close"]) / sigma        # the martingale's input
        X.append([s["pct_from_prior"], z, d["vol_30d"], d["ret_1d"], d["ret_5d"],
                  d["brent_spread"] if d["brent_spread"] is not None else np.nan,
                  d["rbob_crack"] if d["rbob_crack"] is not None else np.nan,
                  DOW.get(d["weekday"], 5), d["prior_close"]])
        y.append(1 if s["above"] else 0)
        meta.append((s["day"], z))
    return np.array(X, float), np.array(y), meta, sorted(days)


NAMES = ["pct_from_prior", "z", "vol_30d", "ret_1d", "ret_5d",
         "brent_spread", "rbob_crack", "weekday", "prior_close"]


def metrics(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    pos, neg = p[y == 1], p[y == 0]
    auc = float(np.mean((pos[:, None] > neg[None, :]) +
                        0.5 * (pos[:, None] == neg[None, :])))
    acc = float(np.mean((p >= 0.5) == y))
    return auc, brier, ll, acc


def main() -> None:
    X, y, meta, days = rows()
    eval_days = set(days[-40:])
    test = np.array([m[0] in eval_days for m in meta])
    train = ~test
    print(f"train {train.sum()} rows / test {test.sum()} rows "
          f"({len(eval_days)} held-out days, strictly later)")

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        objective="binary:logistic", eval_metric="logloss", n_jobs=4)
    model.fit(X[train], y[train])
    p_boost = model.predict_proba(X[test])[:, 1]

    z_test = X[test][:, 1]
    p_mart = 1 - 0.5 * (1 + np.vectorize(math.erf)(z_test / math.sqrt(2)))
    rtj = json.loads((DATA / "pred_512.json").read_text())
    ids = [f"{m[0]}:{s}" for m, s in zip(np.array(meta, dtype=object)[test],
                                         [None] * int(test.sum()))]
    # rebuild ids properly
    strikes = json.loads((DATA / "strikes.json").read_text())
    test_ids = [s["strike_id"] for s, t in zip(strikes, test) if t]
    p_rtj = np.array([rtj.get(i, np.nan) for i in test_ids])
    have = ~np.isnan(p_rtj)

    yt = y[test]
    print(f"\n{'':<14}{'AUROC':>8}{'Brier':>9}{'logloss':>10}{'acc':>8}")
    for name, p in (("xgboost", p_boost), ("martingale", p_mart)):
        a, b, l, ac = metrics(p, yt)
        print(f"{name:<14}{a:>8.3f}{b:>9.4f}{l:>10.4f}{ac:>8.3f}")
    if have.any():
        a, b, l, ac = metrics(p_rtj[have], yt[have])
        print(f"{'RT-J':<14}{a:>8.3f}{b:>9.4f}{l:>10.4f}{ac:>8.3f}")
    base = float(yt.mean())
    print(f"{'base rate':<14}{'-':>8}{float(np.mean((base-yt)**2)):>9.4f}")

    print("\nfeature importance (gain):")
    for n, g in sorted(zip(NAMES, model.feature_importances_),
                       key=lambda kv: -kv[1]):
        print(f"  {n:<16}{g:>7.3f}")


if __name__ == "__main__":
    main()
