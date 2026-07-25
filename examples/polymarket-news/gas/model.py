"""Fit, backtest and quote the AAA daily gas ladder.

The first attempt scored 17 strike-contracts as 17 independent binary
questions and produced a non-monotone ladder — 0.54 on a threshold four cents
below spot. A ladder is a CDF, so the fix is to model the *level* and read the
ladder off a predictive distribution.

    change_t = a + b * momentum_t + c * weekday_t          (fitted, walk-forward)
    P(level > K) = 1 - Phi((K - (level_{t-1} + change_t)) / sigma)

Everything is judged against the closing price of each settled strike, because
that is the number a forecast has to beat to be worth anything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "gas"
DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def load():
    levels = {dt.date.fromisoformat(k): v
              for k, v in json.loads((DATA / "levels.json").read_text()).items()}
    levels[dt.date(2026, 7, 24)] = 4.1050      # AAA-reported, exact
    levels[dt.date(2026, 7, 25)] = 4.1110
    ladders = json.loads((DATA / "ladders.json").read_text())
    book = json.loads((DATA / "book.json").read_text())
    return levels, ladders, book


def features(levels, day):
    """Momentum and weekday, using only days strictly before `day`."""
    past = sorted(d for d in levels if d < day)
    if len(past) < 5:
        return None
    changes = []
    for i in range(1, len(past)):
        if (past[i] - past[i - 1]).days == 1:
            changes.append((past[i], (levels[past[i]] - levels[past[i - 1]]) * 100))
    recent = [c for d, c in changes if (past[-1] - d).days < 3]
    if not recent:
        return None
    return {"last_level": levels[past[-1]],
            "momentum": statistics.mean(recent),
            "dow": day.strftime("%a"),
            "history": changes}


def fit(history, dow):
    """Least squares on change ~ 1 + momentum, plus a weekday offset.

    Deliberately tiny: 60-odd observations cannot support anything richer, and
    an over-parameterized fit here would be indistinguishable from the noise
    it is trying to explain."""
    rows = []
    for i in range(3, len(history)):
        day, change = history[i]
        prior = [c for d, c in history[max(0, i - 3):i]]
        if len(prior) == 3:
            rows.append((statistics.mean(prior), change, day.strftime("%a")))
    if len(rows) < 12:
        return 0.0, 0.0, 1.5
    mx = statistics.mean(r[0] for r in rows)
    my = statistics.mean(r[1] for r in rows)
    var = sum((r[0] - mx) ** 2 for r in rows)
    beta = (sum((r[0] - mx) * (r[1] - my) for r in rows) / var) if var else 0.0
    alpha = my - beta * mx
    same = [r[1] - (alpha + beta * r[0]) for r in rows if r[2] == dow]
    offset = statistics.mean(same) if len(same) >= 3 else 0.0
    resid = [r[1] - (alpha + beta * r[0] +
                     (offset if r[2] == dow else 0.0)) for r in rows]
    sigma = statistics.pstdev(resid) or 1.5
    return alpha + offset, beta, sigma


def predict(levels, day):
    f = features(levels, day)
    if f is None:
        return None
    alpha, beta, sigma = fit(f["history"], f["dow"])
    change = alpha + beta * f["momentum"]
    return {"level": f["last_level"] + change / 100, "change": change,
            "sigma_cents": sigma, "last_level": f["last_level"],
            "momentum": f["momentum"]}


def p_above(pred, strike):
    z = (strike - pred["level"]) * 100 / pred["sigma_cents"]
    return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -(math.log(p) if y else math.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="2026-07-26")
    args = ap.parse_args()
    levels, ladders, book = load()
    target = dt.date.fromisoformat(args.target)

    print("== walk-forward backtest against the closing book ==")
    rows = []
    for day_str, markets in sorted(ladders.items()):
        day = dt.date.fromisoformat(day_str)
        if day >= target:
            continue
        pred = predict(levels, day)
        if pred is None:
            continue
        for m in markets:
            if m.get("result") not in ("yes", "no"):
                continue
            entry = book.get(m["ticker"])
            if not entry:
                continue
            strike = float(m["ticker"].rsplit("-", 1)[1])
            rows.append({"day": day, "y": m["result"] == "yes",
                         "model": p_above(pred, strike),
                         "market": entry["close_price"]})
    if rows:
        n = len(rows)
        print(f"  {n} settled strikes across "
              f"{len({r['day'] for r in rows})} days\n")
        print(f"  {'':<12} {'brier':>8} {'logloss':>9} {'acc':>7}")
        for name in ("market", "model"):
            b = sum(brier(r[name], r["y"]) for r in rows) / n
            l = sum(logloss(r[name], r["y"]) for r in rows) / n
            a = sum((r[name] >= 0.5) == r["y"] for r in rows) / n
            print(f"  {name:<12} {b:>8.4f} {l:>9.4f} {a:>7.3f}")
        always = sum(brier(0.5, r["y"]) for r in rows) / n
        print(f"  {'coin flip':<12} {always:>8.4f} {math.log(2):>9.4f}")

    print(f"\n== quote for {target} ==")
    pred = predict(levels, target)
    print(f"  last level {pred['last_level']:.4f}   momentum "
          f"{pred['momentum']:+.2f}c   fitted change {pred['change']:+.2f}c"
          f"   sigma {pred['sigma_cents']:.2f}c")
    print(f"  => expected {pred['level']:.4f}\n")
    tomorrow = ladders.get(str(target), [])
    print(f"  {'strike':>8} {'model':>7} {'market':>7}")
    for m in sorted(tomorrow, key=lambda m: float(m["ticker"].rsplit("-", 1)[1])):
        strike = float(m["ticker"].rsplit("-", 1)[1])
        print(f"  {strike:>8.3f} {p_above(pred, strike):>7.3f} {'':>7}")


if __name__ == "__main__":
    main()
