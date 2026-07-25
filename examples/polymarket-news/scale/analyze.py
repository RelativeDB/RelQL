"""Scoring and comparison over cached predictions — no model, no GPU.

`resolve.py` pays for the forward passes and writes one parquet per arm; this
reads them back, so trying a different baseline, blend or subgroup costs a
second instead of an hour.

    python -m scale.analyze --split dev
    python -m scale.analyze --split holdout
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pyarrow.parquet as pq

from scale.build import SCALE, build
from scale.resolve import SPLIT_AT, accuracy, auroc, brier, logloss

PRICE_WINDOW = 24        # hours of tape averaged into the baseline price


def clip(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return min(hi, max(lo, x))


def market_price(m: dict, hours: int = PRICE_WINDOW) -> float:
    """The crowd's forecast at the anchor.

    Averaged over the last day of trading rather than taken from the final
    print: on a thin contract the last fill can be a single dollar at a stale
    level, and a baseline should be the strongest honest version of itself,
    not the most convenient one. Chosen on dev (AUROC 0.780 vs 0.760 for the
    last VWAP, 0.735 for the last close) and then frozen."""
    bars = m["bars"][-hours:]
    return clip(sum(b["vwap"] for b in bars) / len(bars))


def logit(p: float) -> float:
    p = clip(p, 1e-4, 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def paired_bootstrap(a, b, truth, *, rounds: int = 2000, seed: int = 0):
    """Interval on the AUROC *difference*, resampling markets — the two arms
    are scored on the same contracts, so their errors are correlated and two
    separate intervals overstate the uncertainty of the gap."""
    rng = random.Random(seed)
    n, gaps = len(truth), []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        t = [truth[i] for i in idx]
        if not (0 < sum(t) < n):
            continue
        gaps.append(auroc([a[i] for i in idx], t) - auroc([b[i] for i in idx], t))
    gaps.sort()
    return gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps))]


def report(name, scores, truth):
    area = auroc(scores, truth)
    print(f"  {name:<32} acc {accuracy(scores, truth):.3f}   auroc {area:.3f}   "
          f"brier {brier(scores, truth):.4f}   logloss {logloss(scores, truth):.4f}")
    return {"name": name, "acc": accuracy(scores, truth), "auroc": area,
            "brier": brier(scores, truth), "logloss": logloss(scores, truth)}


def load_arm(split: str, arm: str, cells: int, tag: str = "core") -> dict | None:
    path = SCALE / f"pred_{tag}_{split}_{arm}_{cells}.parquet"
    if not path.exists():
        return None
    return {r["market_id"]: r["p"] for r in pq.read_table(path).to_pylist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", action="store_true",
                    help="the enlarged Aug-Dec universe")
    ap.add_argument("--split", choices=["dev", "holdout"], default="dev")
    ap.add_argument("--context-cells", type=int, default=2048)
    args = ap.parse_args()
    tag = "wide" if args.wide else "core"
    if args.wide:
        from scale.build import WIDE, use_sources
        use_sources(**WIDE)

    _, _, corpus = build()
    chosen = [m for m in corpus.markets
              if (m["closes_at"] < SPLIT_AT) == (args.split == "dev")]
    arms = {a: load_arm(args.split, a, args.context_cells, tag)
            for a in ("news", "nonews")}
    scored = arms["news"]
    if scored is None:
        raise SystemExit(f"no cached predictions for {args.split} — "
                         f"run `python -m scale.resolve --split {args.split}`")
    chosen = [m for m in chosen if m["market_id"] in scored]
    ids = [m["market_id"] for m in chosen]
    truth = [m["resolved_yes"] for m in chosen]
    price = [market_price(m) for m in chosen]
    rtj = [scored[i] for i in ids]

    print(f"== {args.split}: {len(chosen)} settled contracts, {sum(truth)} YES "
          f"({sum(truth) / len(truth):.1%}) ==")
    rows = [report("market price (24h vwap)", price, truth),
            report("always NO (p=0.5)", [0.5] * len(ids), truth),
            report("RT-J (markets+tape+news)", rtj, truth)]
    if arms["nonews"]:
        rows.append(report("RT-J (news ablated)",
                           [arms["nonews"][i] for i in ids], truth))
    blend = [sigmoid(0.5 * (logit(p) + logit(r))) for p, r in zip(price, rtj)]
    rows.append(report("price x RT-J (logit mean)", blend, truth))

    lo, hi = paired_bootstrap(blend, price, truth)
    print(f"\n  AUROC gain, blend over price:  {auroc(blend, truth) - auroc(price, truth):+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]")
    if arms["nonews"]:
        ablated = [arms["nonews"][i] for i in ids]
        lo, hi = paired_bootstrap(rtj, ablated, truth)
        print(f"  AUROC gain, news over ablated: "
              f"{auroc(rtj, truth) - auroc(ablated, truth):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n  by venue:")
    for venue in ("polymarket", "kalshi"):
        keep = [i for i, m in enumerate(chosen) if m["venue"] == venue]
        if len(keep) < 30:
            continue
        t = [truth[i] for i in keep]
        print(f"    {venue:<11} n={len(keep):<5} price {auroc([price[i] for i in keep], t):.3f}   "
              f"RT-J {auroc([rtj[i] for i in keep], t):.3f}   "
              f"blend {auroc([blend[i] for i in keep], t):.3f}")

    print("\n  by how sure the market was:")
    for label, confident in (("confident (<0.2 or >0.8)", True),
                             ("uncertain (0.2-0.8)", False)):
        keep = [i for i, p in enumerate(price)
                if (min(p, 1 - p) < 0.2) is confident]
        t = [truth[i] for i in keep]
        if len(keep) < 30 or not (0 < sum(t) < len(t)):
            continue
        print(f"    {label:<26} n={len(keep):<5} "
              f"price {auroc([price[i] for i in keep], t):.3f}   "
              f"RT-J {auroc([rtj[i] for i in keep], t):.3f}   "
              f"blend {auroc([blend[i] for i in keep], t):.3f}")

    out = SCALE / f"analysis_{tag}_{args.split}_{args.context_cells}.json"
    out.write_text(json.dumps({"split": args.split, "n": len(chosen),
                               "rows": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
