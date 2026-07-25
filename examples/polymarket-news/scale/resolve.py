"""Can a frozen relational transformer beat the market's own price?

For every contract that settled inside the window, cut the database two to
three days before it closes and ask:

    PREDICT markets.resolved_yes
    FROM markets
    WHERE markets.market_id IN :ids
    RETURN PROBABILITY

The yardstick is not a coin flip — it is the market price at that same moment,
which is a real forecast made by people with money at stake. Beating it means
finding something the crowd had not priced.

Splits are decided by close date and fixed before anything runs:

    dev      settles 2025-11-01 .. 2025-12-05   protocol choices allowed here
    holdout  settles 2025-12-06 .. 2025-12-31   scored once, at the end

Predictions are written to data/scale/pred_<split>_<arm>.parquet and reused on
the next run — an arm costs real compute, and re-running it to re-print a
table is waste.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from relativedb import ContextPolicy, Engine, ExecutionInput, RtNativeBackend
from relativedb.rt_native import ContextTruncationWarning

from scale.build import SCALE, build

QUERY = ("PREDICT markets.resolved_yes FROM markets "
         "WHERE markets.market_id IN :ids RETURN PROBABILITY")

SPLIT_AT = datetime(2025, 12, 6, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def auroc(scores, truth):
    pos = [s for s, y in zip(scores, truth) if y]
    neg = [s for s, y in zip(scores, truth) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brier(scores, truth):
    return sum((s - y) ** 2 for s, y in zip(scores, truth)) / len(truth)


def logloss(scores, truth):
    eps = 1e-6
    return -sum(math.log(max(eps, min(1 - eps, s if y else 1 - s)))
                for s, y in zip(scores, truth)) / len(truth)


def accuracy(scores, truth):
    return sum((s >= 0.5) == y for s, y in zip(scores, truth)) / len(truth)


def report(name, scores, truth):
    area = auroc(scores, truth)
    print(f"  {name:<30} n={len(truth):<5} acc {accuracy(scores, truth):.3f}   "
          f"auroc {'n/a' if area is None else f'{area:.3f}'}   "
          f"brier {brier(scores, truth):.4f}   logloss {logloss(scores, truth):.4f}")
    return {"name": name, "n": len(truth), "acc": accuracy(scores, truth),
            "auroc": area, "brier": brier(scores, truth),
            "logloss": logloss(scores, truth)}


def bootstrap_auroc(scores, truth, *, rounds: int = 400, seed: int = 0):
    """Percentile interval over resampled markets — with thousands of
    correlated contracts, the reported gap needs an error bar or it is a
    decoration."""
    rng = random.Random(seed)
    n = len(truth)
    areas = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        area = auroc([scores[i] for i in idx], [truth[i] for i in idx])
        if area is not None:
            areas.append(area)
    areas.sort()
    return areas[int(0.025 * len(areas))], areas[int(0.975 * len(areas))]


# ---------------------------------------------------------------------------
def score(corpus_markets, schema, wiring, ids, *, context_cells, batch_size,
          seed=0):
    engine = Engine(
        schema, wiring,
        model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                      max_seq_len=context_cells,
                                      batch_size=batch_size),
        context_policy=ContextPolicy(max_context_cells=context_cells,
                                     local_context_cells=context_cells // 2,
                                     bfs_width=32, max_hops=3, seed=seed))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = engine.execute(ExecutionInput(
            query=QUERY, per_entity_anchor=True, params={"ids": list(ids)}))
    truncated = sum(1 for w in caught
                    if isinstance(w.message, ContextTruncationWarning))
    return ({p.id: float(p.probability) for p in result.predictions}, truncated)


def cached_arm(path: Path, force: bool):
    if path.exists() and not force:
        table = pq.read_table(path).to_pylist()
        return {r["market_id"]: r["p"] for r in table}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", action="store_true",
                    help="the enlarged Aug-Dec universe")
    ap.add_argument("--split", choices=["dev", "holdout"], default="dev")
    ap.add_argument("--limit", type=int, default=600,
                    help="markets to score (sampled deterministically)")
    ap.add_argument("--context-cells", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--arms", default="news,nonews",
                    help="comma-separated: news, nonews")
    ap.add_argument("--force", action="store_true", help="ignore cached arms")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # The universe tags every cached file. Without it a --wide run would load
    # the narrow run's predictions and silently report them as its own.
    tag = "wide" if args.wide else "core"
    if args.wide:
        from scale.build import WIDE, use_sources
        use_sources(**WIDE)

    schema, wiring, corpus = build()
    markets = corpus.markets
    dev = [m for m in markets if m["closes_at"] < SPLIT_AT]
    holdout = [m for m in markets if m["closes_at"] >= SPLIT_AT]
    chosen = dev if args.split == "dev" else holdout
    print(f"universe: {len(markets)} settled contracts "
          f"({len(dev)} dev / {len(holdout)} holdout)")
    print(f"tables: " + ", ".join(f"{k}={v}" for k, v in corpus.counts.items()))

    rng = random.Random(args.seed)
    sample = sorted(chosen, key=lambda m: m["market_id"])
    if args.limit and len(sample) > args.limit:
        sample = rng.sample(sample, args.limit)
    ids = [m["market_id"] for m in sample]
    truth = [m["resolved_yes"] for m in sample]
    price = [min(0.99, max(0.01, m["price_at_anchor"])) for m in sample]
    print(f"\n== {args.split}: {len(sample)} contracts, "
          f"{sum(truth)} YES ({sum(truth) / len(truth):.1%}), "
          f"anchors {min(m['anchor_at'] for m in sample):%Y-%m-%d}.."
          f"{max(m['anchor_at'] for m in sample):%Y-%m-%d} ==")

    signals = {"market price at anchor": dict(zip(ids, price)),
               "always NO (p=0.5)": {i: 0.5 for i in ids}}

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        path = SCALE / f"pred_{tag}_{args.split}_{arm}_{args.context_cells}.parquet"
        scored = cached_arm(path, args.force)
        label = ("RT-J (markets+tape+news)" if arm == "news"
                 else "RT-J (markets+tape only)")
        if scored is None or any(i not in scored for i in ids):
            if arm == "nonews":
                s, w, _ = build(with_news=False)
            else:
                s, w = schema, wiring
            print(f"   scoring {label} ...", flush=True)
            scored, truncated = score(sample, s, w, ids,
                                      context_cells=args.context_cells,
                                      batch_size=args.batch_size,
                                      seed=args.seed)
            pq.write_table(pa.table({"market_id": list(scored),
                                     "p": [scored[i] for i in scored]}), path)
            print(f"   wrote {path.name} ({truncated} contexts truncated)")
        signals[label] = scored

    print()
    table = [report(name, [signal[i] for i in ids], truth)
             for name, signal in signals.items()]

    rtj = signals.get("RT-J (markets+tape+news)")
    if rtj is not None:
        blend = {i: 0.5 * (rtj[i] + p) for i, p in zip(ids, price)}
        table.append(report("price + RT-J, averaged",
                            [blend[i] for i in ids], truth))
        lo, hi = bootstrap_auroc([rtj[i] for i in ids], truth)
        plo, phi = bootstrap_auroc(price, truth)
        print(f"\n  AUROC 95% interval — RT-J [{lo:.3f}, {hi:.3f}]   "
              f"price [{plo:.3f}, {phi:.3f}]")

    out = SCALE / f"report_{tag}_{args.split}_{args.context_cells}.json"
    out.write_text(json.dumps({"split": args.split, "n": len(sample),
                               "rows": table}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
