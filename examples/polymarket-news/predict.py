"""Six-hour direction, priced by a relational transformer that has never seen
a prediction market.

The task
--------
Cut the database at an anchor. For every market in the universe, ask:

    PREDICT SUM(price_ticks.ret) OVER (6 HOURS FOLLOWING) > 0
    FROM markets
    WHERE markets.market_id IN :ids
    RETURN PROBABILITY

``ret`` is the hourly change in the YES price, so the sum of the next six
hourly returns is exactly the six-hour move, and ``> 0`` is "the market
reprices upward". The same aggregate without the comparison is the regression
form — how *far* it moves — and both run against the identical database.

Why it is hard: a prediction market's price is already the crowd's forecast.
Beating a coin flip on its six-hour direction means finding information the
price has not absorbed yet — which is where the news tables come in.

Nothing is trained. RT-J is frozen, no head is fitted, and the only thing that
changes between the two arms of the ablation is whether the news tables exist.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

from relativedb import ContextPolicy, Engine, ExecutionInput, RtNativeBackend
from relativedb.rt_native import ContextTruncationWarning

import db

HORIZON = 6                                   # hours

CLASSIFY = (f"PREDICT SUM(price_ticks.ret) OVER ({HORIZON} HOURS FOLLOWING) > 0 "
            "FROM markets WHERE markets.market_id IN :ids "
            "RETURN PROBABILITY")
REGRESS = (f"PREDICT SUM(price_ticks.ret) OVER ({HORIZON} HOURS FOLLOWING) "
           "FROM markets WHERE markets.market_id IN :ids "
           "RETURN EXPECTED VALUE")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def auroc(scores: list[float], truth: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, truth) if y]
    neg = [s for s, y in zip(scores, truth) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def accuracy(scores: list[float], truth: list[bool], cut: float = 0.5) -> float:
    return sum((s >= cut) == y for s, y in zip(scores, truth)) / len(truth)


def brier(scores: list[float], truth: list[bool]) -> float:
    return sum((s - y) ** 2 for s, y in zip(scores, truth)) / len(truth)


def logloss(scores: list[float], truth: list[bool]) -> float:
    eps = 1e-6
    return -sum(math.log(max(eps, s if y else 1 - s))
                for s, y in zip(scores, truth)) / len(truth)


def report(name: str, scores: list[float], truth: list[bool]) -> dict:
    area = auroc(scores, truth)
    row = {"name": name, "n": len(truth), "acc": accuracy(scores, truth),
           "auroc": area, "brier": brier(scores, truth),
           "logloss": logloss(scores, truth)}
    print(f"  {name:<34} acc {row['acc']:.3f}   "
          f"auroc {'  n/a' if area is None else f'{area:.3f}'}   "
          f"brier {row['brier']:.3f}   logloss {row['logloss']:.3f}")
    return row


# ---------------------------------------------------------------------------
# baselines — everything the price alone can tell you
# ---------------------------------------------------------------------------
def baselines(snapshot, anchor, market_ids):
    """Squash a signal into a probability with a logistic on the raw move;
    the scale only affects calibration, never the ranking (so never AUROC)."""
    momentum, level = {}, {}
    for market_id in market_ids:
        bars = sorted(snapshot["candles"][market_id], key=lambda b: b["ts"])
        now = db.price_at(bars, anchor)
        then = db.price_at(bars, anchor - timedelta(hours=HORIZON))
        move = 0.0 if (now is None or then is None) else now - then
        momentum[market_id] = 1 / (1 + math.exp(-move / 0.02))
        # A price near 0 has more room above than below, and vice versa: the
        # "mean reversion toward 0.5" prior, with no history at all.
        level[market_id] = 1 - (now if now is not None else 0.5)
    return {"momentum (last 6h move)": momentum,
            "distance from 0.5": level,
            # Uninformative constant: its accuracy IS the up-rate of the
            # window, which is the number every other row has to beat.
            "always up (p=0.5)": {m: 0.5 for m in market_ids}}


# ---------------------------------------------------------------------------
def run_query(snapshot, market_ids, anchor, query, *, with_news, context_cells,
              batch_size):
    """Execute one RelQL query against one arm of the database.

    Truncation warnings are counted rather than printed: with 40 markets they
    would bury the report, but a context that did not fit is a real caveat on
    the numbers, so the count is surfaced."""
    schema, wiring, rows = db.build(snapshot, with_news=with_news)
    engine = Engine(
        schema, wiring,
        model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                      max_seq_len=context_cells,
                                      batch_size=batch_size),
        context_policy=ContextPolicy(max_context_cells=context_cells,
                                     local_context_cells=context_cells // 2,
                                     bfs_width=32, max_hops=3, seed=0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = engine.execute(ExecutionInput(
            query=query, anchor_time=anchor,
            params={"ids": list(market_ids)}))
    truncated = sum(1 for w in caught
                    if isinstance(w.message, ContextTruncationWarning))
    counts = {name: len(table_rows) for name, table_rows in rows.items()}
    if truncated:
        counts["(contexts truncated)"] = truncated
    return result, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(db.SNAPSHOT))
    ap.add_argument("--context-cells", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dump", help="write per-market predictions to this path")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-regression", action="store_true")
    args = ap.parse_args()

    snapshot = db.load(Path(args.snapshot))
    fetched = datetime.fromisoformat(snapshot["fetched_at"])
    # The last hourly candle is only complete once its hour has closed, so the
    # anchor sits HORIZON hours behind the last closed hour: everything after
    # it is the held-out future the model must not see.
    anchor = db.floor_hour(fetched) - timedelta(hours=HORIZON)

    truth_by_id = db.labels(snapshot, anchor, HORIZON)
    market_ids = [m["market_id"] for m in snapshot["markets"]
                  if m["market_id"] in truth_by_id]
    truth = [truth_by_id[m][3] for m in market_ids]
    moves = [truth_by_id[m][2] for m in market_ids]
    question = {m["market_id"]: m["question"] for m in snapshot["markets"]}

    print(f"snapshot fetched {fetched:%Y-%m-%d %H:%M} UTC   "
          f"anchor {anchor:%Y-%m-%d %H:%M} UTC   horizon {HORIZON}h")
    print(f"universe: {len(market_ids)} markets   "
          f"{sum(truth)} up / {len(truth) - sum(truth)} down   "
          f"median |move| {sorted(abs(x) for x in moves)[len(moves) // 2]:.4f}")
    news_before = sum(1 for a in snapshot["articles"]
                      if db.ts(a["published_at"]) <= anchor)
    print(f"news: {len(snapshot['articles'])} articles "
          f"({news_before} of them before the anchor, the only ones visible), "
          f"{len(snapshot['mentions'])} event mentions")

    signals = baselines(snapshot, anchor, market_ids)

    print(f"\n== RT-J zero-shot ==\n  {CLASSIFY}")
    result, counts = run_query(snapshot, market_ids, anchor, CLASSIFY,
                               with_news=True,
                               context_cells=args.context_cells,
                               batch_size=args.batch_size)
    print("  database: " + ", ".join(f"{n}={c}" for n, c in counts.items()))
    scored = {p.id: float(p.probability) for p in result.predictions}
    signals["RT-J  (markets + news)"] = scored

    if not args.skip_ablation:
        result_no_news, _ = run_query(snapshot, market_ids, anchor, CLASSIFY,
                                      with_news=False,
                                      context_cells=args.context_cells,
                                      batch_size=args.batch_size)
        signals["RT-J  (markets only, ablated)"] = {
            p.id: float(p.probability) for p in result_no_news.predictions}

    print("\n== direction over the held-out window ==")
    table = [report(name, [signal[m] for m in market_ids], truth)
             for name, signal in signals.items()]

    # A market whose price is unchanged to the tick has no direction to get
    # right; scored as "down" it flatters or punishes at random. Re-score
    # without them and see whether the ordering survives.
    movers = [m for m in market_ids if truth_by_id[m][2] != 0]
    if 0 < len(movers) < len(market_ids):
        moved_truth = [truth_by_id[m][3] for m in movers]
        print(f"\n== same, restricted to the {len(movers)} markets that moved "
              f"({sum(moved_truth)} up / {len(movers) - sum(moved_truth)} down) ==")
        for name, signal in signals.items():
            report(name, [signal[m] for m in movers], moved_truth)

    if not args.skip_regression:
        print(f"\n== regression: how far does it move ==\n  {REGRESS}")
        reg, _ = run_query(snapshot, market_ids, anchor, REGRESS,
                           with_news=True, context_cells=args.context_cells,
                           batch_size=args.batch_size)
        predicted = {p.id: float(p.value) for p in reg.predictions}
        errors = [abs(predicted[m] - truth_by_id[m][2]) for m in market_ids]
        naive = [abs(truth_by_id[m][2]) for m in market_ids]          # "no move"
        signs = sum((predicted[m] > 0) == truth_by_id[m][3]
                    for m in market_ids)
        print(f"  MAE {sum(errors)/len(errors):.4f}   "
              f"vs predicting zero move {sum(naive)/len(naive):.4f}   "
              f"direction from the sign: {signs}/{len(market_ids)} "
              f"= {signs/len(market_ids):.0%}")

    if args.dump:
        # Per-market predictions, so the aggregate table can be taken apart
        # afterwards (by move size, by liquidity, by news coverage) without
        # paying for another pass over the model.
        Path(args.dump).write_text(json.dumps({
            "anchor": anchor.isoformat(),
            "horizon_hours": HORIZON,
            "context_cells": args.context_cells,
            "rows": [{"market_id": m, "question": question[m],
                      "price_before": truth_by_id[m][0],
                      "price_after": truth_by_id[m][1],
                      "move": truth_by_id[m][2], "up": truth_by_id[m][3],
                      **{name: signal[m] for name, signal in signals.items()}}
                     for m in market_ids]}, indent=1))
        print(f"\nwrote {args.dump}")

    print("\n== most confident calls (RT-J with news) ==")
    ranked = sorted(market_ids, key=lambda m: -abs(scored[m] - 0.5))
    for market_id in ranked[:12]:
        p = scored[market_id]
        before, after, move, up = truth_by_id[market_id]
        ok = "OK " if (p >= 0.5) == up else "XX "
        print(f"  {ok} p(up)={p:.3f}  {before:.3f} -> {after:.3f} "
              f"({move:+.3f})  {question[market_id][:60]}")

    return table


if __name__ == "__main__":
    main()
