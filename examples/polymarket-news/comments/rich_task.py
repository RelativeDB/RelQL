"""The chatter experiment again, on a better graph.

Same population, same target, same ablation as `comments.chatter_task` — the
only thing that changes is how the text is wired to the markets:

    lexical (chatter_task)   two shared distinctive words -> event -> market
    rich    (this)           MiniLM similarity -> market question, a subject
                             bridge, hourly rollups, thread grouping

Comparing "gain from the words" between the two answers a question the first
run could not: was the text useless, or was it merely badly attached?

    python -m comments.rich_task --horizon 6 --per-split 1200
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from comments.links import build_rich, attach, hour_floor
from comments.run import price_at, score, report
from scale.analyze import paired_bootstrap
from scale.build import load_venues, tokens
from scale.resolve import auroc

CHATTER = Path(__file__).resolve().parent.parent / "data" / "chatter" / "chatter.parquet"
SPLIT_AT = datetime(2025, 12, 6, tzinfo=timezone.utc)


def prefilter(markets, posts, *, max_df: float = 0.03, min_hits: int = 3,
              min_len: int = 6):
    """Only posts that plausibly discuss a market are worth embedding.

    One shared word is far too weak a gate: with a few thousand market
    questions the distinctive vocabulary is broad enough that 588k of 708k
    posts qualified, which is most of an afternoon in MiniLM for nothing.
    Two rarer words cuts that by roughly an order of magnitude and loses
    almost nothing the encoder would have rescued."""
    from collections import Counter
    frequency = Counter()
    for m in markets:
        frequency.update(tokens(m["question"]))
    cutoff = max(2, int(len(markets) * max_df))
    vocabulary = {t for t, n in frequency.items()
                  if n <= cutoff and len(t) >= min_len}
    return [p for p in posts if len(tokens(p["body"]) & vocabulary) >= min_hits]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--per-split", type=int, default=1200)
    ap.add_argument("--context-cells", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--floor", type=float, default=0.35)
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--tick-history", type=int, default=336,
                    help="hours of tape to keep before the first post")
    ap.add_argument("--chunk", type=int, default=200,
                    help="posts per scoring chunk (checkpointed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    market_meta, _, ticks = load_venues(("polymarket", "kalshi"))
    markets = [m for m in market_meta if len(ticks.get(m["market_id"], ())) >= 72]
    bars = {}
    for m in markets:
        series, previous = [], None
        for b in sorted(ticks[m["market_id"]], key=lambda b: b["hour"]):
            series.append({"at": b["hour"], "close": b["close"],
                           "ret": None if previous is None else b["close"] - previous,
                           "volume": b["usd"] or b["fills"] or 0.0})
            previous = b["close"]
        bars[m["market_id"]] = series
    print(f">> {len(markets)} markets with a usable tape")

    posts = pq.read_table(CHATTER).to_pylist()
    for p in posts:
        p["posted_at"] = p["posted_at"].replace(tzinfo=timezone.utc)
    candidates = prefilter(markets, posts)
    print(f">> {len(posts):,} posts -> {len(candidates):,} candidates to embed")

    attachment = attach(markets, candidates, top_k=args.top_k, floor=args.floor)
    print(f">> {len(attachment):,} posts attached to a market by meaning")

    # Trim the graph to what the population can actually reach. Carrying 2,291
    # markets and 778k hourly bars through context assembly cost minutes per
    # post and enough memory to get the process killed; the markets nobody
    # posted about contribute nothing to a prediction about a post.
    live = {market_id for market_id, _ in attachment.values()}
    stamps = [candidates[i]["posted_at"] for i in attachment]
    lo = min(stamps) - timedelta(hours=args.tick_history)
    hi = max(stamps) + timedelta(hours=args.horizon + 1)
    markets = [m for m in markets if m["market_id"] in live]
    bars = {k: [b for b in v if lo <= b["at"] <= hi]
            for k, v in bars.items() if k in live}
    print(f">> trimmed to {len(markets)} markets, "
          f"{sum(len(v) for v in bars.values()):,} bars "
          f"({lo:%m-%d} .. {hi:%m-%d})")

    # Labels: the move of the market a post is actually about.
    labels = []
    for post_i, (market_id, similarity) in attachment.items():
        post = candidates[post_i]
        series = bars[market_id]
        posted = hour_floor(post["posted_at"])
        before = price_at(series, posted)
        after = price_at(series, posted + timedelta(hours=args.horizon))
        if before is None or after is None or after == before:
            continue
        if series[-1]["at"] < posted + timedelta(hours=args.horizon):
            continue
        earlier = price_at(series, posted - timedelta(hours=args.horizon))
        labels.append({"comment_id": post["chatter_id"], "posted_at": posted,
                       "price_up": after > before, "before": before,
                       "similarity": similarity,
                       "momentum": 0.0 if earlier is None else before - earlier})
    print(f">> {len(labels):,} labelled posts "
          f"({sum(c['price_up'] for c in labels) / max(1, len(labels)):.1%} up)")

    rng = random.Random(args.seed)
    splits = {}
    for name in ("dev", "holdout"):
        group = [c for c in labels
                 if (c["posted_at"] < SPLIT_AT) == (name == "dev")]
        if len(group) > args.per_split:
            group = rng.sample(group, args.per_split)
        splits[name] = group
    ids = [c["comment_id"] for g in splits.values() for c in g]
    print(f">> scoring {len(ids):,} "
          f"({len(splits['dev'])} dev / {len(splits['holdout'])} holdout)")

    arms = {}
    for arm, muted in (("full", False), ("muted", True)):
        path = (CHATTER.parent /
                f"rich_{arm}_{args.horizon}h_{args.context_cells}.parquet")
        if path.exists() and not args.force:
            arms[arm] = {r["comment_id"]: r["p"]
                         for r in pq.read_table(path).to_pylist()}
            print(f"   reusing {path.name}")
            continue
        schema, wiring, rows = build_rich(
            markets, bars, candidates, labels, horizon=args.horizon,
            muted=muted, top_k=args.top_k, floor=args.floor,
            attachment=attachment)
        if arm == "full":
            print("   tables: " + ", ".join(f"{k}={len(v)}" for k, v in rows.items()))
        # Scored in checkpointed chunks: an arm that dies at 80% keeps its
        # 80%, and a resumed run skips what is already on disk.
        partial = path.with_suffix(".partial.parquet")
        scored = ({r["comment_id"]: r["p"]
                   for r in pq.read_table(partial).to_pylist()}
                  if partial.exists() and not args.force else {})
        todo = [i for i in ids if i not in scored]
        print(f"   scoring {arm}: {len(todo)} to go "
              f"({len(scored)} already cached)", flush=True)
        for start in range(0, len(todo), args.chunk):
            piece = todo[start:start + args.chunk]
            got, truncated = score(schema, wiring, piece,
                                   cells=args.context_cells,
                                   batch=args.batch_size)
            scored.update(got)
            pq.write_table(pa.table({"comment_id": list(scored),
                                     "p": [scored[i] for i in scored]}), partial)
            print(f"     {min(start + args.chunk, len(todo)):>5}/{len(todo)} "
                  f"({truncated} truncated)", flush=True)
        pq.write_table(pa.table({"comment_id": list(scored),
                                 "p": [scored[i] for i in scored]}), path)
        partial.unlink(missing_ok=True)
        print(f"   wrote {path.name}")
        arms[arm] = scored

    for name, group in splits.items():
        if len(group) < 30:
            continue
        gids = [c["comment_id"] for c in group]
        truth = [c["price_up"] for c in group]
        full = [arms["full"][i] for i in gids]
        muted = [arms["muted"][i] for i in gids]
        print(f"\n== rich graph, {name}: {len(group)} posts, "
              f"{sum(truth) / len(truth):.1%} up ==")
        report("always up (p=0.5)", [0.5] * len(group), truth)
        report("price level (1 - price)", [1 - c["before"] for c in group], truth)
        report("RT-J, chatter text", full, truth)
        report("RT-J, text muted", muted, truth)
        lo, hi = paired_bootstrap(full, muted, truth)
        print(f"  AUROC gain from the words: "
              f"{auroc(full, truth) - auroc(muted, truth):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        strong = [i for i, c in enumerate(group) if c["similarity"] >= 0.5]
        if len(strong) >= 40:
            t = [truth[i] for i in strong]
            if 0 < sum(t) < len(t):
                print(f"  well-matched posts only (cos>=0.5, n={len(strong)}): "
                      f"text {auroc([full[i] for i in strong], t):.3f} vs "
                      f"muted {auroc([muted[i] for i in strong], t):.3f}")


if __name__ == "__main__":
    main()
