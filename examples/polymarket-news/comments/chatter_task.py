"""Does public chatter move a prediction market in the next few hours?

Population: one row per post that talks about a market's subject. Target: a
masked boolean — did that market's price go up over the next H hours?

    PREDICT comments.price_up
    FROM comments
    WHERE comments.comment_id IN :ids
    RETURN PROBABILITY

The whole experiment is one ablation. Both arms see the poster, the channel,
the timestamp, the market and its tape; only one of them sees **the words**.
Whatever separates them is what the sentence was worth. That is the cleanest
question you can ask a model that encodes sentences.

Chatter comes from `comments.pull_chatter` (Reddit + Hacker News, Nov–Dec
2025); prices come from the trade tapes already pulled by `scale.pull_*`.

    python -m comments.chatter_task --horizon 6 --per-split 1200
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq

from scale.build import SCALE, load_venues, tokens

CHATTER = Path(__file__).resolve().parent.parent / "data" / "chatter" / "chatter.parquet"
SPLIT_AT = datetime(2025, 12, 6, tzinfo=timezone.utc)


def one_market_per_event(markets, ticks):
    """The event's most actively traded market carries its price series."""
    best: dict[str, dict] = {}
    for m in markets:
        bars = ticks.get(m["market_id"], ())
        if len(bars) < 72:
            continue
        current = best.get(m["event_id"])
        if current is None or len(bars) > len(ticks[current["market_id"]]):
            best[m["event_id"]] = m
    return best


def link(events: dict[str, str], posts, *, min_overlap: int, max_df: float,
         per_event_cap: int):
    """Same distinctive-vocabulary matching the news study uses: a token in
    more than `max_df` of event titles identifies nothing, so it is dropped."""
    index: dict[str, set[str]] = {}
    for event_id, title in events.items():
        for token in tokens(title):
            index.setdefault(token, set()).add(event_id)
    cutoff = max(2, int(len(events) * max_df))
    index = {t: e for t, e in index.items() if len(e) <= cutoff}

    per_event: dict[str, int] = {}
    linked = []
    for post in posts:
        hits: dict[str, int] = {}
        for token in tokens(post["body"]):
            for event_id in index.get(token, ()):
                hits[event_id] = hits.get(event_id, 0) + 1
        if not hits:
            continue
        event_id, overlap = max(hits.items(), key=lambda kv: kv[1])
        if overlap < min_overlap or per_event.get(event_id, 0) >= per_event_cap:
            continue
        per_event[event_id] = per_event.get(event_id, 0) + 1
        linked.append((event_id, overlap, post))
    return linked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--min-overlap", type=int, default=2)
    ap.add_argument("--max-df", type=float, default=0.02)
    ap.add_argument("--per-event-cap", type=int, default=400)
    ap.add_argument("--per-split", type=int, default=1200)
    ap.add_argument("--context-cells", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    markets_meta, event_titles, ticks = load_venues(("polymarket", "kalshi"))
    chosen = one_market_per_event(markets_meta, ticks)
    titles = {e: event_titles[e] for e in chosen}
    print(f">> {len(chosen)} events with a traded market and a usable tape")

    posts = pq.read_table(CHATTER).to_pylist()
    print(f">> {len(posts):,} posts, "
          f"{posts[0]['posted_at']:%Y-%m-%d} .. {posts[-1]['posted_at']:%Y-%m-%d}")
    linked = link(titles, posts, min_overlap=args.min_overlap,
                  max_df=args.max_df, per_event_cap=args.per_event_cap)
    print(f">> {len(linked):,} posts mention an event "
          f"({len({e for e, _, _ in linked})} events)")

    # Shape the linked posts like the Polymarket-comment snapshot so the same
    # builder, retrievers and leakage rules apply unchanged.
    snapshot = {"markets": [], "candles": {}, "comments": [], "authors": []}
    seen_markets, seen_authors = set(), set()
    for event_id, overlap, post in linked:
        market = chosen[event_id]
        market_id = market["market_id"]
        if market_id not in seen_markets:
            seen_markets.add(market_id)
            snapshot["markets"].append({
                "market_id": market_id, "event_id": event_id,
                "question": market["question"],
                "outcome_label": market["outcome_label"]})
            snapshot["candles"][market_id] = [
                {"ts": int(b["hour"].timestamp() * 1000), "close": b["close"],
                 "open": b["vwap"], "high": b["high"], "low": b["low"],
                 "volume": b["usd"] or b["fills"] or 0.0}
                for b in sorted(ticks[market_id], key=lambda b: b["hour"])]
        author = f'{post["source"]}:{post["author"]}'
        if author not in seen_authors:
            seen_authors.add(author)
            snapshot["authors"].append({"author_id": author,
                                        "name": post["author"]})
        snapshot["comments"].append({
            "comment_id": post["chatter_id"], "market_id": market_id,
            "event_id": event_id, "author_id": author,
            "created_at": post["posted_at"].replace(tzinfo=timezone.utc).isoformat(),
            "body": f'[{post["channel"]}] {post["body"]}',
            "reactions": post["score"], "reports": 0})

    from comments.run import build, score, report
    from scale.analyze import paired_bootstrap
    from scale.resolve import auroc

    schema, wiring, kept, counts = build(snapshot, horizon=args.horizon,
                                         muted=False)
    print(f">> labelled: {len(kept):,} posts "
          f"({sum(c['price_up'] for c in kept) / max(1, len(kept)):.1%} up)")
    print("   tables: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    rng = random.Random(args.seed)
    splits = {}
    for name in ("dev", "holdout"):
        group = [c for c in kept
                 if (c["posted_at"] < SPLIT_AT) == (name == "dev")]
        if len(group) > args.per_split:
            group = rng.sample(group, args.per_split)
        splits[name] = group
    ids = [c["comment_id"] for g in splits.values() for c in g]
    print(f">> scoring {len(ids):,} posts "
          f"({len(splits['dev'])} dev / {len(splits['holdout'])} holdout)")

    arms = {}
    for arm, muted in (("full", False), ("muted", True)):
        path = (CHATTER.parent /
                f"pred_{arm}_{args.horizon}h_{args.context_cells}.parquet")
        if path.exists() and not args.force:
            arms[arm] = {r["comment_id"]: r["p"]
                         for r in pq.read_table(path).to_pylist()}
            print(f"   reusing {path.name}")
            continue
        s, w, _, _ = build(snapshot, horizon=args.horizon, muted=muted)
        print(f"   scoring {arm} ...", flush=True)
        scored, truncated = score(s, w, ids, cells=args.context_cells,
                                  batch=args.batch_size)
        import pyarrow as pa
        pq.write_table(pa.table({"comment_id": list(scored),
                                 "p": [scored[i] for i in scored]}), path)
        print(f"   wrote {path.name} ({truncated} truncated)")
        arms[arm] = scored

    for name, group in splits.items():
        if len(group) < 30:
            continue
        gids = [c["comment_id"] for c in group]
        truth = [c["price_up"] for c in group]
        full = [arms["full"][i] for i in gids]
        muted = [arms["muted"][i] for i in gids]
        print(f"\n== {name}: {len(group)} posts, "
              f"{sum(truth) / len(truth):.1%} followed by an up move ==")
        report("always up (p=0.5)", [0.5] * len(group), truth)
        report("momentum (prior window)",
               [1 / (1 + 2.718 ** (-c["momentum"] / 0.02)) for c in group], truth)
        # The control that matters: a contract at 0.03 has almost no room to
        # fall and every tick up is a big relative move, so the price *level*
        # alone predicts the sign of the next move. If this scores as well as
        # the model, the model is reading the level, not the market.
        report("price level (1 - price)", [1 - c["before"] for c in group], truth)
        report("RT-J, chatter text", full, truth)
        report("RT-J, text muted", muted, truth)
        lo, hi = paired_bootstrap(full, muted, truth)
        print(f"  AUROC gain from the words: "
              f"{auroc(full, truth) - auroc(muted, truth):+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
