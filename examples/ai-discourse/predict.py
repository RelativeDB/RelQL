"""Which of today's AI posts will matter — scored without trusting the likes.

The complaint this answers
--------------------------
"I want to know what's worth reading in AI, and I can't just sort by likes."
Correct, and the snapshot says why: across a whole day of AI Bluesky, the
overwhelming majority of engagement on a post comes from accounts that already
follow its author. Like count is mostly a readout of audience size. It tells
you who has a big following, not who said something.

So the target is not "will this get engagement". It is:

    PREDICT COUNT(engagements.* WHERE engagements.from_follower = FALSE)
            OVER (48 HOURS FOLLOWING) >= K
    FROM posts
    WHERE posts.post_id IN :ids
    RETURN PROBABILITY

*"Will this post reach past the author's own audience?"* — did it get picked up
by people who were not already listening. That is the thing a big account
cannot buy with follower count, and it is only expressible because the database
carries the follow graph next to the engagement graph.

Two contrast targets run against the identical database: ``replies`` (does it
start a conversation rather than collect applause) and ``volume`` (plain
engagement count — the naive target, included to show what an easy target looks
like next to a hard one).

Nothing is trained. RT-J is frozen, no head is fitted. The only difference
between the two arms of the ablation is whether the follow graph exists.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

from relativedb import ContextPolicy, Engine, ExecutionInput, RtNativeBackend
from relativedb.scoring import ContextTruncationWarning

import db
import topics

HORIZON = 48                                       # hours

TARGETS = {
    "reach": {
        "blurb": "engagement from accounts that do NOT already follow the author",
        "query": ("PREDICT COUNT(engagements.* WHERE "
                  "engagements.from_follower = FALSE) "
                  f"OVER ({HORIZON} HOURS FOLLOWING) >= {{k}} "
                  "FROM posts WHERE posts.post_id IN :ids RETURN PROBABILITY"),
        "count": lambda e, follower: not follower,
    },
    "replies": {
        "blurb": "replies — conversation, which costs more than a like",
        "query": ("PREDICT COUNT(engagements.* WHERE "
                  "engagements.kind = 'reply') "
                  f"OVER ({HORIZON} HOURS FOLLOWING) >= {{k}} "
                  "FROM posts WHERE posts.post_id IN :ids RETURN PROBABILITY"),
        "count": lambda e, follower: e["kind"] == "reply",
    },
    "volume": {
        "blurb": "any engagement at all — the naive target, for contrast",
        "query": (f"PREDICT COUNT(engagements.*) "
                  f"OVER ({HORIZON} HOURS FOLLOWING) >= {{k}} "
                  "FROM posts WHERE posts.post_id IN :ids RETURN PROBABILITY"),
        "count": lambda e, follower: True,
    },
}


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


def accuracy(scores, truth, cut=0.5):
    return sum((s >= cut) == y for s, y in zip(scores, truth)) / len(truth)


def brier(scores, truth):
    return sum((s - y) ** 2 for s, y in zip(scores, truth)) / len(truth)


def logloss(scores, truth):
    eps = 1e-6
    return -sum(math.log(max(eps, s if y else 1 - s))
                for s, y in zip(scores, truth)) / len(truth)


def report(name, scores, truth):
    area = auroc(scores, truth)
    row = {"name": name, "n": len(truth), "acc": accuracy(scores, truth),
           "auroc": area, "brier": brier(scores, truth),
           "logloss": logloss(scores, truth)}
    print(f"  {name:<38} acc {row['acc']:.3f}   "
          f"auroc {'  n/a' if area is None else f'{area:.3f}'}   "
          f"brier {row['brier']:.3f}   logloss {row['logloss']:.3f}")
    return row


# ---------------------------------------------------------------------------
# labels, computed from the held-out window only
# ---------------------------------------------------------------------------
def split_engagements(snapshot, anchor):
    """post_id -> (rows at or before the anchor, rows inside the horizon)."""
    end = anchor + timedelta(hours=HORIZON)
    before, after = {}, {}
    for edge in snapshot["engagements"]:
        when = db.iso(edge.get("at"))
        if when is None:                    # undated: unusable on both sides
            continue
        if when <= anchor:
            before.setdefault(edge["post_id"], []).append(edge)
        elif when <= end:
            after.setdefault(edge["post_id"], []).append(edge)
    return before, after


def label_counts(snapshot, future, post_ids, counter):
    """How many qualifying engagements each post got in the held-out window."""
    author_of = {p["post_id"]: p["author_did"] for p in snapshot["posts"]}
    flags = snapshot.get("follower_flags", {})
    counts = {}
    for post_id in post_ids:
        total = 0
        for edge in future.get(post_id, []):
            follower = flags.get(
                f'{author_of.get(post_id)}|{edge["actor_did"]}', False)
            total += bool(counter(edge, follower))
        counts[post_id] = total
    return counts


def pick_threshold(counts, requested):
    """``auto`` puts the split at the median so the task is not degenerate.

    This is a single global constant, not a per-post choice, and it is reported
    alongside the base rate. A threshold that makes 95%% of posts negative
    yields an AUROC anyone can beat by predicting the author's follower count,
    which measures nothing."""
    if requested != "auto":
        return int(requested)
    values = sorted(counts.values())
    if not values:
        return 1
    median = values[len(values) // 2]
    return max(1, median + 1)


# ---------------------------------------------------------------------------
# baselines — everything you could do without the model
# ---------------------------------------------------------------------------
def baselines(snapshot, past, post_ids, anchor):
    """The signals a person would actually reach for, including the one the
    friend distrusts (early likes) and the one that secretly drives it
    (follower count)."""
    members = {m["did"]: m for m in snapshot["members"]}
    author_of = {p["post_id"]: p["author_did"] for p in snapshot["posts"]}
    created = {p["post_id"]: db.iso(p["created_at"]) for p in snapshot["posts"]}
    flags = snapshot.get("follower_flags", {})

    early, early_rate, followers, age, early_outside = {}, {}, {}, {}, {}
    for post_id in post_ids:
        seen = past.get(post_id, [])
        early[post_id] = len(seen)
        hours = max(0.5, (anchor - created[post_id]).total_seconds() / 3600.0)
        age[post_id] = hours
        early_rate[post_id] = len(seen) / hours
        author = author_of.get(post_id)
        followers[post_id] = (members.get(author, {}) or {}).get("followers") or 0
        early_outside[post_id] = sum(
            1 for e in seen
            if not flags.get(f'{author}|{e["actor_did"]}', False))

    def squash(raw, scale):
        top = max(raw.values()) or 1.0
        return {k: 1 / (1 + math.exp(-(v / top - 0.5) / scale))
                for k, v in raw.items()}

    return {
        "early likes (engagement so far)": squash(early, 0.25),
        "early engagement rate (per hour)": squash(early_rate, 0.25),
        "author follower count": squash(followers, 0.25),
        "early NON-follower engagement": squash(early_outside, 0.25),
        "constant (p=0.5)": {p: 0.5 for p in post_ids},
    }


# ---------------------------------------------------------------------------
def run_query(snapshot, post_ids, anchor, query, *, context_cells, batch_size,
              with_follows=True, with_flag=True, bfs_width=32, max_hops=2,
              local_cells=256, num_walks=10_000, walk_length=20):
    """Defaults here are the reference RT-J evaluation settings.

    ``rt.eval_utils.build_evaluator`` uses ``local_ctx_size=256``,
    ``bfs_width=32``, ``num_walks=10_000``, ``walk_length=20`` at
    ``ctx_size=8192``. The graph reach comes from those 10,000 random walks of
    length 20 — ``ReferenceTraversal`` builds the walk graph by *unbounded* BFS
    over the reachable component, so ``max_hops`` does not cap it. What does cap
    it is ``local_context_cells``: every cell spent on rows adjacent to the
    target is a cell the walk-ranked rows do not get."""
    schema, wiring, rows = db.build(snapshot, with_follows=with_follows,
                                    with_flag=with_flag)
    engine = Engine(
        schema, wiring,
        model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                      max_seq_len=context_cells,
                                      batch_size=batch_size),
        context_policy=ContextPolicy(max_context_cells=context_cells,
                                     local_context_cells=local_cells,
                                     bfs_width=bfs_width, max_hops=max_hops,
                                     num_walks=num_walks,
                                     walk_length=walk_length,
                                     seed=0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = engine.execute(ExecutionInput(
            query=query, anchor_time=anchor, params={"ids": list(post_ids)}))
    truncated = sum(1 for w in caught
                    if isinstance(w.message, ContextTruncationWarning))
    counts = {name: len(table_rows) for name, table_rows in rows.items()}
    if truncated:
        counts["(contexts truncated)"] = truncated
    return result, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(db.SNAPSHOT))
    ap.add_argument("--target", default="reach", choices=[*TARGETS, "all"])
    ap.add_argument("--threshold", default="auto",
                    help="'auto' splits at the median, or give an integer K")
    ap.add_argument("--context-cells", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--dump", help="write per-post predictions here")
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N posts (smoke tests)")
    ap.add_argument("--ai-only", action="store_true",
                    help="narrow the universe to posts that explicitly mention "
                         "AI (crude keyword filter — see topics.py)")
    args = ap.parse_args()

    snapshot = db.load(Path(args.snapshot))
    anchor = db.iso(snapshot["anchor"])
    past, future = split_engagements(snapshot, anchor)

    universe = [p for p in snapshot["posts"] if p.get("in_universe")]
    if args.ai_only:
        kept = topics.filter_posts(universe, True)
        print(f"--ai-only: {len(kept)}/{len(universe)} posts mention AI "
              f"explicitly (keyword filter; see topics.py)")
        universe = kept
    if args.limit:
        universe = universe[:args.limit]
    post_ids = [p["post_id"] for p in universe]
    text = {p["post_id"]: p["text"] for p in universe}
    members = {m["did"]: m for m in snapshot["members"]}
    handle = {p["post_id"]: (members.get(p["author_did"], {}) or {})
              .get("handle", "?") for p in universe}

    print(f"snapshot fetched {snapshot['fetched_at'][:16]}   "
          f"anchor {anchor:%Y-%m-%d %H:%M} UTC   horizon {HORIZON}h")
    print(f"universe: {len(post_ids)} posts from "
          f"{len({p['author_did'] for p in universe})} authors, created in the "
          f"{snapshot['universe_hours']}h before the anchor")

    # The premise, measured rather than asserted.
    flags = snapshot.get("follower_flags", {})
    if flags:
        share = sum(flags.values()) / len(flags)
        print(f"premise check: {share:.1%} of all (author, engager) pairs in "
              f"the snapshot are people who ALREADY follow the author")

    targets = list(TARGETS) if args.target == "all" else [args.target]
    output = {}

    for name in targets:
        spec = TARGETS[name]
        counts = label_counts(snapshot, future, post_ids, spec["count"])
        k = pick_threshold(counts, args.threshold)
        truth = [counts[p] >= k for p in post_ids]
        base = sum(truth) / len(truth) if truth else 0.0
        query = spec["query"].format(k=k)

        print(f"\n{'=' * 74}\n== target '{name}': {spec['blurb']}")
        print(f"   held-out counts: median {sorted(counts.values())[len(counts)//2]}, "
              f"max {max(counts.values())}, threshold K={k}, "
              f"base rate {base:.1%} ({sum(truth)}/{len(truth)})")
        print(f"   {query}")
        if base in (0.0, 1.0):
            print("   !! degenerate label at this threshold — skipping")
            continue

        signals = baselines(snapshot, past, post_ids, anchor)
        result, table_counts = run_query(
            snapshot, post_ids, anchor, query,
            context_cells=args.context_cells, batch_size=args.batch_size)
        print("   database: " + ", ".join(f"{n}={c}"
                                          for n, c in table_counts.items()))
        scored = {p.id: float(p.probability) for p in result.predictions}
        signals["RT-J  (full graph)"] = scored

        if not args.skip_ablation:
            # A true ablation: identical query, identical labels, and the
            # `follows` table taken away. Any gap is what walking the raw edges
            # was worth.
            ablated, _ = run_query(
                snapshot, post_ids, anchor, query, with_follows=False,
                context_cells=args.context_cells, batch_size=args.batch_size)
            signals["RT-J  (ablated: no follows table)"] = {
                p.id: float(p.probability) for p in ablated.predictions}

            # NOT an ablation — the naive question, asked of a database with no
            # graph in it at all, then scored against the same labels. It says
            # how far "predict engagement" gets you when what you wanted was
            # "predict reach". Kept visibly separate for that reason.
            naive_query = TARGETS["volume"]["query"].format(k=k)
            naive, _ = run_query(
                snapshot, post_ids, anchor, naive_query, with_follows=False,
                with_flag=False, context_cells=args.context_cells,
                batch_size=args.batch_size)
            signals["RT-J  (naive volume query, no graph)"] = {
                p.id: float(p.probability) for p in naive.predictions}

        print(f"\n== target '{name}' on the held-out window (n={len(truth)}, "
              f"{sum(truth)} positive) ==")
        output[name] = {
            "k": k, "base_rate": base, "query": query,
            "rows": [report(sig, [signals[sig][p] for p in post_ids], truth)
                     for sig in signals],
        }

        if name == args.target or args.target != "all":
            print(f"\n== the candidates: top {args.show} posts RT-J picked ==")
            for post_id in sorted(post_ids, key=lambda p: -scored[p])[:args.show]:
                mark = "HIT " if counts[post_id] >= k else "miss"
                print(f"  {mark} p={scored[post_id]:.3f}  outside={counts[post_id]:<3} "
                      f"likes_so_far={len(past.get(post_id, [])):<4} "
                      f"@{handle[post_id][:24]:<24} "
                      f"{text[post_id][:88].replace(chr(10), ' ')}")

            print(f"\n== what sorting by early likes would have given you ==")
            by_likes = sorted(post_ids, key=lambda p: -len(past.get(p, [])))
            for post_id in by_likes[:args.show]:
                mark = "HIT " if counts[post_id] >= k else "miss"
                print(f"  {mark} likes_so_far={len(past.get(post_id, [])):<4} "
                      f"outside={counts[post_id]:<3} "
                      f"@{handle[post_id][:24]:<24} "
                      f"{text[post_id][:88].replace(chr(10), ' ')}")

        if args.dump:
            output[name]["per_post"] = [
                {"post_id": p, "handle": handle[p], "text": text[p],
                 "label_count": counts[p], "label": counts[p] >= k,
                 **{s: signals[s][p] for s in signals}} for p in post_ids]

    if args.dump:
        Path(args.dump).write_text(json.dumps(
            {"anchor": anchor.isoformat(), "horizon_hours": HORIZON,
             "context_cells": args.context_cells, "targets": output}, indent=1))
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
