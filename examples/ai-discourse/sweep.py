"""Is RT-J losing to the baselines, or losing to the context budget?

The headline run truncates every single context: the database has ~23k
engagement rows and ~15k accounts, and a breadth-first walk from a post can
spend its whole budget on like-rows before it reaches anything that
discriminates. That is a claim about the retrieval policy, not about the model,
and it is cheap to test — so this sweeps the two knobs that decide what lands
in the context (cell budget, and how wide/deep the walk goes) against a fixed
subsample, and prints the best baseline on the same subsample as the bar.

Run it before believing any single number in the README.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import db
import predict
import topics

CONFIGS = [
    # (context_cells, bfs_width, max_hops)
    (2048, 32, 3),
    (8192, 32, 3),
    (8192, 8, 2),      # narrow + shallow: fewer rows, closer to the post
    (8192, 64, 2),     # wide + shallow: many neighbours, no third hop
    (16384, 16, 3),
]


def subsample(post_ids, n):
    """Evenly strided, not the first N — the feed is grouped by author, so a
    prefix is a handful of authors and tells you nothing."""
    if not n or n >= len(post_ids):
        return post_ids
    step = len(post_ids) / n
    return [post_ids[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(db.SNAPSHOT))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--ai-only", action="store_true")
    args = ap.parse_args()

    snapshot = db.load(Path(args.snapshot))
    anchor = db.iso(snapshot["anchor"])
    past, future = predict.split_engagements(snapshot, anchor)
    universe = topics.filter_posts(
        [p for p in snapshot["posts"] if p.get("in_universe")], args.ai_only)
    post_ids = subsample([p["post_id"] for p in universe], args.n)

    spec = predict.TARGETS["reach"]
    counts = predict.label_counts(snapshot, future, post_ids, spec["count"])
    k = predict.pick_threshold(counts, "auto")
    truth = [counts[p] >= k for p in post_ids]
    query = spec["query"].format(k=k)
    print(f"{len(post_ids)} posts, K={k}, base rate "
          f"{sum(truth)/len(truth):.1%} ({sum(truth)}/{len(truth)})\n")

    signals = predict.baselines(snapshot, past, post_ids, anchor)
    print("baselines on this subsample:")
    best = 0.0
    for name, signal in signals.items():
        area = predict.auroc([signal[p] for p in post_ids], truth)
        if area:
            best = max(best, area)
        print(f"   {name:<38} auroc "
              f"{'n/a' if area is None else f'{area:.3f}'}")
    print(f"\nbar to clear: {best:.3f}\n")

    print(f"{'cells':>7} {'bfs':>5} {'hops':>5} {'auroc':>7} {'acc':>7} "
          f"{'trunc':>7}")
    for cells, width, hops in CONFIGS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, table_counts = predict.run_query(
                snapshot, post_ids, anchor, query, context_cells=cells,
                batch_size=args.batch_size, bfs_width=width, max_hops=hops)
        scored = {p.id: float(p.probability) for p in result.predictions}
        area = predict.auroc([scored[p] for p in post_ids], truth)
        acc = predict.accuracy([scored[p] for p in post_ids], truth)
        trunc = table_counts.get("(contexts truncated)", 0)
        flag = "  <-- clears the bar" if area and area > best else ""
        print(f"{cells:>7} {width:>5} {hops:>5} "
              f"{'n/a' if area is None else f'{area:>7.3f}'} {acc:>7.3f} "
              f"{trunc:>3}/{len(post_ids):<3}{flag}", flush=True)


if __name__ == "__main__":
    main()
