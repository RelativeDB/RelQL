"""Did the headline run starve the random walks?

The reference RT-J evaluation (``rt.eval_utils.build_evaluator``) runs

    ctx_size=8192, local_ctx_size=256, bfs_width=32,
    num_walks=10_000, walk_length=20

The graph reach is the 10,000 walks of length 20, and ``ReferenceTraversal``
builds the walk graph by unbounded BFS over the reachable component, so nothing
caps the hop count. What *does* compete with it is ``local_context_cells``: the
budget handed to rows sitting next to the target. The first run in this example
set it to ``context_cells // 2`` — 4,096 of 8,192 cells, sixteen times the
reference — so half the context was spent before a single walk-ranked row was
admitted.

This sweeps that one knob with everything else at reference values.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import db
import predict
import topics
from sweep import subsample

LOCAL_CELLS = [256, 512, 1024, 2048, 4096]      # 256 = reference, 4096 = the bug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(db.SNAPSHOT))
    ap.add_argument("--n", type=int, default=0, help="0 = the whole universe")
    ap.add_argument("--context-cells", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--walk-length", type=int, default=20)
    ap.add_argument("--num-walks", type=int, default=10_000)
    ap.add_argument("--ai-only", action="store_true")
    ap.add_argument("--locals", default="",
                    help="comma-separated local_context_cells to try")
    args = ap.parse_args()
    locals_to_try = ([int(x) for x in args.locals.split(",")]
                     if args.locals else LOCAL_CELLS)

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
          f"{sum(truth)/len(truth):.1%} ({sum(truth)}/{len(truth)})")
    print(f"ctx={args.context_cells}  bfs_width=32  num_walks={args.num_walks}"
          f"  walk_length={args.walk_length}\n")

    best = 0.0
    for name, signal in predict.baselines(snapshot, past, post_ids,
                                          anchor).items():
        area = predict.auroc([signal[p] for p in post_ids], truth)
        if area:
            best = max(best, area)
    print(f"bar to clear (best baseline): {best:.3f}\n")

    print(f"{'local_cells':>12} {'auroc':>7} {'acc':>7} {'brier':>7} "
          f"{'trunc':>9}  note")
    for local in locals_to_try:
        if local >= args.context_cells:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, table_counts = predict.run_query(
                snapshot, post_ids, anchor, query,
                context_cells=args.context_cells, batch_size=args.batch_size,
                local_cells=local, num_walks=args.num_walks,
                walk_length=args.walk_length)
        scored = {p.id: float(p.probability) for p in result.predictions}
        area = predict.auroc([scored[p] for p in post_ids], truth)
        acc = predict.accuracy([scored[p] for p in post_ids], truth)
        bri = predict.brier([scored[p] for p in post_ids], truth)
        trunc = table_counts.get("(contexts truncated)", 0)
        note = ("reference" if local == 256 else
                "the original run" if local == args.context_cells // 2 else "")
        spread = max(scored.values()) - min(scored.values())
        print(f"{local:>12} {area:>7.3f} {acc:>7.3f} {bri:>7.3f} "
              f"{trunc:>4}/{len(post_ids):<4} {note}"
              f"{'' if note else ''}   (p spread {spread:.3f})", flush=True)


if __name__ == "__main__":
    main()
