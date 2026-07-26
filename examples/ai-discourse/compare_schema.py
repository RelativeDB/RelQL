"""Old sparse-event schema vs new dense multi-table schema, same query.

Prints the context composition for both so you can see where the cell budget
actually goes, then (with --score) runs the real query against each.

The query, labels and anchor are identical between arms. The only thing that
changes is the shape of the database.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from relativedb import ContextPolicy, Engine, ExecutionInput, RtNativeBackend

import db
import db_dense
import predict


def context_report(engine, query, anchor, ids, label):
    ex = engine.execute(ExecutionInput(query="EXPLAIN CONTEXT " + query,
                                       anchor_time=anchor,
                                       params={"ids": ids[:1]}))
    c = ex.context
    total = c["total_cells"] or 1
    print(f"\n  == {label} ==")
    print(f"     total_rows={c['total_rows']}  total_cells={c['total_cells']}  "
          f"hit_budget={c['contexts_hit_cell_budget']}  "
          f"links_traversed={c['links_traversed']}")
    print(f"     {'table':<16}{'rows':>8}{'cells':>8}{'share':>8}")
    for name, v in sorted(c["tables"].items(),
                          key=lambda kv: -kv[1]["cells"]):
        bar = "#" * int(28 * v["cells"] / total)
        print(f"     {name:<16}{v['rows']:>8}{v['cells']:>8}"
              f"{v['cells']/total:>7.0%}  {bar}")
    if c.get("tables_unreachable"):
        print(f"     unreachable: {c['tables_unreachable']}")
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-cells", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--score", action="store_true",
                    help="also run the query and report AUROC (slow)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", default="",
                    help="comma-separated subset to run")
    args = ap.parse_args()

    snapshot = db.load()
    anchor = db.iso(snapshot["anchor"])
    past, future = predict.split_engagements(snapshot, anchor)
    universe = [p for p in snapshot["posts"] if p.get("in_universe")]
    if args.limit:
        universe = universe[:args.limit]
    ids = [p["post_id"] for p in universe]

    spec = predict.TARGETS["reach"]
    counts = predict.label_counts(snapshot, future, ids, spec["count"])
    k = predict.pick_threshold(counts, "auto")
    truth = [counts[p] >= k for p in ids]
    query = spec["query"].format(k=k)

    print(f"{len(ids)} posts, K={k}, base rate {sum(truth)/len(truth):.1%}, "
          f"anchor {anchor:%Y-%m-%d %H:%M}, {args.context_cells} cells")
    print(f"  {query}")

    def make(schema, wiring):
        return Engine(
            schema, wiring,
            model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                          max_seq_len=args.context_cells,
                                          batch_size=args.batch_size),
            # reference RT-J evaluation settings
            context_policy=ContextPolicy(max_context_cells=args.context_cells,
                                         local_context_cells=256,
                                         bfs_width=32, num_walks=10_000,
                                         walk_length=20, seed=0))

    VARIANTS = {
        "old": None,
        "dense": {},
        "dense+fk": {"fk_features": True},
        "dense+followcols": {"follow_columns": True},
        "dense-actorlink": {"with_actor_link": False},
        "dense-actor+followcols": {"with_actor_link": False,
                                   "follow_columns": True},
        "dense-actor+fk+followcols": {"with_actor_link": False,
                                      "fk_features": True,
                                      "follow_columns": True},
    }
    wanted = args.variants.split(",") if args.variants else list(VARIANTS)
    arms = []
    for name in wanted:
        kw = VARIANTS[name]
        if kw is None:
            sc, wi, rr = db.build(snapshot)
        else:
            sc, wi, rr = db_dense.build(snapshot, **kw)
        arms.append((name, make(sc, wi), rr))

    for label, engine, rows in arms:
        print(f"\n{label}: " + ", ".join(f"{n}={len(v)}"
                                         for n, v in rows.items()))
        context_report(engine, query, anchor, ids, label)

    if not args.score:
        print("\n(pass --score to run the query and compare AUROC)")
        return

    signals = predict.baselines(snapshot, past, ids, anchor)
    print(f"\n== held-out AUROC (n={len(truth)}) ==")
    for name, signal in signals.items():
        predict.report(name, [signal[p] for p in ids], truth)
    for label, engine, _ in arms:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = engine.execute(ExecutionInput(query=query,
                                                anchor_time=anchor,
                                                params={"ids": ids}))
        scored = {p.id: float(p.probability) for p in res.predictions}
        vals = [scored[p] for p in ids]
        predict.report(f"RT-J  {label}", vals, truth)
        print(f"         prediction spread {min(vals):.3f}..{max(vals):.3f}")


if __name__ == "__main__":
    main()
