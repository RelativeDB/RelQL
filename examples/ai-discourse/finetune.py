"""Light experiment: does fitting a head on the past beat zero-shot?

Zero-shot RT-J loses to a one-line heuristic on this task (see the README), so
the obvious next question is whether the signal is there and only the readout
is wrong. The cheap answer would be ``Engine.fit_head``, which freezes the
backbone — but it is limited to multiclass and ranking adapters, and this
target is scalar binary, so it refuses. That leaves ``Engine.finetune``, which
differentiates through the whole checkpoint. It is the expensive option, so
this stays deliberately small: a few hundred examples and a couple of epochs.

**Training examples come from the same snapshot, strictly earlier in time.**
Each training post gets its *own* anchor — its creation time plus
``--age-hours``, so it is the same age at its anchor as a test post is at the
test anchor — and its label is read from that anchor's own 48-hour window.
Every training window is required to close before the test anchor:

    created + age + 48h  <=  test anchor

so no training label can see a moment the test anchor has not already reached.
The test set is untouched: the same 211 posts, at the same anchor, as
``predict.py``.

This is a *light* experiment and the honest read on it is in the README: a few
hundred training examples drawn from one 3-week window of one community is not
enough to conclude much, and the training posts are a biased sample (they are
the most recent posts of each author before the universe window, so prolific
authors are over-represented).
"""
from __future__ import annotations

import argparse
import warnings
from datetime import timedelta
from pathlib import Path

from relativedb import (ContextPolicy, Engine, ExecutionInput, ModelConfig,
                        RtNativeBackend)

import db
import predict
import topics

HORIZON = predict.HORIZON


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(db.SNAPSHOT))
    ap.add_argument("--age-hours", type=int, default=18,
                    help="how old a training post is at its own anchor; the "
                         "median age of a test post at the test anchor")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--context-cells", type=int, default=2048,
                    help="used for BOTH training and evaluation, so the "
                         "comparison against zero-shot is like for like")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--ai-only", action="store_true")
    ap.add_argument("--out-dir", default="data/finetuned")
    args = ap.parse_args()

    snapshot = db.load(Path(args.snapshot))
    test_anchor = db.iso(snapshot["anchor"])
    past, future = predict.split_engagements(snapshot, test_anchor)
    spec = predict.TARGETS["reach"]

    # -- test set: exactly what predict.py scores ------------------------
    test_universe = topics.filter_posts(
        [p for p in snapshot["posts"] if p.get("in_universe")], args.ai_only)
    test_ids = [p["post_id"] for p in test_universe]
    test_counts = predict.label_counts(snapshot, future, test_ids,
                                       spec["count"])
    k = predict.pick_threshold(test_counts, "auto")
    truth = [test_counts[p] >= k for p in test_ids]
    query = spec["query"].format(k=k)

    # -- training set: earlier posts, each at its own cut-off ------------
    cutoff = test_anchor - timedelta(hours=args.age_hours + HORIZON)
    train_pairs = []
    for post in snapshot["posts"]:
        created = db.iso(post["created_at"])
        if created is None or created > cutoff:
            continue
        anchor = created + timedelta(hours=args.age_hours)
        _, after = predict.split_engagements(snapshot, anchor)
        count = predict.label_counts(snapshot, after, [post["post_id"]],
                                     spec["count"])[post["post_id"]]
        train_pairs.append((post["post_id"], anchor, count >= k))
    train_pairs.sort(key=lambda t: t[1])
    if args.max_train and args.max_train < len(train_pairs):
        # Strided, not a tail slice: the tail is the most recent few days and
        # is nowhere near class-balanced.
        step = len(train_pairs) / args.max_train
        train_pairs = [train_pairs[int(i * step)]
                       for i in range(args.max_train)]

    positives = sum(1 for _, _, y in train_pairs if y)
    print(f"test  : {len(test_ids)} posts at {test_anchor:%Y-%m-%d %H:%M}, "
          f"K={k}, base rate {sum(truth)/len(truth):.1%}")
    print(f"train : {len(train_pairs)} posts, {positives} positive "
          f"({positives/max(1,len(train_pairs)):.1%}), anchors "
          f"{train_pairs[0][1]:%m-%d} .. {train_pairs[-1][1]:%m-%d}")
    print(f"        every training window closes by "
          f"{max(a for _, a, _ in train_pairs) + timedelta(hours=HORIZON):%m-%d %H:%M}"
          f" <= test anchor {test_anchor:%m-%d %H:%M}")
    if not train_pairs or positives in (0, len(train_pairs)):
        raise SystemExit("degenerate training labels — widen the window")

    schema, wiring, _ = db.build(snapshot)
    engine = Engine(
        schema, wiring,
        model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                      max_seq_len=args.context_cells,
                                      batch_size=args.batch_size),
        context_policy=ContextPolicy(max_context_cells=args.context_cells,
                                     local_context_cells=args.context_cells // 2,
                                     bfs_width=32, max_hops=3, seed=0))

    # Zero-shot on the same test set at the same context budget — the number
    # the fine-tune has to beat, measured here rather than quoted from another
    # run at another budget.
    print("\n>> zero-shot baseline at this context budget")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zero = engine.execute(ExecutionInput(
            query=query, anchor_time=test_anchor,
            params={"ids": list(test_ids)}))
    zero_scores = {p.id: float(p.probability) for p in zero.predictions}

    print(f"\n>> fine-tuning the full checkpoint "
          f"({args.epochs} epochs, lr {args.learning_rate}, "
          f"{len(train_pairs)} examples) — this is the slow part")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = engine.finetune(
            query,
            anchors=[a for _, a, _ in train_pairs],
            entity_ids=[i for i, _, _ in train_pairs],
            # The query filters on :ids; at fit time the entities are named
            # explicitly, so bind it to the training ids to keep the parsed
            # query the same shape it will have at serve time.
            params={"ids": [i for i, _, _ in train_pairs]},
            labels={(i, a): y for i, a, y in train_pairs},
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, output_dir=args.out_dir)
    if checkpoint.losses:
        print(f"   {checkpoint.steps} steps over {checkpoint.examples} "
              f"examples in {checkpoint.seconds:.0f}s, "
              f"loss {checkpoint.losses[0]:.4f} -> {checkpoint.losses[-1]:.4f}")
    print(f"   wrote {checkpoint.path}")

    print("\n>> scoring the held-out test set with the fine-tuned checkpoint")
    served = Engine(
        schema, wiring,
        model_config=ModelConfig(
            classification_model_uri=checkpoint.model_uri,
            normalization_mode=checkpoint.normalization_mode),
        model_backend=RtNativeBackend(
            schema=schema, wiring=wiring, max_seq_len=args.context_cells,
            batch_size=args.batch_size,
            column_stats=checkpoint.column_stats,
            normalization_mode=checkpoint.normalization_mode),
        context_policy=ContextPolicy(max_context_cells=args.context_cells,
                                     local_context_cells=args.context_cells // 2,
                                     bfs_width=32, max_hops=3, seed=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = served.execute(ExecutionInput(
            query=query, anchor_time=test_anchor,
            params={"ids": list(test_ids)}))
    tuned = {p.id: float(p.probability) for p in result.predictions}

    print(f"\n== held-out test set (n={len(truth)}, "
          f"{args.context_cells} cells) ==")
    signals = predict.baselines(snapshot, past, test_ids, test_anchor)
    signals["RT-J  (zero-shot)"] = zero_scores
    signals["RT-J  (fine-tuned)"] = tuned
    for name, signal in signals.items():
        predict.report(name, [signal[p] for p in test_ids], truth)


if __name__ == "__main__":
    main()
