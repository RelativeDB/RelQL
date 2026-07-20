"""Run the benchmark and write results.json.

    python -m olist.run --tasks bad_review,delivery_days --test 400

Every number in the report comes from this file; the HTML step only formats
what it finds here.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D, metrics as M, runners as R, tasks as T

OUT = Path(__file__).resolve().parent / "results.json"
SPLIT = pd.Timestamp("2018-05-01")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=",".join(t.name for t in T.TASKS))
    p.add_argument("--train", type=int, default=3000,
                   help="train rows sampled per task (0 = all)")
    p.add_argument("--test", type=int, default=600,
                   help="test rows scored per task (0 = all)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--split", default=str(SPLIT.date()))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--skip-finetune", action="store_true")
    return p.parse_args()


def _sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if not n or n >= len(frame):
        return frame
    return frame.sample(n=n, random_state=seed).sort_values("anchor")


def main() -> None:
    args = parse_args()
    split = pd.Timestamp(args.split)
    names = [n.strip() for n in args.tasks.split(",") if n.strip()]

    print("loading Olist ...", flush=True)
    o = D.load()
    feats = D.order_features(o)
    data_end = o.orders.order_purchase_timestamp.max()
    print(f"data ends {data_end.date()}", flush=True)
    schema = R.build_schema()

    from harness import datasets as H          # reuse the frame -> retriever glue
    frames = R.build_frames(o, feats)
    t0 = time.perf_counter()
    wiring = H._wire(schema, frames)
    wiring_s = time.perf_counter() - t0

    report = {
        "dataset": "Brazilian E-Commerce Public Dataset by Olist",
        "dataset_url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        "generated": pd.Timestamp.utcnow().isoformat(),
        "precision": "fp32",
        "split": str(split.date()),
        "data_end": str(data_end.date()),
        "machine": {"platform": platform.platform(),
                    "processor": platform.processor() or platform.machine()},
        "table_sizes": o.describe(),
        "wiring_seconds": wiring_s,
        "sparsity": {
            "note": ("Olist keys every order to a fresh customer_id; "
                     "customer_unique_id is the person. 97% of people order "
                     "exactly once, so most rows are cold-start."),
        },
        "tasks": [],
    }

    for name in names:
        task = T.BY_NAME[name]
        print(f"\n=== {task.name} ({task.kind}) ===", flush=True)
        frame = task.frame(o)
        # An outcome is only observable if its window closed before the data
        # ends. Without this the tail of the test split carries no positives
        # and the comparison measures censoring instead of the models.
        horizon = pd.Timedelta(days=task.horizon_days)
        observable = frame.anchor + horizon <= data_end
        dropped = int((~observable).sum())
        frame = frame[observable]
        train_all, test_all = T.temporal_split(frame, split)
        if len(train_all) == 0 or len(test_all) == 0:
            print("  skipped: split leaves one side empty")
            continue
        train = _sample(train_all, args.train, 0)
        test = _sample(test_all, args.test, 1)

        depth = D.history_depth(o, test, task.entity_table, "anchor").to_numpy()
        y = test.label.to_numpy()
        entry = {
            "name": task.name, "kind": task.kind, "title": task.title,
            "question": task.question, "query": task.query, "notes": task.notes,
            "entity_table": task.entity_table,
            "n_train_available": len(train_all), "n_test_available": len(test_all),
            "horizon_days": task.horizon_days,
            "rows_dropped_unobservable": dropped,
            "n_train": len(train), "n_test": len(test),
            "label_balance": (pd.Series(y).value_counts(normalize=True)
                              .sort_index().round(4).to_dict()),
            "depth_counts": {str(k): int(v) for k, v in
                             D.depth_bucket(depth).value_counts().sort_index().items()},
            "systems": [],
        }

        runs = [("xgboost", lambda: R.run_xgboost(task, train, test)),
                ("relativedb-zero-shot",
                 lambda: R.run_relativedb(task, train, test, schema, wiring,
                                          finetune=False))]
        if not args.skip_finetune:
            runs.append(("relativedb-finetuned",
                         lambda: R.run_relativedb(task, train, test, schema,
                                                  wiring, finetune=True,
                                                  epochs=args.epochs)))

        for label, fn in runs:
            print(f"  {label} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                res = fn()
            except Exception as e:                       # keep the sweep going
                print(f"FAILED: {type(e).__name__}: {str(e)[:90]}")
                entry["systems"].append({"system": label, "error":
                                         f"{type(e).__name__}: {e}"})
                continue
            wall = time.perf_counter() - t0
            entry["systems"].append({
                "system": res.system,
                "train_seconds": res.train_seconds,
                "inference_ms_per_row": res.inference_ms_per_row,
                "wall_seconds": wall,
                "overall": M.score(task.kind, y, res.pred),
                "by_depth": M.by_depth(task.kind, y, res.pred, depth),
                "detail": res.detail,
            })
            head = M.headline(task.kind)
            got = entry["systems"][-1]["overall"].get(head)
            print(f"{head}={got:.4f}  train={res.train_seconds:.1f}s  "
                  f"{res.inference_ms_per_row:.2f} ms/row" if got is not None
                  else f"done ({wall:.1f}s)")

        entry["headline_metric"] = M.headline(task.kind)
        report["tasks"].append(entry)

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
