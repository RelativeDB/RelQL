#!/usr/bin/env python
"""Fit optional frozen RelativeDB task heads on RelBench validation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.run_native_on_reference import NativeReferenceAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-checkpoint", required=True)
    parser.add_argument("--regression-checkpoint", required=True)
    parser.add_argument("--pre-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--local-ctx-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bfs-width", type=int, default=32)
    parser.add_argument("--num-walks", type=int, default=10_000)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--items-per-task", type=int, default=10_000_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--library")
    parser.add_argument("--native-device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()

    from relativedb import TaskType
    from relativedb.rt_native import RtNativeBackend
    from rt.eval_utils import build_evaluator
    from rt.tasks import eval_tasks

    selected = set(args.tasks or ())
    tasks = [t for t in eval_tasks(args.pre_dir, splits=("val",))
             if not selected or t.db_name in selected
             or f"{t.db_name}/{t.table_name}" in selected]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for kind, checkpoint in (("clf", args.classification_checkpoint),
                             ("reg", args.regression_checkpoint)):
        these = [t for t in tasks if t.task_type == kind]
        if not these:
            continue
        adapter = NativeReferenceAdapter(
            checkpoint, args.library, args.native_device, collect=True)
        evaluator = build_evaluator(
            these, args.pre_dir, embedding_model="all-MiniLM-L12-v2", d_text=384,
            device="cpu", ctx_size=args.ctx_size,
            local_ctx_size=args.local_ctx_size, bfs_width=args.bfs_width,
            num_walks=args.num_walks, walk_length=args.walk_length,
            items_per_task=args.items_per_task, num_workers=0,
            tokens_per_gpu=args.ctx_size * args.batch_size, mmap_populate=True)
        # Driving evaluate_raw collects target features/labels in the adapter.
        list(evaluator.evaluate_raw([(adapter, "")], [args.ctx_size]))
        backend = RtNativeBackend(lib_path=args.library)
        for task in these:
            task_id = f"{task.db_name}/{task.table_name}"
            chunks = adapter.collected.get(task_id, [])
            if not chunks:
                continue
            features = np.concatenate([x for x, _ in chunks])
            labels = np.concatenate([y for _, y in chunks]).astype(np.float32)
            task_type = (TaskType.BINARY_CLASSIFICATION if kind == "clf"
                         else TaskType.REGRESSION)
            if kind == "clf":
                labels = (labels > 0).astype(np.float32)
            head = backend.fit_head(
                adapter.model, task_type, features, labels,
                np.asarray([0, len(labels)], np.int32), 1,
                epochs=args.epochs)
            path = out_dir / f"{task.db_name}__{task.table_name}.head"
            head.save(str(path))
            manifest[task_id] = str(path.resolve())
    manifest_path = out_dir / "heads.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
