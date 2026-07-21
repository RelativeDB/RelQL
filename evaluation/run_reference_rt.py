#!/usr/bin/env python
"""Run the reference PyTorch RT evaluator with an explicit device."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pre-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-type", choices=("clf", "reg"), required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"),
                        default="auto",
                        help="auto preserves the reference model's bfloat16 path")
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--local-ctx-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="examples per evaluator batch; context length is unchanged")
    parser.add_argument("--bfs-width", type=int, default=32)
    parser.add_argument("--num-walks", type=int, default=10_000)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--items-per-task", type=int, default=10_000_000)
    args = parser.parse_args()

    from rt.checkpoints import load_rt_model
    from rt.eval_utils import build_evaluator, run_and_report
    from rt.recipes import get_tasks

    model, config = load_rt_model(args.checkpoint, device=args.device,
                                  compile=False)
    torch = __import__("torch")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = model.to(args.device).to(dtype)
    tasks = [t for t in get_tasks("relbench_eval_test", args.pre_dir)
             if t.task_type == args.task_type]
    if args.tasks:
        selected = set(args.tasks)
        tasks = [t for t in tasks if t.db_name in selected
                 or f"{t.db_name}/{t.table_name}" in selected]
    evaluator = build_evaluator(
        tasks, args.pre_dir, embedding_model=config["embedding_model"],
        d_text=config["d_text"], device=args.device, ctx_size=args.ctx_size,
        local_ctx_size=args.local_ctx_size, bfs_width=args.bfs_width,
        num_walks=args.num_walks, walk_length=args.walk_length,
        items_per_task=args.items_per_task, num_workers=0,
        tokens_per_gpu=args.ctx_size * args.batch_size, mmap_populate=True)
    run_and_report(
        model, tasks, args.pre_dir, ctx_size=args.ctx_size, reg_metric="mae",
        out_dir=args.out_dir, no_csv=False, evaluator=evaluator,
        embedding_model=config["embedding_model"])


if __name__ == "__main__":
    main()
