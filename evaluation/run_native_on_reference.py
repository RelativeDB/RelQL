#!/usr/bin/env python
"""Evaluate RelativeDB's native RT on relational-transformer's exact batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


class NativeReferenceAdapter:
    """The reference Evaluator model protocol backed by RelativeDB C++."""

    def __init__(self, checkpoint: str, library: str | None, native_device: str,
                 heads: dict[str, str] | None = None, collect: bool = False):
        from relativedb.rt_native import (FineTunedHead, RT_DEVICE_CPU,
                                          RT_DEVICE_MPS, load_lib,
                                          resolve_model_path)

        self.model = load_lib(library).load_model(resolve_model_path(checkpoint))
        self.device = {"cpu": RT_DEVICE_CPU, "mps": RT_DEVICE_MPS}[native_device]
        self.heads = {name: FineTunedHead.load(path)
                      for name, path in (heads or {}).items()}
        self.collect = collect
        self.collected: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    def eval(self):
        return self

    @staticmethod
    def _np(tensor, dtype=None):
        array = tensor.detach().float().cpu().numpy()
        return array.astype(dtype, copy=False) if dtype is not None else array

    def predict(self, batch, ctx_sizes, device, task, bool_as_num):
        outputs = {}
        for ctx in ctx_sizes:
            b = {key: value[:, :ctx] for key, value in batch.items()}
            args = dict(
                node_idxs=self._np(b["node_idxs"], np.int64),
                f2p=self._np(b["f2p_nbr_idxs"], np.int64),
                col_idxs=self._np(b["col_name_idxs"], np.int64),
                table_idxs=self._np(b["table_name_idxs"], np.int64),
                is_padding=self._np(b["is_padding"], np.uint8),
                sem_types=self._np(b["sem_types"], np.int64),
                is_target=self._np(b["is_targets"], np.uint8),
                number_v=self._np(b["number_values"]),
                datetime_v=self._np(b["datetime_values"]),
                boolean_v=self._np(b["boolean_values"]),
                text_v=self._np(b["text_values"]),
                col_name_v=self._np(b["col_name_values"]),
            )
            head = self.heads.get(f"{task.db_name}/{task.table_name}")
            features = None
            if head is not None or self.collect:
                features = self.model.encode_targets(**args, device=self.device)
            if self.collect:
                real = b["is_targets"].any(dim=1).cpu().numpy().astype(bool)
                value_key = ("boolean_values"
                             if task.task_type == "clf" and not bool_as_num
                             else "number_values")
                labels = ((b[value_key].squeeze(-1)
                           * b["is_targets"].to(b[value_key].dtype))
                          .sum(dim=1).float().cpu().numpy())
                self.collected.setdefault(
                    f"{task.db_name}/{task.table_name}", []).append(
                        (features[real], labels[real]))
            if head is None:
                values = self.model.forward(**args, device=self.device)
            else:
                values = head.predict(features).reshape(-1)
            outputs[ctx] = torch.from_numpy(np.asarray(values, np.float32))
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pre-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-type", choices=("clf", "reg"), required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--local-ctx-size", type=int, default=256)
    parser.add_argument("--bfs-width", type=int, default=32)
    parser.add_argument("--num-walks", type=int, default=10_000)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--items-per-task", type=int, default=10_000_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="examples per evaluator batch; context length is unchanged")
    parser.add_argument("--library")
    parser.add_argument("--native-device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--heads-json",
                        help="optional task-id -> FineTunedHead path JSON")
    args = parser.parse_args()

    from rt.eval_utils import build_evaluator, run_and_report
    from evaluation.reference_tasks import selected_test_tasks

    tasks = selected_test_tasks(args.pre_dir, args.tasks, args.task_type)
    heads = None
    if args.heads_json:
        import json
        heads = json.loads(Path(args.heads_json).read_text())
    model = NativeReferenceAdapter(args.checkpoint, args.library,
                                   args.native_device, heads)
    evaluator = build_evaluator(
        tasks, args.pre_dir, embedding_model="all-MiniLM-L12-v2", d_text=384,
        device="cpu", ctx_size=args.ctx_size,
        local_ctx_size=args.local_ctx_size, bfs_width=args.bfs_width,
        num_walks=args.num_walks, walk_length=args.walk_length,
        items_per_task=args.items_per_task, num_workers=args.num_workers,
        tokens_per_gpu=args.ctx_size * args.batch_size, mmap_populate=True)
    run_and_report(
        model, tasks, args.pre_dir, ctx_size=args.ctx_size, reg_metric="mae",
        out_dir=args.out_dir, no_csv=False, evaluator=evaluator,
        embedding_model="all-MiniLM-L12-v2")


if __name__ == "__main__":
    main()
