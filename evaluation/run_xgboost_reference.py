#!/usr/bin/env python
"""Run the reference val-tuned XGBoost over its SQL RelBench features."""

from __future__ import annotations

import argparse
import pickle
import struct
import subprocess
import sys
import types


def _install_relbench_legacy_api() -> None:
    """Bridge the reference featurizer to the pinned RelBench 3 loader API.

    The SQL baseline in relational-transformer still imports the former
    ``relbench.datasets``/``relbench.tasks`` modules.  The reference commit's
    own dependency pin now exposes equivalent loaders at package level.
    Keeping this shim in the evaluation wrapper avoids modifying the reference
    checkout while preserving its featurizer and predictor unchanged.
    """
    import relbench

    datasets = types.ModuleType("relbench.datasets")
    datasets.get_dataset = lambda name, download=True: relbench.load_dataset(
        f"stanford-rdl/relbench/{name}"
    )
    tasks = types.ModuleType("relbench.tasks")
    tasks.get_task = lambda name, task, download=True: relbench.load_task(
        f"stanford-rdl/relbench/{name}", task
    )
    sys.modules.setdefault("relbench.datasets", datasets)
    sys.modules.setdefault("relbench.tasks", tasks)


class IsolatedPredictor:
    """Run the unchanged reference predictor outside the sampler process.

    On macOS, loading XGBoost's OpenMP runtime into the process that owns the
    reference Rust iterator reliably segfaults on the next sampled row.  Spawn
    isolation changes no features, labels, hyperparameters, or predictions.
    """

    def __init__(self):
        worker = __file__.replace("run_xgboost_reference.py", "xgboost_worker.py")
        self._process = subprocess.Popen(
            [sys.executable, worker], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def predict(self, *args):
        # Torch's multiprocessing reducer shares tensor storage by descriptor.
        # The source batch is released while XGBoost may still hold a NumPy
        # view, so transfer owning NumPy copies instead.
        import torch
        owned = tuple(value.detach().cpu().numpy().copy()
                      if isinstance(value, torch.Tensor) else value
                      for value in args)
        blob = pickle.dumps(owned, protocol=5)
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(struct.pack("!Q", len(blob)) + blob)
        self._process.stdin.flush()
        size = self._process.stdout.read(8)
        if not size:
            raise RuntimeError("isolated XGBoost worker exited unexpectedly")
        ok, result = pickle.loads(
            self._process.stdout.read(struct.unpack("!Q", size)[0])
        )
        if not ok:
            raise RuntimeError(f"isolated XGBoost predictor failed: {result}")
        return result

    def close(self):
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--local-ctx-size", type=int, default=256)
    parser.add_argument("--bfs-width", type=int, default=32)
    parser.add_argument("--num-walks", type=int, default=10_000)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--items-per-task", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="evaluator batch size; 1 avoids native XGBoost "
                             "booster accumulation on macOS")
    args = parser.parse_args()

    _install_relbench_legacy_api()
    from rel2tab.featurizers.sql_featurizer import SQLFeaturizer
    from rel2tab.featurizers.sql_queries import SQL_REGISTRY
    from rel2tab.model import Rel2TabModel
    from rt.eval_utils import build_evaluator, run_and_report
    from rt.pre import resolve_pre_dir
    import rt.recipes as recipes
    from evaluation.f1_extra_queries import install_sql_queries
    from evaluation.reference_tasks import selected_test_tasks

    tasks = selected_test_tasks(args.pre_dir, args.tasks)
    install_sql_queries(SQL_REGISTRY)
    pre_dir = resolve_pre_dir(
        args.pre_dir, sorted({t.db_name for t in tasks}), "all-MiniLM-L12-v2"
    )
    for database in sorted({t.db_name for t in tasks}):
        db_tasks = [t for t in tasks if t.db_name == database]
        original_get_tasks = recipes.get_tasks
        recipes.get_tasks = lambda _recipe, _pre_dir: db_tasks
        try:
            featurizer = SQLFeaturizer(
                pre_dir=pre_dir, eval_recipe="relbench_eval_test", db=database)
        finally:
            recipes.get_tasks = original_get_tasks
        predictor = IsolatedPredictor()
        model = Rel2TabModel(featurizer, predictor, 4096)
        evaluator = build_evaluator(
            db_tasks, pre_dir, embedding_model="all-MiniLM-L12-v2", d_text=384,
            device="cpu", ctx_size=args.ctx_size,
            local_ctx_size=args.local_ctx_size, bfs_width=args.bfs_width,
            num_walks=args.num_walks, walk_length=args.walk_length,
            items_per_task=args.items_per_task, num_workers=0,
            tokens_per_gpu=args.ctx_size * args.batch_size, mmap_populate=True)
        try:
            run_and_report(
                model, db_tasks, pre_dir, ctx_size=args.ctx_size, reg_metric="mae",
                out_dir=args.out_dir, no_csv=False, evaluator=evaluator,
                embedding_model="all-MiniLM-L12-v2")
        finally:
            predictor.close()


if __name__ == "__main__":
    main()
