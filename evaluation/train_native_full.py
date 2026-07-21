#!/usr/bin/env python3
"""Validation-selected full RT-J fine-tuning in native C++/MPS.

The reference Rust sampler supplies train and validation contexts.  NumPy only
adapts its bf16 buffers; model forward/backward, gradient accumulation, clipping,
and AdamW execute in ``librt_c`` without Torch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import ml_dtypes  # noqa: F401 -- registers NumPy bfloat16 for rustler
import numpy as np


def _task_sampler(pre_dir: str, task_name: str, embedding_model: str, split: str,
                  *, seed: int, ctx_size: int, local_ctx_size: int, bfs_width: int,
                  num_walks: int, walk_length: int, items_per_task: int):
    from rt._rustler import Sampler
    from rt.pre import resolve_pre_dir
    from rt.tasks import eval_tasks

    db, table = task_name.split("/", 1)
    root = resolve_pre_dir(pre_dir, [db], embedding_model)
    matches = [t for t in eval_tasks(root, splits=(split,))
               if t.db_name == db and t.table_name == table]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {split} task for {task_name!r}, found {len(matches)}")
    task = matches[0]
    base = Path(root) / db
    table_info = json.loads((base / "table_info.json").read_text())
    suffix = {"train": "Train", "val": "Val", "test": "Test"}[split]
    info = (table_info[f"{table}:Db"] if f"{table}:Db" in table_info
            else table_info[f"{table}:{suffix}"])
    columns = json.loads((base / "column_index.json").read_text())
    target = columns[f"{task.target_column} of {table}"]
    dropped = [columns[f"{col} of {table}"] for col in task.leakage_columns
               if col != task.target_column and f"{col} of {table}" in columns]
    sampler = Sampler(
        dataset_tuples=[(db, table, info["node_idx_offset"], info["num_nodes"])],
        global_rank=0, local_rank=0, world_size=1,
        local_ctx_sizes=[min(local_ctx_size, ctx_size)], bfs_widths=[bfs_width],
        num_walks=num_walks, walk_length=walk_length, prefer_latest=[True],
        mask_prob_max=0.0, embedding_model=embedding_model, pre_dir=root,
        d_text=384, shuffle_seed=seed, context_seed=seed,
        target_columns=[target], columns_to_drop=[dropped],
        items_per_task=items_per_task, quiet=False, ignore_data_errors=False,
        num_prev_skipped=0, skip_text_cols=False, mmap_populate=True,
        balance_labels=[False], timeout_per_item=60,
        ablate_schema_semantics=False, vector_db_path=None,
        train_only_fallback=False)
    return sampler, task


def _native_batch(raw) -> tuple[dict[str, np.ndarray], np.ndarray]:
    raw = dict(raw)
    seq_len = int(raw.pop("seq_len"))
    batch_mask = np.asarray(raw.pop("batch_mask"), dtype=bool).reshape(-1)
    for key in ("number_values", "datetime_values", "boolean_values",
                "text_values", "col_name_values"):
        raw[key] = np.ascontiguousarray(np.asarray(raw[key], dtype=np.float32))
    sem = np.asarray(raw["sem_types"]).reshape(-1, seq_len)
    number = raw["number_values"].reshape(-1, seq_len, 1)
    boolean = raw["boolean_values"].reshape(-1, seq_len, 1)
    bool_mask = sem == 3
    number[bool_mask] = boolean[bool_mask]
    boolean[bool_mask] = 0
    sem[bool_mask] = 0
    return {
        "node_idxs": np.asarray(raw["node_idxs"], np.int64).reshape(-1, seq_len),
        "f2p": np.asarray(raw["f2p_nbr_idxs"], np.int64).reshape(-1, seq_len, 5),
        "col_idxs": np.asarray(raw["col_name_idxs"], np.int64).reshape(-1, seq_len),
        "table_idxs": np.asarray(raw["table_name_idxs"], np.int64).reshape(-1, seq_len),
        "is_padding": np.asarray(raw["is_padding"], np.uint8).reshape(-1, seq_len),
        "sem_types": np.ascontiguousarray(sem, np.int64),
        "is_target": np.asarray(raw["is_targets"], np.uint8).reshape(-1, seq_len),
        "number_v": np.ascontiguousarray(number, np.float32),
        "datetime_v": np.ascontiguousarray(
            raw["datetime_values"].reshape(-1, seq_len, 1), np.float32),
        "boolean_v": np.ascontiguousarray(boolean, np.float32),
        "text_v": np.ascontiguousarray(
            raw["text_values"].reshape(-1, seq_len, 384), np.float32),
        "col_name_v": np.ascontiguousarray(
            raw["col_name_values"].reshape(-1, seq_len, 384), np.float32),
    }, batch_mask


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = (labels > 0).astype(np.int8)
    pos, neg = int(labels.sum()), int(labels.size - labels.sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _evaluate(model, sampler, task_type: str, *, ctx_size: int,
              batch_size: int, max_items: int) -> dict[str, float | int]:
    from relativedb.rt_native import RT_DEVICE_MPS

    labels, predictions = [], []
    n_items = min(int(sampler.num_items), max_items)
    for batch_idx in range(math.ceil(n_items / batch_size)):
        batch, mask = _native_batch(
            sampler.batch_py(batch_idx, batch_size, ctx_size))
        y = (batch["number_v"].reshape(batch_size, ctx_size)
             * batch["is_target"]).sum(axis=1)
        pred = model.forward(**batch, device=RT_DEVICE_MPS)
        labels.extend(y[mask].astype(np.float64).tolist())
        predictions.extend(pred[mask].astype(np.float64).tolist())
    y = np.asarray(labels[:n_items], dtype=np.float64)
    p = np.asarray(predictions[:n_items], dtype=np.float64)
    if task_type == "clf":
        return {"metric": "roc_auc", "value": _auc(y, p), "items": int(y.size)}
    mae = float(np.mean(np.abs(p - y)))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((p - y) ** 2) / denom) if denom else float("nan")
    return {"metric": "nmae", "value": mae, "r2": r2, "items": int(y.size)}


def _better(task_type: str, candidate: float, incumbent: float,
            minimum_improvement: float) -> bool:
    return (candidate > incumbent + minimum_improvement if task_type == "clf"
            else candidate < incumbent - minimum_improvement)


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def _save_recovery(model, out: Path, state: dict) -> None:
    previous = (json.loads((out / "recovery.json").read_text())
                if (out / "recovery.json").exists() else None)
    step = int(state["optimizer_step"])
    model_name = f"recovery-model-step-{step}.safetensors"
    optimizer_name = f"recovery-optimizer-step-{step}.bin"
    model_tmp = out / f".{model_name}.tmp-{os.getpid()}"
    optimizer_tmp = out / f".{optimizer_name}.tmp-{os.getpid()}"
    model.save(model_tmp)
    model.save_finetune_optimizer(optimizer_tmp)
    os.replace(model_tmp, out / model_name)
    os.replace(optimizer_tmp, out / optimizer_name)
    state = {**state, "model_file": model_name, "optimizer_file": optimizer_name}
    _atomic_json(out / "recovery.json", state)
    if previous:
        for key in ("model_file", "optimizer_file"):
            old = previous.get(key)
            if old and old not in (model_name, optimizer_name):
                (out / old).unlink(missing_ok=True)


def _save_best(model, out: Path, state: dict, *, save_optimizer: bool) -> None:
    best = out / "best"
    best.mkdir(exist_ok=True)
    tmp = best / f".model.safetensors.tmp-{os.getpid()}"
    model.save(tmp)
    os.replace(tmp, best / "model.safetensors")
    optimizer_path = best / "optimizer.bin"
    if save_optimizer:
        optimizer_tmp = best / f".optimizer.bin.tmp-{os.getpid()}"
        model.save_finetune_optimizer(optimizer_tmp)
        os.replace(optimizer_tmp, optimizer_path)
    else:
        optimizer_path.unlink(missing_ok=True)
    _atomic_json(best / "selection.json", state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="database/task-table")
    parser.add_argument("--pre-dir", default="stanford-star/relbench-preprocessed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, required=True,
                        help="maximum optimizer updates")
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="MPS microbatch size")
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--local-ctx-size", type=int, default=256)
    parser.add_argument("--bfs-width", type=int, default=32)
    parser.add_argument("--num-walks", type=int, default=10_000)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-items", type=int, default=10_000_000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--adaptive-lr", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--lr-backoff-factor", type=float, default=0.2)
    parser.add_argument("--lr-backoff-patience", type=int, default=1)
    parser.add_argument("--max-lr-backoffs", type=int, default=3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--baseline-run", default=None,
                        help="reuse step-zero validation from a compatible run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    positive = (args.steps, args.ctx_size, args.batch_size,
                args.effective_batch_size, args.eval_every, args.eval_batch_size)
    if any(v <= 0 for v in positive):
        parser.error("steps, context, batch, and evaluation values must be positive")
    if args.effective_batch_size % args.batch_size:
        parser.error("effective-batch-size must be divisible by batch-size")
    if not 0 < args.lr_backoff_factor < 1:
        parser.error("lr-backoff-factor must be between zero and one")

    from rt.checkpoints import resolve_checkpoint
    from relativedb.rt_native import load_lib

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_config, source_checkpoint = resolve_checkpoint(args.checkpoint)
    train_sampler, task = _task_sampler(
        args.pre_dir, args.task, source_config["embedding_model"], "train",
        seed=args.seed, ctx_size=args.ctx_size, local_ctx_size=args.local_ctx_size,
        bfs_width=args.bfs_width, num_walks=args.num_walks,
        walk_length=args.walk_length, items_per_task=10_000_000)
    val_sampler, val_task = _task_sampler(
        args.pre_dir, args.task, source_config["embedding_model"], "val",
        seed=0, ctx_size=args.ctx_size, local_ctx_size=args.local_ctx_size,
        bfs_width=args.bfs_width, num_walks=args.num_walks,
        walk_length=args.walk_length, items_per_task=args.eval_items)
    if task.task_type != val_task.task_type:
        raise RuntimeError("train/validation task type mismatch")

    recovery_path = out / "recovery.json"
    history: list[dict] = []
    optimizer_step = microbatch_step = stale_evals = backoffs = 0
    current_lr = args.learning_rate
    best_metric = None
    if args.resume:
        recovery = json.loads(recovery_path.read_text())
        native = load_lib()
        model = native.load_model(str(out / recovery["model_file"]))
        model.load_finetune_optimizer(out / recovery["optimizer_file"])
        optimizer_step = int(recovery["optimizer_step"])
        microbatch_step = int(recovery["microbatch_step"])
        best_metric = recovery.get("best_metric")
        stale_evals = int(recovery.get("stale_evals", 0))
        backoffs = int(recovery.get("backoffs", 0))
        current_lr = float(recovery.get("current_learning_rate",
                                        args.learning_rate))
        history = list(recovery.get("validation_history", []))
        train_sampler.set_step_py(microbatch_step)
    else:
        native = load_lib()
        model = native.load_model(str(source_checkpoint))

    config = {
        "backend": "native-mps", "torch": False, "full_model": True,
        "source": str(args.checkpoint), "task": args.task,
        "task_type": task.task_type, "context_size": args.ctx_size,
        "micro_batch_size": args.batch_size,
        "effective_batch_size": args.effective_batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm, "maximum_steps": args.steps,
        "eval_every": args.eval_every, "eval_items": args.eval_items,
        "patience": args.patience, "seed": args.seed,
        "adaptive_lr": args.adaptive_lr,
        "lr_backoff_factor": args.lr_backoff_factor,
        "lr_backoff_patience": args.lr_backoff_patience,
        "max_lr_backoffs": args.max_lr_backoffs,
        "min_learning_rate": args.min_learning_rate,
    }
    if args.resume and (out / "training-config.json").exists():
        previous_config = json.loads((out / "training-config.json").read_text())
        invariant = ("task", "task_type", "context_size", "micro_batch_size",
                     "effective_batch_size", "learning_rate", "weight_decay",
                     "grad_clip_norm", "seed")
        changed = {key: (previous_config.get(key), config.get(key))
                   for key in invariant
                   if previous_config.get(key) != config.get(key)}
        if changed:
            raise RuntimeError(f"resume configuration mismatch: {changed}")
    _atomic_json(out / "training-config.json", config)

    if best_metric is None:
        if args.baseline_run:
            baseline_dir = Path(args.baseline_run).expanduser().resolve()
            baseline_config = json.loads(
                (baseline_dir / "training-config.json").read_text())
            if (baseline_config.get("task") != args.task or
                    baseline_config.get("context_size") != args.ctx_size or
                    baseline_config.get("source") != str(args.checkpoint)):
                raise RuntimeError("baseline-run task/context/source mismatch")
            baseline_history = json.loads(
                (baseline_dir / "validation.json").read_text())["history"]
            baseline = next(row for row in baseline_history
                            if row.get("kind") == "zero_shot")
            baseline = {key: baseline[key]
                        for key in ("metric", "value", "items")}
        else:
            baseline = _evaluate(model, val_sampler, task.task_type,
                                 ctx_size=args.ctx_size,
                                 batch_size=args.eval_batch_size,
                                 max_items=args.eval_items)
        best_metric = float(baseline["value"])
        history.append({"optimizer_step": 0, "kind": "zero_shot", **baseline})
        _save_best(model, out, history[-1], save_optimizer=False)
        print(json.dumps(history[-1]), flush=True)

    accumulation = args.effective_batch_size // args.batch_size
    losses, grad_norms, step_seconds = [], [], []
    started = time.monotonic()
    stop_reason = "maximum_steps"
    while optimizer_step < args.steps:
        for micro in range(accumulation):
            batch, mask = _native_batch(
                train_sampler.batch_py(None, args.batch_size, args.ctx_size))
            if not bool(mask.all()):
                raise RuntimeError("training sampler produced a phantom row")
            result = model.finetune_step(
                **batch, learning_rate=current_lr,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                apply_update=micro == accumulation - 1)
            microbatch_step += 1
            losses.append(float(result["loss"]))
            grad_norms.append(float(result["grad_norm"]))
            step_seconds.append(float(result["seconds"]))
        optimizer_step += 1
        train_record = {"optimizer_step": optimizer_step,
                        "model_optimizer_step": int(result["step"]),
                        "microbatch_step": microbatch_step,
                        "learning_rate": current_lr,
                        "loss": float(np.mean(losses[-accumulation:])),
                        "grad_norm": result["grad_norm"],
                        "seconds": float(np.sum(step_seconds[-accumulation:]))}
        with (out / "train.jsonl").open("a") as log:
            log.write(json.dumps(train_record) + "\n")
        print(json.dumps(train_record), flush=True)

        if optimizer_step % args.eval_every == 0 or optimizer_step == args.steps:
            metrics = _evaluate(model, val_sampler, task.task_type,
                                ctx_size=args.ctx_size,
                                batch_size=args.eval_batch_size,
                                max_items=args.eval_items)
            record = {"optimizer_step": optimizer_step, "kind": "trained",
                      "learning_rate": current_lr, **metrics}
            history.append(record)
            improved = _better(task.task_type, float(metrics["value"]),
                               float(best_metric), args.minimum_improvement)
            if improved:
                best_metric = float(metrics["value"])
                stale_evals = 0
                _save_best(model, out, record, save_optimizer=True)
            else:
                stale_evals += 1
            rolled_back = False
            if (not improved and args.adaptive_lr
                    and stale_evals >= args.lr_backoff_patience
                    and backoffs < args.max_lr_backoffs
                    and current_lr > args.min_learning_rate):
                next_lr = max(args.min_learning_rate,
                              current_lr * args.lr_backoff_factor)
                if next_lr < current_lr:
                    current_lr = next_lr
                    backoffs += 1
                    stale_evals = 0
                    model.close()
                    model = native.load_model(str(out / "best" / "model.safetensors"))
                    best_optimizer = out / "best" / "optimizer.bin"
                    if best_optimizer.exists():
                        model.load_finetune_optimizer(best_optimizer)
                    else:
                        model.reset_finetune_optimizer()
                    rolled_back = True
                    history.append({
                        "optimizer_step": optimizer_step, "kind": "lr_backoff",
                        "learning_rate": current_lr, "backoffs": backoffs,
                        "restored_best_metric": best_metric,
                    })
            state = {
                "optimizer_step": optimizer_step,
                "microbatch_step": microbatch_step,
                "best_metric": best_metric, "stale_evals": stale_evals,
                "current_learning_rate": current_lr, "backoffs": backoffs,
                "validation_history": history,
            }
            _save_recovery(model, out, state)
            _atomic_json(out / "validation.json", {"history": history})
            print(json.dumps({**record, "promoted": improved,
                              "best_metric": best_metric,
                              "stale_evals": stale_evals,
                              "rolled_back": rolled_back,
                              "next_learning_rate": current_lr}), flush=True)
            if stale_evals >= args.patience:
                stop_reason = "early_stopping"
                break

    best_model = out / "best" / "model.safetensors"
    final_config = dict(source_config)
    final_config["checkpoint_file"] = "model.safetensors"
    final_config["finetune"] = {
        **config, "completed_optimizer_steps": optimizer_step,
        "completed_microbatches": microbatch_step,
        "best_validation_metric": best_metric,
        "final_learning_rate": current_lr,
        "learning_rate_backoffs": backoffs,
        "stop_reason": stop_reason,
        "wall_seconds": time.monotonic() - started,
    }
    shutil.copyfile(best_model, out / "model.safetensors")
    _atomic_json(out / "config.json", final_config)
    _atomic_json(out / "training.json", {
        "losses": losses, "grad_norms": grad_norms,
        "step_seconds": step_seconds, "validation_history": history,
        "config": final_config["finetune"],
    })


if __name__ == "__main__":
    main()
