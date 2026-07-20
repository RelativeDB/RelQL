"""Score the retention task with the reference implementation.

Runs under ``~/relational-transformer``'s own interpreter, not this one. It is
the three BYOD steps inlined -- write the dataset directory, write the task
directory, preprocess with the reference's Rust preprocessor, then score with
the reference's evaluator -- because the shipped scripts read a ``config.py``
next to themselves, and this control is not worth editing their tree for. The
model path is untouched: ``rt.checkpoints.load_rt_model``,
``rt.eval_utils.build_evaluator``, ``evaluate_raw``, sigmoid, AUROC, exactly as
``3_predict.py`` does it.

Everything it reads was written by :mod:`gh.export_ref`, so both systems see
the same tables, the same rows, the same anchors and the same labels.

What this can and cannot compare:

  * it CAN compare the task. Same graph, same split, same checkpoint, same
    text encoder.
  * it CANNOT compare the number this benchmark reports. The reference has no
    fine-tuning anywhere in ``src/rt`` or ``scripts`` -- it is zero-shot, and
    relativedb's headline is a fitted head. The comparable pair is zero-shot
    against zero-shot.
  * it CANNOT reproduce the depth ablation. The reference has no hop limit to
    vary: it builds a local BFS context of ``local_ctx_size`` cells and fills
    the rest of the window with random walks over the whole database. Depth is
    not a knob there, so ``max_hops`` has no counterpart to sweep.

    python -m gh.run_ref
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REF = Path.home() / "relational-transformer"
SRC = REF / "examples" / "inference" / "out"        # written by export_ref
DB_NAME = "gh-retention"
TASK_NAME = "account-returns"

TABLES = None   # filled from export_ref, so the two cannot drift apart


def stage(work: Path) -> Path:
    """The dataset and task directories the reference expects."""
    import pandas as pd
    import yaml

    from .export_ref import TABLES as T

    db = work / DB_NAME
    (db / "db").mkdir(parents=True, exist_ok=True)
    import duckdb
    con = duckdb.connect(str(SRC / f"{DB_NAME}.duckdb"), read_only=True)
    try:
        for name, cfg in T.items():
            df = con.execute(f'SELECT * FROM "{name}"').df()
            if cfg.get("time_col"):
                df[cfg["time_col"]] = pd.to_datetime(df[cfg["time_col"]])
            df.to_parquet(db / "db" / f"{name}.parquet", index=False)
            print(f"  {name:15} {len(df):>9,} rows", flush=True)
    finally:
        con.close()
    yaml.safe_dump({"name": DB_NAME, "tables": T},
                   open(db / "manifest.yaml", "w"), sort_keys=False)

    task = db / "tasks" / TASK_NAME
    task.mkdir(parents=True, exist_ok=True)
    for s in ("train", "test"):
        df = pd.read_parquet(SRC / "labels" / f"{s}.parquet")
        df["ts"] = pd.to_datetime(df["ts"])
        df.to_parquet(task / f"{s}.parquet", index=False)
        print(f"  {s+' labels':15} {len(df):>9,} rows "
              f"({df.came_back.mean():.1%} positive)", flush=True)
    yaml.safe_dump({"entity_table": "actors", "entity_col": "actor_login",
                    "target_col": "came_back",
                    "task_type": "binary_classification", "time_col": "ts"},
                   open(task / "manifest.yaml", "w"), sort_keys=False)
    return db


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default=None,
                    help="scratch directory for the staged dataset")
    ap.add_argument("--checkpoint", default="stanford-star/rt-j/classification")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ctx-size", type=int, default=8192)
    # eval_bs = tokens_per_gpu // ctx_size. The reference's 2**18 means a batch
    # of 32, and the materialized (B, S, S) attention masks at ctx 8192 then
    # ask for a 10 GiB buffer, which this machine refuses. This is a batching
    # knob only: local_ctx_size, bfs_width, num_walks, walk_length and the
    # context size are all left at the reference's own defaults, so the
    # contexts the model sees are unchanged.
    ap.add_argument("--tokens-per-gpu", type=int, default=2**14)
    ap.add_argument("--keep", action="store_true",
                    help="reuse an existing preprocessed directory")
    args = ap.parse_args()

    work = Path(args.work or (Path(__file__).resolve().parent / "_ref_work"))
    if not args.keep and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] staging {DB_NAME} from {SRC}", flush=True)
    db = stage(work)

    pre = work / "pre"
    print(f"\n[2/3] preprocessing with the reference's Rust preprocessor",
          flush=True)
    if not (pre / DB_NAME / "table_info.json").exists():
        subprocess.run([sys.executable, str(REF / "scripts" / "preprocess.py"),
                        "one", "--dataset", str(db), "--out-dir", str(pre)],
                       check=True)
    else:
        print("   cached", flush=True)

    print(f"\n[3/3] zero-shot inference (device={args.device})", flush=True)
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score

    from rt.checkpoints import load_rt_model
    from rt.eval_utils import build_evaluator
    from rt.tasks import tasks_from_preprocessed

    model, cfg = load_rt_model(args.checkpoint, device=args.device)
    model = model.to(torch.bfloat16)
    print(f"   {args.checkpoint}  task_type={cfg.get('task_type')} "
          f"embed={cfg.get('embedding_model')} d_text={cfg.get('d_text')}",
          flush=True)

    tasks = [t for t in tasks_from_preprocessed(str(pre), splits=("test",),
                                                dbs=[DB_NAME])
             if t.table_name == TASK_NAME]
    assert tasks, f"task {TASK_NAME!r} not found in {pre}"
    ctx = cfg.get("ctx_len", args.ctx_size)
    ev = build_evaluator(tasks, str(pre),
                         embedding_model=cfg.get("embedding_model",
                                                 "all-MiniLM-L12-v2"),
                         d_text=cfg.get("d_text", 384), device=args.device,
                         ctx_size=ctx, items_per_task=10_000_000,
                         tokens_per_gpu=args.tokens_per_gpu)
    print(f"   ctx={ctx} local_ctx=256 bfs_width=32 walks=10000x20 "
          f"eval_bs={max(1, args.tokens_per_gpu // ctx)}", flush=True)
    ((task, _c, _l, out, _n, node_idxs),) = ev.evaluate_raw(
        [(model, "")], [ctx], with_node_idxs=True)

    info = json.loads((pre / DB_NAME / "table_info.json").read_text())
    off = info[f"{TASK_NAME}:Test"]["node_idx_offset"]
    rows = np.asarray(node_idxs) - off
    df = pd.read_parquet(db / "tasks" / TASK_NAME / "test.parquet")
    df = df.iloc[rows].reset_index(drop=True)
    raw = np.asarray(out[""], dtype=float)
    df["prediction"] = 1 / (1 + np.exp(-raw))
    y = (df.came_back.astype(float) > 0).astype(int)
    auc = roc_auc_score(y, df.prediction)
    print(f"\n[result] reference, zero-shot, ctx={ctx}: AUROC = {auc:.4f} "
          f"(n={len(df)}, {y.mean():.1%} positive, "
          f"{len(np.unique(np.round(df.prediction, 5)))} distinct scores)")
    outp = work / "ref_predictions.parquet"
    df.to_parquet(outp, index=False)
    (work / "ref_result.json").write_text(json.dumps(
        {"auc": float(auc), "n": int(len(df)), "ctx_size": int(ctx),
         "checkpoint": args.checkpoint, "device": args.device,
         "positive_rate": float(y.mean())}, indent=2))
    print(f"   per-row predictions -> {outp}")


if __name__ == "__main__":
    main()
