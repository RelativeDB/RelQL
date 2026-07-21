# RT vs. XGBoost vs. RelativeDB evaluation

This directory provides a fresh head-to-head harness over the same 21 RelBench
tasks used by `relational-transformer` at commit
`eece04847de7b52d6fe7a718c277abec7bb18c83`.

The four runners are independently optional:

- `rt`: the reference RT evaluation script and classification/regression
  checkpoints;
- `xgboost`: the reference val-tuned XGBoost predictor over its SQL features;
- `ours`: RelativeDB zero-shot, either on exact reference tensors or through an
  in-process engine factory;
- `ours_finetuned`: native inference from complete task-specific checkpoints
  produced by end-to-end backbone training.

Every runner emits the official keyed RelBench CSV shape
`<database>__<task>.csv`. The harness then scores all files through
`relbench.leaderboard.evaluate_task` and writes one task matrix to
`results.json` and `results.md`. Missing optional runners or task files are
reported as missing; they are not silently replaced or included in macro means.

## Provenance

The task catalog and evaluator behavior were reconciled against these reference
files:

- `src/rt/tasks.py`: curated 12 classification + 9 regression tasks;
- `src/rt/eval_utils.py`: entity/time-keyed submission shape, official scoring,
  classification probability conversion, and regression de-normalization;
- `scripts/eval.py`: RT runner defaults;
- `scripts/baseline.py`: XGBoost/rel2tab runner defaults.

The reference repository has no tracked license file, so its implementation
files were not copied wholesale. This harness invokes its scripts in place and
contains an independently written orchestration/scoring adapter. The factual
task catalog is recorded with its source commit.

## Dependencies

RT and XGBoost should use the reference repository's Python 3.12/Pixi
environment because they depend on its compiled `rt._rustler`, PyTorch,
RelBench, and (for XGBoost) rel2tab/XGBoost stack.

Official scoring also needs the same RelBench revision used by the reference:

```bash
pip install "git+https://github.com/snap-stanford/relbench.git@6e4a3a3d271f981b7f61e2ecbb6fd1c0f1cd3eeb"
pip install "xgboost==2.1.4"
```

The reference leaves XGBoost unpinned. Version 3.3.0 segfaulted during repeated
label ingestion on the Apple Silicon evaluation host, so the reproduced run
pins 2.1.4. The wrapper also isolates the pure NumPy/XGBoost predictor from the
reference process because importing XGBoost beside its Torch/Rust stack causes
an OpenMP crash on that host. SQL features, validation-tuned hyperparameters,
contexts, labels, and predictions are otherwise the reference baseline's.

The `reference_context_native` runner feeds the exact reference Rust batches to
RelativeDB's C++ runtime. This is the strongest model/runtime comparison because
RT and RelativeDB see identical tensors. The optional in-process RelativeDB
factories instead exercise the whole RelQL/Schema/Wiring path.

The RelativeDB factories run in the Python process launching the harness, so
launch with the RelativeDB virtual environment and ensure the factory module is
importable. If the reference environment differs, set each reference runner's
`python` path in the configuration.

## RelativeDB factory contract

A factory returns `RelativeDBEvalTask` objects. It owns raw RelBench loading,
Schema/Wiring construction, the task's RelQL translation, and—on the fine-tuned
path—head loading. This keeps dataset-specific semantics explicit instead of
guessing a RelQL target from a task-table name.

```python
from evaluation import EvalSample, RelativeDBEvalTask

def zero_shot_tasks():
    engine = build_zero_shot_engine()
    return [RelativeDBEvalTask(
        database="rel-f1",
        table="driver-top3",
        target_column="qualifying",
        query=(
            "PREDICT ... FROM drivers "
            "WHERE drivers.driver_id = :eval_entity_id"
        ),
        engine=engine,
        samples=tuple(
            EvalSample(
                entity_id=row.driver_id,
                anchor=row.timestamp,
                key={"driverId": row.driver_id, "timestamp": row.timestamp},
            )
            for row in test_rows
        ),
    )]

def finetuned_tasks(heads_dir):
    return build_tasks_with_heads(heads_dir)
```

The key dictionary must use the exact entity/time column names expected by the
RelBench task. Binary predictions are exported from `probability`; regression
predictions use `value`. Each execution must return exactly one prediction for
the sample entity.

For an external producer, use runner type `command`. Its command may contain
`{output_dir}` and `{tasks}` placeholders and must write the same CSV filenames.

## Run

Copy and edit `config.example.json`, enable any subset, then run:

```bash
PYTHONPATH=python/src:. python -m evaluation.head_to_head \
  --config evaluation/config.json
```

Override enabled runners or select a smaller task slice without changing the
configuration:

```bash
PYTHONPATH=python/src:. python -m evaluation.head_to_head \
  --config evaluation/config.json \
  --runners rt xgboost ours ours_finetuned \
  --tasks rel-f1/driver-top3 rel-f1/driver-position
```

Use `--no-score` to validate runner execution and submission coverage before
installing/downloading the official datasets. A scored comparison should use
identical task selectors and complete test splits for every runner.

Use `--score-existing` to rebuild a combined Markdown/JSON report from already
emitted CSVs without executing expensive runners again.

The checked-in reproducible slice is `config.rel-f1.json`. Generated reports,
predictions, checkpoints, and optimizer files stay in the ignored
`evaluation/runs/` directory.

## Native full-checkpoint training

`train_native_full.py` fine-tunes the complete model directly from the
reference Rust sampler. Forward, backward, gradient clipping, and AdamW run in
C++ on Metal/MPS. Torch is not used.

```bash
PYTHONPATH=python/src:/path/to/relational-transformer/src \
  .venv-eval/bin/python evaluation/train_native_full.py \
  --task rel-f1/driver-top3 \
  --checkpoint stanford-star/rt-j/classification \
  --output-dir evaluation/runs/driver-top3-native-full \
  --steps 43 --ctx-size 8192 \
  --batch-size 2 --effective-batch-size 32 \
  --eval-every 4 --learning-rate 1e-5
```

`--batch-size` is the physical MPS batch. `--effective-batch-size` is the batch
used for the optimizer update after gradient accumulation. Lower the physical
batch for one model if it does not fit; do not lower every model or silently
change the number of cells.

The released zero-shot model is the first best checkpoint. At each validation
boundary, the trainer promotes an improvement or restores the best model and
reduces the learning rate. Recovery files include the full model, AdamW state,
sampler position, current learning rate, and validation history. Use `--resume`
with the same task, context, batch, and optimizer settings to continue from the
latest recovery point.

For the checked-in long-run configuration, this helper resumes a live suspended
process when one exists and otherwise uses the latest recovery checkpoint:

```bash
./evaluation/continue_native_full_driver_top3.sh
```

If a process dies before its first validation boundary, there is no durable
optimizer checkpoint to recover. Preserve the incomplete attempt and restart
from the last valid full-model checkpoint; never append a fresh trajectory to
the old `train.jsonl`.

The selected output directory contains a complete `model.safetensors` and
`config.json`. Point `ours_finetuned.task_checkpoints` at that directory only
after validation promotion. A short execution check proves that the code runs;
it is not a quality result.

## Frozen-head diagnostic (not `ours_finetuned`)

Task heads can still be fit on the validation split for adapter diagnostics:

```bash
PYTHONPATH=/path/to/relational-transformer/src:/path/to/relational-transformer:\
python/src:. .venv-eval/bin/python evaluation/train_heads_on_reference.py \
  --classification-checkpoint hf://stanford-star/rt-j/classification \
  --regression-checkpoint hf://stanford-star/rt-j/regression \
  --pre-dir stanford-star/relbench-preprocessed \
  --out-dir evaluation/runs/heads \
  --library cpp/build/librt_c.dylib
```

The resulting `heads.json` is deliberately not accepted by the
`ours_finetuned` runner. To enable that runner, populate `task_checkpoints` with
one complete end-to-end-trained model checkpoint for each selected task.

All reference-context runners request 8,192 cells by default. Evaluator batch
size controls throughput/memory only and never changes cells per example. Set
`context_size: 8192` and an optional per-runner
`oom_fallback_context_sizes` list in the configuration. A detected OOM retries
only that command/model and writes its effective size to
`<output>/<runner>/context_usage.json`; other runners remain at 8,192.
