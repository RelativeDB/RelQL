#!/usr/bin/env python
"""Run real F1 RelQL queries through RelativeDB's public Engine path."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from relativedb import parse, validate

from evaluation.f1_relql import (DATASET, TASKS, build_engine, execute_group,
                                 _python_value)


def _gate(engine, spec, frame, context_size: int) -> dict:
    first = frame.iloc[0]
    entity_id = _python_value(first[spec.id_column])
    anchor = _python_value(first["date"])
    pq = validate(parse(spec.query), engine.schema).query.bind_params(
        {"ids": [entity_id]})
    context = engine.assemble_context(
        pq.entity_key.table, entity_id, anchor, query=pq)
    sequence = engine.model_backend._build_sequences(
        pq, pq.task_type(engine.schema), [context])[0][0]
    parent_keys = set()
    for row in context.rows:
        for link in engine.schema.links_from(row.table):
            value = row.parents.get(link.fk_column)
            values = value if isinstance(value, (list, tuple)) else (value,)
            parent_keys.update((link.to_table, one) for one in values
                               if one is not None)
    future_rows = [row for row in context.rows
                   if row.timestamp is not None and row.timestamp > context.anchor]
    # The reference sampler always follows F->P edges. In F1 a qualifying row
    # points to tomorrow's scheduled race; that parent's metadata is known at
    # qualifying time even though its event timestamp is later. Future child
    # or outcome rows remain forbidden.
    unsafe_future_rows = [row for row in future_rows if row.key not in parent_keys]
    target_positions = [i for i, value in enumerate(sequence.is_tgt) if value]
    checks = {
        "parsed": True,
        "native_sampler_bound": hasattr(
            engine.model_backend._native._lib
            if hasattr(engine.model_backend, "_native") else object(),
            "rt_reference_walk_counts"),
        "mps_device": engine.model_backend.device == 1,
        "physical_batch": engine.model_backend.batch_size == 4,
        "sequence_within_context": len(sequence) <= context_size,
        "one_target": target_positions == [0],
        "no_future_child_or_outcome_rows": not unsafe_future_rows,
    }
    # Loading the model is lazy, while the traversal ABI is bound by
    # relativedb.rt_native.load_lib(). Assert it directly instead of relying on
    # the backend's model handle.
    from relativedb.rt_native import load_lib
    checks["native_sampler_bound"] = hasattr(
        load_lib()._lib, "rt_reference_walk_counts")
    if not all(checks.values()):
        raise RuntimeError(f"end-to-end validation gate failed: {checks}")
    return {
        "query": spec.query,
        "entity_table": pq.entity_key.table,
        "entity_id": entity_id,
        "anchor": anchor.isoformat(),
        "context_rows": len(context.rows),
        "sequence_cells": len(sequence),
        "future_parent_rows": [f"{row.table}:{row.id}" for row in future_rows],
        "unsafe_future_rows": [f"{row.table}:{row.id}"
                               for row in unsafe_future_rows],
        "checks": checks,
    }


def run_one(dataset, name: str, output: Path, library: str | None,
            context_size: int, shared: bool = False) -> dict:
    from relbench import load_task

    spec = TASKS[name]
    task = load_task(DATASET, name)
    target = task.get_table("test", mask_input_cols=False).df.copy()
    engine = build_engine(dataset, spec, context_size=context_size, batch_size=4,
                          library=library)
    gate = _gate(engine, spec, target, context_size)
    print(f"{name}: gate passed ({gate['sequence_cells']} cells)", flush=True)

    keyed: dict[tuple[object, object], float] = {}
    stats = {"contexts_hit_cell_budget": 0, "contexts_truncated": 0}
    captured_warnings: list[str] = []
    started = time.perf_counter()
    groups = list(target.groupby("date", sort=False))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if shared:
            from relativedb import ExecutionInput
            batch = [ExecutionInput(
                query=spec.query,
                anchor_time=_python_value(date),
                per_entity_anchor=spec.per_entity_anchor,
                params={"ids": [_python_value(v)
                                for v in group[spec.id_column]]},
                shared_context=True,
            ) for date, group in groups]
            outcomes = engine.execute_many(batch)
        else:
            outcomes = None
        for group_index, (date, group) in enumerate(groups, 1):
            result = (outcomes[group_index - 1] if outcomes is not None
                      else execute_group(engine, spec,
                                         group[spec.id_column].tolist(), date,
                                         shared=shared))
            for prediction in result.predictions:
                value = (prediction.probability if spec.classification
                         else prediction.value)
                keyed[(_python_value(date), prediction.id)] = float(value)
            for key in stats:
                stats[key] += int(result.stats.get(key, 0))
            if group_index == 1 or group_index % 25 == 0 or group_index == len(groups):
                print(f"{name}: {group_index}/{len(groups)} timestamps", flush=True)
        captured_warnings = sorted({str(item.message) for item in seen})

    predictions = np.asarray([
        keyed[(_python_value(row.date), _python_value(getattr(row, spec.id_column)))]
        for row in target.itertuples(index=False)], dtype=np.float32)
    scores = {key: float(value) for key, value in task.evaluate(predictions).items()}
    diagnostic_r2 = None
    if not spec.classification:
        labels = target[spec.target_column].to_numpy(dtype=float)
        denom = float(np.square(labels - labels.mean()).sum())
        diagnostic_r2 = (1.0 - float(np.square(labels - predictions).sum()) / denom
                         if denom else float("nan"))

    output.mkdir(parents=True, exist_ok=True)
    submission = target[["date", spec.id_column, spec.target_column]].copy()
    submission["prediction"] = predictions
    submission.to_csv(output / f"rel-f1__{name}.csv", index=False)
    record = {
        "task": f"rel-f1/{name}",
        "query": spec.query,
        "context_size": context_size,
        "batch_size": 4,
        "shared_context": shared,
        "fine_tuned": False,
        "elapsed_seconds": time.perf_counter() - started,
        "scores": scores,
        "diagnostic_r2": diagnostic_r2,
        "gate": gate,
        "stats": stats,
        "warnings": captured_warnings,
        "removed_columns": [list(value) for value in spec.remove_columns],
    }
    (output / f"rel-f1__{name}.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n")
    print(f"{name}: {scores}", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evaluation/runs/rel-f1-relql-128")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS),
                        default=list(TASKS))
    parser.add_argument("--library", default="cpp/build/librt_c.dylib")
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--shared", action="store_true",
                        help="score each cohort in one shared context")
    args = parser.parse_args()

    from relbench import load_dataset
    dataset = load_dataset(DATASET)
    output = Path(args.output)
    records = [run_one(dataset, name, output, args.library,
                       args.context_size, shared=args.shared)
               for name in args.tasks]
    (output / "relativedb-results.json").write_text(
        json.dumps(records, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
