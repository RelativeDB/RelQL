"""In-process adapter from RelativeDB engines to RelBench submission CSVs."""

from __future__ import annotations

import csv
import importlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from relativedb import ExecutionInput, TaskType


@dataclass(frozen=True)
class EvalSample:
    """One test task row and its official submission key."""

    entity_id: Any
    anchor: datetime
    key: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelativeDBEvalTask:
    """Everything needed to emit one RelativeDB submission table.

    The query should pin its population with ``:eval_entity_id`` (or provide
    equivalent values in each sample's ``params``). The factory owns database
    loading, Schema/Wiring construction, and optional fine-tuned-head loading.
    """

    database: str
    table: str
    target_column: str
    query: Any
    engine: Any
    samples: tuple[EvalSample, ...]

    @property
    def id(self) -> str:
        return f"{self.database}/{self.table}"


def _load_factory(path: str):
    try:
        module_name, attr = path.split(":", 1)
    except ValueError as exc:
        raise ValueError("factory must be 'python.module:function'") from exc
    return getattr(importlib.import_module(module_name), attr)


def _prediction_value(result, entity_id):
    matches = [p for p in result.predictions if p.id == entity_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one prediction for {entity_id!r}, got {len(matches)}")
    pred = matches[0]
    if result.task_type is TaskType.BINARY_CLASSIFICATION:
        value = pred.probability
    elif result.task_type in (TaskType.REGRESSION, TaskType.FORECASTING):
        value = pred.value
    else:
        raise RuntimeError(
            f"RelBench head-to-head supports binary/regression tasks, got "
            f"{result.task_type.value}")
    if value is None:
        raise RuntimeError(f"empty prediction for {entity_id!r}")
    return float(value)


def run(config: dict, output_dir: Path, selected_ids: set[str]) -> list[str]:
    factory = _load_factory(config["factory"])
    kwargs = dict(config.get("factory_kwargs") or {})
    tasks: Iterable[RelativeDBEvalTask] = factory(**kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    emitted: list[str] = []
    for task in tasks:
        if selected_ids and task.id not in selected_ids:
            continue
        if not task.samples:
            continue
        key_columns = tuple(task.samples[0].key)
        path = output_dir / f"{task.database}__{task.table}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=[*key_columns, task.target_column])
            writer.writeheader()
            for sample in task.samples:
                if tuple(sample.key) != key_columns:
                    raise RuntimeError(
                        f"{task.id}: inconsistent submission key columns")
                params = {"eval_entity_id": sample.entity_id, **sample.params}
                result = task.engine.execute(ExecutionInput(
                    query=task.query, anchor_time=sample.anchor, params=params))
                writer.writerow({**sample.key,
                                 task.target_column: _prediction_value(
                                     result, sample.entity_id)})
        emitted.append(task.id)
    manifest = {"runner": config.get("name"), "tasks": emitted,
                "factory": config["factory"]}
    (output_dir / "runner_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return emitted
