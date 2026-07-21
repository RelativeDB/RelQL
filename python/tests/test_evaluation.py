from __future__ import annotations

import csv
import json
import sys
import types
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from evaluation.catalog import EVAL_TASKS, select_tasks
from evaluation.head_to_head import run
from evaluation.head_to_head import _command, _run_with_oom_fallback
from evaluation.relativedb_runner import EvalSample, RelativeDBEvalTask
from relativedb import EntityPrediction, PredictionResult, TaskType


class _Engine:
    def __init__(self, offset=0.0):
        self.offset = offset

    def execute(self, execution_input):
        entity = execution_input.params["eval_entity_id"]
        return PredictionResult(
            TaskType.BINARY_CLASSIFICATION,
            (EntityPrediction(entity, probability=float(entity) / 10
                              + self.offset),))


def _factory(offset=0.0):
    samples = tuple(
        EvalSample(i, datetime(2026, 1, i, tzinfo=timezone.utc),
                   {"entity": i, "timestamp": f"2026-01-{i:02d}"})
        for i in (1, 2)
    )
    return [RelativeDBEvalTask(
        "rel-f1", "driver-top3", "qualifying", "unused", _Engine(offset),
        samples)]


def test_reference_catalog_is_the_curated_12_clf_9_reg_set():
    assert len(EVAL_TASKS) == 21
    assert sum(t.task_type == "clf" for t in EVAL_TASKS) == 12
    assert sum(t.task_type == "reg" for t in EVAL_TASKS) == 9
    assert [t.id for t in select_tasks(["rel-f1/driver-top3"])] == [
        "rel-f1/driver-top3"]
    with pytest.raises(ValueError, match="unknown task"):
        select_tasks(["not-a-task"])


def test_extra_f1_scalar_tasks_require_explicit_selection():
    assert all(t.id not in {
        "rel-f1/qualifying-position", "rel-f1/results-position"
    } for t in select_tasks(None))
    assert [t.id for t in select_tasks([
        "rel-f1/qualifying-position", "rel-f1/results-position"
    ])] == ["rel-f1/qualifying-position", "rel-f1/results-position"]


def test_optional_ours_and_finetuned_emit_comparable_submissions(
        tmp_path, monkeypatch):
    module = types.ModuleType("eval_test_factory")
    module.zero = lambda: _factory(0.0)
    module.tuned = lambda: _factory(0.2)
    monkeypatch.setitem(sys.modules, "eval_test_factory", module)
    config = {
        "output_dir": str(tmp_path),
        "tasks": ["rel-f1/driver-top3"],
        "runners": {
            "ours": {"enabled": True, "type": "relativedb",
                     "factory": "eval_test_factory:zero"},
            "ours_finetuned": {"enabled": True, "type": "relativedb",
                               "factory": "eval_test_factory:tuned"},
        },
    }
    output = run(config, score=False)
    for runner, expected in (("ours", [0.1, 0.2]),
                             ("ours_finetuned", [0.3, 0.4])):
        path = output / runner / "rel-f1__driver-top3.csv"
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert [float(r["qualifying"]) for r in rows] == pytest.approx(expected)
    report = json.loads((output / "results.json").read_text())
    assert report["runners"] == ["ours", "ours_finetuned"]
    assert all(r["status"] == "emitted" for r in report["results"])


def test_scoring_is_shared_across_runners(tmp_path, monkeypatch):
    module = types.ModuleType("eval_score_factory")
    module.zero = lambda: _factory(0.0)
    monkeypatch.setitem(sys.modules, "eval_score_factory", module)
    monkeypatch.setattr("evaluation.head_to_head._source",
                        lambda config, task: "source")
    monkeypatch.setattr("evaluation.head_to_head._score",
                        lambda task, path, source: ("roc_auc", 0.75))
    config = {
        "output_dir": str(tmp_path),
        "tasks": ["rel-f1/driver-top3"],
        "runners": {"ours": {"enabled": True, "type": "relativedb",
                                "factory": "eval_score_factory:zero"}},
    }
    run(config)
    result = json.loads((tmp_path / "results.json").read_text())["results"][0]
    assert result["metric"] == "roc_auc"
    assert result["value"] == 0.75


def test_oom_fallback_changes_only_the_failing_command(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        rc = 1 if command[command.index("--ctx-size") + 1] == "8192" else 0
        return SimpleNamespace(returncode=rc, stdout="",
                               stderr="MPS backend out of memory" if rc else "")

    monkeypatch.setattr("evaluation.head_to_head.subprocess.run", fake_run)
    result = _run_with_oom_fallback(
        ["python", "runner.py", "--ctx-size", "8192"],
        {"context_size": 8192, "oom_fallback_context_sizes": [4096]},
        cwd=None, env={})
    assert [c[c.index("--ctx-size") + 1] for c in calls] == ["8192", "4096"]
    assert result["effective_context_size"] == 4096
    assert result["fallback"] is True


def test_full_finetuned_runner_requires_task_specific_complete_checkpoint(tmp_path):
    task = select_tasks(["rel-f1/driver-top3"])[0]
    cfg = {
        "type": "reference_context_native_full_finetuned",
        "python": "python",
        "pre_dir": "pre",
        "task_checkpoints": {task.id: "/models/full-task.safetensors"},
    }
    command = _command(cfg, "ours_finetuned", tmp_path, [task])[0]
    assert "--heads-json" not in command
    assert command[command.index("--checkpoint") + 1] == "/models/full-task.safetensors"
    assert command[command.index("--tasks") + 1] == task.id
