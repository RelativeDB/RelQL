"""Run optional RT, XGBoost, RelativeDB, and fine-tuned RelativeDB evaluations."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from .catalog import EvalTask, select_tasks
from .relativedb_runner import run as run_relativedb


RUNNER_ORDER = ("rt", "xgboost", "ours", "ours_finetuned")
OOM_MARKERS = ("out of memory", "oom", "mps backend out of memory",
               "cannot allocate memory", "allocation failed")


def _with_context_size(command: list[str], size: int) -> list[str]:
    command = list(command)
    if "--ctx-size" in command:
        command[command.index("--ctx-size") + 1] = str(size)
    else:
        command.extend(["--ctx-size", str(size)])
    return command


def _run_with_oom_fallback(command: list[str], config: dict, *, cwd, env) -> dict:
    """Retry only this command/model at explicitly configured context sizes."""
    requested = int(config.get("context_size", 8192))
    candidates = [requested, *[int(x) for x in
                               config.get("oom_fallback_context_sizes", [])]]
    for attempt, size in enumerate(candidates):
        current = _with_context_size(command, size)
        proc = subprocess.run(current, cwd=cwd, env=env,
                              text=True, capture_output=True)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        if proc.returncode == 0:
            return {"requested_context_size": requested,
                    "effective_context_size": size,
                    "fallback": size != requested,
                    "command": current}
        diagnostic = f"{proc.stdout}\n{proc.stderr}".lower()
        is_oom = (any(marker in diagnostic for marker in OOM_MARKERS)
                  or proc.returncode in (-9, 137))
        if not is_oom or attempt == len(candidates) - 1:
            raise subprocess.CalledProcessError(
                proc.returncode, current, proc.stdout, proc.stderr)
        print(f"OOM at {size} cells; retrying only this model/command at "
              f"{candidates[attempt + 1]} cells", file=sys.stderr)
    raise AssertionError("unreachable")


def _command(config: dict, runner: str, output_dir: Path,
             tasks: list[EvalTask]) -> list[list[str]]:
    kind = config.get("type", runner)
    if kind == "reference_rt":
        script = Path(__file__).with_name("run_reference_rt.py")
        python = config.get("python", sys.executable)
        common = [python, str(script),
                  "--pre-dir", config["pre_dir"], "--out-dir", str(output_dir),
                  "--device", config.get("device", "mps"), "--tasks",
                  *[t.id for t in tasks],
                  *[str(x) for x in config.get("extra_args", [])]]
        commands = []
        checkpoints = config.get("checkpoints") or {}
        for task_type in ("clf", "reg"):
            checkpoint = checkpoints.get(task_type)
            if checkpoint and any(t.task_type == task_type for t in tasks):
                commands.append([*common, "--task-type", task_type,
                                 "--checkpoint", checkpoint])
        return commands
    if kind == "reference_xgboost":
        script = Path(__file__).with_name("run_xgboost_reference.py")
        return [[config.get("python", sys.executable),
                 str(script),
                 "--pre-dir", config["pre_dir"],
                 "--out-dir", str(output_dir),
                 "--tasks", *[t.id for t in tasks],
                 *[str(x) for x in config.get("extra_args", [])]]]
    if kind == "reference_context_native":
        python = config.get("python", sys.executable)
        script = Path(__file__).with_name("run_native_on_reference.py")
        common = [python, str(script), "--pre-dir", config["pre_dir"],
                  "--out-dir", str(output_dir), "--tasks",
                  *[t.id for t in tasks],
                  "--native-device", config.get("native_device", "mps"),
                  *[str(x) for x in config.get("extra_args", [])]]
        if config.get("library"):
            common.extend(["--library", config["library"]])
        if config.get("heads_json"):
            common.extend(["--heads-json", config["heads_json"]])
        commands = []
        for task_type in ("clf", "reg"):
            checkpoint = (config.get("checkpoints") or {}).get(task_type)
            if checkpoint and any(t.task_type == task_type for t in tasks):
                commands.append([*common, "--task-type", task_type,
                                 "--checkpoint", checkpoint])
        return commands
    if kind == "reference_context_native_full_finetuned":
        # A true fine-tuned result is a complete task-specific model
        # checkpoint. One invocation per task prevents accidental reuse of a
        # frozen shared backbone or a validation-fitted linear head.
        python = config.get("python", sys.executable)
        script = Path(__file__).with_name("run_native_on_reference.py")
        checkpoints = config.get("task_checkpoints") or {}
        commands = []
        for task in tasks:
            checkpoint = checkpoints.get(task.id)
            if not checkpoint:
                continue
            command = [python, str(script), "--pre-dir", config["pre_dir"],
                       "--out-dir", str(output_dir), "--tasks", task.id,
                       "--task-type", task.task_type,
                       "--checkpoint", checkpoint,
                       "--native-device", config.get("native_device", "mps"),
                       *[str(x) for x in config.get("extra_args", [])]]
            if config.get("library"):
                command.extend(["--library", config["library"]])
            commands.append(command)
        if not commands:
            raise ValueError(
                f"runner {runner!r}: no full-model task_checkpoints match selected tasks")
        return commands
    if kind == "command":
        values = {"output_dir": str(output_dir),
                  "tasks": " ".join(t.id for t in tasks)}
        raw = config["command"]
        parts = shlex.split(raw) if isinstance(raw, str) else list(raw)
        return [[str(part).format(**values) for part in parts]]
    raise ValueError(f"runner {runner!r}: unsupported type {kind!r}")


def _source(config: dict, task: EvalTask) -> str:
    explicit = (config.get("dataset_sources") or {}).get(task.database)
    if explicit:
        return explicit
    for runner in config.get("runners", {}).values():
        pre_dir = runner.get("pre_dir")
        if not pre_dir or str(pre_dir).startswith(("hf://", "stanford-")):
            continue
        meta = Path(pre_dir).expanduser() / task.database / "meta.json"
        if meta.exists():
            value = json.loads(meta.read_text()).get("source")
            if value:
                return value
    raise RuntimeError(
        f"no RelBench dataset source for {task.database}; add "
        f"dataset_sources.{task.database!s} to the config")


def _score(task: EvalTask, csv_path: Path, source: str) -> tuple[str, float]:
    try:
        from relbench.leaderboard import evaluate_task
    except Exception as exc:
        raise RuntimeError(
            "official scoring requires the reference-pinned RelBench package; "
            "see evaluation/README.md") from exc
    metrics = evaluate_task(task.id, str(csv_path), dataset=source)
    if len(metrics) != 1:
        raise RuntimeError(f"{task.id}: expected one metric, got {metrics}")
    name, value = next(iter(metrics.items()))
    return str(name), float(value)


def _regression_r2(task: EvalTask, csv_path: Path, source: str) -> float:
    """Key-aligned diagnostic R²; official RelBench scoring remains primary."""
    import numpy as np
    import pandas as pd
    import relbench

    truth = relbench.load_task(source, task.table).get_table(
        "test", mask_input_cols=False).df.copy()
    pred = pd.read_csv(csv_path)
    keys = [column for column in pred.columns if column != task.target]
    for column in keys:
        if pd.api.types.is_datetime64_any_dtype(truth[column]):
            pred[column] = pd.to_datetime(pred[column])
    aligned = truth.merge(pred, on=keys, suffixes=("_true", "_pred"),
                          validate="one_to_one")
    if len(aligned) != len(truth):
        raise RuntimeError(f"{task.id}: R² alignment kept {len(aligned)}/{len(truth)}")
    y = aligned[f"{task.target}_true"].to_numpy(float)
    yhat = aligned[f"{task.target}_pred"].to_numpy(float)
    return float(1.0 - np.square(y - yhat).sum() /
                 np.square(y - y.mean()).sum())


def _write_report(output: Path, results: list[dict], runners: list[str],
                  config: dict, elapsed: dict[str, float]) -> None:
    payload = {"runners": runners, "tasks": config.get("tasks"),
               "elapsed_seconds": elapsed, "results": results}
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Head-to-head evaluation results", "",
             "All values below were computed from each runner's complete keyed "
             "submission CSV by the same official `relbench.leaderboard.evaluate_task` "
             "scorer. A dash means the runner did not produce that task.", "",
             "## Scores", "",
             "| Task | Metric | " + " | ".join(runners) + " |",
             "|---|---|" + "---:|" * len(runners)]
    by_task = defaultdict(dict)
    metrics = {}
    for row in results:
        by_task[row["task"]][row["runner"]] = row
        if row.get("metric"):
            metrics[row["task"]] = row["metric"]
    for task in sorted(by_task):
        vals = []
        for runner in runners:
            row = by_task[task].get(runner)
            vals.append("—" if not row or "value" not in row
                        else f"{row['value']:.6f}")
        lines.append(f"| {task} | {metrics.get(task, '—')} | "
                     + " | ".join(vals) + " |")
    runner_configs = config.get("runners") or {}
    if "rt" in runners and runner_configs.get("rt", {}).get("valid", True):
        lines.extend(["", "## Gain versus RT", "",
                      "Positive is better: ROC AUC uses `runner - RT`; NMAE uses "
                      "`RT - runner`.", "",
                      "| Task | " + " | ".join(r for r in runners if r != "rt") + " |",
                      "|---|" + "---:|" * (len(runners) - 1)])
        for task in sorted(by_task):
            baseline = by_task[task].get("rt", {}).get("value")
            values = []
            for runner in runners:
                if runner == "rt":
                    continue
                value = by_task[task].get(runner, {}).get("value")
                if baseline is None or value is None:
                    values.append("—")
                else:
                    gain = (value - baseline if metrics.get(task) == "roc_auc"
                            else baseline - value)
                    values.append(f"{gain:+.6f}")
            lines.append(f"| {task} | " + " | ".join(values) + " |")
    lines.extend(["", "## Macro means", "",
                  "Means include only tasks successfully scored by that runner.", "",
                  "| Runner | Metric | Mean | Tasks |", "|---|---|---:|---:|"])
    grouped = defaultdict(list)
    for row in results:
        if "value" in row and row.get("valid", True):
            grouped[(row["runner"], row["metric"])].append(row["value"])
    for (runner, metric), values in sorted(grouped.items()):
        lines.append(f"| {runner} | {metric} | "
                     f"{sum(values) / len(values):.6f} | {len(values)} |")
    r2_rows = [row for row in results if "diagnostic_r2" in row]
    if r2_rows:
        lines.extend(["", "## Regression diagnostic", "",
                      "R² is recomputed after exact key alignment so the rel-f1 "
                      "result can be compared with the paper. It is additional "
                      "to the official NMAE score.", "",
                      "| Task | Runner | R² | Valid |", "|---|---|---:|---|"])
        for row in r2_rows:
            lines.append(f"| {row['task']} | {row['runner']} | "
                         f"{row['diagnostic_r2']:.6f} | "
                         f"{'yes' if row.get('valid', True) else 'no'} |")
    lines.extend(["", "## Execution", "",
                  "| Runner | Wall time (s) | Valid scored | Missing |",
                  "|---|---:|---:|---:|"])
    for runner in runners:
        rows = [r for r in results if r["runner"] == runner]
        lines.append(f"| {runner} | {elapsed.get(runner, 0.0):.3f} | "
                     f"{sum(r.get('status') == 'ok' for r in rows)} | "
                     f"{sum(r.get('status') == 'missing' for r in rows)} |")
    context_rows = []
    for runner in runners:
        path = output / runner / "context_usage.json"
        if path.exists():
            for item in json.loads(path.read_text()):
                command = item.get("command", [])
                kind = (command[command.index("--task-type") + 1]
                        if "--task-type" in command else "all")
                selected = (kind if kind != "all" else
                            (command[command.index("--tasks") + 1]
                             if "--tasks" in command else kind))
                context_rows.append((runner, selected, item))
    if context_rows:
        lines.extend(["", "## Effective context", "",
                      "OOM fallback is isolated to the command/model shown; it "
                      "does not alter any other runner.", "",
                      "| Runner | Command/task | Requested cells | Effective cells | Fallback |",
                      "|---|---|---:|---:|---|"])
        for runner, selected, item in context_rows:
            lines.append(
                f"| {runner} | {selected} | "
                f"{item['requested_context_size']} | "
                f"{item['effective_context_size']} | "
                f"{'yes' if item['fallback'] else 'no'} |")
    report = config.get("report") or {}
    if report:
        lines.extend(["", "## Protocol", ""])
        for note in report.get("protocol", []):
            lines.append(f"- {note}")
        lines.extend(["", "## Runner definitions", ""])
        for runner in runners:
            definition = (report.get("runners") or {}).get(runner)
            if definition:
                lines.append(f"- **{runner}:** {definition}")
        if report.get("limitations"):
            lines.extend(["", "## Limitations", ""])
            for note in report["limitations"]:
                lines.append(f"- {note}")
    invalid = [(runner, runner_configs.get(runner, {}).get("invalid_reason"))
               for runner in runners
               if not runner_configs.get(runner, {}).get("valid", True)]
    if invalid:
        lines.extend(["", "## Invalid diagnostics", "",
                      "These files are retained for diagnosis but excluded "
                      "from valid macro means and baseline-gain claims.", ""])
        for runner, reason in invalid:
            lines.append(
                f"- **{runner}:** {reason or 'marked invalid by configuration'}")
    lines.extend(["", "## Reproduction", "",
                  "Configuration: `evaluation/config.rel-f1.json`", "",
                  "```bash",
                  "PYTHONPATH=python/src:. .venv-eval/bin/python -m evaluation.head_to_head \\",
                  "  --config evaluation/config.rel-f1.json \\",
                  "  --runners " + " ".join(runners),
                  "```", "",
                  "To regenerate only the combined report from existing CSVs, add "
                  "`--score-existing`."])
    (output / "results.md").write_text("\n".join(lines) + "\n")


def run(config: dict, *, enabled: list[str] | None = None,
        selectors: list[str] | None = None, score: bool = True,
        execute: bool = True) -> Path:
    tasks = select_tasks(selectors or config.get("tasks"))
    selected_ids = {t.id for t in tasks}
    runner_configs = config.get("runners") or {}
    names = enabled or [n for n in RUNNER_ORDER
                        if runner_configs.get(n, {}).get("enabled", False)]
    if not names:
        raise ValueError("no runners enabled")
    unknown = set(names) - set(RUNNER_ORDER)
    if unknown:
        raise ValueError(f"unknown runner(s): {', '.join(sorted(unknown))}")
    output = Path(config.get("output_dir", "evaluation/runs/latest")).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    previous_elapsed = {}
    if not execute and (output / "results.json").exists():
        previous_elapsed = json.loads(
            (output / "results.json").read_text()
        ).get("elapsed_seconds", {})
    elapsed: dict[str, float] = {}
    for name in names:
        rcfg = {**runner_configs.get(name, {}), "name": name}
        runner_dir = output / name
        runner_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        if execute:
            if rcfg.get("type") == "relativedb":
                run_relativedb(rcfg, runner_dir, selected_ids)
            else:
                contexts = []
                for command in _command(rcfg, name, runner_dir, tasks):
                    contexts.append(_run_with_oom_fallback(
                        command, rcfg, cwd=rcfg.get("cwd") or None,
                        env={**os.environ,
                             **{str(k): str(v) for k, v in
                                (rcfg.get("env") or {}).items()}}))
                (runner_dir / "context_usage.json").write_text(
                    json.dumps(contexts, indent=2) + "\n")
        elapsed[name] = (time.monotonic() - started if execute
                         else float(previous_elapsed.get(name, 0.0)))
    results = []
    for task in tasks:
        source = _source(config, task) if score else None
        for name in names:
            csv_path = output / name / task.filename
            row = {"task": task.id, "task_type": task.task_type,
                   "runner": name, "submission": str(csv_path)}
            if not csv_path.exists():
                row["status"] = "missing"
            elif score:
                metric, value = _score(task, csv_path, source)
                valid = runner_configs.get(name, {}).get("valid", True)
                row.update(status=("ok" if valid else "diagnostic_invalid"),
                           metric=metric, value=value, valid=bool(valid))
                if task.task_type == "reg" and config.get("diagnostic_r2", False):
                    row["diagnostic_r2"] = _regression_r2(
                        task, csv_path, source)
            else:
                row["status"] = "emitted"
            results.append(row)
    _write_report(output, results, names, config, elapsed)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runners", nargs="+", choices=RUNNER_ORDER)
    parser.add_argument("--tasks", nargs="+",
                        help="database or database/task selectors")
    parser.add_argument("--no-score", action="store_true",
                        help="only run and inventory submission CSVs")
    parser.add_argument("--score-existing", action="store_true",
                        help="do not execute runners; score their existing CSVs")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = run(config, enabled=args.runners, selectors=args.tasks,
                 score=not args.no_score, execute=not args.score_existing)
    print(output / "results.md")


if __name__ == "__main__":
    main()
