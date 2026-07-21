"""Select curated or explicitly requested task tables from preprocessing."""

from __future__ import annotations


def selected_test_tasks(pre_dir: str, selectors: list[str] | None,
                        task_type: str | None = None):
    from rt.pre import resolve_pre_dir
    from rt.recipes import get_tasks
    from rt.tasks import tasks_from_preprocessed

    curated = list(get_tasks("relbench_eval_test", pre_dir))
    if selectors:
        databases = sorted({value.split("/", 1)[0] for value in selectors})
        root = resolve_pre_dir(
            pre_dir, databases, "all-MiniLM-L12-v2")
        available = tasks_from_preprocessed(
            root, splits=("test",), dbs=databases)
        by_id = {f"{task.db_name}/{task.table_name}": task
                 for task in (*curated, *available)}
        missing = [value for value in selectors
                   if "/" in value and value not in by_id]
        if missing:
            raise ValueError(f"unknown preprocessed task(s): {missing}")
        selected = [task for key, task in by_id.items()
                    if task.db_name in selectors or key in selectors]
    else:
        selected = curated
    if task_type:
        selected = [task for task in selected if task.task_type == task_type]
    return sorted(selected, key=lambda task: (task.db_name, task.table_name))
