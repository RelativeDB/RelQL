"""The 21-task RelBench evaluation set used by relational-transformer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalTask:
    database: str
    table: str
    target: str
    task_type: str  # clf | reg

    @property
    def id(self) -> str:
        return f"{self.database}/{self.table}"

    @property
    def filename(self) -> str:
        return f"{self.database}__{self.table}.csv"


# Ported from rt.tasks.RELBENCH_EVAL_TASKS at reference commit
# eece04847de7b52d6fe7a718c277abec7bb18c83.
EVAL_TASKS: tuple[EvalTask, ...] = (
    EvalTask("rel-amazon", "user-churn", "churn", "clf"),
    EvalTask("rel-hm", "user-churn", "churn", "clf"),
    EvalTask("rel-stack", "user-badge", "WillGetBadge", "clf"),
    EvalTask("rel-amazon", "item-churn", "churn", "clf"),
    EvalTask("rel-stack", "user-engagement", "contribution", "clf"),
    EvalTask("rel-avito", "user-visits", "num_click", "clf"),
    EvalTask("rel-avito", "user-clicks", "num_click", "clf"),
    EvalTask("rel-event", "user-ignore", "target", "clf"),
    EvalTask("rel-trial", "study-outcome", "outcome", "clf"),
    EvalTask("rel-f1", "driver-dnf", "did_not_finish", "clf"),
    EvalTask("rel-event", "user-repeat", "target", "clf"),
    EvalTask("rel-f1", "driver-top3", "qualifying", "clf"),
    EvalTask("rel-hm", "item-sales", "sales", "reg"),
    EvalTask("rel-amazon", "user-ltv", "ltv", "reg"),
    EvalTask("rel-amazon", "item-ltv", "ltv", "reg"),
    EvalTask("rel-stack", "post-votes", "popularity", "reg"),
    EvalTask("rel-trial", "site-success", "success_rate", "reg"),
    EvalTask("rel-trial", "study-adverse", "num_of_adverse_events", "reg"),
    EvalTask("rel-event", "user-attendance", "target", "reg"),
    EvalTask("rel-f1", "driver-position", "position", "reg"),
    EvalTask("rel-avito", "ad-ctr", "num_click", "reg"),
)

# Additional scalar task tables shipped by rel-f1 but not included in the
# curated 21-task paper benchmark. They are available for explicit query
# experiments and are never added to the default catalog selection.
EXTRA_TASKS: tuple[EvalTask, ...] = (
    EvalTask("rel-f1", "qualifying-position", "position", "reg"),
    EvalTask("rel-f1", "results-position", "position", "reg"),
)


def select_tasks(selectors: list[str] | None) -> list[EvalTask]:
    if not selectors:
        return list(EVAL_TASKS)
    wanted = set(selectors)
    selected = [t for t in (*EVAL_TASKS, *EXTRA_TASKS)
                if t.database in wanted or t.id in wanted]
    unknown = wanted - {t.database for t in selected} - {t.id for t in selected}
    if unknown:
        raise ValueError(f"unknown task selector(s): {', '.join(sorted(unknown))}")
    return selected
