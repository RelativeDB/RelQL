"""Generalizability suite: run the task catalogue across every dataset and
report a matrix — headline metric, best naive baseline, and whether the engine
beats it — plus per-split (per-anchor) stability (mean ± std of the headline
metric across anchors).

The point is to judge a backend change across the *whole grid*, not on a single
dataset's single number. A change "generalizes" only if it moves the
`beats_naive` fraction up without wrecking stability across datasets/splits.
"""
from __future__ import annotations

import dataclasses
import statistics
from typing import Callable, Optional

from . import backtest as B
from .datasets import Dataset


# (task label, kind, builder(ds) -> TaskResult). Builders close over the
# dataset's child table / item / value column.
def _specs(ds: Dataset) -> list[tuple[str, str, Callable[[Dataset], B.TaskResult]]]:
    meta = {
        "online_retail": dict(child="purchases", item="stock_code", value="amount",
                              churn=(90, 90), count=(90, 90), rank=(90, 90)),
        "movielens": dict(child="ratings", item="movie_id", value=None,
                         churn=(60, 180), count=(60, 180), rank=(180, 365)),
        "brightkite": dict(child="checkins", item="location_id", value=None,
                          churn=(60, 90), count=(60, 90), rank=(60, 90)),
    }[ds.name]
    child = meta["child"]
    specs = [
        ("churn", "binary",
         lambda d: B.churn_task(d, child, *meta["churn"])),
        ("activity_count", "regression",
         lambda d: B.count_task(d, child, *meta["count"])),
        ("buy_it_again", "ranking",
         lambda d: B.ranking_task(d, child, meta["item"], *meta["rank"], k=10)),
    ]
    if meta["value"] is not None:
        specs.append(("forward_value", "regression",
                      lambda d: B.value_task(d, child, meta["value"], *meta["churn"])))
    return specs


def _headline(r: B.TaskResult) -> tuple[str, Optional[float], Optional[float], str]:
    """(metric name, engine value, best-naive value, better-direction)."""
    m = r.metrics
    if r.kind == "binary":
        naive = max((b["auroc"] for b in r.baselines.values()
                     if b["auroc"] is not None), default=None)
        return "AUROC", m["auroc"], naive, "higher"
    if r.kind == "regression":
        naive = min((b["mae"] for b in r.baselines.values()
                     if b["mae"] is not None), default=None)
        return "MAE", m["mae"], naive, "lower"
    naive = r.baselines.get("global_popularity", {}).get("recall_at_k")
    return "Recall@10", m["recall_at_k"], naive, "higher"


def _beats(eng: Optional[float], naive: Optional[float], direction: str) -> Optional[bool]:
    if eng is None or naive is None:
        return None
    return eng > naive + 1e-9 if direction == "higher" else eng < naive - 1e-9


def _stability(ds: Dataset, builder: Callable[[Dataset], B.TaskResult]) -> dict:
    """Headline metric per anchor (single-anchor dataset views) → mean ± std."""
    vals = []
    for a in ds.anchors:
        one = dataclasses.replace(ds, anchors=[a])
        _, eng, _, _ = _headline(builder(one))
        if eng is not None:
            vals.append(eng)
    if not vals:
        return {"mean": None, "std": None, "n_splits": 0}
    return {"mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n_splits": len(vals)}


def run(datasets: list[Dataset], with_stability: bool = True) -> dict:
    cells = []
    for ds in datasets:
        for label, kind, builder in _specs(ds):
            r = builder(ds)
            metric, eng, naive, direction = _headline(r)
            cell = {
                "dataset": ds.name, "task": label, "kind": kind,
                "horizon": r.horizon, "metric": metric,
                "engine": eng, "naive": naive, "direction": direction,
                "beats_naive": _beats(eng, naive, direction),
                "n": r.n_entities,
            }
            if with_stability:
                cell["stability"] = _stability(ds, builder)
            cells.append(cell)
    beatable = [c for c in cells if c["beats_naive"] is not None]
    won = sum(1 for c in beatable if c["beats_naive"])
    return {"cells": cells,
            "beats_naive_fraction": (won / len(beatable)) if beatable else None,
            "n_cells": len(cells), "n_beatable": len(beatable), "n_won": won}
