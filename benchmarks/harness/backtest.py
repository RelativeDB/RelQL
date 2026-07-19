"""The point-in-time backtest protocol and the task catalogue.

For each task and each anchor T we call ``Engine.execute`` exactly as a user
would, collect the per-entity prediction, then score it against ground truth
computed independently from the raw event arrays. We also compute simple
*naive* baselines on the same split — persistence, popularity, global mean —
so a metric only counts as "signal" if the engine beats the one-liner a user
could have written without it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

from relativedb import ExecutionInput

from . import metrics as M
from .datasets import Dataset, EntityEvents, to_epoch


@dataclass
class TaskResult:
    dataset: str
    task: str
    kind: str                       # binary | regression | ranking
    query: str
    horizon: str
    metrics: dict
    baselines: dict                 # name -> metrics dict
    n_entities: int
    n_anchors: int
    seconds: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Binary: activity churn
# ---------------------------------------------------------------------------
def churn_task(ds: Dataset, child: str, horizon_days: int, lookback_days: int) -> TaskResult:
    q = (f"PREDICT COUNT({child}.*) OVER ({horizon_days} DAYS FOLLOWING) = 0 "
         f"FROM {ds.entity_table} "
         f"WHERE COUNT({child}.*) OVER ({lookback_days} DAYS PRECEDING) > 0")
    y_true, y_prob = [], []
    recency, past_cnt = [], []       # naive churn scores
    t0 = time.time()
    for T in ds.anchors:
        res = ds.engine.execute(ExecutionInput(query=q, anchor_time=T))
        for p in res.predictions:
            if p.probability is None:
                continue
            truth = 1 if ds.events.count(p.id, T, 0, horizon_days, "days") == 0 else 0
            y_true.append(truth)
            y_prob.append(float(p.probability))
            recency.append(_recency_gap(ds.events, p.id, T))
            past_cnt.append(ds.events.count(p.id, T, -lookback_days, 0, "days"))
    mets = M.binary_metrics(y_true, y_prob)
    base = {
        "recency_gap": M.binary_metrics(y_true, recency),
        "neg_recent_activity": M.binary_metrics(y_true, [-c for c in past_cnt]),
        "constant_prevalence": M.binary_metrics(
            y_true, [float(np.mean(y_true) if y_true else 0.0)] * len(y_true)),
    }
    notes = []
    if mets["distinct_scores"] is not None and mets["distinct_scores"] <= 6:
        notes.append(f"engine emits only {mets['distinct_scores']} distinct "
                     f"probabilities → step-function ranking, coarse calibration")
    return TaskResult(ds.name, "churn", "binary", q, f"{horizon_days}d",
                      mets, base, len(y_true), len(ds.anchors),
                      time.time() - t0, notes)


# ---------------------------------------------------------------------------
# Regression: forward activity count
# ---------------------------------------------------------------------------
def count_task(ds: Dataset, child: str, horizon_days: int, lookback_days: int) -> TaskResult:
    q = (f"PREDICT COUNT({child}.*) OVER ({horizon_days} DAYS FOLLOWING) "
         f"FROM {ds.entity_table} "
         f"WHERE COUNT({child}.*) OVER ({lookback_days} DAYS PRECEDING) > 0")
    y_true, y_pred, past, gmean_src = [], [], [], []
    t0 = time.time()
    for T in ds.anchors:
        res = ds.engine.execute(ExecutionInput(query=q, anchor_time=T))
        for p in res.predictions:
            if p.value is None:
                continue
            y_true.append(ds.events.count(p.id, T, 0, horizon_days, "days"))
            y_pred.append(float(p.value))
            past.append(ds.events.count(p.id, T, -horizon_days, 0, "days"))
    gm = float(np.mean(y_true)) if y_true else 0.0
    base = {"persistence_last_window": M.regression_metrics(y_true, past),
            "global_mean": M.regression_metrics(y_true, [gm] * len(y_true))}
    return TaskResult(ds.name, "activity_count", "regression", q, f"{horizon_days}d",
                      M.regression_metrics(y_true, y_pred), base,
                      len(y_true), len(ds.anchors), time.time() - t0)


# ---------------------------------------------------------------------------
# Regression: forward monetary value (CLV-style)
# ---------------------------------------------------------------------------
def value_task(ds: Dataset, child: str, value_col: str, horizon_days: int,
               lookback_days: int) -> TaskResult:
    q = (f"PREDICT SUM({child}.{value_col}) OVER ({horizon_days} DAYS FOLLOWING) "
         f"FROM {ds.entity_table} "
         f"WHERE COUNT({child}.*) OVER ({lookback_days} DAYS PRECEDING) > 0")
    y_true, y_pred, past = [], [], []
    t0 = time.time()
    for T in ds.anchors:
        res = ds.engine.execute(ExecutionInput(query=q, anchor_time=T))
        for p in res.predictions:
            if p.value is None:
                continue
            y_true.append(ds.events.sum_value(p.id, T, 0, horizon_days, "days"))
            y_pred.append(float(p.value))
            past.append(ds.events.sum_value(p.id, T, -horizon_days, 0, "days"))
    gm = float(np.mean(y_true)) if y_true else 0.0
    base = {"persistence_last_window": M.regression_metrics(y_true, past),
            "global_mean": M.regression_metrics(y_true, [gm] * len(y_true))}
    return TaskResult(ds.name, "forward_value", "regression", q, f"{horizon_days}d",
                      M.regression_metrics(y_true, y_pred), base,
                      len(y_true), len(ds.anchors), time.time() - t0)


# ---------------------------------------------------------------------------
# Ranking: buy-it-again / next-items
# ---------------------------------------------------------------------------
def ranking_task(ds: Dataset, child: str, item_col: str, horizon_days: int,
                 lookback_days: int, k: int = 10) -> TaskResult:
    q = (f"PREDICT LIST_DISTINCT({child}.{item_col}) OVER ({horizon_days} DAYS FOLLOWING) "
         f"RANK TOP {k} "
         f"FROM {ds.entity_table} "
         f"WHERE COUNT({child}.*) OVER ({lookback_days} DAYS PRECEDING) > 0")
    recs, rels, pop_recs = [], [], []
    t0 = time.time()
    for T in ds.anchors:
        res = ds.engine.execute(ExecutionInput(query=q, anchor_time=T))
        for p in res.predictions:
            rel = ds.events.item_set(p.id, T, 0, horizon_days, "days")
            recs.append(list(p.ranked))
            rels.append(rel)
            pop_recs.append(ds.global_top_items[:k])
    mets = M.ranking_metrics(recs, rels, k)
    base = {"global_popularity": M.ranking_metrics(pop_recs, rels, k)}
    notes = []
    # Does the ground truth ever contain a *new* item the entity never had?
    reuse = _repeat_rate(ds, child, item_col)
    if reuse is not None and reuse < 0.05:
        notes.append(f"only {reuse:.1%} of future items were previously seen by "
                     f"the entity → a 'recommend past items' ranker cannot recall "
                     f"future items here (structural ceiling, not a tuning issue)")
    return TaskResult(ds.name, "buy_it_again", "ranking", q, f"{horizon_days}d",
                      mets, base, mets.get("n", 0), len(ds.anchors),
                      time.time() - t0, notes)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _recency_gap(events: EntityEvents, eid, T: datetime) -> float:
    ts = events.times.get(eid)
    a = to_epoch(T)
    past = ts[ts <= a] if ts is not None else None
    if past is None or past.size == 0:
        return 1e12
    return a - float(past[-1])


def _repeat_rate(ds: Dataset, child: str, item_col: str) -> Optional[float]:
    """Fraction of (entity, future-item) pairs where the item was already in
    the entity's history — the recall ceiling of a repeat-only ranker."""
    seen_total = seen_repeat = 0
    for eid in ds.entity_ids:
        items = ds.events.items.get(eid)
        if items is None or items.size < 2:
            continue
        seen = set()
        for it in items.tolist():
            seen_total += 1
            if it in seen:
                seen_repeat += 1
            seen.add(it)
    return seen_repeat / seen_total if seen_total else None
