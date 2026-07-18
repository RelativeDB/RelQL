"""Scoring metrics for the backtest, thin wrappers over sklearn + numpy.

All functions are defensive: degenerate inputs (single class, empty, all-NaN)
return ``None`` for the affected metric rather than raising, because a real
backtest hits those cases constantly (an anchor where every user churned, a
window where nobody was active, etc.).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             roc_auc_score)


# --- binary classification -------------------------------------------------
def binary_metrics(y_true: Sequence[int], y_prob: Sequence[float]) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    out = {"n": int(y.size), "positive_rate": None, "auroc": None,
           "pr_auc": None, "brier": None, "lift_at_10pct": None,
           "distinct_scores": None}
    if y.size == 0:
        return out
    out["positive_rate"] = float(y.mean())
    out["distinct_scores"] = int(np.unique(p).size)
    if 0 < y.sum() < y.size:                       # both classes present
        out["auroc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    out["brier"] = float(np.mean((p - y) ** 2))
    out["lift_at_10pct"] = _lift_at_k(y, p, 0.10)
    return out


def _lift_at_k(y: np.ndarray, p: np.ndarray, frac: float) -> Optional[float]:
    base = y.mean()
    if base == 0:
        return None
    k = max(1, int(round(frac * y.size)))
    top = np.argsort(-p)[:k]
    return float(y[top].mean() / base)


# --- regression ------------------------------------------------------------
def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    out = {"n": int(y.size), "mae": None, "rmse": None, "r2": None,
           "spearman": None, "true_mean": None, "pred_mean": None}
    if y.size == 0:
        return out
    out["true_mean"] = float(y.mean())
    out["pred_mean"] = float(p.mean())
    out["mae"] = float(mean_absolute_error(y, p))
    out["rmse"] = float(math.sqrt(np.mean((y - p) ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out["r2"] = float(1 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else None
    out["spearman"] = _spearman(y, p)
    return out


def _spearman(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if y.size < 3 or np.unique(p).size < 2 or np.unique(y).size < 2:
        return None
    ry, rp = _rankdata(y), _rankdata(p)
    ry, rp = ry - ry.mean(), rp - rp.mean()
    denom = math.sqrt(float(np.sum(ry ** 2) * np.sum(rp ** 2)))
    return float(np.sum(ry * rp) / denom) if denom > 0 else None


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, a.size + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


# --- ranking / recommendation ---------------------------------------------
def ranking_metrics(recommended: Sequence[Sequence], relevant: Sequence[Sequence],
                    k: int) -> dict:
    """recommended[i], relevant[i] are the ranked list and the ground-truth
    set for entity i. Reports Recall@K, Precision@K, MAP@K, NDCG@K, plus the
    hit rate and mean coverage (did we even produce K items?)."""
    recalls, precs, aps, ndcgs, hits, covs = [], [], [], [], [], []
    for rec, rel in zip(recommended, relevant):
        rel_set = set(rel)
        if not rel_set:
            continue
        rec_k = list(rec)[:k]
        covs.append(len(rec_k) / k)
        hitset = [1 if item in rel_set else 0 for item in rec_k]
        n_hit = sum(hitset)
        recalls.append(n_hit / len(rel_set))
        precs.append(n_hit / k)
        hits.append(1 if n_hit > 0 else 0)
        aps.append(_ap(hitset, len(rel_set)))
        ndcgs.append(_ndcg(hitset, len(rel_set), k))
    n = len(recalls)
    if n == 0:
        return {"n": 0, "recall_at_k": None, "precision_at_k": None,
                "map_at_k": None, "ndcg_at_k": None, "hit_rate": None,
                "list_coverage": None, "k": k}
    return {"n": n, "k": k,
            "recall_at_k": float(np.mean(recalls)),
            "precision_at_k": float(np.mean(precs)),
            "map_at_k": float(np.mean(aps)),
            "ndcg_at_k": float(np.mean(ndcgs)),
            "hit_rate": float(np.mean(hits)),
            "list_coverage": float(np.mean(covs))}


def _ap(hitset: list[int], n_rel: int) -> float:
    if not hitset:
        return 0.0
    hits, score = 0, 0.0
    for i, h in enumerate(hitset, 1):
        if h:
            hits += 1
            score += hits / i
    return score / min(n_rel, len(hitset)) if hits else 0.0


def _ndcg(hitset: list[int], n_rel: int, k: int) -> float:
    dcg = sum(h / math.log2(i + 1) for i, h in enumerate(hitset, 1))
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(n_rel, k) + 1))
    return dcg / ideal if ideal > 0 else 0.0
