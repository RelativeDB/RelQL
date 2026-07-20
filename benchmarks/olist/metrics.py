"""Scoring, overall and sliced by how much history the entity had.

The sliced view is the point of the benchmark: an aggregate metric hides the
cold-start rows, and on Olist those are 97% of customers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             mean_absolute_error, r2_score, roc_auc_score)

from . import data as D


def score(kind: str, y: np.ndarray, pred: np.ndarray) -> dict:
    ok = np.isfinite(pred).all(axis=1) if pred.ndim > 1 else np.isfinite(pred)
    if ok.sum() == 0:
        return {"n": 0}
    y, pred = y[ok], pred[ok]

    if kind == "regression":
        return {"n": int(len(y)),
                "mae": float(mean_absolute_error(y, pred)),
                "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
                "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan")}

    if kind == "multiclass":
        p = np.clip(pred, 1e-9, 1)
        p = p / p.sum(axis=1, keepdims=True)
        hard = p.argmax(axis=1)
        out = {"n": int(len(y)),
               "accuracy": float(accuracy_score(y, hard)),
               "macro_f1": float(f1_score(y, hard, average="macro",
                                          zero_division=0))}
        try:
            out["cross_entropy"] = float(log_loss(y, p, labels=list(range(p.shape[1]))))
        except ValueError:
            pass
        return out

    out = {"n": int(len(y)), "positive_rate": float(np.mean(y))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, pred))
        out["log_loss"] = float(log_loss(y, np.clip(pred, 1e-9, 1 - 1e-9)))
    out["accuracy"] = float(accuracy_score(y, (pred >= 0.5).astype(int)))
    out["f1"] = float(f1_score(y, (pred >= 0.5).astype(int), zero_division=0))
    return out


def by_depth(kind: str, y: np.ndarray, pred: np.ndarray,
             depth: np.ndarray) -> dict:
    """The same metric, per sparsity bucket."""
    buckets = D.depth_bucket(depth)
    out = {}
    for b in buckets.cat.categories:
        m = (buckets == b).to_numpy()
        if m.sum() == 0:
            continue
        out[str(b)] = score(kind, y[m], pred[m])
    return out


def headline(kind: str) -> str:
    """The metric the report leads with for this task kind."""
    return {"regression": "mae", "multiclass": "accuracy"}.get(kind, "auc")
