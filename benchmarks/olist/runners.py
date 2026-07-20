"""The two systems under test, behind one interface.

Both receive the same rows, the same chronological split, and the same labels.
XGBoost sees a flat encoded matrix; RelativeDB sees the relational graph through
retrievers and answers a RelQL query. Every runner reports the same timings, so
train cost and inference cost are comparable rather than incidental.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from relativedb import (ContextPolicy, Engine, ExecutionInput, LinkDef,
                        RtNativeBackend, SamplerMode, Schema, TableDef,
                        ValueType)

# fp32 only: the quantized variants are opt-in through this variable, so the
# benchmark clears it rather than trusting the ambient environment.
os.environ.pop("RELATIVEDB_RT_QUANTIZED", None)

POLICY = ContextPolicy(max_context_cells=200_000, bfs_width=64, max_hops=2)


@dataclass
class Result:
    system: str
    task: str
    kind: str
    pred: np.ndarray                      # score / class-probabilities / value
    train_seconds: float
    inference_ms_per_row: float
    n_train: int
    n_test: int
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def run_xgboost(task, train: pd.DataFrame, test: pd.DataFrame) -> Result:
    from sklearn.preprocessing import OrdinalEncoder
    from xgboost import XGBClassifier, XGBRegressor

    t0 = time.perf_counter()
    num = [c for c in task.numeric if c in train.columns]
    cat = [c for c in task.categorical if c in train.columns]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    Xtr = np.hstack([train[num].astype(float).fillna(np.nan).to_numpy(),
                     enc.fit_transform(train[cat].astype(str))])
    Xte = np.hstack([test[num].astype(float).fillna(np.nan).to_numpy(),
                     enc.transform(test[cat].astype(str))])
    prep_s = time.perf_counter() - t0

    ytr = train.label.to_numpy()
    common = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                  tree_method="hist", n_jobs=0)
    if task.kind == "regression":
        model = XGBRegressor(**common)
    elif task.kind == "multiclass":
        model = XGBClassifier(**common, objective="multi:softprob",
                              num_class=int(ytr.max()) + 1)
    else:
        model = XGBClassifier(**common, objective="binary:logistic")

    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if task.kind == "regression":
        pred = model.predict(Xte)
    elif task.kind == "multiclass":
        pred = model.predict_proba(Xte)
    else:
        pred = model.predict_proba(Xte)[:, 1]
    infer_s = time.perf_counter() - t0

    return Result("XGBoost", task.name, task.kind, np.asarray(pred),
                  train_seconds=prep_s + fit_s,
                  inference_ms_per_row=1000 * infer_s / max(1, len(test)),
                  n_train=len(train), n_test=len(test),
                  detail={"encoded_features": Xtr.shape[1],
                          "preprocessing_seconds": prep_s,
                          "fit_seconds": fit_s,
                          "n_estimators": 400, "max_depth": 6})


# ---------------------------------------------------------------------------
# RelativeDB: schema + retrievers over the same rows
# ---------------------------------------------------------------------------

def build_schema() -> Schema:
    return (Schema.new_schema()
        .table(TableDef.new_table("customers")
               .column("customer_state", ValueType.TEXT)
               .column("customer_city", ValueType.TEXT)
               .column("customer_zip_code_prefix", ValueType.NUMBER)
               .primary_key("customer_unique_id").build())
        .table(TableDef.new_table("orders")
               .column("order_status", ValueType.TEXT)
               .column("estimated_delivery_days", ValueType.NUMBER)
               .column("purchase_hour", ValueType.NUMBER)
               .column("purchase_dow", ValueType.NUMBER)
               .column("ts", ValueType.DATETIME)
               .primary_key("order_id").time_column("ts").build())
        .table(TableDef.new_table("items")
               .column("price", ValueType.NUMBER)
               .column("freight_value", ValueType.NUMBER)
               .column("ts", ValueType.DATETIME)
               .primary_key("item_key").time_column("ts").build())
        .table(TableDef.new_table("products")
               .column("category", ValueType.TEXT)
               .column("product_weight_g", ValueType.NUMBER)
               .primary_key("product_id").build())
        .table(TableDef.new_table("sellers")
               .column("seller_state", ValueType.TEXT)
               .primary_key("seller_id").build())
        .table(TableDef.new_table("payments")
               .column("payment_type", ValueType.TEXT)
               .column("payment_value", ValueType.NUMBER)
               .column("payment_installments", ValueType.NUMBER)
               .column("ts", ValueType.DATETIME)
               .primary_key("payment_key").time_column("ts").build())
        .table(TableDef.new_table("reviews")
               .column("review_score", ValueType.NUMBER)
               # the same outcome as a discrete label: a NUMBER target infers
               # regression, so multiclass needs a categorical column
               .column("star", ValueType.TEXT)
               .column("ts", ValueType.DATETIME)
               .primary_key("review_id").time_column("ts").build())
        .link(LinkDef("orders", "customer_unique_id", "customers"))
        .link(LinkDef("items", "order_id", "orders"))
        .link(LinkDef("items", "product_id", "products"))
        .link(LinkDef("items", "seller_id", "sellers"))
        .link(LinkDef("payments", "order_id", "orders"))
        .link(LinkDef("reviews", "order_id", "orders"))
        .build())


def build_frames(o, feats: pd.DataFrame) -> dict:
    """The same rows XGBoost sees, shaped as tables rather than one matrix."""
    orders = feats[["order_id", "customer_unique_id", "order_status",
                    "estimated_delivery_days", "purchase_hour", "purchase_dow",
                    "delivery_days"]].copy()
    orders["ts"] = feats.order_purchase_timestamp
    return {
        "customers": o.customers[["customer_unique_id", "customer_state",
                                  "customer_city", "customer_zip_code_prefix"]],
        "orders": orders,
        "items": o.items[["item_key", "order_id", "product_id", "seller_id",
                          "price", "freight_value", "ts"]],
        "products": o.products[["product_id", "category", "product_weight_g"]],
        "sellers": o.sellers[["seller_id", "seller_state"]],
        "payments": o.payments[["payment_key", "order_id", "payment_type",
                                "payment_value", "payment_installments", "ts"]],
        "reviews": o.reviews.assign(
            star=o.reviews.review_score.astype("Int64").astype(str))[
                ["review_id", "order_id", "review_score", "star", "ts"]],
    }


def _engine(schema, wiring, head=None) -> Engine:
    return Engine(schema, wiring, sampler_mode=SamplerMode.CSC,
                  context_policy=POLICY,
                  model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                                head=head))


def run_relativedb(task, train: pd.DataFrame, test: pd.DataFrame, schema,
                   wiring, *, finetune: bool, epochs: int = 300,
                   learning_rate: float = 1e-2,
                   train_anchors: int = 4) -> Result:
    """Zero-shot, or fine-tuned on anchors drawn from the train split only."""
    eng = _engine(schema, wiring)
    ids = list(test.entity_id)
    head, train_s, detail = None, 0.0, {}

    if finetune:
        # anchors spread across the train window; every label the head sees
        # comes from before the evaluation split
        qs = np.linspace(0.25, 0.95, train_anchors)
        anchors = [train.anchor.quantile(q).to_pydatetime() for q in qs]
        t0 = time.perf_counter()
        head = eng.finetune(task.query, anchors,
                            params={"ids": list(train.entity_id)},
                            epochs=epochs, learning_rate=learning_rate)
        train_s = time.perf_counter() - t0
        detail = {"epochs": epochs, "learning_rate": learning_rate,
                  "anchors": [str(a.date()) for a in anchors],
                  "examples": head.n_examples,
                  "loss_before": head.initial_loss,
                  "loss_after": head.final_loss,
                  "head_fit_seconds": head.seconds}
        eng = _engine(schema, wiring, head=head)

    # One execution per entity, at that entity's own anchor. A shared anchor
    # would bound every context at the latest test timestamp, letting an order
    # see reviews that arrived after it — which scores a perfect AUC and means
    # nothing.
    by_id, task_type, model_uri = {}, None, None
    t0 = time.perf_counter()
    for eid, anchor in zip(ids, test.anchor):
        r = eng.execute(ExecutionInput(query=task.query,
                                       anchor_time=anchor.to_pydatetime(),
                                       params={"ids": [eid]}))
        task_type = r.task_type.value
        model_uri = r.model_uri
        for p in r.predictions:
            by_id[p.id] = p
    infer_s = time.perf_counter() - t0

    pred = _extract(task, by_id, ids)
    return Result("RelativeDB (fine-tuned)" if finetune else "RelativeDB (zero-shot)",
                  task.name, task.kind, pred,
                  train_seconds=train_s,
                  inference_ms_per_row=1000 * infer_s / max(1, len(ids)),
                  n_train=len(train), n_test=len(test),
                  detail={**detail, "task_type": task_type,
                          "model_uri": model_uri, "precision": "fp32",
                          "scored": len(by_id),
                          "anchoring": "per-entity"})


def _extract(task, by_id: dict, ids: list) -> np.ndarray:
    """Pull the per-row number the metrics need, in test order."""
    if task.kind == "regression":
        return np.array([_num(by_id.get(i), "value") for i in ids], float)
    if task.kind == "multiclass":
        rows = []
        for i in ids:
            p = by_id.get(i)
            probs = getattr(p, "class_probs", {}) or {}
            vec = np.zeros(5)
            for k, v in probs.items():
                try:
                    idx = int(float(k)) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < 5:
                    vec[idx] = v
            rows.append(vec if vec.sum() > 0 else np.full(5, 0.2))
        return np.vstack(rows)
    return np.array([_num(by_id.get(i), "probability") for i in ids], float)


def _num(pred, attr: str) -> float:
    if pred is None:
        return float("nan")
    v = getattr(pred, attr, None)
    if v is None:
        v = getattr(pred, "value", None)
    return float(v) if v is not None else float("nan")
