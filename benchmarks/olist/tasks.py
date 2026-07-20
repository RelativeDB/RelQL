"""The benchmark's task registry.

One entry per prediction problem. Each carries the RelQL the engine runs, the
label the same problem gives XGBoost, and the flat feature columns XGBoost is
allowed to see. Keeping both sides in one object is what stops the comparison
drifting: if a column is a feature here, it is a feature for both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import data as D

# Everything below is knowable at order-purchase time. Outcomes that arrive
# later (review score, delivery date) are labels, never features.
ORDER_NUMERIC = [
    "customer_zip_code_prefix", "purchase_year", "purchase_month",
    "purchase_dow", "purchase_hour", "estimated_delivery_days",
    "item_count", "total_price", "total_freight", "mean_item_price",
    "distinct_products", "distinct_sellers", "product_weight_g",
    "product_photos_qty", "product_desc_len", "payment_value",
    "payment_installments", "payment_count",
]
ORDER_CATEGORICAL = ["customer_state", "category", "seller_state", "payment_type"]


@dataclass
class Task:
    name: str
    kind: str                       # binary | multiclass | regression | ranking
    title: str
    question: str
    query: str                      # the RelQL the engine executes
    entity_table: str
    anchor_col: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    build: Optional[Callable] = None   # (Olist) -> frame with id/anchor/label
    horizon_days: int = 120            # forward window the label is read from
    notes: str = ""

    def frame(self, o: D.Olist) -> pd.DataFrame:
        return self.build(o)


# --------------------------------------------------------------------------
# order-level tasks: anchored at purchase, labelled by what arrived later
# --------------------------------------------------------------------------

def _order_base(o: D.Olist) -> pd.DataFrame:
    f = D.order_features(o)
    # a review is only a label if it landed after the order was placed
    f = f[f.review_score.notna()]
    f = f[f.review_ts >= f.order_purchase_timestamp]
    return f.reset_index(drop=True)


def _bad_review(o: D.Olist) -> pd.DataFrame:
    f = _order_base(o)
    f["label"] = (f.review_score <= 2).astype(int)
    f["entity_id"] = f.order_id
    f["anchor"] = f.order_purchase_timestamp
    return f


def _stars(o: D.Olist) -> pd.DataFrame:
    f = _order_base(o)
    f["label"] = f.review_score.astype(int) - 1        # 0..4
    f["entity_id"] = f.order_id
    f["anchor"] = f.order_purchase_timestamp
    return f


def _delivery_days(o: D.Olist) -> pd.DataFrame:
    f = D.order_features(o)
    f = f[f.delivery_days.notna() & (f.delivery_days >= 0)
          & (f.delivery_days < 120)]
    f["label"] = f.delivery_days.astype(float)
    f["entity_id"] = f.order_id
    f["anchor"] = f.order_purchase_timestamp
    return f.reset_index(drop=True)


# --------------------------------------------------------------------------
# customer-level task: the sparse one. 97% of Olist customers never return,
# so a model that only sees per-customer aggregates has almost nothing.
# --------------------------------------------------------------------------

def _future_spend(o: D.Olist) -> pd.DataFrame:
    f = D.order_features(o)
    first = (f.sort_values("order_purchase_timestamp")
             .groupby("customer_unique_id").first().reset_index())
    pay = f[["customer_unique_id", "order_purchase_timestamp", "payment_value"]]
    j = pay.merge(first[["customer_unique_id", "order_purchase_timestamp"]]
                  .rename(columns={"order_purchase_timestamp": "t0"}),
                  on="customer_unique_id", how="left")
    win = j[(j.order_purchase_timestamp > j.t0)
            & (j.order_purchase_timestamp <= j.t0 + pd.Timedelta(days=90))]
    spend = win.groupby("customer_unique_id").payment_value.sum()
    first["label"] = first.customer_unique_id.map(spend).fillna(0.0).astype(float)
    first["entity_id"] = first.customer_unique_id
    first["anchor"] = first.order_purchase_timestamp
    return first.reset_index(drop=True)


def _repeat_purchase(o: D.Olist) -> pd.DataFrame:
    f = D.order_features(o)
    first = (f.sort_values("order_purchase_timestamp")
             .groupby("customer_unique_id").first().reset_index())
    later = f.merge(first[["customer_unique_id", "order_purchase_timestamp"]]
                    .rename(columns={"order_purchase_timestamp": "t0"}),
                    on="customer_unique_id", how="left")
    win = later[(later.order_purchase_timestamp > later.t0)
                & (later.order_purchase_timestamp
                   <= later.t0 + pd.Timedelta(days=90))]
    repeat = set(win.customer_unique_id)
    first["label"] = first.customer_unique_id.isin(repeat).astype(int)
    first["entity_id"] = first.customer_unique_id
    first["anchor"] = first.order_purchase_timestamp
    return first.reset_index(drop=True)


TASKS = [
    Task(
        name="bad_review", kind="binary",
        title="Bad review at purchase",
        question="Will this order be reviewed 1 or 2 stars?",
        query=("PREDICT LAST(reviews.review_score) OVER (120 DAYS FOLLOWING) <= 2 "
               "FROM orders WHERE orders.order_id IN :ids"),
        entity_table="orders", anchor_col="order_purchase_timestamp",
        numeric=ORDER_NUMERIC, categorical=ORDER_CATEGORICAL,
        build=_bad_review,
        notes="Imbalanced: roughly one order in six draws a 1-2 star review."),
    Task(
        name="review_stars", kind="multiclass",
        title="Star rating at purchase",
        question="How many stars will this order be reviewed?",
        query=("PREDICT LAST(reviews.star) OVER (120 DAYS FOLLOWING) "
               "FROM orders WHERE orders.order_id IN :ids"),
        entity_table="orders", anchor_col="order_purchase_timestamp",
        numeric=ORDER_NUMERIC, categorical=ORDER_CATEGORICAL,
        build=_stars,
        notes="Five classes, heavily skewed to five stars."),
    Task(
        name="future_spend", kind="regression",
        title="Next-year spend after the first order (sparse)",
        question="How much will this customer spend in the year after their "
                 "first order?",
        query=("PREDICT SUM(payments.payment_value) OVER (90 DAYS FOLLOWING) "
               "FROM customers WHERE customers.customer_unique_id IN :ids"),
        entity_table="customers", anchor_col="order_purchase_timestamp",
        numeric=ORDER_NUMERIC, categorical=ORDER_CATEGORICAL,
        build=_future_spend, horizon_days=90,
        notes="A 90-day future-window aggregate over a linked table, so the "
              "target cannot appear in the entity's own row. Zero for the "
              "majority who never return."),
    Task(
        name="repeat_purchase", kind="binary",
        title="Repeat purchase (sparse)",
        question="After their first order, will this customer ever order again?",
        query=("PREDICT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
               "FROM customers WHERE customers.customer_unique_id IN :ids"),
        entity_table="customers", anchor_col="order_purchase_timestamp",
        numeric=ORDER_NUMERIC, categorical=ORDER_CATEGORICAL,
        build=_repeat_purchase, horizon_days=90,
        notes="The sparse case: most customers have exactly one order, so "
              "per-customer aggregates carry almost no signal."),
]

BY_NAME = {t.name: t for t in TASKS}


def temporal_split(frame: pd.DataFrame, split: pd.Timestamp):
    """Chronological split. Never random: a future row in train leaks."""
    train = frame[frame.anchor < split]
    test = frame[frame.anchor >= split]
    return train.reset_index(drop=True), test.reset_index(drop=True)
