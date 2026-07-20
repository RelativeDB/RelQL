"""Olist (Brazilian e-commerce) loaded as a relational graph.

XGBoost needs one flat table, so it gets a denormalized feature frame. RelativeDB
reads the graph directly, so it gets a schema plus retrievers over the same rows.
Both sides are built from the same source CSVs and the same temporal split, so
the comparison is about the models rather than the data prep.
"""
from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
DEFAULT_DIR = Path("/private/tmp/olist-data")

FILES = [
    "olist_customers_dataset.csv", "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv", "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv", "olist_products_dataset.csv",
    "olist_sellers_dataset.csv", "product_category_name_translation.csv",
]


def ensure_data(data_dir: Path = DEFAULT_DIR) -> Path:
    """Download and unzip Olist on first use; a no-op once cached."""
    if all((data_dir / f).exists() for f in FILES):
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "olist.zip"
    print(f"downloading Olist -> {data_dir}", flush=True)
    urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    missing = [f for f in FILES if not (data_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Olist archive is missing {missing}")
    return data_dir


@dataclass
class Olist:
    """The loaded tables, already typed and joined where the join is 1:1."""

    customers: pd.DataFrame
    orders: pd.DataFrame
    items: pd.DataFrame
    products: pd.DataFrame
    sellers: pd.DataFrame
    payments: pd.DataFrame
    reviews: pd.DataFrame

    def describe(self) -> dict:
        return {name: int(len(getattr(self, name))) for name in
                ("customers", "orders", "items", "products", "sellers",
                 "payments", "reviews")}


def load(data_dir: Path = DEFAULT_DIR, max_orders: Optional[int] = None) -> Olist:
    d = ensure_data(data_dir)

    orders = pd.read_csv(d / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date"])
    orders = orders[orders.order_purchase_timestamp.notna()]
    orders = orders.sort_values("order_purchase_timestamp")
    if max_orders:
        orders = orders.head(max_orders)
    keep = set(orders.order_id)

    customers = pd.read_csv(d / "olist_customers_dataset.csv")
    items = pd.read_csv(d / "olist_order_items_dataset.csv",
                        parse_dates=["shipping_limit_date"])
    products = pd.read_csv(d / "olist_products_dataset.csv")
    sellers = pd.read_csv(d / "olist_sellers_dataset.csv")
    payments = pd.read_csv(d / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(d / "olist_order_reviews_dataset.csv", parse_dates=[
        "review_creation_date", "review_answer_timestamp"])

    tr = pd.read_csv(d / "product_category_name_translation.csv")
    products = products.merge(tr, on="product_category_name", how="left")
    products["category"] = (products.product_category_name_english
                            .fillna(products.product_category_name)
                            .fillna("unknown"))

    items = items[items.order_id.isin(keep)]
    payments = payments[payments.order_id.isin(keep)]
    reviews = reviews[reviews.order_id.isin(keep)]
    # `customer_id` is unique per order in Olist; `customer_unique_id` is the
    # actual person. Keying on the former makes every customer look new and
    # erases the repeat-purchase signal entirely.
    orders = orders.merge(customers[["customer_id", "customer_unique_id"]],
                          on="customer_id", how="left")
    customers = (customers.sort_values("customer_id")
                 .drop_duplicates("customer_unique_id")
                 .reset_index(drop=True))
    customers = customers[customers.customer_unique_id.isin(
        set(orders.customer_unique_id))]

    # every row needs a stable primary key and a time column where it has one
    items = items.reset_index(drop=True)
    items["item_key"] = items.order_id + "-" + items.order_item_id.astype(str)
    payments = payments.reset_index(drop=True)
    payments["payment_key"] = (payments.order_id + "-"
                               + payments.payment_sequential.astype(str))

    # an order's time is its purchase; items/payments inherit it, since neither
    # carries a usable timestamp of its own
    ts = orders.set_index("order_id").order_purchase_timestamp
    items["ts"] = items.order_id.map(ts)
    payments["ts"] = payments.order_id.map(ts)
    reviews["ts"] = reviews.review_creation_date

    return Olist(customers=customers, orders=orders, items=items,
                 products=products, sellers=sellers, payments=payments,
                 reviews=reviews)


def order_features(o: Olist) -> pd.DataFrame:
    """One row per order: everything knowable at purchase time, plus the
    outcomes tasks may use as labels. Feature/label separation is the task's
    job, not this function's."""
    f = o.orders[["order_id", "customer_id", "order_status",
                  "order_purchase_timestamp", "order_estimated_delivery_date",
                  "order_delivered_customer_date", "order_approved_at"]].copy()
    # customers is deduped by customer_unique_id, so customer_id no longer joins
    # against it. orders already carries customer_unique_id; attributes come
    # across on that key instead.
    f["customer_unique_id"] = o.orders.set_index("order_id").customer_unique_id \
        .reindex(f.order_id).to_numpy()
    f = f.merge(o.customers[["customer_unique_id", "customer_state",
                             "customer_city", "customer_zip_code_prefix"]],
                on="customer_unique_id", how="left")

    agg = o.items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        mean_item_price=("price", "mean"),
        distinct_products=("product_id", "nunique"),
        distinct_sellers=("seller_id", "nunique"))
    f = f.merge(agg, on="order_id", how="left")

    # the modal category and its seller stand in for the order's contents
    first = o.items.sort_values("order_item_id").groupby("order_id").first()
    f = f.merge(first[["product_id", "seller_id"]], on="order_id", how="left")
    f = f.merge(o.products[["product_id", "category", "product_weight_g",
                            "product_photos_qty", "product_description_lenght"]]
                .rename(columns={"product_description_lenght": "product_desc_len"}),
                on="product_id", how="left")
    f = f.merge(o.sellers[["seller_id", "seller_state"]], on="seller_id",
                how="left")

    pay = o.payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_count=("payment_sequential", "count"))
    f = f.merge(pay, on="order_id", how="left")
    ptype = (o.payments.sort_values("payment_sequential")
             .groupby("order_id").payment_type.first())
    f["payment_type"] = f.order_id.map(ptype)

    rev = o.reviews.sort_values("review_creation_date").groupby("order_id").last()
    f["review_score"] = f.order_id.map(rev.review_score)
    f["review_ts"] = f.order_id.map(rev.ts)

    p = f.order_purchase_timestamp
    f["purchase_year"] = p.dt.year
    f["purchase_month"] = p.dt.month
    f["purchase_dow"] = p.dt.dayofweek
    f["purchase_hour"] = p.dt.hour
    f["estimated_delivery_days"] = (f.order_estimated_delivery_date - p).dt.days
    f["approval_hours"] = (f.order_approved_at - p).dt.total_seconds() / 3600
    f["delivery_days"] = (f.order_delivered_customer_date - p).dt.days
    return f


def history_depth(o: Olist, frame: pd.DataFrame, entity: str,
                  anchor_col: str) -> pd.Series:
    """How many prior orders the entity had at its anchor time.

    This is the sparsity axis: a model that only sees aggregates has nothing to
    aggregate at depth 0, while a relational model still sees the order itself
    and its neighbours.
    """
    ord_ts = (o.orders[["order_id", "customer_unique_id",
                        "order_purchase_timestamp"]]
              .rename(columns={"order_purchase_timestamp": "_hist_ts",
                               "order_id": "_hist_order"}))
    if entity == "customers":
        key = "customer_unique_id"
        merged = frame[[key, anchor_col]].merge(ord_ts, on=key, how="left")
    else:
        key = "order_id"
        cust = o.orders.set_index("order_id").customer_unique_id
        f = frame[[key, anchor_col]].copy()
        f["customer_unique_id"] = f[key].map(cust)
        merged = f.merge(ord_ts, on="customer_unique_id", how="left")
    prior = merged[merged._hist_ts < merged[anchor_col]]
    return (prior.groupby(key).size()
            .reindex(frame[key].values, fill_value=0).astype(int))


def depth_bucket(n: np.ndarray | pd.Series) -> pd.Series:
    """Sparsity buckets used for every per-slice metric in the report."""
    n = pd.Series(np.asarray(n))
    return pd.cut(n, bins=[-1, 0, 1, 5, 10**9],
                  labels=["0 prior", "1 prior", "2-5 prior", "6+ prior"])
