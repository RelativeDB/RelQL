"""Olist review-score benchmark: RT-J Metal fine-tuning vs XGBoost.

Task
----
At the moment an order is delivered, predict its later 1--5 star review.  All
features are known by delivery time; rows whose review predates delivery are
excluded.  The split is chronological at 2018-05-01.

Caveat
------
Olist is *not* a clean zero-shot dataset for RT-J.  The released pretraining
recipe contains ``join-spider2-brazilian-e-commerce``.  The benchmark is still
useful as a supervised adaptation comparison, but the released-head result may
be contaminated by pretraining.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
MODEL_COMMIT = "1f552a738a0f8dada8af77db913b2b90511e2f00"
D_MODEL, D_TEXT, MAX_F2P = 512, 384, 5
SEM_NUMBER, SEM_TEXT = 0, 1
RT_DEVICE_MPS = 1
RT_FINETUNE_MULTICLASS = 2
SPLIT_DATE = pd.Timestamp("2018-05-01")
CLASS_NAMES = ["one star", "two stars", "three stars", "four stars", "five stars"]

NUMERIC = [
    "customer_zip_code_prefix",
    "purchase_year", "purchase_month", "purchase_day_of_week", "purchase_hour",
    "estimated_delivery_days", "approval_hours", "carrier_handoff_days",
    "delivery_days", "late_delivery_days",
    "item_count", "total_price", "total_freight", "mean_item_price",
    "distinct_products", "distinct_sellers",
    "product_name_length", "product_description_length", "product_photos_qty",
    "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
    "payment_count", "payment_value", "payment_installments",
]
CATEGORICAL = [
    "customer_state", "customer_city", "product_category", "seller_state",
    "payment_type",
]

# (column, table, node, parent-node). Target is added separately.
FEATURE_SPECS = [
    ("customer_zip_code_prefix", "customers", 0, -1),
    ("customer_state", "customers", 0, -1),
    ("customer_city", "customers", 0, -1),
    ("purchase_year", "orders", 1, 0),
    ("purchase_month", "orders", 1, 0),
    ("purchase_day_of_week", "orders", 1, 0),
    ("purchase_hour", "orders", 1, 0),
    ("estimated_delivery_days", "orders", 1, 0),
    ("approval_hours", "orders", 1, 0),
    ("carrier_handoff_days", "orders", 1, 0),
    ("delivery_days", "orders", 1, 0),
    ("late_delivery_days", "orders", 1, 0),
    ("item_count", "order_items", 2, 1),
    ("total_price", "order_items", 2, 1),
    ("total_freight", "order_items", 2, 1),
    ("mean_item_price", "order_items", 2, 1),
    ("distinct_products", "order_items", 2, 1),
    ("distinct_sellers", "order_items", 2, 1),
    ("product_category", "order_items", 2, 1),
    ("seller_state", "order_items", 2, 1),
    ("product_name_length", "order_items", 2, 1),
    ("product_description_length", "order_items", 2, 1),
    ("product_photos_qty", "order_items", 2, 1),
    ("product_weight_g", "order_items", 2, 1),
    ("product_length_cm", "order_items", 2, 1),
    ("product_height_cm", "order_items", 2, 1),
    ("product_width_cm", "order_items", 2, 1),
    ("payment_count", "payments", 3, 1),
    ("payment_value", "payments", 3, 1),
    ("payment_installments", "payments", 3, 1),
    ("payment_type", "payments", 3, 1),
]


def parse_args():
    default_ckpt = (Path.home() / ".cache/huggingface/hub/"
                    "models--stanford-star--rt-j/snapshots" / MODEL_COMMIT /
                    "classification/model.safetensors")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "benchmarks/corpus/olist")
    p.add_argument("--lib", type=Path, default=ROOT / "cpp/build/librt_c.dylib")
    p.add_argument("--checkpoint", type=Path, default=default_ckpt)
    p.add_argument("--adapter", type=Path,
                   default=ROOT / "benchmarks/task_fit/olist_review_head.safetensors")
    p.add_argument("--results", type=Path,
                   default=ROOT / "benchmarks/task_fit/olist_results.json")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256,
                   help="RT-J feature-extraction batch size")
    p.add_argument("--max-train", type=int, default=0,
                   help="optional newest-N training rows; 0 uses all")
    p.add_argument("--max-test", type=int, default=0,
                   help="optional earliest-N test rows; 0 uses all")
    return p.parse_args()


def ensure_data(data_dir: Path):
    required = [
        "olist_customers_dataset.csv", "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv", "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv", "olist_products_dataset.csv",
        "olist_sellers_dataset.csv", "product_category_name_translation.csv",
    ]
    if all((data_dir / f).exists() for f in required):
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "olist-brazilian-ecommerce.zip"
    print(f"downloading Olist from {DATA_URL}", flush=True)
    urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Olist archive missing files: {missing}")


def prepare_frame(data_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
    reviews = pd.read_csv(data_dir / "olist_order_reviews_dataset.csv",
                          usecols=["order_id", "review_score", "review_creation_date"],
                          parse_dates=["review_creation_date"])
    # A handful of orders have duplicate reviews. Keep the first review that
    # would have become observable in time.
    reviews = reviews.sort_values("review_creation_date").drop_duplicates("order_id")

    products = pd.read_csv(data_dir / "olist_products_dataset.csv")
    products = products.rename(columns={
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    })
    translations = pd.read_csv(data_dir / "product_category_name_translation.csv")
    products = products.merge(translations, on="product_category_name", how="left")
    products["product_category"] = products["product_category_name_english"].fillna(
        products["product_category_name"])
    sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv")
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
    items = items.merge(products, on="product_id", how="left").merge(
        sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
    items = items.sort_values(["order_id", "order_item_id"])
    item_agg = items.groupby("order_id", sort=False).agg(
        item_count=("order_item_id", "size"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        mean_item_price=("price", "mean"),
        distinct_products=("product_id", "nunique"),
        distinct_sellers=("seller_id", "nunique"),
        product_category=("product_category", "first"),
        seller_state=("seller_state", "first"),
        product_name_length=("product_name_length", "mean"),
        product_description_length=("product_description_length", "mean"),
        product_photos_qty=("product_photos_qty", "mean"),
        product_weight_g=("product_weight_g", "mean"),
        product_length_cm=("product_length_cm", "mean"),
        product_height_cm=("product_height_cm", "mean"),
        product_width_cm=("product_width_cm", "mean"),
    ).reset_index()

    payments = pd.read_csv(data_dir / "olist_order_payments_dataset.csv")
    payments = payments.sort_values(["order_id", "payment_sequential"])
    payment_agg = payments.groupby("order_id", sort=False).agg(
        payment_count=("payment_sequential", "size"),
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", "first"),
    ).reset_index()

    frame = (orders.merge(customers, on="customer_id", how="inner")
             .merge(reviews, on="order_id", how="inner")
             .merge(item_agg, on="order_id", how="inner")
             .merge(payment_agg, on="order_id", how="inner"))
    # The prediction anchor is delivery. No future actual-delivery values are
    # exposed: reviews that happened before delivery cannot be valid labels.
    frame = frame[
        frame["order_status"].eq("delivered") &
        frame["order_delivered_customer_date"].notna() &
        (frame["review_creation_date"] >= frame["order_delivered_customer_date"])
    ].copy()
    purchase = frame["order_purchase_timestamp"]
    frame["purchase_year"] = purchase.dt.year
    frame["purchase_month"] = purchase.dt.month
    frame["purchase_day_of_week"] = purchase.dt.dayofweek
    frame["purchase_hour"] = purchase.dt.hour
    hours = lambda end, start: (end - start).dt.total_seconds() / 3600.0
    frame["estimated_delivery_days"] = hours(
        frame["order_estimated_delivery_date"], purchase) / 24.0
    frame["approval_hours"] = hours(frame["order_approved_at"], purchase)
    frame["carrier_handoff_days"] = hours(
        frame["order_delivered_carrier_date"], purchase) / 24.0
    frame["delivery_days"] = hours(
        frame["order_delivered_customer_date"], purchase) / 24.0
    frame["late_delivery_days"] = hours(
        frame["order_delivered_customer_date"],
        frame["order_estimated_delivery_date"]) / 24.0
    frame = frame.sort_values("order_delivered_customer_date").reset_index(drop=True)
    return frame


def clean_features(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    stats = {}
    for col in NUMERIC:
        tr = pd.to_numeric(train[col], errors="coerce")
        median = float(tr.median())
        tr = tr.fillna(median)
        te = pd.to_numeric(test[col], errors="coerce").fillna(median)
        mean, std = float(tr.mean()), float(tr.std(ddof=0))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        train[col], test[col] = tr, te
        stats[col] = (mean, std)
    for col in CATEGORICAL:
        train[col] = train[col].fillna("unknown").astype(str)
        test[col] = test[col].fillna("unknown").astype(str)
    return train, test, stats


def bind(path: Path):
    lib = ctypes.CDLL(str(path))
    f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    lib.rt_device_available.argtypes = [ctypes.c_int32]
    lib.rt_device_available.restype = ctypes.c_int
    lib.rt_model_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_model_load.restype = ctypes.c_void_p
    lib.rt_model_free.argtypes = [ctypes.c_void_p]
    lib.rt_encode_targets_device.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        i64p, i64p, i64p, i64p, u8p, i64p, u8p,
        f32p, f32p, f32p, f32p, f32p,
        ctypes.c_int32, ctypes.c_int32, f32p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_encode_targets_device.restype = ctypes.c_int
    lib.rt_finetune_head_create.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, f32p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_finetune_head_create.restype = ctypes.c_void_p
    lib.rt_finetune_head_free.argtypes = [ctypes.c_void_p]
    lib.rt_finetune_head_predict.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, f32p, f32p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_finetune_head_predict.restype = ctypes.c_int
    lib.rt_finetune_head_fit_metal.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, f32p, f32p,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_finetune_head_fit_metal.restype = ctypes.c_int
    lib.rt_finetune_head_save.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.rt_finetune_head_save.restype = ctypes.c_int
    return lib


def check(rc: int, err, what: str):
    if rc:
        raise RuntimeError(f"{what}: {err.value.decode('utf-8', 'replace')}")


def display_path(path: Path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def text_embeddings(train: pd.DataFrame, test: pd.DataFrame):
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L12-v2", local_files_only=True)
    phrases = [f"{col} of {table}" for col, table, _, _ in FEATURE_SPECS]
    phrases.append("review_score of task")
    col_emb = np.asarray(encoder.encode(phrases, show_progress_bar=False),
                         dtype=np.float32)
    values = []
    for col in CATEGORICAL:
        values.extend(pd.concat([train[col], test[col]]).unique().tolist())
    values = list(dict.fromkeys(values))
    value_emb = np.asarray(encoder.encode(
        values, batch_size=256, show_progress_bar=False), dtype=np.float32)
    lookup = {value: i for i, value in enumerate(values)}
    class_emb = np.asarray(encoder.encode(
        CLASS_NAMES, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32)
    return col_emb, value_emb, lookup, class_emb


def rt_arrays(frame: pd.DataFrame, stats, lookup):
    numeric = np.empty((len(frame), len(NUMERIC)), np.float32)
    for j, col in enumerate(NUMERIC):
        mean, std = stats[col]
        numeric[:, j] = ((frame[col].to_numpy(np.float32) - mean) / std)
    num_pos = {c: i for i, c in enumerate(NUMERIC)}
    cat_codes = np.empty((len(frame), len(CATEGORICAL)), np.int32)
    for j, col in enumerate(CATEGORICAL):
        cat_codes[:, j] = np.fromiter((lookup[v] for v in frame[col]),
                                      dtype=np.int32, count=len(frame))
    cat_pos = {c: i for i, c in enumerate(CATEGORICAL)}
    return numeric, num_pos, cat_codes, cat_pos


def encode_rt(lib, model, numeric, num_pos, cat_codes, cat_pos,
              value_emb, col_emb, batch_size):
    S = len(FEATURE_SPECS) + 1
    N = len(numeric)
    result = np.empty((N, D_MODEL), np.float32)
    # These structural arrays are identical for every order.
    node_row = np.array([n for _, _, n, _ in FEATURE_SPECS] + [4], np.int64)
    table_names = list(dict.fromkeys([t for _, t, _, _ in FEATURE_SPECS] + ["task"]))
    table_id = {t: i for i, t in enumerate(table_names)}
    table_row = np.array([table_id[t] for _, t, _, _ in FEATURE_SPECS] +
                         [table_id["task"]], np.int64)
    f2p_row = np.full((S, MAX_F2P), -1, np.int64)
    for s, (_, _, _, parent) in enumerate(FEATURE_SPECS):
        if parent >= 0:
            f2p_row[s, 0] = parent
    f2p_row[-1, 0] = 1
    sem_row = np.array([
        SEM_TEXT if col in CATEGORICAL else SEM_NUMBER
        for col, _, _, _ in FEATURE_SPECS
    ] + [SEM_TEXT], np.int64)
    target_row = np.zeros(S, np.uint8)
    target_row[-1] = 1
    col_row = np.arange(S, dtype=np.int64)

    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        B = stop - start
        node = np.broadcast_to(node_row, (B, S)).copy()
        f2p = np.broadcast_to(f2p_row, (B, S, MAX_F2P)).copy()
        col = np.broadcast_to(col_row, (B, S)).copy()
        table = np.broadcast_to(table_row, (B, S)).copy()
        padding = np.zeros((B, S), np.uint8)
        sem = np.broadcast_to(sem_row, (B, S)).copy()
        target = np.broadcast_to(target_row, (B, S)).copy()
        number = np.zeros((B, S), np.float32)
        text = np.zeros((B, S, D_TEXT), np.float32)
        for s, (name, _, _, _) in enumerate(FEATURE_SPECS):
            if name in num_pos:
                number[:, s] = numeric[start:stop, num_pos[name]]
            else:
                text[:, s] = value_emb[cat_codes[start:stop, cat_pos[name]]]
        zeros = np.zeros((B, S), np.float32)
        col_names = np.ascontiguousarray(
            np.broadcast_to(col_emb, (B, S, D_TEXT)), np.float32)
        out = np.empty((B, D_MODEL), np.float32)
        err = ctypes.create_string_buffer(1024)
        check(lib.rt_encode_targets_device(
            model, B, S, node, f2p, col, table, padding, sem, target,
            number, zeros, zeros, text, col_names, 0, RT_DEVICE_MPS, out,
            err, len(err)), err, "rt_encode_targets_device")
        result[start:stop] = out
        if stop == N or stop % (batch_size * 20) == 0:
            print(f"RT-J feature extraction {stop:6d}/{N}", flush=True)
    return result


def head_predict(lib, head, features, classes=5):
    features = np.ascontiguousarray(features, np.float32)
    logits = np.empty((len(features), classes), np.float32)
    err = ctypes.create_string_buffer(1024)
    check(lib.rt_finetune_head_predict(head, len(features), features, logits,
                                       err, len(err)), err, "head predict")
    return logits


def softmax(logits):
    z = np.asarray(logits, np.float64)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def metrics(y, prob):
    pred = prob.argmax(axis=1)
    order = np.argsort(-prob, axis=1)
    ranks = np.argmax(order == y[:, None], axis=1) + 1
    bad = y <= 1
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "cross_entropy": float(log_loss(y, prob, labels=np.arange(5))),
        "star_mae": float(np.mean(np.abs(pred - y))),
        "bad_review_auc": float(roc_auc_score(bad, prob[:, :2].sum(axis=1))),
        "class_ranking_mrr": float(np.mean(1.0 / ranks)),
        "class_ranking_recall_at_3": float(np.mean(ranks <= 3)),
    }


def xgboost_run(train, test, y_train):
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20),
         CATEGORICAL),
    ], sparse_threshold=1.0)
    t0 = time.perf_counter()
    X_train = pre.fit_transform(train)
    X_test = pre.transform(test)
    prep_seconds = time.perf_counter() - t0
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss",
        tree_method="hist", n_jobs=8, random_state=1729,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0
    return model.predict_proba(X_test), prep_seconds, train_seconds, X_train.shape[1]


def main():
    a = parse_args()
    ensure_data(a.data_dir)
    frame = prepare_frame(a.data_dir)
    train = frame[frame["order_delivered_customer_date"] < SPLIT_DATE].copy()
    test = frame[frame["order_delivered_customer_date"] >= SPLIT_DATE].copy()
    if a.max_train:
        train = train.tail(a.max_train).copy()
    if a.max_test:
        test = test.head(a.max_test).copy()
    train, test, stats = clean_features(train, test)
    y_train = train["review_score"].to_numpy(np.int64) - 1
    y_test = test["review_score"].to_numpy(np.int64) - 1
    print(f"Olist chronological split: train={len(train)} test={len(test)}", flush=True)
    print("train class counts", np.bincount(y_train, minlength=5).tolist(), flush=True)
    print("test  class counts", np.bincount(y_test, minlength=5).tolist(), flush=True)

    # Majority/prior baseline.
    priors = np.bincount(y_train, minlength=5).astype(np.float64)
    priors /= priors.sum()
    majority_prob = np.broadcast_to(priors, (len(test), 5)).copy()

    # XGBoost gets the same source columns as RT-J.
    xgb_prob, xgb_prep_s, xgb_train_s, xgb_features = xgboost_run(
        train, test, y_train)
    print(f"XGBoost trained in {xgb_train_s:.2f}s ({xgb_features} encoded features)",
          flush=True)

    lib = bind(a.lib)
    if not lib.rt_device_available(RT_DEVICE_MPS):
        raise RuntimeError("Metal device unavailable (run outside a restricted sandbox)")
    err = ctypes.create_string_buffer(1024)
    model = lib.rt_model_load(os.fsencode(a.checkpoint), err, len(err))
    if not model:
        raise RuntimeError(f"model load: {err.value.decode()}")
    head = None
    try:
        col_emb, value_emb, lookup, class_emb = text_embeddings(train, test)
        tr_num, num_pos, tr_cat, cat_pos = rt_arrays(train, stats, lookup)
        te_num, _, te_cat, _ = rt_arrays(test, stats, lookup)
        t0 = time.perf_counter()
        tr_features = encode_rt(lib, model, tr_num, num_pos, tr_cat, cat_pos,
                                value_emb, col_emb, a.batch_size)
        te_features = encode_rt(lib, model, te_num, num_pos, te_cat, cat_pos,
                                value_emb, col_emb, a.batch_size)
        feature_seconds = time.perf_counter() - t0
        head = lib.rt_finetune_head_create(
            model, RT_FINETUNE_MULTICLASS, 5,
            np.ascontiguousarray(class_emb), err, len(err))
        if not head:
            raise RuntimeError(f"head create: {err.value.decode()}")
        rt_before = softmax(head_predict(lib, head, te_features))
        initial, final = ctypes.c_float(), ctypes.c_float()
        train_seconds = ctypes.c_double()
        check(lib.rt_finetune_head_fit_metal(
            head, len(tr_features), np.ascontiguousarray(tr_features),
            np.ascontiguousarray(y_train, np.float32), None, 0, a.epochs,
            a.learning_rate, a.weight_decay,
            ctypes.byref(initial), ctypes.byref(final),
            ctypes.byref(train_seconds), err, len(err)), err, "Metal fine-tune")
        rt_after = softmax(head_predict(lib, head, te_features))
        check(lib.rt_finetune_head_save(
            head, os.fsencode(a.adapter), err, len(err)), err, "head save")

        result = {
            "dataset": "Brazilian E-Commerce Public Dataset by Olist",
            "dataset_url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
            "pretraining_caveat": (
                "potentially contaminated: RT-J recipe contains "
                "join-spider2-brazilian-e-commerce"),
            "task": "predict 1-5 star review at order delivery",
            "temporal_protocol": {
                "split": str(SPLIT_DATE.date()),
                "train": len(train), "test": len(test),
                "review_must_be_at_or_after_delivery": True,
                "feature_statistics": "train split only",
            },
            "checkpoint": f"stanford-star/rt-j@{MODEL_COMMIT}/classification",
            "rt_method": "frozen RT-J backbone + trainable 5x512 linear head",
            "rt_trainable_parameters": 5 * D_MODEL + 5,
            "rt_feature_extraction_seconds": feature_seconds,
            "rt_feature_batch_size": a.batch_size,
            "rt_epochs": a.epochs,
            "rt_head_training_seconds": train_seconds.value,
            "rt_training_loss": {"before": initial.value, "after": final.value},
            "xgboost": {
                "encoded_features": xgb_features,
                "preprocessing_seconds": xgb_prep_s,
                "training_seconds": xgb_train_s,
                "parameters": {
                    "n_estimators": 500, "max_depth": 6,
                    "learning_rate": 0.05,
                },
            },
            "metrics": {
                "train_prior": metrics(y_test, majority_prob),
                "rt_j_released_head": metrics(y_test, rt_before),
                "rt_j_metal_finetuned": metrics(y_test, rt_after),
                "xgboost": metrics(y_test, xgb_prob),
            },
            "adapter": display_path(a.adapter),
        }
        a.results.write_text(json.dumps(result, indent=2) + "\n")
        print("\n" + json.dumps(result, indent=2))
    finally:
        if head:
            lib.rt_finetune_head_free(head)
        lib.rt_model_free(model)


if __name__ == "__main__":
    main()
