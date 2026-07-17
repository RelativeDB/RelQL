"""End-to-end: DataFrames -> schema -> context -> churn prediction,
mirroring the kb/example.md scenario."""
from __future__ import annotations

import pandas as pd
import pytest

import relativedb
from relativedb import SamplerMode, TaskType, ValueType

T0 = pd.Timestamp("2026-07-01")


@pytest.fixture
def frames():
    customers = pd.DataFrame({
        "customer_id": ["C1", "C7", "C9"],
        "age": [34, 52, 27],
        "signup_date": pd.to_datetime(["2026-02-10", "2026-01-20", "2026-03-05"]),
    })
    products = pd.DataFrame({
        "product_id": ["P1", "P2", "P3"],
        "price": [25.0, 90.0, 35.0],
        "name": ["running shoes", "espresso machine", "yoga mat"],
    })
    orders = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4"],
        "customer_id": ["C7", "C7", "C1", "C7"],
        "product_id": ["P2", "P1", "P3", "P3"],
        "qty": [1, 2, 1, 1],
        "order_date": pd.to_datetime(
            ["2026-03-10", "2026-05-02", "2026-06-20", "2026-07-05"]),
    })
    return {"customers": customers, "products": products, "orders": orders}


@pytest.fixture
def dataset(frames):
    return relativedb.from_dataframes(
        frames,
        links=[("orders", "customer_id", "customers"),
               ("orders", "product_id", "products")])


def test_schema_inference(dataset):
    s = dataset.schema
    customers = s.table("customers")
    assert customers.primary_key == "customer_id"
    assert customers.column("age").type is ValueType.NUMBER
    assert customers.column("signup_date").type is ValueType.DATETIME
    assert customers.time_column is None or customers.time_column == "signup_date"
    orders = s.table("orders")
    assert orders.primary_key == "order_id"
    assert orders.time_column == "order_date"
    # FK/PK columns are edges, not cells (F17)
    assert orders.column("customer_id") is None
    assert orders.column("order_id") is None
    assert s.table("products").column("name").type is ValueType.TEXT
    assert len(s.links) == 2


def test_context_assembly_excludes_future(dataset):
    ctx = dataset.assemble_context("customers", "C7", anchor_time=T0)
    keys = ctx.row_keys
    assert ("customers", "C7") in keys
    assert ("orders", "O1") in keys and ("orders", "O2") in keys
    assert ("orders", "O4") not in keys       # after t0: leakage guard
    assert ("orders", "O3") not in keys       # belongs to C1
    assert ("products", "P1") in keys and ("products", "P2") in keys
    # FK values never appear as cells on assembled rows
    for r in ctx.rows:
        assert "customer_id" not in r.cells and "product_id" not in r.cells


def test_churn_predict_end_to_end(dataset):
    df = dataset.predict(
        "PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id",
        anchor_time=T0)
    assert set(df["entity_id"]) == {"C1", "C7", "C9"}
    assert "probability" in df.columns
    probs = dict(zip(df["entity_id"], df["probability"]))
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    # C9 has never ordered -> every history window is empty -> churn-certain
    assert probs["C9"] == 1.0
    # C7 ordered recently (O2, 2026-05-02 within the trailing 90d window)
    assert probs["C7"] < 1.0


def test_predict_with_indices_and_csc(dataset):
    df = dataset.predict(
        "PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR EACH customers.customer_id",
        anchor_time=T0, indices=["C7"], sampler_mode=SamplerMode.CSC)
    assert list(df["entity_id"]) == ["C7"]


def test_regression_and_where_filter(dataset):
    df = dataset.predict(
        "PREDICT SUM(orders.qty, 0, 90, days) FOR EACH customers.customer_id "
        "WHERE COUNT(orders.*, -INF, 0) > 0",
        anchor_time=T0)
    # C9 (no orders ever) is filtered out by the WHERE clause
    assert set(df["entity_id"]) == {"C1", "C7"}
    assert "value" in df.columns


def test_static_where_on_entity_cells(dataset):
    df = dataset.predict(
        "PREDICT SUM(orders.qty, 0, 90, days) FOR EACH customers.customer_id "
        "WHERE customers.age > 30",
        anchor_time=T0)
    assert set(df["entity_id"]) == {"C1", "C7"}


def test_explicit_overrides(frames):
    ds = relativedb.from_dataframes(
        frames,
        links=[("orders", "customer_id", "customers"),
               ("orders", "product_id", "products")],
        primary_keys={"customers": "customer_id"},
        time_columns={"orders": "order_date"})
    assert ds.schema.table("orders").time_column == "order_date"
