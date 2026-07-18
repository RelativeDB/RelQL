"""Shared fixtures: the worked churn example from kb/example.md as a toy graph."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from relativedb import (ColumnDef, EntityPrediction, LinkDef, RetrieverWiring,
                      Row, Schema, TableDef, TaskType, TemporalBound, ValueType)


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


class StubBackend:
    """Tiny deterministic test-only ModelBackend. The engine ships no scorer;
    plumbing tests (routing, AS OF, CSC execute, EXPLAIN ANALYZE) use this so
    they stay fast and offline without a real checkpoint. RETURN output-shaping
    is a native-backend concern and is exercised in test_rt_native.py."""

    def score(self, query, task_type, contexts, model_uri, config):
        binary = task_type is TaskType.BINARY_CLASSIFICATION
        return [EntityPrediction(c.entity_id,
                                 probability=0.5 if binary else None,
                                 value=None if binary else 1.0)
                for c in contexts]


@pytest.fixture
def stub_backend() -> StubBackend:
    return StubBackend()


@pytest.fixture
def churn_schema() -> Schema:
    return (Schema.new_schema()
            .table(TableDef.new_table("customers")
                   .column("age", ValueType.NUMBER)
                   .column("signup_date", ValueType.DATETIME)
                   .primary_key("customer_id").build())
            .table(TableDef.new_table("products")
                   .column("price", ValueType.NUMBER)
                   .column("name", ValueType.TEXT)
                   .primary_key("product_id").build())
            .table(TableDef.new_table("orders")
                   .column("qty", ValueType.NUMBER)
                   .column("order_date", ValueType.DATETIME)
                   .primary_key("order_id")
                   .time_column("order_date").build())
            .link(LinkDef("orders", "customer_id", "customers"))
            .link(LinkDef("orders", "product_id", "products"))
            .build())


def churn_rows() -> dict[str, list[Row]]:
    """The kb/example.md database. O4 (2026-07-05) is AFTER the anchor t0 =
    2026-07-01 and must never enter context."""
    customers = [
        Row("customers", "C1", {"age": 34.0, "signup_date": dt("2026-02-10")}),
        Row("customers", "C7", {"age": 52.0, "signup_date": dt("2026-01-20")}),
        Row("customers", "C9", {"age": 27.0, "signup_date": dt("2026-03-05")}),
    ]
    products = [
        Row("products", "P1", {"price": 25.0, "name": "running shoes"}),
        Row("products", "P2", {"price": 90.0, "name": "espresso machine"}),
        Row("products", "P3", {"price": 35.0, "name": "yoga mat"}),
    ]
    orders = [
        Row("orders", "O1", {"qty": 1.0, "order_date": dt("2026-03-10")},
            timestamp=dt("2026-03-10"),
            parents={"customer_id": "C7", "product_id": "P2"}),
        Row("orders", "O2", {"qty": 2.0, "order_date": dt("2026-05-02")},
            timestamp=dt("2026-05-02"),
            parents={"customer_id": "C7", "product_id": "P1"}),
        Row("orders", "O3", {"qty": 1.0, "order_date": dt("2026-06-20")},
            timestamp=dt("2026-06-20"),
            parents={"customer_id": "C1", "product_id": "P3"}),
        Row("orders", "O4", {"qty": 1.0, "order_date": dt("2026-07-05")},
            timestamp=dt("2026-07-05"),  # future of t0!
            parents={"customer_id": "C7", "product_id": "P3"}),
    ]
    return {"customers": customers, "products": products, "orders": orders}


def in_memory_wiring(rows: dict[str, list[Row]], *,
                     honor_bound: bool = True) -> RetrieverWiring:
    """Well-behaved (or, with honor_bound=False, deliberately leaky)
    retrievers + scanners over an in-memory row dict."""
    by_id = {t: {r.id: r for r in rs} for t, rs in rows.items()}

    def entity(table, ids, bound: TemporalBound):
        out = []
        for i in ids:
            r = by_id[table].get(i)
            if r is None:
                continue
            if honor_bound and not bound.admits_row(r):
                continue
            out.append(r)
        return out

    def links(link, parent_id, bound: TemporalBound, limit):
        kids = [r for r in rows[link.from_table]
                if r.parents.get(link.fk_column) == parent_id]
        if honor_bound:
            kids = [r for r in kids if bound.admits_row(r)]
        kids.sort(key=lambda r: (r.timestamp is None,
                                 -(r.timestamp.timestamp() if r.timestamp
                                   else 0.0)))
        return kids[:limit] if honor_bound else kids

    def make_scanner(table):
        def scan(t, bound: TemporalBound):
            for r in rows[table]:
                if not honor_bound or bound.admits_row(r):
                    yield r
        return scan

    wb = RetrieverWiring.new_wiring().default_links(links)
    for t in rows:
        wb.entities(t, entity)
        wb.scanner(t, make_scanner(t))
    return wb.build()


@pytest.fixture
def churn_wiring() -> RetrieverWiring:
    return in_memory_wiring(churn_rows())
