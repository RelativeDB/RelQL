"""The retriever wiring the columnar store fabricates over its arrays."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from relativedb import ColumnDef, LinkDef, Schema, TableDef, ValueType
from relativedb.retrieve import TemporalBound
from relativedb.traversal import ColumnarStore


def _dt(day):
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def _store():
    schema = Schema(
        tables=(
            TableDef("customers", columns=(ColumnDef("age", ValueType.NUMBER),),
                     primary_key="id"),
            TableDef("orders", columns=(ColumnDef("amount", ValueType.NUMBER),
                                        ColumnDef("placed", ValueType.DATETIME)),
                     primary_key="id", time_column="placed"),
        ),
        links=(LinkDef("orders", "customer_id", "customers"),),
    )
    frames = {
        "customers": {"id": np.asarray([1, 2]), "age": np.asarray([30.0, 40.0])},
        "orders": {"id": np.asarray([10, 11, 12]),
                   "amount": np.asarray([5.0, 6.0, 7.0]),
                   "placed": np.asarray([np.datetime64("2024-01-01", "us"),
                                         np.datetime64("2024-01-02", "us"),
                                         np.datetime64("2024-01-03", "us")]),
                   "customer_id": np.asarray([1, 1, 2])},
    }
    return schema, ColumnarStore(schema, frames)


def test_wiring_speaks_the_standard_retriever_contract():
    schema, store = _store()
    wiring = store.wiring()
    bound = TemporalBound.at_or_before(_dt(2))

    entities = wiring.entity_retriever("customers")("customers", [1, 99], bound)
    assert [r.id for r in entities] == [1]

    link = schema.links[0]
    kids = wiring.link_retriever("orders")(link, 1, bound, limit=5)
    assert [r.id for r in kids] == [11, 10]        # newest-first, day 3 cut
    assert wiring.link_retriever("orders")(link, 99, bound, 5) == []

    scanned = list(wiring.scanner("orders")("orders", bound))
    assert [r.id for r in scanned] == [10, 11]     # bound honored by the scan
    assert all(r.cells["amount"] for r in scanned)
