"""End-to-end smoke test of the published PyPI artifact.

Runs the worked churn example (the repo's kb/example.md toy graph, same data
as python/tests/conftest.py) through the real RT-J model via the native
engine bundled in the wheel:

    PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers

Passes iff the native library resolves from site-packages (not a repo
build tree) and every customer gets a finite probability in [0, 1].
"""
from datetime import datetime, timezone

from relativedb import (Engine, ExecutionInput, LinkDef, RetrieverWiring,
                        Row, Schema, TableDef, TemporalBound, ValueType)
from relativedb.rt_native import RtNativeBackend, load_lib


def dt(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# ---- schema (README quickstart / kb/example.md) ---------------------------
schema = (Schema.new_schema()
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

# ---- in-memory database ---------------------------------------------------
ROWS = {
    "customers": [
        Row("customers", "C1", {"age": 34.0, "signup_date": dt("2026-02-10")}),
        Row("customers", "C7", {"age": 52.0, "signup_date": dt("2026-01-20")}),
        Row("customers", "C9", {"age": 27.0, "signup_date": dt("2026-03-05")}),
    ],
    "products": [
        Row("products", "P1", {"price": 25.0, "name": "running shoes"}),
        Row("products", "P2", {"price": 90.0, "name": "espresso machine"}),
        Row("products", "P3", {"price": 35.0, "name": "yoga mat"}),
    ],
    "orders": [
        Row("orders", "O1", {"qty": 1.0, "order_date": dt("2026-03-10")},
            timestamp=dt("2026-03-10"),
            parents={"customer_id": "C7", "product_id": "P2"}),
        Row("orders", "O2", {"qty": 2.0, "order_date": dt("2026-05-02")},
            timestamp=dt("2026-05-02"),
            parents={"customer_id": "C7", "product_id": "P1"}),
        Row("orders", "O3", {"qty": 1.0, "order_date": dt("2026-06-20")},
            timestamp=dt("2026-06-20"),
            parents={"customer_id": "C1", "product_id": "P3"}),
        # O4 is after the anchor and must never enter context
        Row("orders", "O4", {"qty": 1.0, "order_date": dt("2026-07-05")},
            timestamp=dt("2026-07-05"),
            parents={"customer_id": "C7", "product_id": "P3"}),
    ],
}
BY_ID = {t: {r.id: r for r in rs} for t, rs in ROWS.items()}


def entity(table, ids, bound: TemporalBound):
    rows = (BY_ID[table].get(i) for i in ids)
    return [r for r in rows if r is not None and bound.admits_row(r)]


def links(link, parent_id, bound: TemporalBound, limit):
    kids = [r for r in ROWS[link.from_table]
            if r.parents.get(link.fk_column) == parent_id
            and bound.admits_row(r)]
    kids.sort(key=lambda r: (r.timestamp is None,
                             -(r.timestamp.timestamp() if r.timestamp else 0)))
    return kids[:limit]


def make_scanner(table):
    def scan(t, bound: TemporalBound):
        for r in ROWS[table]:
            if bound.admits_row(r):
                yield r
    return scan


wiring = RetrieverWiring.new_wiring().default_links(links)
for t in ROWS:
    wiring.entities(t, entity)
    wiring.scanner(t, make_scanner(t))
wiring = wiring.build()

# ---- the actual check -----------------------------------------------------
lib = load_lib()
assert "site-packages" in lib.path, (
    f"expected the wheel's bundled librt_c, got {lib.path}")
print(f"native engine: ...{lib.path.split('site-packages/')[-1]}")

engine = Engine(schema, wiring, model_backend=RtNativeBackend(schema=schema))
result = engine.execute(ExecutionInput(
    query=("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
           "FROM customers WHERE customers.customer_id IN :ids"),
    params={"ids": ["C1", "C7", "C9"]},
    anchor_time=dt("2026-07-01")))

print("\nP(no order in next 90 days), anchored 2026-07-01:")
probs = {}
for p in result.predictions:
    probs[p.id] = p.probability
    print(f"  {p.id}: {p.probability:.4f}")

assert set(probs) == {"C1", "C7", "C9"}, f"missing predictions: {probs}"
for cid, pr in probs.items():
    assert pr is not None and 0.0 <= pr <= 1.0, f"{cid}: bad probability {pr}"
# The model must actually read the differing contexts — identical outputs
# would mean the per-entity context assembly is broken. (No ordering
# assertion: zero-shot ranking on a 4-row toy graph is model opinion, not
# artifact correctness; accuracy is covered by the repo's golden tests.)
assert len(set(probs.values())) == 3, f"undifferentiated outputs: {probs}"

print("\nPYPI SMOKE TEST PASS")
