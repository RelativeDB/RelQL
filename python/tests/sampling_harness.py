"""Deterministic fixtures and fingerprints for the sampling regression harness.

Context assembly is a sampled walk over a graph, seeded from the context
policy and from node indices. That makes it exact but fragile: reordering a
loop, renumbering nodes, or changing how a strategy is dispatched can silently
change which rows land in a context -- and therefore every prediction -- while
every behavioural test still passes. columnar.py was removed for exactly this
class of divergence.

This module builds a graph big enough for the sampler to make real choices,
then fingerprints what it sampled. The fingerprint is the ORDERED sequence of
context rows, not a summary: order is what the model sees and what the RNG
determines, so it is the property worth pinning.

Nothing here needs a model. Sampling is decided before scoring, so the whole
harness runs in the unit tier with a stub backend.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from relativedb import (ContextPolicy, Engine, EntityPrediction, LinkDef,
                        RetrieverWiring, Row, Schema, TableDef, TaskType,
                        TemporalBound, ValueType)
from relativedb.engine import SamplerMode
from relativedb.traversal import BreadthFirstTraversal, ReferenceTraversal

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Sampling geometry only bites when there is more to sample than the caps
# admit, so the graph is sized so that customers exceed bfs_width and orders
# per customer exceed the fanout.
N_CUSTOMERS = 12
N_PRODUCTS = 5
N_ORDERS = 90


class _Lcg:
    """Fixed-seed LCG (Numerical Recipes constants). The fixture must not move
    when Python's hash randomization or the platform RNG changes."""

    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFF_FFFF_FFFF_FFFF

    def next(self) -> int:
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) \
            & 0xFFFF_FFFF_FFFF_FFFF
        return self.s

    def range(self, n: int) -> int:
        return self.next() % n if n > 0 else 0


def build_schema() -> Schema:
    return (Schema.new_schema()
            .table(TableDef.new_table("customers")
                   .column("age", ValueType.NUMBER)
                   .column("tier", ValueType.TEXT)
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


def build_rows() -> dict[str, list[Row]]:
    """A fixed graph. Every value is derived from the LCG or the index, so the
    fixture is byte-identical on every machine and every Python build."""
    rng = _Lcg(0xC0FFEE)
    customers = [
        Row("customers", f"C{i:02d}",
            {"age": float(20 + (i * 7) % 45),
             "tier": ("gold", "silver", "bronze")[i % 3]})
        for i in range(N_CUSTOMERS)
    ]
    products = [
        Row("products", f"P{i}",
            {"price": float(10 + i * 15),
             "name": ("shoes", "espresso machine", "yoga mat", "kettle",
                      "lamp")[i]})
        for i in range(N_PRODUCTS)
    ]
    orders = []
    for i in range(N_ORDERS):
        cust = rng.range(N_CUSTOMERS)
        prod = rng.range(N_PRODUCTS)
        day = rng.range(180)
        ts = EPOCH + timedelta(days=day)
        orders.append(Row("orders", f"O{i:03d}",
                          {"qty": float(1 + rng.range(4)), "order_date": ts},
                          timestamp=ts,
                          parents={"customer_id": f"C{cust:02d}",
                                   "product_id": f"P{prod}"}))
    return {"customers": customers, "products": products, "orders": orders}


def build_wiring(rows: dict[str, list[Row]]) -> RetrieverWiring:
    by_id = {t: {r.id: r for r in rs} for t, rs in rows.items()}

    def entity(table, ids, bound: TemporalBound):
        out = []
        for i in ids:
            r = by_id[table].get(i)
            if r is not None and bound.admits_row(r):
                out.append(r)
        return out

    def links(link, parent_id, bound: TemporalBound, limit):
        kids = [r for r in rows[link.from_table]
                if r.parents.get(link.fk_column) == parent_id
                and bound.admits_row(r)]
        kids.sort(key=lambda r: (r.timestamp is None,
                                 -(r.timestamp.timestamp() if r.timestamp
                                   else 0.0)))
        return kids[:limit]

    def make_scanner(table):
        def scan(t, bound: TemporalBound):
            for r in rows[table]:
                if bound.admits_row(r):
                    yield r
        return scan

    wb = RetrieverWiring.new_wiring().default_links(links)
    for t in rows:
        wb.entities(t, entity)
        wb.scanner(t, make_scanner(t))
    return wb.build()


class RecordingBackend:
    """Records the order contexts arrive in. Deterministic and model-free: the
    value is a pure function of the entity id, so a prediction can never mask a
    context-ordering change."""

    def __init__(self, batch_size: int = 0):
        if batch_size:
            self.batch_size = batch_size
        self.seen: list[str] = []

    def score(self, query, task_type, contexts, model_uri, config):
        binary = task_type is TaskType.BINARY_CLASSIFICATION
        out = []
        for c in contexts:
            self.seen.append(str(c.entity_id))
            v = (int(hashlib.sha256(str(c.entity_id).encode()).hexdigest()[:8],
                     16) % 1000) / 1000.0
            out.append(EntityPrediction(c.entity_id,
                                        probability=v if binary else None,
                                        value=None if binary else v))
        return out


# --------------------------------------------------------------------------
# the case matrix
# --------------------------------------------------------------------------

CHURN = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")
# A different window as well as a different target: with the same 90-day frame
# a COUNT fingerprints identically to the NOT EXISTS above, so it would add no
# regression-detection power over `reference-default`.
COUNT = ("PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")

# A direct target (a column of the entity table) skips the derived-task graph
# and takes ReferenceTraversal's non-shared path, where the walk seeds from
# `physical_node_ids` -- i.e. from the ORDER rows are enumerated in, not just
# their count. Without a case here the harness cannot see a node-renumbering
# change at all, which a mutation test showed the hard way.
DIRECT = ("PREDICT customers.age FROM customers "
          "WHERE customers.customer_id IN :ids")

# `customers` is declared first AND sorts first, so its node indices are the
# same under either ordering -- a renumbering bug is invisible from a customer
# target. `products` is declared second but sorts last, so its indices do move.
# This is the case that can actually see a renumbering change.
DIRECT_PRODUCTS = "PREDICT products.price FROM products"
PRODUCT_COHORT = [f"P{i}" for i in range(N_PRODUCTS)]

COHORT = [f"C{i:02d}" for i in range(6)]
ANCHOR = EPOCH + timedelta(days=120)


def _policy(**kw) -> ContextPolicy:
    base = dict(max_context_cells=256, bfs_width=3, max_hops=2,
                local_context_cells=64, num_walks=200, walk_length=8, seed=0)
    base.update(kw)
    return ContextPolicy(**base)


def cases() -> dict[str, dict]:
    """Name -> engine/query configuration. Each name is a fingerprint key, so
    renaming one is a deliberate act that shows up as a missing entry."""
    return {
        "reference-default": dict(policy=_policy(), traversal="reference"),
        "reference-seed7": dict(policy=_policy(seed=7), traversal="reference"),
        "reference-wide": dict(policy=_policy(bfs_width=8, max_hops=3),
                               traversal="reference"),
        "reference-no-prefer-latest": dict(
            policy=_policy(prefer_latest=False), traversal="reference"),
        "reference-tight-budget": dict(
            policy=_policy(max_context_cells=64), traversal="reference"),
        "bfs-retriever": dict(policy=_policy(), traversal="bfs",
                              sampler_mode=SamplerMode.RETRIEVER),
        "bfs-csc": dict(policy=_policy(), traversal="bfs",
                        sampler_mode=SamplerMode.CSC),
        "reference-regression-target": dict(
            policy=_policy(), traversal="reference", query=COUNT),
        # An earlier anchor hides most of the graph, so this pins the temporal
        # bound's effect on sampling rather than just the geometry's.
        "reference-early-anchor": dict(
            policy=_policy(), traversal="reference",
            anchor=EPOCH + timedelta(days=45)),
        # Pipelined assembly must sample and score in the same order as the
        # serial path; a batch size smaller than the cohort turns it on.
        "reference-pipelined": dict(
            policy=_policy(), traversal="reference", batch_size=2),
        # Direct target -> the non-shared traversal path, whose walk seed
        # depends on physical node ORDER. This is the only case that can see a
        # renumbering change.
        # (No BFS equivalent: BreadthFirstTraversal is target-agnostic, so a
        # direct target fingerprints identically to the cases above and would
        # add no detection power.)
        "reference-direct-target": dict(
            policy=_policy(), traversal="reference", query=DIRECT),
        "reference-direct-target-products": dict(
            policy=_policy(), traversal="reference", query=DIRECT_PRODUCTS,
            entity_table="products", cohort=PRODUCT_COHORT),
    }


def make_engine(schema, wiring, spec, backend=None) -> Engine:
    traversal = (ReferenceTraversal() if spec["traversal"] == "reference"
                 else BreadthFirstTraversal())
    return Engine(schema, wiring,
                  model_backend=backend,
                  context_policy=spec["policy"],
                  traversal=traversal,
                  sampler_mode=spec.get("sampler_mode",
                                        SamplerMode.RETRIEVER))


# --------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------

def _row_key(row: Row) -> str:
    return f"{row.table}:{row.id}"


def context_fingerprint(engine: Engine, query, ids, anchor,
                        entity_table: str = "customers") -> dict:
    """The ordered context each entity is given.

    ``rows`` is the sampled sequence in order -- the exact thing the model
    consumes and the exact thing an RNG or traversal change perturbs.
    """
    out: dict[str, dict] = {}
    for eid in ids:
        ctx = engine.assemble_context(entity_table, eid, anchor, query=query)
        out[str(eid)] = {
            "rows": [_row_key(r) for r in ctx.rows],
            "focal": sorted(f"{t}:{i}" for t, i in ctx.focal_row_keys),
            "cells": ctx.cell_count,
            "truncated_children": bool(ctx.truncated_children),
            "hit_cell_budget": bool(ctx.hit_cell_budget),
        }
    return out


def execution_fingerprint(engine: Engine, backend: RecordingBackend,
                          query: str, ids, anchor) -> dict:
    """The order execute() delivers contexts to the backend, and the order and
    values of the predictions that come back."""
    from relativedb import ExecutionInput
    params = {"ids": list(ids)} if ":ids" in query else None
    result = engine.execute(ExecutionInput(query=query, anchor_time=anchor,
                                           params=params))
    return {
        "scored_order": list(backend.seen),
        "prediction_order": [str(p.id) for p in result.predictions],
        "values": [round(p.probability if p.probability is not None
                         else p.value, 6) for p in result.predictions],
    }


def compute_all() -> dict:
    """Every case's fingerprint, ready to compare against the committed file."""
    schema = build_schema()
    rows = build_rows()
    wiring = build_wiring(rows)
    out: dict = {}
    from relativedb import parse, validate
    for name, spec in cases().items():
        query = spec.get("query", CHURN)
        # assemble_context takes a bound ParsedQuery; the cohort pin is a
        # WHERE concern and irrelevant to how one entity's context is sampled.
        bare = query.replace(" WHERE customers.customer_id IN :ids", "")
        pq = validate(parse(bare), schema).query
        anchor = spec.get("anchor", ANCHOR)
        cohort = spec.get("cohort", COHORT)
        entity_table = spec.get("entity_table", "customers")
        ctx_engine = make_engine(schema, wiring, spec)
        backend = RecordingBackend(batch_size=spec.get("batch_size", 0))
        exec_engine = make_engine(schema, wiring, spec, backend=backend)
        out[name] = {
            "context": context_fingerprint(ctx_engine, pq, cohort, anchor,
                                           entity_table),
            "execution": execution_fingerprint(exec_engine, backend, query,
                                               cohort, anchor),
        }
    return out


def digest(fingerprints: dict) -> str:
    return hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
