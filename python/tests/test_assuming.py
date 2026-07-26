"""ASSUMING must change what the model sees — by exactly the assumed value.

The bug these tests pin down: ``_apply_assumptions`` used to overwrite EVERY
context row of the assigned table. Combined with zero-shot normalization —
which derives column statistics inside each context — the assumed column
became a constant, and a constant numeric column normalizes to zero regardless
of which constant it was. Result: ``ASSUMING x = TRUE`` and ``ASSUMING
x = FALSE`` produced byte-identical model inputs, while both differed from the
no-ASSUMING baseline (the factual variation had been destroyed). The clause's
*presence* changed the prediction; its *value* was silently erased.

The fix is semantic: an assignment on the entity table intervenes on the
entity's own row only, keeping sibling rows factual — which both matches what
a counterfactual means and preserves the in-context scale the assumed value is
normalized against. Assignments on other tables keep the documented broad
meaning, and the cases where a value still cannot survive (flattened numeric
column, missing entity row) now warn instead of degrading silently.
"""
from __future__ import annotations

import warnings

import pytest

from relativedb import AssumptionNotAppliedWarning, Engine, ExecutionInput
from relativedb.engine import _apply_assumptions, _warn_inert_assumptions
from relativedb.plan import assumptions
from relativedb.relql.parser import parse, validate

from conftest import churn_rows, dt, in_memory_wiring

ANCHOR = dt("2026-07-01")


def _engine(churn_schema, stub_backend):
    return Engine(churn_schema, in_memory_wiring(churn_rows()),
                  model_backend=stub_backend)


def _context(engine, query, entity_id="C7"):
    """Parse the query and hand-build a context holding the entity row, a
    sibling customer, and the entity's pre-anchor orders. Synthetic on
    purpose: the reference traversal does not expand children of an undated
    entity row, and these tests are about assumption semantics, not
    traversal reach."""
    from relativedb.engine import EntityContext
    pq = validate(parse(query), engine.schema,
                  {"ids": [entity_id]}).query.bind_params({"ids": [entity_id]})
    rows = churn_rows()
    picked = [r for r in rows["customers"] if r.id in (entity_id, "C1")]
    picked += [r for r in rows["orders"]
               if r.parents.get("customer_id") == entity_id
               and r.timestamp is not None and r.timestamp <= ANCHOR]
    ctx = EntityContext(entity_id=entity_id, anchor=ANCHOR,
                        rows=tuple(picked), focal_row_keys=frozenset(),
                        node_ids={})
    return pq, ctx


# ---------------------------------------------------------------------------
# semantics: who gets the assumed value
# ---------------------------------------------------------------------------

def test_entity_table_assumption_touches_only_the_entity_row(
        churn_schema, stub_backend):
    engine = _engine(churn_schema, stub_backend)
    query = ("PREDICT customers.age FROM customers "
             "WHERE customers.customer_id IN :ids "
             "ASSUMING customers.age = 99 RETURN EXPECTED VALUE")
    pq, ctx = _context(engine, query)
    out = _apply_assumptions(assumptions(pq.assuming), ctx, "customers")
    ages = {r.id: r.cells["age"] for r in out.rows if r.table == "customers"}
    assert ages["C7"] == 99          # the entity: intervened
    assert ages["C1"] == 34.0        # the sibling: factual


def test_non_entity_table_assumption_keeps_broad_semantics(
        churn_schema, stub_backend):
    engine = _engine(churn_schema, stub_backend)
    query = ("PREDICT customers.age FROM customers "
             "WHERE customers.customer_id IN :ids "
             "ASSUMING orders.qty = 7 RETURN EXPECTED VALUE")
    pq, ctx = _context(engine, query)
    out = _apply_assumptions(assumptions(pq.assuming), ctx, "customers")
    qtys = [r.cells["qty"] for r in out.rows if r.table == "orders"]
    assert qtys, "fixture context should hold the customer's orders"
    assert all(q == 7 for q in qtys)   # "these orders": all of them


def test_assumed_value_survives_into_distinct_model_inputs(
        churn_schema, stub_backend):
    """TRUE vs FALSE (here: 99 vs 18) must not collapse to the same context."""
    engine = _engine(churn_schema, stub_backend)
    cells = {}
    for value in (99, 18):
        query = ("PREDICT customers.age FROM customers "
                 "WHERE customers.customer_id IN :ids "
                 f"ASSUMING customers.age = {value} RETURN EXPECTED VALUE")
        pq, ctx = _context(engine, query)
        out = _apply_assumptions(assumptions(pq.assuming), ctx, "customers")
        cells[value] = {(r.table, r.id): dict(r.cells) for r in out.rows}
    assert cells[99] != cells[18]
    assert cells[99][("customers", "C7")]["age"] == 99
    assert cells[18][("customers", "C7")]["age"] == 18


# ---------------------------------------------------------------------------
# loud failure: the ways an assumption still cannot reach the model
# ---------------------------------------------------------------------------

def test_flattened_numeric_column_warns(churn_schema, stub_backend):
    """A non-entity assignment that leaves the column constant in-context is
    erased by zero-shot normalization — that must be said, not swallowed."""
    engine = _engine(churn_schema, stub_backend)
    query = ("PREDICT customers.age FROM customers "
             "WHERE customers.customer_id IN :ids "
             "ASSUMING orders.qty = 7 RETURN EXPECTED VALUE")
    pq, ctx = _context(engine, query)
    assigned = assumptions(pq.assuming)
    out = _apply_assumptions(assigned, ctx, "customers")
    with pytest.warns(AssumptionNotAppliedWarning,
                      match="constant across the context"):
        _warn_inert_assumptions(assigned, [out], "customers", engine.schema)


def test_missing_table_still_warns(churn_schema, stub_backend):
    engine = _engine(churn_schema, stub_backend)
    query = ("PREDICT customers.age FROM customers "
             "WHERE customers.customer_id IN :ids "
             "ASSUMING products.price = 1 RETURN EXPECTED VALUE")
    pq, ctx = _context(engine, query, entity_id="C9")   # C9 has no orders
    assigned = assumptions(pq.assuming)
    out = _apply_assumptions(assigned, ctx, "customers")
    assert not any(r.table == "products" for r in out.rows)
    with pytest.warns(AssumptionNotAppliedWarning, match="no rows"):
        _warn_inert_assumptions(assigned, [out], "customers", engine.schema)


def test_entity_row_missing_from_context_warns(churn_schema):
    from relativedb.engine import EntityContext
    row = churn_rows()["orders"][0]
    ctx = EntityContext(entity_id="C7", anchor=ANCHOR, rows=(row,),
                        focal_row_keys=frozenset(), node_ids={})
    with pytest.warns(AssumptionNotAppliedWarning,
                      match="entity's own row is missing"):
        _warn_inert_assumptions([("customers", "age", 99)], [ctx],
                                "customers", None)


def test_end_to_end_execute_applies_and_scores(churn_schema, stub_backend):
    """The full path: execute() with ASSUMING routes through the per-entity
    strategy, applies the assumption, and returns a prediction."""
    engine = _engine(churn_schema, stub_backend)
    result = engine.execute(ExecutionInput(
        query=("PREDICT customers.age FROM customers "
               "WHERE customers.customer_id IN :ids "
               "ASSUMING customers.age = 99 RETURN EXPECTED VALUE"),
        anchor_time=ANCHOR, params={"ids": ["C7"]}))
    assert len(result.predictions) == 1
