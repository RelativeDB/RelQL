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


# ---------------------------------------------------------------------------
# aggregate counterfactuals: ASSUMING COUNT(...) >= k / EXISTS / NOT EXISTS
# ---------------------------------------------------------------------------
# The bound is realized structurally: the entity's own in-window rows are
# cloned (newest first, re-timestamped inside the window) or dropped (oldest
# first) until the count holds. Clones inherit real cells — an empty synthetic
# row would be invisible to the model (see InvisibleTableWarning).

from relativedb.engine import _apply_aggregate_assumptions
from relativedb.errors import ExecutionError
from relativedb.plan import aggregate_assumptions


def _agg_conds(engine, assuming_clause, entity_id="C9"):
    query = ("PREDICT customers.age FROM customers "
             "WHERE customers.customer_id IN :ids "
             f"ASSUMING {assuming_clause} RETURN EXPECTED VALUE")
    pq = validate(parse(query), engine.schema,
                  {"ids": [entity_id]}).query.bind_params({"ids": [entity_id]})
    return aggregate_assumptions(pq.assuming)


def _entity_orders(ctx, entity_id):
    return [r for r in ctx.rows if r.table == "orders"
            and r.parents.get("customer_id") == entity_id]


def test_assume_count_clones_rows_for_an_inactive_entity(
        churn_schema, stub_backend):
    """C9 never ordered; ASSUMING COUNT >= 3 must build the history."""
    engine = _engine(churn_schema, stub_backend)
    ctx = engine.assemble_context("customers", "C9", ANCHOR)
    assert not _entity_orders(ctx, "C9")
    conds = _agg_conds(engine,
                       "COUNT(orders.*) OVER (180 DAYS PRECEDING) >= 3")
    out = _apply_aggregate_assumptions(conds, ctx, "customers", engine.schema)
    mine = _entity_orders(out, "C9")
    assert len(mine) == 3
    for r in mine:
        assert r.cells.get("qty") is not None        # cloned, not empty
        assert r.timestamp is not None and r.timestamp <= ANCHOR
        assert r.timestamp > ANCHOR - __import__("datetime").timedelta(days=180)
        assert r.key in out.focal_row_keys           # counts as the entity's


def test_assume_count_already_satisfied_is_a_no_op(churn_schema, stub_backend):
    engine = _engine(churn_schema, stub_backend)
    ctx = engine.assemble_context("customers", "C7", ANCHOR)
    conds = _agg_conds(engine,
                       "COUNT(orders.*) OVER (365 DAYS PRECEDING) >= 1", "C7")
    out = _apply_aggregate_assumptions(conds, ctx, "customers", engine.schema)
    assert [r.key for r in out.rows] == [r.key for r in ctx.rows]


def test_assume_exact_count_drops_oldest_first(churn_schema, stub_backend):
    """C7 has O1 (March) and O2 (May); COUNT = 1 keeps the newest."""
    engine = _engine(churn_schema, stub_backend)
    ctx = engine.assemble_context("customers", "C7", ANCHOR)
    conds = _agg_conds(engine,
                       "COUNT(orders.*) OVER (365 DAYS PRECEDING) = 1", "C7")
    out = _apply_aggregate_assumptions(conds, ctx, "customers", engine.schema)
    kept = _entity_orders(out, "C7")
    assert [r.id for r in kept] == ["O2"]


def test_assume_not_exists_clears_the_entitys_window(
        churn_schema, stub_backend):
    engine = _engine(churn_schema, stub_backend)
    ctx = engine.assemble_context("customers", "C7", ANCHOR)
    conds = _agg_conds(engine,
                       "NOT EXISTS(orders.*) OVER (365 DAYS PRECEDING)", "C7")
    out = _apply_aggregate_assumptions(conds, ctx, "customers", engine.schema)
    assert not _entity_orders(out, "C7")
    # peers' orders are context evidence, not the entity's history: kept
    assert any(r.table == "orders" for r in out.rows)


def test_assume_count_end_to_end(churn_schema, stub_backend):
    engine = _engine(churn_schema, stub_backend)
    result = engine.execute(ExecutionInput(
        query=("PREDICT customers.age FROM customers "
               "WHERE customers.customer_id IN :ids "
               "ASSUMING COUNT(orders.*) OVER (180 DAYS PRECEDING) >= 3 "
               "RETURN EXPECTED VALUE"),
        anchor_time=ANCHOR, params={"ids": ["C9"]}))
    assert len(result.predictions) == 1


def test_unsupported_aggregate_assumptions_fail_loudly(
        churn_schema, stub_backend):
    engine = _engine(churn_schema, stub_backend)
    ctx = engine.assemble_context("customers", "C7", ANCHOR)

    # only COUNT/EXISTS bounds are buildable; other aggregates never were
    with pytest.raises(ExecutionError, match="possible worlds"):
        _agg_conds(engine, "SUM(orders.qty) OVER (90 DAYS PRECEDING) >= 10",
                   "C7")
    # a filtered COUNT extracts but cannot be synthesized: loud, with the fix
    conds = _agg_conds(engine, "COUNT(orders.* WHERE orders.qty = 1) "
                               "OVER (90 DAYS PRECEDING) >= 2", "C7")
    with pytest.raises(ExecutionError, match="filtered"):
        _apply_aggregate_assumptions(conds, ctx, "customers", engine.schema)
    # inequality on a plain column is still not a buildable world
    with pytest.raises(ExecutionError, match="possible worlds"):
        _agg_conds(engine, "customers.age >= 30", "C7")
