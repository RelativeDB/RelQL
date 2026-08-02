"""ABLATE TABLE applies, and EXPLAIN ABLATE ranks what matters.

``ABLATE TABLE t`` used to parse, validate, and then silently change nothing —
the plan even printed "declared, not applied". It now drops every row of ``t``
from each scored context. ``EXPLAIN ABLATE`` (alias: ``EXPLAIN ABLATION``)
turns that mechanism into a report: score the query as written, re-score once
per candidate ablation — each non-entity table in the contexts and each
declared non-key column carrying values — and rank candidates by how far
dropping them moves the predictions. Near-zero movement marks a good
ablation; large movement marks a load-bearing input.

The backend here scores a context by its total cell count, so every drop has
an exactly predictable effect on the prediction.
"""
from __future__ import annotations

import pytest

from relativedb import ReferenceTraversal, Engine, ExecutionInput
from relativedb.engine import EntityPrediction, ExecutionError
from relativedb.relql.parser import parse

from conftest import churn_rows, dt, in_memory_wiring

ANCHOR = dt("2026-07-01")
# A dated entity: the reference walk expands children only under a dated
# seed, so orders (not customers) is the entity table that pulls its
# customer and product parents into context.
Q = ("PREDICT orders.qty FROM orders WHERE orders.order_id IN :ids "
     "RETURN EXPECTED VALUE")
IDS = {"ids": ["O2"]}


class _CellCountBackend:
    """value = total cells in the context: every dropped row or column moves
    the prediction by exactly its cell count."""

    def score(self, query, task_type, contexts, model_uri, config):
        return [EntityPrediction(c.entity_id, probability=None,
                                 value=float(sum(len(r.cells)
                                                 for r in c.rows)))
                for c in contexts]


def _engine(churn_schema):
    # These assertions count cells in the assembled context, so the traversal
    # that assembles it is part of the fixture rather than an incidental
    # default. Pinned here now that the engine defaults to pull-per-hop.
    return Engine(churn_schema, in_memory_wiring(churn_rows()),
                  model_backend=_CellCountBackend(),
                  traversal=ReferenceTraversal())


def _value(engine, query):
    res = engine.execute(ExecutionInput(query=query, anchor_time=ANCHOR,
                                        params=IDS))
    return res.predictions[0].value


# ---------------------------------------------------------------------------
# grammar: EXPLAIN ABLATE is EXPLAIN ABLATION
# ---------------------------------------------------------------------------

def test_explain_ablate_spelling_is_an_alias():
    for kw in ("ABLATE", "ABLATION"):
        pq = parse(f"EXPLAIN {kw} " + Q)
        assert pq.explain is not None and pq.explain.mode == "ABLATION"


# ---------------------------------------------------------------------------
# ABLATE TABLE actually ablates
# ---------------------------------------------------------------------------

def test_ablate_table_changes_what_gets_scored(churn_schema):
    engine = _engine(churn_schema)
    full = _value(engine, Q)
    without_products = _value(engine, Q + " ABLATE TABLE products")
    assert without_products < full
    # exactly the product rows' cells are gone, nothing else: P1 and P2
    # (price + name each) reach O2's context through its sibling orders
    assert full - without_products == 4.0


def test_ablate_unknown_table_is_an_error(churn_schema):
    engine = _engine(churn_schema)
    with pytest.raises(ExecutionError, match="unknown table"):
        _value(engine, Q + " ABLATE TABLE typo_table")


def test_ablate_entity_table_is_an_error(churn_schema):
    engine = _engine(churn_schema)
    with pytest.raises(ExecutionError, match="cannot ablate the entity"):
        _value(engine, Q + " ABLATE TABLE orders")


# ---------------------------------------------------------------------------
# EXPLAIN ABLATE: the impact ranking
# ---------------------------------------------------------------------------

def _report(churn_schema, query=Q):
    engine = _engine(churn_schema)
    res = engine.execute(ExecutionInput(query="EXPLAIN ABLATE " + query,
                                        anchor_time=ANCHOR, params=IDS))
    assert res.mode == "ABLATION"
    return res


def test_candidates_cover_tables_and_columns(churn_schema):
    rep = _report(churn_schema).ablation
    named = {(e["kind"], e["name"]) for e in rep["candidates"]}
    assert ("table", "customers") in named
    assert ("table", "products") in named
    assert ("column", "customers.age") in named
    # the target column IS a candidate: the entity's own cell is masked, so
    # its ablation measures reliance on sibling rows' past outcomes
    assert ("column", "orders.qty") in named
    # the entity table and time columns are not candidates
    assert ("table", "orders") not in named
    assert ("column", "orders.order_date") not in named


def test_deltas_are_exact_and_ranked(churn_schema):
    rep = _report(churn_schema).ablation
    assert rep["metric"] == "expected_value"
    by_name = {(e["kind"], e["name"]): e for e in rep["candidates"]}
    # dropping a table removes exactly its cells from the cell-count score
    assert by_name[("table", "products")]["mean_abs_delta"] == 4.0
    assert by_name[("table", "products")]["mean_delta"] == -4.0
    assert by_name[("column", "customers.age")]["mean_abs_delta"] == 1.0
    ranks = [e["mean_abs_delta"] for e in rep["candidates"]]
    assert ranks == sorted(ranks, reverse=True)
    assert rep["model_forwards"] == 1 + len(rep["candidates"])


def test_declared_ablation_shapes_the_baseline(churn_schema):
    rep = _report(churn_schema, Q + " ABLATE TABLE products")
    full = _report(churn_schema)
    # the declared ablation is applied to the baseline and not re-explored
    assert (rep.ablation["baseline"]["mean"]
            == full.ablation["baseline"]["mean"] - 4.0)
    assert not any(e["name"] == "products"
                   for e in rep.ablation["candidates"])


def test_render_includes_the_ranking(churn_schema):
    text = _report(churn_schema).render()
    assert "ABLATION" in text
    assert "mean_abs_delta" in text
