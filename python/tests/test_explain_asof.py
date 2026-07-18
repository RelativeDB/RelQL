"""AS OF anchor binding and EXPLAIN (PLAN/CONTEXT/ANALYZE) execution.

See scratchpad EXPLAIN_ASOF_CONTRACT.md — the authoritative spec.
"""
from __future__ import annotations

import json

import pytest

from relativedb import (Engine, EntityPrediction, ExecutionError,
                        ExecutionInput, ExplainResult)

from conftest import dt

T0 = dt("2026-07-01")

CHURN = ("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
         "FOR EACH customers.customer_id")
REG = ("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
       "FOR customers.customer_id = 'C7'")


class SpyBackend:
    """Records whether the model was ever asked to score."""

    def __init__(self):
        self.calls = 0

    def score(self, query, task_type, contexts, model_uri, config):
        self.calls += 1
        return [EntityPrediction(c.entity_id) for c in contexts]


# ---------------------------------------------------------------------------
# AS OF
# ---------------------------------------------------------------------------

def test_as_of_date_overrides_anchor_time(churn_schema, churn_wiring,
                                          stub_backend):
    """AS OF <date> binds the anchor and overrides ExecutionInput.anchor_time.
    With T0 (2026-07-01) O4 (2026-07-05) is hidden; AS OF 2026-08-01 admits it,
    proving the date wins over the passed anchor_time and threads through the
    temporal bound."""
    eng = Engine(churn_schema, churn_wiring, model_backend=stub_backend)
    later = eng.explain(ExecutionInput(
        query="EXPLAIN CONTEXT " + CHURN + " AS OF 2026-08-01", anchor_time=T0))
    base = eng.explain(ExecutionInput(
        query="EXPLAIN CONTEXT " + CHURN, anchor_time=T0))
    # the later date anchor pulls in O4 (and product P3), so strictly more rows
    assert later.context["total_rows"] > base.context["total_rows"]
    assert later.context["anchor"].startswith("2026-08-01")
    # execution still produces predictions for all entities under the new anchor
    res = eng.execute(ExecutionInput(
        query=CHURN + " AS OF 2026-08-01", anchor_time=T0))
    assert {p.id for p in res.predictions} == {"C1", "C7", "C9"}


def test_as_of_param_binds_from_params(churn_schema, churn_wiring, stub_backend):
    eng = Engine(churn_schema, churn_wiring, model_backend=stub_backend)
    res = eng.execute(ExecutionInput(
        query=CHURN + " AS OF :t", params={"t": dt("2026-08-01")}))
    assert {p.id for p in res.predictions} == {"C1", "C7", "C9"}


def test_as_of_param_missing_raises(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring)
    with pytest.raises(ExecutionError) as ei:
        eng.execute(ExecutionInput(query=CHURN + " AS OF :t"))
    assert "t" in str(ei.value)


def test_as_of_param_falls_back_to_anchor_time(churn_schema, churn_wiring,
                                               stub_backend):
    """No param binding but an anchor_time present -> fall back to it."""
    eng = Engine(churn_schema, churn_wiring, model_backend=stub_backend)
    res = eng.execute(ExecutionInput(query=CHURN + " AS OF :t", anchor_time=T0))
    base = eng.execute(ExecutionInput(query=CHURN, anchor_time=T0))
    assert ({p.id: p.probability for p in res.predictions} ==
            {p.id: p.probability for p in base.predictions})


def test_as_of_now_equals_no_as_of(churn_schema, churn_wiring, stub_backend):
    eng = Engine(churn_schema, churn_wiring, model_backend=stub_backend)
    now = eng.execute(ExecutionInput(query=CHURN + " AS OF NOW", anchor_time=T0))
    base = eng.execute(ExecutionInput(query=CHURN, anchor_time=T0))
    assert ({p.id: p.probability for p in now.predictions} ==
            {p.id: p.probability for p in base.predictions})


# ---------------------------------------------------------------------------
# EXPLAIN PLAN — no scoring
# ---------------------------------------------------------------------------

def test_explain_plan_does_not_invoke_model(churn_schema, churn_wiring):
    spy = SpyBackend()
    eng = Engine(churn_schema, churn_wiring, model_backend=spy)
    res = eng.explain(ExecutionInput(query="EXPLAIN PLAN " + CHURN,
                                     anchor_time=T0))
    assert isinstance(res, ExplainResult)
    assert res.mode == "PLAN"
    assert res.context is None
    assert res.predictions is None
    assert spy.calls == 0                     # the model was never touched


def test_explain_plan_fields(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring)
    res = eng.explain(ExecutionInput(query="EXPLAIN PLAN " + CHURN,
                                     anchor_time=T0))
    plan = res.plan
    assert plan["task_type"] == "binary_classification"
    assert plan["output"] == "probability"
    assert plan["entity"] == {"table": "customers", "pk": "customer_id",
                              "selector": "FOR EACH"}
    assert plan["as_of"]["source"] == "execution-anchor"
    # exactly one target window: (0, 90] days
    tgt = [w for w in plan["windows"] if w["role"] == "target"]
    assert len(tgt) == 1
    w = tgt[0]
    assert w["start"] == 0 and w["end"] == 90 and w["unit"] == "days"


def test_bare_explain_defaults_to_plan(churn_schema, churn_wiring):
    spy = SpyBackend()
    eng = Engine(churn_schema, churn_wiring, model_backend=spy)
    res = eng.explain(ExecutionInput(query="EXPLAIN " + CHURN, anchor_time=T0))
    assert res.mode == "PLAN"
    assert spy.calls == 0


def test_explain_on_non_explain_query_defaults_plan(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring)
    res = eng.explain(ExecutionInput(query=CHURN, anchor_time=T0))
    assert res.mode == "PLAN"
    assert res.predictions is None


def test_execute_routes_explain_query(churn_schema, churn_wiring):
    """execute() on an EXPLAIN query delegates to explain() (Python)."""
    eng = Engine(churn_schema, churn_wiring)
    res = eng.execute(ExecutionInput(query="EXPLAIN PLAN " + CHURN,
                                     anchor_time=T0))
    assert isinstance(res, ExplainResult)


# ---------------------------------------------------------------------------
# EXPLAIN CONTEXT — assemble, no scoring
# ---------------------------------------------------------------------------

def test_explain_context_populates_counts_no_predictions(churn_schema,
                                                         churn_wiring):
    spy = SpyBackend()
    eng = Engine(churn_schema, churn_wiring, model_backend=spy)
    res = eng.explain(ExecutionInput(query="EXPLAIN CONTEXT " + CHURN,
                                     anchor_time=T0))
    assert res.mode == "CONTEXT"
    assert res.predictions is None
    assert spy.calls == 0
    ctx = res.context
    assert ctx["entities_covered"] == 3
    assert ctx["total_rows"] > 0
    assert ctx["total_cells"] > 0
    assert "customers" in ctx["tables"]


# ---------------------------------------------------------------------------
# EXPLAIN ANALYZE — assemble + score
# ---------------------------------------------------------------------------

def test_explain_analyze_has_predictions(churn_schema, churn_wiring,
                                         stub_backend):
    eng = Engine(churn_schema, churn_wiring, model_backend=stub_backend)
    res = eng.explain(ExecutionInput(query="EXPLAIN ANALYZE " + CHURN,
                                     anchor_time=T0))
    assert res.mode == "ANALYZE"
    assert res.context is not None
    assert res.predictions is not None
    assert {p.id for p in res.predictions} == {"C1", "C7", "C9"}


def test_explain_ablation_warns_not_implemented(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring)
    res = eng.explain(ExecutionInput(
        query="EXPLAIN ABLATION PREDICT COUNT(orders.*) OVER (90 DAYS "
              "FOLLOWING) = 0 FOR EACH customers.customer_id "
              "ABLATE TABLE products", anchor_time=T0))
    assert res.mode == "ABLATION"
    assert any("ablation not implemented" in w for w in res.plan["warnings"])
    assert any(a["table"] == "products" for a in res.plan["ablations"])


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------

def test_render_json_parses(churn_schema, churn_wiring, stub_backend):
    eng = Engine(churn_schema, churn_wiring, model_backend=stub_backend)
    res = eng.explain(ExecutionInput(
        query="EXPLAIN ANALYZE FORMAT JSON " + CHURN, anchor_time=T0))
    obj = json.loads(res.render())
    assert obj["mode"] == "ANALYZE"
    assert obj["plan"]["task_type"] == "binary_classification"
    assert "predictions" in obj and len(obj["predictions"]) == 3


def test_render_text_contains_target_and_task(churn_schema, churn_wiring):
    eng = Engine(churn_schema, churn_wiring)
    res = eng.explain(ExecutionInput(query="EXPLAIN PLAN " + CHURN,
                                     anchor_time=T0))
    text = res.render()
    assert "binary_classification" in text
    assert "COUNT(orders.*)" in text        # the target expression
    assert "PLAN" in text
