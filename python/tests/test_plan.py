"""The planner, and the guarantee it exists to provide.

Before QueryPlan, Engine.execute derived task type, the pinned entity selector
and the AS OF provenance inline while Engine._build_plan derived them again for
EXPLAIN. Nothing tied the two together, so EXPLAIN could describe a run that
differed from the real one -- and it described only the logical query, never
which strategy, sampler or batching would actually be used.

These tests pin both halves: the derivations themselves, and the property that
explain() and execute() see the same plan.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from relativedb import (BreadthFirstTraversal, Engine, ExecutionInput,
                        TaskType, parse, validate)
from relativedb.engine import SamplerMode
from relativedb.plan import QueryPlan, build_plan, pinned_ids, pure_pin

from conftest import churn_rows, dt, in_memory_wiring

CHURN = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")
ANCHOR = dt("2026-07-01")


@pytest.fixture
def engine(churn_schema, stub_backend):
    return Engine(churn_schema, in_memory_wiring(churn_rows()),
                  model_backend=stub_backend)


def _plan_for(engine, query, **kw):
    input = ExecutionInput(query=query, anchor_time=ANCHOR, **kw)
    pq = validate(parse(query), engine.schema).query.bind_params(input.params)
    return engine._plan(pq, input, ANCHOR)


# --------------------------------------------------------------------------
# logical derivations
# --------------------------------------------------------------------------

def test_plan_reports_task_type_entity_and_output(engine):
    plan = _plan_for(engine, CHURN, params={"ids": ["C1", "C7"]})
    assert plan.task_type is TaskType.BINARY_CLASSIFICATION
    assert plan.entity_table == "customers"
    assert plan.entity_pk == "customer_id"
    assert plan.output == "probability"          # task default, no RETURN


def test_pinned_cohort_becomes_the_entity_selector(engine):
    plan = _plan_for(engine, CHURN, params={"ids": ["C1", "C7"]})
    assert plan.entity_selector == ["C1", "C7"]
    assert plan.cohort_size == 2                 # known without touching data


def test_unpinned_query_selects_all_and_leaves_cohort_unknown(engine):
    q = "PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers"
    plan = _plan_for(engine, q)
    assert plan.entity_selector == "ALL"
    # Describing a query must not enumerate a table; unknown is reported as
    # unknown rather than guessed at zero.
    assert plan.cohort_size is None
    assert plan.pipelined is None


def test_as_of_provenance_distinguishes_its_three_sources(engine):
    base = "PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers"
    assert _plan_for(engine, base).as_of["source"] == "execution-anchor"
    assert _plan_for(engine, base + " AS OF 2026-05-01"
                     ).as_of["source"] == "query-date"
    p = _plan_for(engine, base + " AS OF :t", params={"t": ANCHOR})
    assert p.as_of["source"] == "query-param"
    assert p.as_of["param"] == "t"


def test_windows_are_collected_with_their_role(engine):
    plan = _plan_for(engine, CHURN, params={"ids": ["C1"]})
    roles = {w["role"] for w in plan.windows}
    assert "target" in roles
    assert all(w["table"] == "orders" for w in plan.windows)


# --------------------------------------------------------------------------
# physical decisions -- what EXPLAIN could not previously tell you
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expected", [
    ({}, "per-entity"),
    ({"shared_context": True}, "shared-context"),
    ({"hurdle_gate": 0.5}, "hurdle"),
    # hurdle wins: _execute_hurdle runs its own sub-queries and never
    # consults shared_context.
    ({"hurdle_gate": 0.5, "shared_context": True}, "hurdle"),
])
def test_strategy_mirrors_the_dispatch_in_execute(engine, kwargs, expected):
    plan = _plan_for(engine, CHURN, params={"ids": ["C1"]}, **kwargs)
    assert plan.strategy == expected


def test_plan_names_the_sampler_and_traversal(engine, churn_schema,
                                              stub_backend):
    """The default Engine uses ReferenceTraversal, which always samples from
    the CSC snapshot regardless of sampler_mode; an explicit BFS traversal in
    RETRIEVER mode is the pull-per-hop path."""
    default = _plan_for(engine, CHURN, params={"ids": ["C1"]})
    assert default.sampler == "csc"
    assert default.traversal == "ReferenceTraversal"

    bfs = Engine(churn_schema, in_memory_wiring(churn_rows()),
                 model_backend=stub_backend,
                 traversal=BreadthFirstTraversal())
    plan = _plan_for(bfs, CHURN, params={"ids": ["C1"]})
    assert plan.sampler == "retriever"
    assert plan.traversal == "BreadthFirstTraversal"


def test_pipelining_needs_a_cohort_larger_than_the_scoring_batch(engine):
    """pipelined mirrors execute()'s guard exactly: a batch size, a cohort
    that exceeds it, and a task whose scoring is a plain batched forward."""
    pq = validate(parse(CHURN), engine.schema).query.bind_params(
        {"ids": ["C1", "C7", "C9"]})
    input = ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                           params={"ids": ["C1", "C7", "C9"]})

    def plan_with(batch, cohort):
        return build_plan(engine.schema, pq, input, effective=ANCHOR,
                          traversal=engine.traversal, sampler="retriever",
                          cohort_size=cohort, scoring_batch_size=batch)

    assert plan_with(2, 5).pipelined is True
    assert plan_with(8, 5).pipelined is False    # cohort fits in one batch
    assert plan_with(0, 5).pipelined is False    # backend has no batching
    assert plan_with(None, 5).pipelined is None  # unknown, not False


# --------------------------------------------------------------------------
# the guarantee: EXPLAIN describes the run that would happen
# --------------------------------------------------------------------------

def test_explain_surfaces_the_execution_section(engine):
    result = engine.explain(ExecutionInput(
        query="EXPLAIN " + CHURN, anchor_time=ANCHOR,
        params={"ids": ["C1", "C7"]}))
    ex = result.plan["execution"]
    assert ex["strategy"] == "per-entity"
    assert ex["sampler"] == "csc"
    assert ex["traversal"] == "ReferenceTraversal"
    assert ex["cohort_size"] == 2


def test_explain_and_execute_agree_on_the_plan(engine, monkeypatch):
    """The whole point of the planner. Capture the plan execute() builds and
    compare it against the one explain() reports for the same input."""
    seen: list[QueryPlan] = []
    real = Engine._plan

    def capture(self, pq, input, effective, **kw):
        plan = real(self, pq, input, effective, **kw)
        seen.append(plan)
        return plan

    monkeypatch.setattr(Engine, "_plan", capture)

    params = {"ids": ["C1", "C7", "C9"]}
    engine.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                  params=params))
    executed = seen[-1]
    seen.clear()
    engine.explain(ExecutionInput(query="EXPLAIN " + CHURN, anchor_time=ANCHOR,
                                  params=params))
    explained = seen[-1]

    for field in ("task_type", "entity_table", "entity_pk", "entity_selector",
                  "output", "strategy", "sampler", "traversal", "model_uri",
                  "cohort_size", "target", "as_of"):
        assert getattr(executed, field) == getattr(explained, field), field


def test_explain_plan_keys_are_unchanged_for_existing_consumers(engine):
    """The execution section is additive; the logical keys and shapes are the
    ones ExplainResult._render_text and external consumers already read."""
    plan = engine.explain(ExecutionInput(
        query="EXPLAIN " + CHURN, anchor_time=ANCHOR,
        params={"ids": ["C1"]})).plan
    for key in ("target", "task_type", "entity", "output", "windows",
                "where_present", "assuming_present", "assuming", "as_of",
                "ablations", "warnings"):
        assert key in plan, key
    assert set(plan["entity"]) == {"table", "pk", "selector"}
    assert isinstance(plan["task_type"], str)   # serialized, not the enum


# --------------------------------------------------------------------------
# WHERE-clause analysis
# --------------------------------------------------------------------------

def test_pinned_ids_intersects_anded_pins(churn_schema):
    q = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers "
         "WHERE customers.customer_id IN ('C1','C7') "
         "AND customers.customer_id IN ('C7','C9')")
    pq = validate(parse(q), churn_schema).query
    assert pinned_ids(pq.where, pq.entity_key) == ["C7"]
    assert pure_pin(pq.where, pq.entity_key)


def test_or_does_not_pin_a_cohort(churn_schema):
    q = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers "
         "WHERE customers.customer_id = 'C1' OR customers.age > 30")
    pq = validate(parse(q), churn_schema).query
    assert pinned_ids(pq.where, pq.entity_key) is None
    assert not pure_pin(pq.where, pq.entity_key)
