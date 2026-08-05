"""Execution-strategy selection: the paths the planner now chooses between.

These were the least-covered part of the engine -- the pipelined producer/
consumer path, the hurdle composition and the shared-context guards are all
selected by flags on ExecutionInput and were never exercised by a test.
"""
from __future__ import annotations

import threading
import time

import pytest

from relativedb import (BreadthFirstTraversal, ReferenceTraversal, Engine,
                        ExecutionInput, TaskType)
from relativedb.engine import EntityPrediction, ExecutionError

from conftest import churn_rows, dt, in_memory_wiring

CHURN = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")
COUNT = ("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) "
         "FROM customers WHERE customers.customer_id IN :ids")
ANCHOR = dt("2026-07-01")
IDS = ["C1", "C7", "C9"]


class BatchedStub:
    """A stub that declares a scoring batch size, which is what turns on the
    pipelined assembly path. Records the batches it was handed so a test can
    tell chunked scoring from one big call."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.batches: list[int] = []

    def task_spec(self, query, task_type):
        # Real backends provide this, and shared-context chunk sizing needs it.
        # Without it the engine warns and guesses a split -- correct behaviour,
        # but it means the test was not exercising the sizing path at all.
        from relativedb.task import TaskSpec
        return TaskSpec.from_query(query, task_type)

    def score(self, query, task_type, contexts, model_uri, config):
        self.batches.append(len(contexts))
        binary = task_type is TaskType.BINARY_CLASSIFICATION
        out = []
        for i, c in enumerate(contexts):
            # vary by entity so a reordering bug cannot pass unnoticed
            v = (hash(c.entity_id) % 97) / 97.0
            out.append(EntityPrediction(c.entity_id,
                                        probability=v if binary else None,
                                        value=None if binary else float(v)))
        return out


def _engine(churn_schema, backend):
    # The shared-context path is defined over the reference graph, so these
    # engines ask for it rather than relying on it being the default.
    return Engine(churn_schema, in_memory_wiring(churn_rows()),
                  model_backend=backend, traversal=ReferenceTraversal())


# --------------------------------------------------------------------------
# pipelined assembly
# --------------------------------------------------------------------------

def test_pipelined_and_serial_scoring_agree(churn_schema):
    """The pipeline overlaps context assembly with the model forward on a
    producer thread. It is an optimization, so it must produce byte-identical
    predictions to the serial path -- including their order."""
    piped = _engine(churn_schema, BatchedStub(batch_size=2))     # 3 ids > 2
    serial = _engine(churn_schema, BatchedStub(batch_size=99))   # 3 ids < 99

    args = dict(query=CHURN, anchor_time=ANCHOR, params={"ids": IDS})
    a = piped.execute(ExecutionInput(**args))
    b = serial.execute(ExecutionInput(**args))

    assert [p.id for p in a.predictions] == [p.id for p in b.predictions]
    assert [p.probability for p in a.predictions] == \
           [p.probability for p in b.predictions]
    # and the pipeline really did chunk rather than falling through
    assert piped.model_backend.batches == [2, 1]
    assert serial.model_backend.batches == [3]


def test_pipeline_engages_only_when_the_cohort_exceeds_the_batch(churn_schema):
    eng = _engine(churn_schema, BatchedStub(batch_size=3))
    eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                               params={"ids": IDS}))
    assert eng.model_backend.batches == [3]      # one call, not pipelined


def test_producer_thread_errors_surface_to_the_caller(churn_schema):
    """A failure inside the producer must propagate, not hang the consumer
    waiting on a queue that will never fill."""
    eng = _engine(churn_schema, BatchedStub(batch_size=2))

    def boom(*a, **k):
        raise RuntimeError("assembly exploded")

    eng.assemble_context = boom
    with pytest.raises(RuntimeError, match="assembly exploded"):
        eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                   params={"ids": IDS}))


def test_breadth_first_context_builds_use_configured_workers(
        churn_schema, monkeypatch):
    eng = Engine(churn_schema, in_memory_wiring(churn_rows()),
                 model_backend=BatchedStub(batch_size=99),
                 traversal=BreadthFirstTraversal())
    lock = threading.Lock()
    active = 0
    max_active = 0

    def build(eid):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.01)
            return eid
        finally:
            with lock:
                active -= 1

    monkeypatch.setenv("RELATIVEDB_CONTEXT_WORKERS", "3")
    result = eng._map_context_builds(["a", "b", "c"], build)

    assert result == ["a", "b", "c"]
    assert max_active == 3


def test_reference_context_builds_remain_serial(churn_schema, monkeypatch):
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    owner = threading.get_ident()
    seen = []
    monkeypatch.setenv("RELATIVEDB_CONTEXT_WORKERS", "3")

    assert eng._map_context_builds(IDS, lambda eid: seen.append(
        threading.get_ident()) or eid) == IDS
    assert seen == [owner] * len(IDS)


# --------------------------------------------------------------------------
# hurdle
# --------------------------------------------------------------------------

def test_hurdle_rejects_non_regression_targets(churn_schema):
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    with pytest.raises(ExecutionError, match="regression targets only"):
        eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                   params={"ids": IDS}, hurdle_gate=0.5))


def test_hurdle_zeroes_predictions_below_the_gate(churn_schema):
    """The hurdle runs the regression query and a derived existence query on
    the same cohort, and forces exactly 0 wherever existence falls short."""
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    res = eng.execute(ExecutionInput(query=COUNT, anchor_time=ANCHOR,
                                     params={"ids": IDS}, hurdle_gate=1.1))
    # a gate above every possible probability zeroes the whole cohort
    assert res.task_type is TaskType.REGRESSION
    assert [p.value for p in res.predictions] == [0.0] * len(IDS)


def test_hurdle_gate_of_zero_keeps_every_regression_value(churn_schema):
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    gated = eng.execute(ExecutionInput(query=COUNT, anchor_time=ANCHOR,
                                       params={"ids": IDS}, hurdle_gate=0.0))
    plain = eng.execute(ExecutionInput(query=COUNT, anchor_time=ANCHOR,
                                       params={"ids": IDS}))
    assert ([p.value for p in gated.predictions]
            == [p.value for p in plain.predictions])


# --------------------------------------------------------------------------
# shared-context scoring through the Scorer protocol
#
# These run the REAL SequenceBackend over a stub Scorer, because the engine's
# shared path reaches into the payload prepare_shared returns
# (payload["batch"] / payload["model_uri"] -> scorer.forward). A stub backend
# with its own prepare_shared would never catch the engine and the backend
# drifting apart -- which happened: the engine kept reading the pre-split
# payload["model"].forward_tokens(...) and only a live benchmark noticed.
# --------------------------------------------------------------------------

class ZeroScorer:
    """Scorer whose every token score is 0 (so classification decodes to
    probability 0.5) and whose embeddings are all-zero vectors."""

    def __init__(self):
        self.outputs_seen: list[str] = []

    def forward(self, batch, *, model_uri, output):
        import numpy as np
        from relativedb.scoring import ForwardResult
        self.outputs_seen.append(output)
        if output == "token_scores":
            return ForwardResult(
                scores=np.zeros((batch.b, batch.s), np.float32))
        return ForwardResult(scores=np.zeros(batch.b, np.float32))

    def embed(self, texts, *, normalize=True):
        import numpy as np
        return np.zeros((len(texts), 384), np.float32)


def _shared_engine(churn_schema):
    from relativedb.scoring import SequenceBackend
    scorer = ZeroScorer()
    backend = SequenceBackend(scorer, schema=churn_schema)
    return _engine(churn_schema, backend), scorer


def test_shared_context_scores_through_the_scorer_payload(churn_schema):
    eng, scorer = _shared_engine(churn_schema)
    res = eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                     params={"ids": IDS},
                                     shared_context=True))
    # shared predictions come back in cohort/context order, not input order
    assert sorted(p.id for p in res.predictions) == sorted(IDS)
    # the shared path really ran (fallback would score per-entity instead)
    assert "token_scores" in scorer.outputs_seen
    for p in res.predictions:
        assert p.probability == pytest.approx(0.5)


@pytest.mark.parametrize("concurrent", [False, True])
def test_execute_many_pipelines_shared_cohorts_through_the_scorer(
        churn_schema, concurrent):
    """execute_many's producer/consumer path has its own copy of the forward
    call; it must consume the same payload contract as execute(). The
    concurrent variant marks the scorer remote-like (a ``url`` attribute),
    which turns on the in-flight forward executor."""
    eng, scorer = _shared_engine(churn_schema)
    if concurrent:
        scorer.url = "stub://remote"
    inputs = [ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                             params={"ids": IDS}, shared_context=True),
              ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                             params={"ids": IDS[:2]}, shared_context=True)]
    results = eng.execute_many(inputs)
    assert sorted(p.id for p in results[0].predictions) == sorted(IDS)
    assert sorted(p.id for p in results[1].predictions) == sorted(IDS[:2])
    assert scorer.outputs_seen.count("token_scores") >= 2
    for res in results:
        for p in res.predictions:
            assert p.probability == pytest.approx(0.5)


# --------------------------------------------------------------------------
# shared-context guards
#
# The shared path needs a backend that can prepare shared state. These pin the
# refusals, which are the part that protects correctness: scoring a cohort in
# one sequence is only sound for scalar tasks with a shared anchor and a
# key-pinned WHERE.
# --------------------------------------------------------------------------

def test_shared_context_refuses_without_a_capable_backend(churn_schema):
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    with pytest.raises(ExecutionError, match="needs a model backend"):
        eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                   params={"ids": IDS}, shared_context=True))


def test_shared_context_refuses_a_non_key_where(churn_schema):
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    q = ("PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers "
         "WHERE customers.age > 30")
    with pytest.raises(ExecutionError, match="primary-key pin"):
        eng.execute(ExecutionInput(query=q, anchor_time=ANCHOR,
                                   shared_context=True))


def test_shared_context_refuses_assuming(churn_schema):
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    q = CHURN + " ASSUMING customers.age = 40"
    with pytest.raises(ExecutionError, match="does not support ASSUMING"):
        eng.execute(ExecutionInput(query=q, anchor_time=ANCHOR,
                                   params={"ids": IDS}, shared_context=True))


# --------------------------------------------------------------------------
# backend requirement
# --------------------------------------------------------------------------

def test_scoring_without_a_backend_says_so(churn_schema):
    eng = Engine(churn_schema, in_memory_wiring(churn_rows()))
    with pytest.raises(ExecutionError, match="requires a model backend"):
        eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                   params={"ids": IDS}))


def test_explain_plan_needs_no_backend(churn_schema):
    """PLAN/CONTEXT never score, so they must work on a backend-less engine --
    that is what makes EXPLAIN usable for inspecting a wiring."""
    eng = Engine(churn_schema, in_memory_wiring(churn_rows()))
    out = eng.explain(ExecutionInput(query="EXPLAIN " + CHURN,
                                     anchor_time=ANCHOR,
                                     params={"ids": IDS}))
    assert out.plan["execution"]["strategy"] == "per-entity"
    assert out.predictions is None


# --------------------------------------------------------------------------
# strategy dispatch
#
# execute() no longer branches on ExecutionInput flags; it hands the plan to
# relativedb.strategies, which selects from a registry. These pin that the
# selection really is plan-driven.
# --------------------------------------------------------------------------

def test_dispatch_selects_the_strategy_the_plan_names(churn_schema,
                                                      monkeypatch):
    from relativedb import strategies

    called: list[str] = []

    def spy(name, real):
        def wrapper(engine, req):
            called.append(name)
            return real(engine, req)
        return wrapper

    monkeypatch.setitem(strategies._REGISTRY, "per-entity",
                        spy("per-entity", strategies.run_per_entity))
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                               params={"ids": IDS}))
    assert called == ["per-entity"]


def test_shared_context_declining_falls_back_to_per_entity(churn_schema,
                                                           monkeypatch):
    """A strategy that returns None means "not applicable", and the dispatcher
    must fall through rather than returning nothing."""
    from relativedb import strategies

    monkeypatch.setitem(strategies._REGISTRY, "shared-context",
                        lambda engine, req: None)
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    res = eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                     params={"ids": IDS},
                                     shared_context=True))
    assert [p.id for p in res.predictions] == IDS


def test_a_strategy_that_must_not_decline_is_an_error(churn_schema,
                                                      monkeypatch):
    """per-entity always produces a result. If it ever returns None that is a
    bug in the strategy, not a signal to fall back -- silently retrying would
    loop or mask it."""
    from relativedb import strategies

    monkeypatch.setitem(strategies._REGISTRY, "per-entity",
                        lambda engine, req: None)
    eng = _engine(churn_schema, BatchedStub(batch_size=99))
    with pytest.raises(RuntimeError, match="returned no result"):
        eng.execute(ExecutionInput(query=CHURN, anchor_time=ANCHOR,
                                   params={"ids": IDS}))


def test_every_plan_strategy_has_a_registered_handler():
    """The planner and the registry must not drift apart: a plan naming a
    strategy nothing implements would fail only at execution time."""
    from relativedb import strategies
    from relativedb.plan import _strategy_for

    class _Input:
        def __init__(self, hurdle_gate=None, shared_context=False):
            self.hurdle_gate = hurdle_gate
            self.shared_context = shared_context

    reachable = {
        _strategy_for(_Input()),
        _strategy_for(_Input(shared_context=True)),
        _strategy_for(_Input(hurdle_gate=0.5)),
    }
    assert reachable <= set(strategies._REGISTRY)
