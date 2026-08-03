"""Execution strategies, selected by the plan.

``Engine.execute`` used to decide how to run a query with an if-chain over
``ExecutionInput`` flags and then inline all three implementations, so the
method carried the hurdle composition, the shared-context handoff and both
halves of per-entity scoring at once. The planner already names the strategy
(``QueryPlan.strategy``); this module is the registry it selects from, so
adding one is a new entry rather than another branch in the middle of the
execution path.

The order contexts are assembled and handed to the backend is load-bearing --
it is what the sampler's RNG determines and what the model consumes -- so these
bodies were moved verbatim. ``python/tests/test_sampling_regression.py`` pins
that order across eleven engine configurations and is the guard on this file.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # engine imports this module; annotations only
    from .engine import Engine, EntityContext, PredictionResult

__all__ = ["ExecutionRequest", "run"]


class ExecutionRequest:
    """Everything a strategy needs, resolved once by ``Engine.execute``."""

    __slots__ = ("pq", "input", "plan", "entity_table", "ids", "assumed",
                 "backend")

    def __init__(self, pq, input, plan, entity_table, ids, assumed, backend):
        self.pq = pq
        self.input = input
        self.plan = plan
        self.entity_table = entity_table
        self.ids = ids
        self.assumed = assumed
        self.backend = backend


def _context_builder(engine: "Engine", req: ExecutionRequest):
    """Assemble one entity's context, or None when WHERE excludes it."""
    from .engine import (_apply_ablations, _apply_aggregate_assumptions,
                         _apply_assumptions)
    from .plan import aggregate_assumptions

    agg_assumed = aggregate_assumptions(req.pq.assuming)
    # WHERE stays at the query's factual anchor even when
    # context_anchor_time decouples the context "now".
    where_anchor = (req.input.anchor_time
                    if req.input.context_anchor_time is not None else None)

    def build(eid) -> Optional["EntityContext"]:
        anchor = engine._anchor_for(req.entity_table, eid, req.input)
        ctx = engine.assemble_context(req.entity_table, eid, anchor,
                                      query=req.pq)
        # WHERE selects who to score and is factual; the counterfactual is
        # applied afterwards, to the context that actually gets scored.
        if (req.pq.where is not None
                and not engine._where_ok(req.pq, ctx, req.entity_table,
                                         where_anchor)):
            return None
        # Ablate first, then assume: assuming on an ablated table should
        # warn as inert, not resurrect the rows.
        ctx = _apply_ablations(req.pq.ablations, ctx)
        ctx = _apply_aggregate_assumptions(
            agg_assumed, ctx, req.entity_table, engine.schema,
            engine._assumption_template_fetch(ctx, req.entity_table))
        return _apply_assumptions(req.assumed, ctx, req.entity_table)

    return build


# ---------------------------------------------------------------------------
# per-entity
# ---------------------------------------------------------------------------

def _score_serial(engine: "Engine", req: ExecutionRequest, build):
    contexts = [ctx for ctx in engine._map_context_builds(req.ids, build)
                if ctx is not None]
    preds = req.backend.score(req.pq, req.plan.task_type, contexts,
                              req.plan.model_uri, engine.model_config)
    return contexts, list(preds)


def _score_pipelined(engine: "Engine", req: ExecutionRequest, build,
                     chunk_size: int):
    """Overlap the next chunk's context assembly (pure Python) with the current
    chunk's model forward (native, releases the GIL).

    Scoring normalizes each sequence independently, so chunking a cohort gives
    identical results to one call -- this is an optimization and must stay
    invisible in the output, which the regression harness asserts by requiring
    the pipelined and serial fingerprints to be equal.
    """
    import queue as _queue
    import threading

    contexts: list = []
    preds: list = []
    feed: _queue.Queue = _queue.Queue(maxsize=2)
    stop = threading.Event()

    def produce():
        try:
            buf: list = []
            for eid in req.ids:
                if stop.is_set():
                    return
                ctx = build(eid)
                if ctx is None:
                    continue
                buf.append(ctx)
                if len(buf) == chunk_size:
                    feed.put(("chunk", buf))
                    buf = []
            if buf:
                feed.put(("chunk", buf))
            feed.put(("done", None))
        except BaseException as error:   # surfaced on the main thread
            feed.put(("error", error))

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        while True:
            kind, payload = feed.get()
            if kind == "error":
                raise payload
            if kind == "done":
                break
            contexts.extend(payload)
            preds.extend(req.backend.score(req.pq, req.plan.task_type, payload,
                                           req.plan.model_uri,
                                           engine.model_config))
    finally:
        stop.set()
        while producer.is_alive():
            try:
                feed.get_nowait()
            except _queue.Empty:
                producer.join(timeout=0.05)
    return contexts, preds


def run_per_entity(engine: "Engine", req: ExecutionRequest):
    from .engine import (PredictionResult, _warn_context_health,
                         _warn_inert_assumptions)

    build = _context_builder(engine, req)
    chunk_size = req.plan.scoring_batch_size or 0
    if req.plan.pipelined:
        contexts, preds = _score_pipelined(engine, req, build, chunk_size)
    else:
        contexts, preds = _score_serial(engine, req, build)
    _warn_inert_assumptions(req.assumed, contexts,
                            req.entity_table, engine.schema)
    _warn_context_health(contexts, engine.schema, engine.context_policy)
    return PredictionResult(
        task_type=req.plan.task_type, predictions=tuple(preds),
        model_uri=req.plan.model_uri,
        stats=engine._collect_stats(req.pq, req.plan.task_type, contexts))


# ---------------------------------------------------------------------------
# hurdle and shared context
# ---------------------------------------------------------------------------

def run_hurdle(engine: "Engine", req: ExecutionRequest):
    return engine._execute_hurdle(req.pq, req.input, req.plan.task_type,
                                  req.plan.model_uri, req.entity_table,
                                  req.ids)


def run_shared(engine: "Engine", req: ExecutionRequest):
    """Returns None when no shared state matched -- e.g. a traversal that
    cannot produce cohort targets -- and the caller falls back to per-entity
    scoring."""
    return engine._execute_shared(req.pq, req.input, req.plan.task_type,
                                  req.plan.model_uri, req.entity_table,
                                  req.ids, req.backend)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

# Strategies that may decline the query by returning None; the dispatcher then
# falls through to per-entity, which always produces a result.
_MAY_DECLINE = {"shared-context"}

_REGISTRY = {
    "hurdle": run_hurdle,
    "shared-context": run_shared,
    "per-entity": run_per_entity,
}


def run(engine: "Engine", req: ExecutionRequest) -> "PredictionResult":
    """Execute a query with the strategy its plan selected."""
    strategy = req.plan.strategy
    try:
        handler = _REGISTRY[strategy]
    except KeyError:  # pragma: no cover - a plan can only name a known one
        raise RuntimeError(
            f"no execution strategy registered for {strategy!r}") from None
    result = handler(engine, req)
    if result is None:
        if strategy not in _MAY_DECLINE:
            raise RuntimeError(
                f"execution strategy {strategy!r} returned no result")
        # Declining is legitimate, but it is a PROTOCOL CHANGE, not a detail:
        # a cohort scored per-entity is not the cohort scored in one shared
        # context, and the two give different predictions. Say so.
        from .engine import _fallback
        _fallback(f"{strategy} declined this query; scoring per-entity "
                  f"instead, which produces different predictions than the "
                  f"shared context that was requested")
        result = run_per_entity(engine, req)
    return result
