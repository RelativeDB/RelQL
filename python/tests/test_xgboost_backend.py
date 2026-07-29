"""The XGBoost flat-feature backend: eligibility routing, feature extraction
through the C ABI, and end-to-end fit/score on the churn toy graph.

The numeric semantics of every feature column are pinned in C++
(cpp/src/test_flat.cpp); these tests cover the Python wiring — context
encoding (focal rows, epoch datetimes, temporal admissibility), the fit/score
round trip, and persistence. Tests that train a real model skip without the
xgboost package; everything else needs only librt_c, like the rest of the
unit tier.

The model tests run their scenario in a SPAWNED subprocess (this file's
``__main__`` block). Not for style points: on macOS, constructing XGBoost
boosters in a process that has also exec'd the reference eval adapter
(test_rt_native.py loads evaluation/run_native_on_reference.py) segfaults in
DMatrix construction — the same OpenMP-runtime clash that forced
evaluation/run_xgboost_reference.py to isolate its predictor in a worker
process. In-process the scenarios pass alone and die under the full suite.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest

from conftest import churn_rows, dt
from relativedb import ContextPolicy, Engine, ExecutionError, TaskType
from relativedb.xgb import XgboostBackend, analyze_flat

REG_QUERY = "PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) FROM customers"
BIN_QUERY = ("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
             "FROM customers RETURN PROBABILITY")


@pytest.fixture
def engine(churn_schema, churn_wiring):
    return Engine(churn_schema, churn_wiring,
                  context_policy=ContextPolicy(max_context_cells=512))


# ---------------------------------------------------------------------------
# eligibility: the C++ planner's answer surfaces unchanged
# ---------------------------------------------------------------------------

def test_analyze_flat_eligible(churn_schema):
    a = analyze_flat(REG_QUERY, churn_schema)
    assert a.eligible
    assert a.task_type == "regression"
    assert a.entity_table == "customers"
    names = set(a.feature_names)
    assert "entity.age" in names
    assert "orders.count_30d" in names
    assert "orders.recency_days" in names
    assert any(n.startswith("hist1:") for n in names)


def test_analyze_flat_declines_graph_shapes(churn_schema):
    assuming = ("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                "FROM customers "
                "ASSUMING COUNT(orders.*) OVER (7 DAYS FOLLOWING) > 2")
    a = analyze_flat(assuming, churn_schema)
    assert not a.eligible and "ASSUMING" in a.reason

    rank = ("PREDICT LIST_DISTINCT(orders.product_id) OVER "
            "(30 DAYS FOLLOWING RANK TOP 5) FROM customers")
    assert not analyze_flat(rank, churn_schema).eligible


def test_analyze_flat_bad_query_raises(churn_schema):
    with pytest.raises(ExecutionError):
        analyze_flat("PREDICT COUNT(nonexistent.*) FROM customers",
                     churn_schema)


# ---------------------------------------------------------------------------
# feature extraction over engine-assembled contexts
# ---------------------------------------------------------------------------

def test_features_respect_the_anchor(engine, churn_schema):
    """O4 (2026-07-05) is after the anchor and must not count."""
    from relativedb.xgb import _features
    a = analyze_flat(REG_QUERY, churn_schema)
    t0 = dt("2026-07-01")
    ctx = engine.assemble_context("customers", "C7", t0)
    x = _features(REG_QUERY, churn_schema, None, [ctx],
                  len(a.feature_names))
    by_name = dict(zip(a.feature_names, x[0]))
    assert by_name["entity.age"] == 52.0
    assert by_name["orders.count_all"] == 2.0        # O1, O2; never O4
    # O2 is 2026-05-02, 60 days before the anchor
    assert by_name["orders.recency_days"] == 60.0
    assert by_name["orders.qty_sum_all"] == 3.0      # 1 + 2
    # customers is static: signup age comes from the anchor
    assert by_name["entity.signup_date_age_days"] == pytest.approx(
        (t0 - dt("2026-01-20")).days, abs=1)


def test_missing_history_is_nan_not_zero(engine, churn_schema):
    from relativedb.xgb import _features
    a = analyze_flat(REG_QUERY, churn_schema)
    ctx = engine.assemble_context("customers", "C9", dt("2026-07-01"))
    x = _features(REG_QUERY, churn_schema, None, [ctx],
                  len(a.feature_names))
    by_name = dict(zip(a.feature_names, x[0]))
    assert by_name["orders.count_all"] == 0.0        # a real zero
    assert math.isnan(by_name["orders.recency_days"])  # never ordered
    assert math.isnan(by_name["orders.qty_avg_all"])


# ---------------------------------------------------------------------------
# end-to-end model scenarios, each in a clean subprocess (see module docstring)
# ---------------------------------------------------------------------------

def _fresh_engine():
    import conftest
    schema_fn = getattr(conftest.churn_schema, "__wrapped__",
                        conftest.churn_schema)
    schema = schema_fn()
    return schema, Engine(schema, conftest.in_memory_wiring(churn_rows()),
                          context_policy=ContextPolicy(max_context_cells=512))


def _scenario_regression():
    schema, engine = _fresh_engine()
    backend = XgboostBackend(schema, device="cpu")
    result = backend.fit(engine, REG_QUERY,
                         anchors=[dt("2026-04-01"), dt("2026-06-01")])
    assert result.n_examples == 6                    # 3 customers x 2 anchors
    assert result.n_features == len(result.feature_names)
    assert result.task_type is TaskType.REGRESSION

    engine.model_backend = backend
    out = engine.execute(REG_QUERY, anchor_time=dt("2026-06-01"))
    assert out.task_type is TaskType.REGRESSION
    assert len(out.predictions) == 3
    assert all(p.value is not None for p in out.predictions)


def _scenario_engine_fit_xgboost():
    """The Engine method: the adaptation path next to fit_head/finetune."""
    _, engine = _fresh_engine()
    backend = engine.fit_xgboost(REG_QUERY, anchors=[dt("2026-04-01"),
                                                     dt("2026-06-01")],
                                 device="cpu")
    assert backend.fit_result is not None
    assert backend.fit_result.n_examples == 6
    engine.model_backend = backend
    out = engine.execute(REG_QUERY, anchor_time=dt("2026-06-01"))
    assert len(out.predictions) == 3


def _scenario_binary():
    schema, engine = _fresh_engine()
    backend = XgboostBackend(schema, device="cpu")
    backend.fit(engine, BIN_QUERY, anchors=[dt("2026-04-01")])
    engine.model_backend = backend
    out = engine.execute(BIN_QUERY, anchor_time=dt("2026-04-01"))
    assert out.task_type is TaskType.BINARY_CLASSIFICATION
    for p in out.predictions:
        assert p.probability is not None and 0.0 <= p.probability <= 1.0


def _scenario_ineligible_fit():
    schema, engine = _fresh_engine()
    backend = XgboostBackend(schema, device="cpu")
    try:
        backend.fit(
            engine,
            "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FROM customers "
            "ASSUMING COUNT(orders.*) OVER (7 DAYS FOLLOWING) > 2",
            anchors=[dt("2026-06-01")])
    except ExecutionError as e:
        assert "not flat-eligible" in str(e)
    else:
        raise AssertionError("ineligible fit did not raise")


def _scenario_score_without_fit():
    schema, engine = _fresh_engine()
    engine.model_backend = XgboostBackend(schema, device="cpu")
    try:
        engine.execute(REG_QUERY, anchor_time=dt("2026-06-01"))
    except ExecutionError as e:
        assert "no fitted model" in str(e)
    else:
        raise AssertionError("scoring without a fitted model did not raise")


def _scenario_save_load():
    import tempfile
    schema, engine = _fresh_engine()
    backend = XgboostBackend(schema, device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        backend.fit(engine, REG_QUERY,
                    anchors=[dt("2026-04-01"), dt("2026-06-01")],
                    save_path=os.path.join(tmp, "xgb_model"))
        engine.model_backend = backend
        before = engine.execute(REG_QUERY, anchor_time=dt("2026-06-01"))

        loaded = XgboostBackend(schema, device="cpu").load(
            os.path.join(tmp, "xgb_model"))
        engine.model_backend = loaded
        after = engine.execute(REG_QUERY, anchor_time=dt("2026-06-01"))
    assert [p.value for p in before.predictions] == \
        [p.value for p in after.predictions]


_SCENARIOS = {
    "regression": _scenario_regression,
    "engine_fit_xgboost": _scenario_engine_fit_xgboost,
    "binary": _scenario_binary,
    "ineligible_fit": _scenario_ineligible_fit,
    "score_without_fit": _scenario_score_without_fit,
    "save_load": _scenario_save_load,
}


def _run_isolated(name: str) -> None:
    pytest.importorskip("xgboost")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), name],
        capture_output=True, text=True, timeout=300,
        env=os.environ.copy(), cwd=os.path.dirname(os.path.abspath(__file__)))
    assert proc.returncode == 0, (
        f"isolated scenario {name!r} failed "
        f"(rc={proc.returncode})\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}")
    assert f"SCENARIO-OK {name}" in proc.stdout


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_model_scenarios_isolated(name):
    _run_isolated(name)


if __name__ == "__main__":
    _name = sys.argv[1]
    _SCENARIOS[_name]()
    print(f"SCENARIO-OK {_name}")


def test_peer_rows_never_leak_into_features(churn_schema):
    """Two entities sharing a context sample must not share features: the
    encoder sends focal rows only."""
    from relativedb.xgb import _encode_context
    from relativedb.engine import EntityContext
    rows = churn_rows()
    ctx = EntityContext(
        entity_id="C1", anchor=dt("2026-07-01"),
        rows=rows["customers"] + rows["orders"],
        focal_row_keys=frozenset({("customers", "C1"), ("orders", "O3")}))
    encoded = _encode_context(ctx)
    assert {r["id"] for r in encoded["rows"]} == {"C1", "O3"}
