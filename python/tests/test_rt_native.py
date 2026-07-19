"""Native RT backend: golden ctypes regression + end-to-end churn scenario.

The golden test feeds the raw PRE-sort arrays dumped from the PyTorch
reference (cpp/testdata, B=5 S=16) straight through the ctypes layer and
checks the target scores of BOTH checkpoints; it is the acceptance gate for
the binding. The e2e test runs the README churn scenario with
:class:`RtNativeBackend` (the engine's only real scoring backend).
"""
import json
import math
import os

import numpy as np
import pytest

from relativedb.rt_native import (RtNativeBackend, RtNativeUnavailableError,
                                load_lib, resolve_model_path)
from conftest import churn_rows, dt, in_memory_wiring

TESTDATA = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "cpp", "testdata"))

GOLDEN_SCORES = {
    "classification": [-0.18470, -0.33108, +0.43363, -0.14449, +0.46848],
    "regression": [-0.27052, -0.41538, +0.39998, -0.30649, +0.26804],
}


def _lib_or_skip():
    try:
        return load_lib()
    except RtNativeUnavailableError as e:
        pytest.skip(f"librt_c not available: {e}")


def _checkpoint_or_skip(variant: str) -> str:
    try:
        return resolve_model_path(f"hf://stanford-star/rt-j/{variant}")
    except Exception as e:
        pytest.skip(f"rt-j {variant} checkpoint not available: {e}")


def _load_golden_batch():
    if not os.path.isfile(os.path.join(TESTDATA, "manifest.json")):
        pytest.skip(f"golden testdata not found at {TESTDATA}")
    man = json.load(open(os.path.join(TESTDATA, "manifest.json")))

    def arr(name):
        m = man[name]
        return np.fromfile(os.path.join(TESTDATA, f"{name}.bin"),
                           dtype=m["dtype"]).reshape(m["shape"])

    return dict(
        node_idxs=arr("node_idxs"), f2p=arr("f2p_nbr_idxs"),
        col_idxs=arr("col_name_idxs"), table_idxs=arr("table_name_idxs"),
        is_padding=arr("is_padding"), sem_types=arr("sem_types"),
        is_target=arr("is_targets"), number_v=arr("number_values"),
        datetime_v=arr("datetime_values"), boolean_v=arr("boolean_values"),
        text_v=arr("text_values"), col_name_v=arr("col_name_values"))


@pytest.mark.parametrize("variant", ["classification", "regression"])
def test_golden_scores_through_ctypes(variant):
    """Raw golden arrays -> rt_forward -> scores match PyTorch within 2e-3."""
    lib = _lib_or_skip()
    path = _checkpoint_or_skip(variant)
    batch = _load_golden_batch()
    model = lib.load_model(path)
    assert model.num_params > 80_000_000
    scores = model.forward(**batch)
    assert scores.shape == (5,)
    for got, want in zip(scores, GOLDEN_SCORES[variant]):
        assert abs(float(got) - want) < 2e-3, (variant, scores)


def test_load_lib_missing_is_clear_error():
    with pytest.raises(RtNativeUnavailableError, match="Searched"):
        load_lib("/nonexistent/librt_c.dylib")


def test_resolve_model_path_local_and_hf(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"x")
    assert resolve_model_path(str(f)) == str(f)
    assert resolve_model_path(str(tmp_path)) == str(f)
    with pytest.raises(RtNativeUnavailableError):
        resolve_model_path("gs://nope")


def test_resolve_model_path_prefers_quantized_only_when_opted_in(
        tmp_path, monkeypatch):
    f = tmp_path / "model.safetensors"
    q8 = tmp_path / "model.q8.safetensors"
    q4 = tmp_path / "model.q4.safetensors"
    f.write_bytes(b"x")
    q8.write_bytes(b"q")
    q4.write_bytes(b"q")
    monkeypatch.delenv("RELATIVEDB_RT_QUANTIZED", raising=False)
    assert resolve_model_path(str(tmp_path)) == str(f)     # default: fp32
    monkeypatch.setenv("RELATIVEDB_RT_QUANTIZED", "1")
    assert resolve_model_path(str(tmp_path)) == str(q8)    # 1 -> q8
    monkeypatch.setenv("RELATIVEDB_RT_QUANTIZED", "q4")
    assert resolve_model_path(str(tmp_path)) == str(q4)    # explicit variant
    assert resolve_model_path(str(f)) == str(f)            # explicit file wins
    monkeypatch.setenv("RELATIVEDB_RT_QUANTIZED", "f16")   # variant missing
    assert resolve_model_path(str(tmp_path)) == str(f)


# quantized formats: (file variant, score tolerance vs fp32 golden)
QUANT_TOL = {"q8": 5e-2, "q4": 1e-1, "f16": 5e-3}


@pytest.mark.parametrize("variant", ["classification", "regression"])
@pytest.mark.parametrize("qv", ["q8", "q4", "f16"])
def test_quantized_checkpoint_scores_track_fp32(variant, qv):
    """Quantized checkpoints load through the same C ABI (weights stay
    quantized-resident; kernels dequantize) and score within per-format
    tolerance of the PyTorch golden scores."""
    lib = _lib_or_skip()
    fp32 = _checkpoint_or_skip(variant)
    q = os.path.join(os.path.dirname(fp32), f"model.{qv}.safetensors")
    if not os.path.isfile(q):
        pytest.skip(f"no model.{qv}.safetensors (run cpp/rt_quantize)")
    batch = _load_golden_batch()
    scores = lib.load_model(q).forward(**batch)
    for got, want in zip(scores, GOLDEN_SCORES[variant]):
        assert abs(float(got) - want) < QUANT_TOL[qv], (variant, qv, scores)


# --------------------------------------------------------------------------
# End-to-end: the README churn scenario, scored by the native RT engine.
# --------------------------------------------------------------------------

def test_churn_end_to_end_with_native_backend(churn_schema):
    pytest.importorskip("sentence_transformers")
    _lib_or_skip()
    _checkpoint_or_skip("classification")
    from relativedb import Engine, ExecutionInput

    backend = RtNativeBackend(schema=churn_schema)
    engine = Engine(churn_schema, in_memory_wiring(churn_rows()),
                    model_backend=backend)
    result = engine.execute(ExecutionInput(
        query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FOR EACH customers.customer_id",
        anchor_time=dt("2026-07-01")))

    probs = {p.id: p.probability for p in result.predictions}
    assert set(probs) == {"C1", "C7", "C9"}
    for p in probs.values():
        assert 0.0 < p < 1.0 and math.isfinite(p)
    # soft ranking check: the long-inactive C9 (only order 2026-01-15-ish era,
    # none in context window) should look riskier than recently-active C1
    assert probs["C9"] > probs["C1"], probs


# --------------------------------------------------------------------------
# RETURN output-shaping — moved from the deleted history baseline onto the
# native backend's model probability.
# --------------------------------------------------------------------------

def _native_engine(schema):
    pytest.importorskip("sentence_transformers")
    _lib_or_skip()
    _checkpoint_or_skip("classification")
    from relativedb import Engine
    return Engine(schema, in_memory_wiring(churn_rows()),
                  model_backend=RtNativeBackend(schema=schema))


def test_return_class_emits_hard_label(churn_schema):
    from relativedb import ExecutionInput
    eng = _native_engine(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FOR EACH customers.customer_id RETURN CLASS",
        entity_ids=['C7'], anchor_time=dt("2026-07-01")))
    pred = res.predictions[0]
    assert pred.predicted_class in ("true", "false")   # hard label at 0.5
    assert pred.probability is None                     # not the score
    assert not pred.class_probs


def test_return_distribution_two_key_dist(churn_schema):
    from relativedb import ExecutionInput
    eng = _native_engine(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FOR EACH customers.customer_id RETURN DISTRIBUTION",
        entity_ids=['C7'], anchor_time=dt("2026-07-01")))
    pred = res.predictions[0]
    assert set(pred.class_probs) == {"true", "false"}
    assert abs(sum(pred.class_probs.values()) - 1.0) < 1e-9


# --------------------------------------------------------------------------
# Multiclass classification & ranking, end-to-end through the native engine.
# Both enumerate their domain via the wired TableScanner (CONTRACT.md §2/§3).
# --------------------------------------------------------------------------

def _native_engine_with_wiring(schema):
    pytest.importorskip("sentence_transformers")
    _lib_or_skip()
    _checkpoint_or_skip("classification")
    from relativedb import Engine
    wiring = in_memory_wiring(churn_rows())
    return Engine(schema, wiring,
                  model_backend=RtNativeBackend(schema=schema, wiring=wiring))


def test_multiclass_classification_end_to_end(churn_schema):
    """FIRST(products.name) -> masked TEXT target -> rt_forward_ex text head ->
    cosine vs the L2-normed MiniLM label embeddings. class_probs is a full,
    normalized distribution over the distinct product names."""
    from relativedb import ExecutionInput
    eng = _native_engine_with_wiring(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT FIRST(products.name) FOR EACH customers.customer_id",
        anchor_time=dt("2026-07-01")))
    assert res.task_type.name == "MULTICLASS_CLASSIFICATION"
    assert res.model_uri == "hf://stanford-star/rt-j/classification"
    labels = {"espresso machine", "running shoes", "yoga mat"}
    assert {p.id for p in res.predictions} == {"C1", "C7", "C9"}
    for p in res.predictions:
        assert set(p.class_probs) == labels                 # full K-way domain
        assert abs(sum(p.class_probs.values()) - 1.0) < 1e-6
        assert all(0.0 <= v <= 1.0 for v in p.class_probs.values())
        assert p.predicted_class in labels
        # argmax must agree with the class_probs argmax (both are cosine order)
        top = max(p.class_probs, key=p.class_probs.get)
        assert p.predicted_class == top


def test_multiclass_return_class_and_distribution(churn_schema):
    from relativedb import ExecutionInput
    eng = _native_engine_with_wiring(churn_schema)
    for ret in ("CLASS", "DISTRIBUTION"):
        res = eng.execute(ExecutionInput(
            query="PREDICT FIRST(products.name) "
                  f"FOR EACH customers.customer_id RETURN {ret}",
            entity_ids=["C7"], anchor_time=dt("2026-07-01")))
        pred = res.predictions[0]
        assert pred.predicted_class in {"espresso machine", "running shoes",
                                        "yoga mat"}
        assert abs(sum(pred.class_probs.values()) - 1.0) < 1e-6


def test_ranking_end_to_end_top_k(churn_schema):
    """LIST_DISTINCT(orders.product_id) RANK TOP 2 -> per-candidate existence
    contexts over the products (parent) ids -> sigmoid -> top-k ids."""
    from relativedb import ExecutionInput
    eng = _native_engine_with_wiring(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT LIST_DISTINCT(orders.product_id) OVER (30 DAYS FOLLOWING) "
              "RANK TOP 2 FOR EACH customers.customer_id",
        anchor_time=dt("2026-07-01")))
    assert res.task_type.name == "MULTILABEL_RANKING"
    all_products = {"P1", "P2", "P3"}
    assert {p.id for p in res.predictions} == {"C1", "C7", "C9"}
    for p in res.predictions:
        assert isinstance(p.ranked, tuple)
        assert 1 <= len(p.ranked) <= 2                      # top-k, k=2
        assert len(set(p.ranked)) == len(p.ranked)          # no duplicates
        assert set(p.ranked) <= all_products                # real parent ids


def test_ranking_top_k_clamped_to_candidate_count(churn_schema):
    """k larger than the candidate pool yields at most #candidates ids (3)."""
    from relativedb import ExecutionInput
    eng = _native_engine_with_wiring(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT LIST_DISTINCT(orders.product_id) OVER (30 DAYS FOLLOWING) "
              "RANK TOP 25 FOR EACH customers.customer_id",
        entity_ids=["C7"], anchor_time=dt("2026-07-01")))
    ranked = res.predictions[0].ranked
    assert len(ranked) == 3 and set(ranked) == {"P1", "P2", "P3"}
