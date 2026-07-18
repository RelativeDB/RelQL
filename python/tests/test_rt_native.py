"""Native RT backend: golden ctypes regression + end-to-end churn scenario.

The golden test feeds the raw PRE-sort arrays dumped from the PyTorch
reference (cpp/testdata, B=5 S=16) straight through the ctypes layer and
checks the target scores of BOTH checkpoints; it is the acceptance gate for
the binding. The e2e test runs the README churn scenario with
:class:`RtNativeBackend` instead of the history baseline.
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
