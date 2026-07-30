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

from relativedb.scoring import ContextConnectivityWarning
from relativedb_engine import (RtNativeBackend, RtNativeUnavailableError,
                               load_lib, resolve_model_path)
from conftest import (churn_rows, dt, in_memory_wiring, require_checkpoint,
                      require_native, require_text_embedder, _unavailable)

TESTDATA = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "cpp", "testdata"))

GOLDEN_SCORES = {
    "classification": [-0.18470, -0.33108, +0.43363, -0.14449, +0.46848],
    "regression": [-0.27052, -0.41538, +0.39998, -0.30649, +0.26804],
}

# Second golden, cpp/testdata/large (B=2, S=70). The B=5 S=16 batch above is
# structurally blind to long-context defects: at S=16 every attention group is
# a singleton or near it, and the node ids are 0..N. The large batch clears the
# S=36 regime, carries 35 multi-token same-node (feat) groups and 8 multi-token
# column groups, and uses node ids around 4.49e7 -- past 2**24, where distinct
# integers stop being representable in float32. See cpp/tools/dump_golden.py.
LARGE_GOLDEN_SCORES = {
    "classification": [+0.46119, +1.27047],
}

# Backwards-compatible aliases; both route through conftest so that
# RELATIVEDB_REQUIRE_NATIVE=1 turns a missing prerequisite into a failure.
_lib_or_skip = require_native
_checkpoint_or_skip = require_checkpoint


def _load_golden_batch(subdir=""):
    root = os.path.join(TESTDATA, subdir) if subdir else TESTDATA
    if not os.path.isfile(os.path.join(root, "manifest.json")):
        # Checked into the repo alongside cpp/; absence means a broken checkout.
        _unavailable(f"golden testdata (cpp/testdata/{subdir})",
                     f"not found at {root}")
    man = json.load(open(os.path.join(root, "manifest.json")))

    # The manifest alone is not enough: cpp/.gitignore used to exclude
    # testdata/*.bin, so a clone got the manifest and none of the arrays. That
    # slipped past a manifest-only check and died in np.fromfile with a bare
    # FileNotFoundError -- an unreadable failure for the binding's acceptance
    # gate. Check the arrays up front and route through _unavailable, so a
    # partial checkout skips (or, under strict mode, fails) legibly.
    # Underscore-prefixed manifest keys (_checkpoint, _note) are provenance
    # strings, not arrays; only the dict-valued entries describe a .bin.
    arrays = [n for n, m in man.items()
              if not n.startswith("_") and isinstance(m, dict)]
    missing = [n for n in arrays
               if not os.path.isfile(os.path.join(root, f"{n}.bin"))]
    if missing:
        _unavailable(
            f"golden testdata arrays (cpp/testdata/{subdir}/*.bin)",
            f"{len(missing)} of {len(arrays)} missing under {root} "
            f"(e.g. {', '.join(sorted(missing)[:3])}); regenerate with "
            f"cpp/tools/dump_golden.py")

    def arr(name):
        m = man[name]
        return np.fromfile(os.path.join(root, f"{name}.bin"),
                           dtype=m["dtype"]).reshape(m["shape"])

    return dict(
        node_idxs=arr("node_idxs"), f2p=arr("f2p_nbr_idxs"),
        col_idxs=arr("col_name_idxs"), table_idxs=arr("table_name_idxs"),
        is_padding=arr("is_padding"), sem_types=arr("sem_types"),
        is_target=arr("is_targets"), number_v=arr("number_values"),
        datetime_v=arr("datetime_values"), boolean_v=arr("boolean_values"),
        text_v=arr("text_values"), col_name_v=arr("col_name_values"))


def _load_golden_reference(subdir=""):
    """PyTorch yhat_number (POST-sort) and the sort permutation that yields it."""
    root = os.path.join(TESTDATA, subdir) if subdir else TESTDATA
    man = json.load(open(os.path.join(root, "manifest.json")))

    def arr(name):
        m = man[name]
        return np.fromfile(os.path.join(root, f"{name}.bin"),
                           dtype=m["dtype"]).reshape(m["shape"])

    return arr("yhat_number"), arr("sort_idxs")


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
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


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
@pytest.mark.parametrize("variant", sorted(LARGE_GOLDEN_SCORES))
def test_large_golden_scores_through_ctypes(variant):
    """Long-context golden (B=2, S=70) -> rt_forward -> scores match PyTorch.

    The B=5 S=16 gate above sits underneath a whole class of defect: it has no
    long sequences, no crowded attention groups and no node ids above 2**24.
    This case covers that regime, so a grouping or masking regression that only
    shows up past S~36 cannot pass the suite green.
    """
    lib = _lib_or_skip()
    path = _checkpoint_or_skip(variant)
    batch = _load_golden_batch("large")
    assert batch["node_idxs"].shape[1] >= 64
    model = lib.load_model(path)

    # Per-token check first: it localizes a regression to a token instead of
    # letting sign-cancellation hide inside a summed score.
    #
    # Tolerances are looser than the S=16 gate's 2e-3, and the reason is the
    # golden, not the native code. The stored yhat_number is the PyTorch model
    # in fp32; measured against an fp64 recomputation of this same batch, the
    # native forward is accurate to 8e-6 per token while the stored fp32
    # reference is off by 3.8e-3. 6e-3 covers the reference's own noise at
    # S=70 and still leaves ~25x headroom over the grouping defect this batch
    # exists to catch, which moved tokens by 1.5e-1 at block 0 alone.
    tokens = model.forward_tokens(**batch)          # pre-sort order
    reference, sort_idxs = _load_golden_reference("large")
    got = np.take_along_axis(tokens, sort_idxs, axis=1)    # -> post-sort order
    keep = np.take_along_axis(batch["is_padding"], sort_idxs, axis=1) == 0
    assert np.abs(got - reference)[keep].max() < 6e-3, (
        variant, np.abs(got - reference)[keep].max())

    # Each row carries six masked targets, so the summed score accumulates six
    # tokens' worth of that same reference drift.
    scores = model.forward(**batch)
    assert scores.shape == (2,)
    for value, expected in zip(scores, LARGE_GOLDEN_SCORES[variant]):
        assert abs(float(value) - expected) < 1e-2, (variant, scores)


def test_large_golden_exercises_crowded_groups_and_wide_node_ids():
    """The large golden is only a real gate if it has the structure to be one.

    Guards the batch itself, not the model: a regenerated golden whose groups
    are all singletons, or whose node ids fit in float32, would be exactly as
    blind as the S=16 set and would still pass every score assertion.
    """
    batch = _load_golden_batch("large")
    keep = batch["is_padding"][0] == 0
    node = batch["node_idxs"][0][keep]
    coltab = [(int(c) << 32) ^ int(t) for c, t in
              zip(batch["col_idxs"][0][keep], batch["table_idxs"][0][keep])]

    _, node_counts = np.unique(node, return_counts=True)
    _, col_counts = np.unique(coltab, return_counts=True)
    assert batch["node_idxs"].shape[1] >= 64, "must clear the S~36 onset"
    assert (node_counts > 1).sum() >= 5, "no multi-token feat groups"
    assert (col_counts > 1).sum() >= 5, "no multi-token column groups"

    # Node ids are global row ids in production, routinely past 2**24. Anything
    # that round-trips them through float32 on the way to the native layer
    # merges distinct nodes and silently fuses their attention groups; the
    # golden has to contain ids that make that mistake observable.
    assert int(node.max()) > 2 ** 24
    assert len(np.unique(node.astype(np.float32))) < len(np.unique(node))


def test_reference_adapter_preserves_wide_integer_ids():
    """The eval adapter must hand the native layer exact int64 ids.

    Regression test for the defect this golden was added alongside: the adapter
    converted every batch tensor with ``.float()`` before casting to int64, so
    node ids and FK parent ids above 2**24 collapsed onto shared float32 values.
    Distinct rows became one node, their feat/nbr attention groups were merged,
    and native diverged from the reference by ~5e-1 in logit space -- with every
    C++ and Python test still green.
    """
    torch = pytest.importorskip("torch")
    import importlib.util
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "evaluation",
        "run_native_on_reference.py"))
    if not os.path.isfile(path):
        pytest.skip("evaluation/run_native_on_reference.py not present")
    spec = importlib.util.spec_from_file_location("_run_native_on_ref", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Float32 spacing at 4.49e7 is 4, so all three ids below round to the same
    # float and the old conversion returned them as one value.
    ids = torch.tensor([[44903103, 44903104, 44903105, -1]], dtype=torch.int64)
    got = module.NativeReferenceAdapter._np(ids, np.int64)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, ids.numpy())
    assert len(np.unique(got)) == 4, "distinct ids must survive the conversion"

    values = torch.tensor([[1.5, -2.25]], dtype=torch.float32)
    np.testing.assert_allclose(module.NativeReferenceAdapter._np(values),
                               values.numpy())


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_native_mps_full_model_step_and_checkpoint(tmp_path):
    """The full path decreases loss and emits a regular model checkpoint."""
    lib = _lib_or_skip()
    from relativedb_engine import RT_DEVICE_MPS
    if not lib.device_available(RT_DEVICE_MPS):
        pytest.skip("full-model fine-tuning needs MPS")
    if not lib.full_finetune_available():
        pytest.skip("full-model fine-tuning needs a Metal 3 GPU; this device "
                    "(paravirtualised GPUs on CI runners and VMs) is not one")
    model = lib.load_model(_checkpoint_or_skip("classification"))
    batch = _load_golden_batch()
    first = model.finetune_step(**batch, learning_rate=1e-6)
    second = model.finetune_step(**batch, learning_rate=1e-6)
    assert math.isfinite(first["grad_norm"])
    assert second["loss"] < first["loss"]
    path = tmp_path / "model.safetensors"
    model.save(path)
    loaded = lib.load_model(str(path))
    assert loaded.num_params == model.num_params


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_native_mps_accumulation_resume_and_gradients(tmp_path):
    lib = _lib_or_skip()
    from relativedb_engine import RT_DEVICE_MPS
    if not lib.device_available(RT_DEVICE_MPS):
        pytest.skip("full-model fine-tuning needs MPS")
    if not lib.full_finetune_available():
        pytest.skip("full-model fine-tuning needs a Metal 3 GPU; this device "
                    "(paravirtualised GPUs on CI runners and VMs) is not one")
    checkpoint = _checkpoint_or_skip("classification")
    batch = _load_golden_batch()
    one = {key: value[:1].copy() for key, value in batch.items()}
    two = {key: np.concatenate((value[:1], value[:1]), axis=0)
           for key, value in batch.items()}

    accumulated = lib.load_model(checkpoint)
    combined = lib.load_model(checkpoint)
    first = accumulated.finetune_step(
        **one, learning_rate=1e-6, weight_decay=0, apply_update=False)
    second = accumulated.finetune_step(
        **one, learning_rate=1e-6, weight_decay=0, apply_update=True)
    combined.finetune_step(
        **two, learning_rate=1e-6, weight_decay=0, apply_update=True)
    assert not first["updated"] and second["updated"]
    assert np.max(np.abs(accumulated.forward(**one, device=RT_DEVICE_MPS)
                         - combined.forward(**one, device=RT_DEVICE_MPS))) < 2e-6

    model_path = tmp_path / "model.safetensors"
    optimizer_path = tmp_path / "optimizer.bin"
    accumulated.save(model_path)
    accumulated.save_finetune_optimizer(optimizer_path)
    resumed = lib.load_model(str(model_path))
    resumed.load_finetune_optimizer(optimizer_path)
    accumulated.finetune_step(**one, learning_rate=1e-6, weight_decay=0)
    resumed.finetune_step(**one, learning_rate=1e-6, weight_decay=0)
    assert np.max(np.abs(accumulated.forward(**one, device=RT_DEVICE_MPS)
                         - resumed.forward(**one, device=RT_DEVICE_MPS))) < 2e-6

    checked = lib.load_model(checkpoint).gradient_check(**batch, epsilon=1e-2)
    assert checked["checked"] >= 8
    assert checked["max_absolute_error"] < 2e-5
    assert checked["max_relative_error"] < 2e-2


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


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
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

@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_churn_end_to_end_with_native_backend(churn_schema):
    require_text_embedder()
    _lib_or_skip()
    _checkpoint_or_skip("classification")
    from relativedb import ContextPolicy, Engine, ExecutionInput

    backend = RtNativeBackend(schema=churn_schema)
    # cohort_size=0: with three customers in the fixture, cohort seeding puts
    # all of them in every context, so this stops being a per-entity ordering
    # check. Cohort behaviour is covered where the graph is large enough.
    engine = Engine(churn_schema, in_memory_wiring(churn_rows()),
                    context_policy=ContextPolicy(cohort_size=0),
                    model_backend=backend)
    result = engine.execute(ExecutionInput(
        query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FROM customers",
        anchor_time=dt("2026-07-01")))

    probs = {p.id: p.probability for p in result.predictions}
    assert set(probs) == {"C1", "C7", "C9"}
    for p in probs.values():
        assert 0.0 < p < 1.0 and math.isfinite(p)
    # This is a transport/integration test, not a quality benchmark. Reference
    # task-row materialization intentionally changes the frozen checkpoint's
    # input distribution, so no fixture-specific ordering is contractual.
    assert len(set(probs.values())) > 1


# --------------------------------------------------------------------------
# RETURN output-shaping — moved from the deleted history baseline onto the
# native backend's model probability.
# --------------------------------------------------------------------------

def _native_engine(schema):
    require_text_embedder()
    _lib_or_skip()
    _checkpoint_or_skip("classification")
    from relativedb import Engine
    return Engine(schema, in_memory_wiring(churn_rows()),
                  model_backend=RtNativeBackend(schema=schema))


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_return_class_emits_hard_label(churn_schema):
    from relativedb import ExecutionInput
    eng = _native_engine(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FROM customers WHERE customers.customer_id IN :ids RETURN CLASS",
        params={"ids": ['C7']}, anchor_time=dt("2026-07-01")))
    pred = res.predictions[0]
    assert pred.predicted_class in ("true", "false")   # hard label at 0.5
    assert pred.probability is None                     # not the score
    assert not pred.class_probs


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_return_distribution_two_key_dist(churn_schema):
    from relativedb import ExecutionInput
    eng = _native_engine(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FROM customers WHERE customers.customer_id IN :ids RETURN DISTRIBUTION",
        params={"ids": ['C7']}, anchor_time=dt("2026-07-01")))
    pred = res.predictions[0]
    assert set(pred.class_probs) == {"true", "false"}
    assert abs(sum(pred.class_probs.values()) - 1.0) < 1e-9


# --------------------------------------------------------------------------
# Multiclass classification & ranking, end-to-end through the native engine.
# Both enumerate their domain via the wired TableScanner (CONTRACT.md §2/§3).
# --------------------------------------------------------------------------

def _native_engine_with_wiring(schema):
    require_text_embedder()
    _lib_or_skip()
    _checkpoint_or_skip("classification")
    from relativedb import Engine
    wiring = in_memory_wiring(churn_rows())
    return Engine(schema, wiring,
                  model_backend=RtNativeBackend(schema=schema, wiring=wiring))


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_multiclass_classification_end_to_end(churn_schema):
    """FIRST(products.name) -> masked TEXT target -> rt_forward_ex text head ->
    cosine vs the L2-normed MiniLM label embeddings. class_probs is a full,
    normalized distribution over the distinct product names."""
    from relativedb import ExecutionInput
    eng = _native_engine_with_wiring(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT FIRST(products.name) FROM customers",
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


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_multiclass_return_class_and_distribution(churn_schema):
    from relativedb import ExecutionInput
    eng = _native_engine_with_wiring(churn_schema)
    for ret in ("CLASS", "DISTRIBUTION"):
        res = eng.execute(ExecutionInput(
            query="PREDICT FIRST(products.name) "
                  f"FROM customers WHERE customers.customer_id IN :ids RETURN {ret}",
            params={"ids": ["C7"]}, anchor_time=dt("2026-07-01")))
        pred = res.predictions[0]
        assert pred.predicted_class in {"espresso machine", "running shoes",
                                        "yoga mat"}
        assert abs(sum(pred.class_probs.values()) - 1.0) < 1e-6


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_ranking_end_to_end_top_k(churn_schema):
    """LIST_DISTINCT(orders.product_id) RANK TOP 2 -> per-candidate existence
    contexts over the products (parent) ids -> sigmoid -> top-k ids."""
    from relativedb import ExecutionInput
    eng = _native_engine_with_wiring(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT LIST_DISTINCT(orders.product_id) OVER (30 DAYS FOLLOWING RANK TOP 2) FROM customers",
        anchor_time=dt("2026-07-01")))
    assert res.task_type.name == "MULTILABEL_RANKING"
    all_products = {"P1", "P2", "P3"}
    assert {p.id for p in res.predictions} == {"C1", "C7", "C9"}
    for p in res.predictions:
        assert isinstance(p.ranked, tuple)
        assert 1 <= len(p.ranked) <= 2                      # top-k, k=2
        assert len(set(p.ranked)) == len(p.ranked)          # no duplicates
        assert set(p.ranked) <= all_products                # real parent ids


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_ranking_top_k_clamped_to_candidate_count(churn_schema):
    """k larger than the candidate pool yields at most #candidates ids (3)."""
    from relativedb import ExecutionInput
    eng = _native_engine_with_wiring(churn_schema)
    res = eng.execute(ExecutionInput(
        query="PREDICT LIST_DISTINCT(orders.product_id) OVER (30 DAYS FOLLOWING RANK TOP 25) FROM customers WHERE customers.customer_id IN :ids",
        params={"ids": ["C7"]}, anchor_time=dt("2026-07-01")))
    ranked = res.predictions[0].ranked
    assert len(ranked) == 3 and set(ranked) == {"P1", "P2", "P3"}


# ---------------------------------------------------------------------------
# Context connectivity: a token-less parent row severs the context
# ---------------------------------------------------------------------------

def _keyless_entity_schema(pk_as_column: bool):
    """`customers` with no feature columns at all — optionally opting its
    primary key in as a feature."""
    from relativedb import LinkDef, Schema, TableDef, ValueType
    ct = TableDef.new_table("customers")
    if pk_as_column:
        ct = ct.column("customer_id", ValueType.TEXT)
    return (Schema.new_schema()
            .table(ct.primary_key("customer_id").build())
            .table(TableDef.new_table("orders")
                   .column("qty", ValueType.NUMBER)
                   .column("order_date", ValueType.DATETIME)
                   .primary_key("order_id").time_column("order_date").build())
            .link(LinkDef("orders", "customer_id", "customers")).build())


def _keyless_rows(pk_as_column: bool):
    """customers rows carrying NO feature cells (or only the pk), plus orders."""
    from relativedb import Row
    cust = [Row("customers", cid,
                {"customer_id": cid} if pk_as_column else {})
            for cid in ("C1", "C7")]
    orders, oid = [], 0
    for cid, n in (("C1", 2), ("C7", 6)):        # different histories
        for k in range(n):
            oid += 1
            orders.append(Row("orders", f"O{oid}",
                              {"qty": float(k + 1),
                               "order_date": dt("2026-06-%02d" % (k + 1))},
                              timestamp=dt("2026-06-%02d" % (k + 1)),
                              parents={"customer_id": cid}))
    return {"customers": cust, "orders": orders}


def _seqs_for(schema, warn_record, pk_as_column: bool):
    """Build model sequences for two customers with different histories."""
    import warnings
    from relativedb import Engine
    from relativedb.relql.ast import TaskType
    from relativedb.relql.parser import parse, validate

    rows = _keyless_rows(pk_as_column)
    wiring = in_memory_wiring(rows)
    backend = RtNativeBackend(schema=schema, wiring=wiring)
    eng = Engine(schema, wiring, model_backend=backend)
    pq = validate(parse("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                        "FROM customers"), schema).query
    ctxs = [eng.assemble_context("customers", cid, dt("2026-07-01"))
            for cid in ("C1", "C7")]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        seqs, _, _ = backend._build_sequences(
            pq, TaskType.BINARY_CLASSIFICATION, ctxs)
    warn_record.extend(x for x in w
                       if x.category is ContextConnectivityWarning)
    return seqs


@pytest.mark.integration
@pytest.mark.checkpoint
def test_tokenless_parent_row_warns():
    """A `customers` row with no feature cells emits no tokens, so the orders
    hanging off it can never reach the prediction. That must not be silent."""
    require_text_embedder()
    got = []
    _seqs_for(_keyless_entity_schema(pk_as_column=False), got,
              pk_as_column=False)
    assert got, "expected a ContextConnectivityWarning"
    assert "customers" in str(got[0].message)


def test_primary_key_declared_as_column_reconnects_the_context():
    """Reference preprocessing always excludes primary-key feature cells."""
    from relativedb import SchemaError
    with pytest.raises(SchemaError, match="cannot also be a feature"):
        _keyless_entity_schema(pk_as_column=True)


def test_schema_rejects_a_primary_key_that_is_also_a_column():
    from relativedb import SchemaError
    with pytest.raises(SchemaError, match="cannot also be a feature"):
        _keyless_entity_schema(pk_as_column=True)


# ---------------------------------------------------------------------------
# Frozen adapter fitting: Engine.fit_head -> FineTunedHead -> backend(head=...)
# ---------------------------------------------------------------------------

def _metal_or_skip():
    """Head fitting runs on Metal; scoring a trained head does not.

    A genuinely absent GPU is a legitimate skip (it is a property of the
    runner, not of the build), so this one stays a skip even under strict
    mode — unlike the librt_c/checkpoint gates.
    """
    from relativedb_engine import RT_DEVICE_MPS
    if not require_native().device_available(RT_DEVICE_MPS):
        pytest.skip("fine-tuning needs a Metal device")


def _ft_engine(schema, head=None):
    from relativedb import Engine
    wiring = in_memory_wiring(churn_rows())
    return Engine(schema, wiring,
                  model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                                head=head))


FT_ANCHORS = None      # filled in per-test; the fixture's orders span 2026


def test_fit_head_rejects_scalar_substitute_for_full_tuning(churn_schema):
    from relativedb import ExecutionError
    eng = _ft_engine(churn_schema)
    with pytest.raises(ExecutionError, match="full-backbone fine-tuning"):
        eng.fit_head(
            "PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) = 0 FROM customers",
            [dt("2026-04-01")])


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_fitted_head_changes_ranking(churn_schema):
    """The head must actually be used at scoring time, not merely loadable."""
    _lib_or_skip()
    _metal_or_skip()
    from relativedb import ExecutionInput
    q = ("PREDICT LIST_DISTINCT(orders.product_id) "
         "OVER (30 DAYS FOLLOWING RANK TOP 2) FROM customers")
    base = _ft_engine(churn_schema)
    before = base.execute(ExecutionInput(query=q, anchor_time=dt("2026-07-01")))
    head = base.fit_head(q, [dt("2026-05-01"), dt("2026-06-01")],
                         epochs=60, learning_rate=1e-2)
    assert head.task_name == "ranking"
    assert head.final_loss < head.initial_loss
    tuned = _ft_engine(churn_schema, head=head)
    after = tuned.execute(ExecutionInput(query=q, anchor_time=dt("2026-07-01")))
    assert ({p.id: p.ranked for p in before.predictions}
            != {p.id: p.ranked for p in after.predictions})


def test_fit_head_rejects_scalar_explicit_labels_too(churn_schema):
    """Explicit labels must not reopen the disabled scalar adapter path."""
    from relativedb import ExecutionError
    eng = _ft_engine(churn_schema)
    anchors = [dt("2026-05-01"), dt("2026-06-01")]
    given = {(cid, t): float(i % 2)
             for i, t in enumerate(anchors) for cid in ("C1", "C7", "C9")}
    with pytest.raises(ExecutionError, match="full-backbone fine-tuning"):
        eng.fit_head(
            "PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) = 0 FROM customers",
            anchors, labels=given)


def test_fit_head_needs_the_native_backend(churn_schema):
    """A stub backend cannot encode frozen features; say so plainly."""
    from relativedb import Engine, ExecutionError
    from conftest import StubBackend
    eng = Engine(churn_schema, in_memory_wiring(churn_rows()),
                 model_backend=StubBackend())
    with pytest.raises(ExecutionError, match="native RT backend"):
        eng.fit_head("PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) = 0 "
                     "FROM customers", [dt("2026-05-01")])


def test_finetune_rejects_frozen_head_substitution(churn_schema):
    """The full-fine-tune name must never silently fit only an output head."""
    from relativedb import Engine, ExecutionError
    from conftest import StubBackend
    eng = Engine(churn_schema, in_memory_wiring(churn_rows()),
                 model_backend=StubBackend())
    with pytest.raises(ExecutionError, match="full-backbone"):
        eng.finetune("unused", [])


def test_fk_columns_are_readable_by_aggregations():
    """`LIST_DISTINCT(orders.product_id)` aggregates a foreign key, which lives
    in Row.parents rather than Row.cells — the evaluator must still see it."""
    from relativedb.evaluate import eval_value
    from relativedb.relql.ast import (AggFunc, Aggregation, ColumnRef, TimeUnit,
                                      Window)
    rows = {"orders": [r for r in churn_rows()["orders"]]}
    agg = Aggregation(AggFunc.LIST_DISTINCT, ColumnRef("orders", "product_id"),
                      None, Window(-3650.0, 0.0, TimeUnit.DAYS))
    got = eval_value(agg, rows, {}, dt("2026-07-01"))
    assert set(got) == {"P1", "P2", "P3"}


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_fit_head_accepts_naive_anchors(churn_schema):
    """Row timestamps are UTC-aware; a naive training anchor must be coerced,
    not blow up comparing offset-naive to offset-aware datetimes."""
    _lib_or_skip()
    _metal_or_skip()
    from datetime import datetime as _dt
    eng = _ft_engine(churn_schema)
    head = eng.fit_head(
        "PREDICT LIST_DISTINCT(orders.product_id) "
        "OVER (30 DAYS FOLLOWING RANK TOP 2) FROM customers",
        [_dt(2026, 5, 1), _dt(2026, 6, 1)],      # naive on purpose
        epochs=20, learning_rate=1e-2)
    assert head.n_examples > 0


def test_fit_head_rejects_regression_adapter(churn_schema):
    from relativedb import ExecutionError
    q = "PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) FROM customers"
    eng = _ft_engine(churn_schema)
    with pytest.raises(ExecutionError, match="full-backbone fine-tuning"):
        eng.fit_head(q, [dt("2026-04-01")])


def test_physical_forward_batch_is_bounded_without_reordering():
    backend = RtNativeBackend(batch_size=2)
    calls = []

    def fake_forward(seqs, model_uri, *, output="target_scores"):
        calls.append(list(seqs))
        return np.asarray(seqs, np.float32)

    backend._forward = fake_forward
    got = backend._forward_batched([0, 1, 2, 3, 4], "unused://model")
    assert [len(chunk) for chunk in calls] == [2, 2, 1]
    assert got.tolist() == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# README quickstart parity. release_smoke.py runs the same shape against an
# INSTALLED WHEEL; these run it in-tree so a regression is caught before a
# wheel is ever built. Assertions are shape/finiteness only — zero-shot values
# on a four-row toy graph are model opinion, not artifact correctness.
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_readme_quickstart_not_exists_with_params(churn_schema):
    """The literal quickstart: NOT EXISTS(...) OVER (90 DAYS FOLLOWING) with a
    bound :ids cohort, driven through the documented top-level import surface
    (`from relativedb import RtNativeBackend`, not relativedb.rt_native).

    The rest of the suite only exercises the equivalent
    `COUNT(orders.*) ... = 0` spelling; NOT EXISTS is what users copy.
    """
    require_text_embedder()
    require_native()
    require_checkpoint("classification")
    import relativedb
    from relativedb import Engine, ExecutionInput

    engine = Engine(churn_schema, in_memory_wiring(churn_rows()),
                    model_backend=relativedb.RtNativeBackend(schema=churn_schema))
    res = engine.execute(ExecutionInput(
        query="PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
              "FROM customers WHERE customers.customer_id IN :ids",
        params={"ids": ["C1", "C7"]},
        anchor_time=dt("2026-07-01")))

    assert res.task_type.name == "BINARY_CLASSIFICATION"
    assert res.model_uri == "hf://stanford-star/rt-j/classification"
    probs = {p.id: p.probability for p in res.predictions}
    assert set(probs) == {"C1", "C7"}            # :ids narrows the cohort
    for p in probs.values():
        assert p is not None and 0.0 < p < 1.0 and math.isfinite(p)


@pytest.mark.integration
@pytest.mark.native
@pytest.mark.checkpoint
def test_regression_checkpoint_end_to_end(churn_schema):
    """The rt-j *regression* weights, end to end through the engine.

    Everywhere else the regression checkpoint is only fed raw golden arrays
    through ctypes, or routed to by a stub backend — the two halves are never
    joined. A README example (`PREDICT SUM(transactions.price) OVER (90 DAYS
    FOLLOWING)`) depends on exactly this path.
    """
    require_text_embedder()
    require_native()
    require_checkpoint("regression")
    from relativedb import ContextPolicy, Engine, ExecutionInput

    engine = Engine(churn_schema, in_memory_wiring(churn_rows()),
                    context_policy=ContextPolicy(cohort_size=0),
                    model_backend=RtNativeBackend(schema=churn_schema))
    res = engine.execute(ExecutionInput(
        query="PREDICT SUM(orders.qty) OVER (90 DAYS FOLLOWING) FROM customers",
        anchor_time=dt("2026-07-01")))

    assert res.task_type.name == "REGRESSION"
    assert res.model_uri == "hf://stanford-star/rt-j/regression"
    values = {p.id: p.value for p in res.predictions}
    assert set(values) == {"C1", "C7", "C9"}
    for v in values.values():
        assert v is not None and math.isfinite(v)
    assert values["C1"] is not None and len(set(values.values())) > 1
