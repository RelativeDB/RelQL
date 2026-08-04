"""CUDA inference dispatch: Triton is primary, native CUDA is explicit legacy."""

import numpy as np
import torch

from relativedb_engine.native import RT_DEVICE_CUDA
from relativedb_engine.scorer import NativeScorer


def materialized():
    scalar = np.zeros((1, 2), np.float32)
    integer = np.zeros((1, 2), np.int64)
    return {
        "node_idxs": integer,
        "f2p": np.zeros((1, 2, 5), np.int64),
        "col_idxs": integer,
        "table_idxs": integer,
        "is_padding": np.zeros((1, 2), np.uint8),
        "sem_types": integer,
        "is_target": np.asarray([[1, 0]], np.uint8),
        "number_v": scalar,
        "datetime_v": scalar,
        "boolean_v": scalar,
        "text_v": np.zeros((1, 2, 384), np.float32),
        "col_name_v": np.zeros((1, 2, 384), np.float32),
    }


def bare_scorer(cuda_backend):
    scorer = object.__new__(NativeScorer)
    scorer.cuda_backend = cuda_backend
    scorer.n_threads = 0
    scorer.device = RT_DEVICE_CUDA
    scorer._resolve_device = lambda: RT_DEVICE_CUDA
    scorer._materialize = lambda batch: materialized()
    return scorer


def test_cuda_target_scores_use_triton_by_default():
    scorer = bare_scorer("triton")
    captured = {}

    class Model:
        def predict(self, raw):
            captured.update(raw)
            return np.asarray([0.25], np.float32)

    scorer._triton_model_for = lambda uri: Model()
    scorer._model_for = lambda uri: (_ for _ in ()).throw(
        AssertionError("native CUDA path must stay cold"))

    result = scorer.forward(object(), model_uri="/checkpoint")

    assert result.scores.tolist() == [0.25]
    assert captured["f2p_nbr_idxs"].shape == (1, 2, 5)
    assert captured["is_targets"].tolist() == [[1, 0]]


def test_native_cuda_requires_explicit_legacy_selection():
    scorer = bare_scorer("native")

    class Model:
        def forward(self, **kwargs):
            assert kwargs["device"] == RT_DEVICE_CUDA
            return np.asarray([-0.5], np.float32)

    scorer._model_for = lambda uri: Model()
    scorer._triton_model_for = lambda uri: (_ for _ in ()).throw(
        AssertionError("Triton must not run under explicit legacy selection"))

    result = scorer.forward(object(), model_uri="/checkpoint")

    assert result.scores.tolist() == [-0.5]


def test_explicit_torch_backend_uses_shared_relational_runtime():
    scorer = bare_scorer("native")
    scorer.inference_backend = "torch"
    captured = {}

    class Result:
        scores = torch.tensor([0.75])

    class Model:
        def forward(self, raw, output):
            captured.update(raw)
            assert output == "target_scores"
            return Result()

    scorer._relational_model_for = lambda uri, backend, device: Model()
    scorer._model_for = lambda uri: (_ for _ in ()).throw(
        AssertionError("native backend must stay cold"))

    result = scorer.forward(object(), model_uri="/checkpoint")

    assert result.scores.tolist() == [0.75]
    assert captured["text_values"].shape == (1, 2, 384)
