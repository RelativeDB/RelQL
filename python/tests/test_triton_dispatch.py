"""Backend dispatch: Triton serves CUDA target scores, torch serves the rest."""

import numpy as np
import torch

from relativedb.rt.models import RT_DEVICE_CUDA
from relativedb.rt.scorer import RelationalScorer


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


def bare_scorer(inference_backend="auto"):
    scorer = object.__new__(RelationalScorer)
    scorer.cuda_backend = "triton"
    scorer.n_threads = 0
    scorer.device = RT_DEVICE_CUDA
    scorer.inference_backend = inference_backend
    scorer.onnx_model_path = None
    scorer._resolve_device = lambda: RT_DEVICE_CUDA
    scorer._materialize = lambda batch: materialized()
    return scorer


def test_cuda_target_scores_use_triton_by_default():
    scorer = bare_scorer()
    captured = {}
    backends = []

    class Model:
        def forward(self, batch, output):
            captured.update(batch.as_dict())
            assert output == "target_scores"
            return type("Result", (), {"scores": torch.tensor([0.25])})()

    def model_for(uri, backend, device):
        backends.append(backend)
        return Model()

    scorer._relational_model_for = model_for

    result = scorer.forward(object(), model_uri="/checkpoint")

    assert backends == ["triton"]
    assert result.scores.tolist() == [0.25]
    assert captured["f2p_nbr_idxs"].shape == (1, 2, 5)
    assert captured["is_targets"].tolist() == [[True, False]]
    assert captured["text_values"].shape == (1, 2, 384)


def test_explicit_torch_backend_uses_shared_relational_runtime():
    scorer = bare_scorer(inference_backend="torch")
    backends = []

    class Model:
        def forward(self, batch, output):
            assert output == "target_scores"
            return type("Result", (), {"scores": torch.tensor([0.75])})()

    def model_for(uri, backend, device):
        backends.append(backend)
        return Model()

    scorer._relational_model_for = model_for

    result = scorer.forward(object(), model_uri="/checkpoint")

    assert backends == ["torch"]
    assert result.scores.tolist() == [0.75]


def test_rich_outputs_fall_back_to_torch_on_cuda():
    scorer = bare_scorer()
    backends = []

    class Model:
        def forward(self, batch, output):
            assert output == "target_features"
            return type("Result", (), {"features": torch.zeros((1, 512))})()

    def model_for(uri, backend, device):
        backends.append(backend)
        return Model()

    scorer._relational_model_for = model_for

    result = scorer.forward(object(), model_uri="/checkpoint",
                            output="target_features")

    assert backends == ["torch"]
    assert result.features.shape == (1, 512)
