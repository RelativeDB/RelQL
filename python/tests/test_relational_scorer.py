"""End-to-end coverage of the shared-runtime scorer and torch head fitting."""

import numpy as np
import pytest
import torch
from relational_transformers import RTJModel
from relational_transformers.checkpoints import save_checkpoint

from relativedb.rt.scorer import RT_DEVICE_CPU
from relativedb.rt.scorer import RelationalScorer, TextEncoder

D_TEXT = 384
D_MODEL = 512


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(5)
    model = RTJModel(num_blocks=1, d_model=D_MODEL, d_text=D_TEXT,
                     num_heads=2, d_ff=32)
    directory = tmp_path_factory.mktemp("tiny-rt")
    save_checkpoint(model, directory, {"model": {
        "num_blocks": 1, "d_model": D_MODEL, "d_text": D_TEXT,
        "num_heads": 2, "d_ff": 32,
    }})
    return directory


def materialized(batch_size=1, cells=3):
    rng = np.random.default_rng(9)
    integer = np.zeros((batch_size, cells), np.int64)
    is_target = np.zeros((batch_size, cells), np.uint8)
    is_target[:, 0] = 1
    return {
        "node_idxs": integer,
        "f2p": np.full((batch_size, cells, 5), -1, np.int64),
        "col_idxs": np.tile(np.arange(cells), (batch_size, 1)),
        "table_idxs": integer,
        "is_padding": np.zeros((batch_size, cells), np.uint8),
        "sem_types": integer,
        "is_target": is_target,
        "number_v": rng.normal(size=(batch_size, cells)).astype(np.float32),
        "datetime_v": np.zeros((batch_size, cells), np.float32),
        "boolean_v": np.zeros((batch_size, cells), np.float32),
        "text_v": rng.normal(size=(batch_size, cells, D_TEXT)).astype(np.float32),
        "col_name_v": rng.normal(size=(batch_size, cells, D_TEXT)).astype(np.float32),
    }


def cpu_scorer():
    scorer = RelationalScorer(device=RT_DEVICE_CPU, inference_backend="torch")
    scorer._materialize = lambda batch: materialized()
    return scorer


def test_every_output_kind_runs_through_torch(checkpoint):
    scorer = cpu_scorer()
    uri = str(checkpoint)
    scores = scorer.forward(object(), model_uri=uri).scores
    tokens = scorer.forward(object(), model_uri=uri, output="token_scores").scores
    features = scorer.forward(object(), model_uri=uri,
                              output="target_features").features
    both = scorer.forward(object(), model_uri=uri,
                          output="target_scores_and_text")
    assert scores.shape == (1,)
    assert tokens.shape == (1, 3)
    assert features.shape == (1, D_MODEL)
    assert both.scores.shape == (1,)
    assert both.target_text.shape == (1, D_TEXT)
    assert np.isfinite(scores).all() and np.isfinite(features).all()


def test_models_are_cached_per_path_backend_device(checkpoint):
    scorer = cpu_scorer()
    uri = str(checkpoint)
    scorer.forward(object(), model_uri=uri)
    scorer.forward(object(), model_uri=uri, output="token_scores")
    assert len(scorer._relational_models) == 1


def test_precomputed_embeddings_are_strict():
    encoder = TextEncoder()
    encoder.install_precomputed({"known": np.ones(D_TEXT, np.float32)})
    assert encoder.encode_one("known").shape == (D_TEXT,)
    with pytest.raises(Exception, match="precomputed embedding table"):
        encoder.encode(["unknown"])
