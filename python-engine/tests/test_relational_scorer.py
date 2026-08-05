"""End-to-end coverage of the shared-runtime scorer and torch head fitting."""

import numpy as np
import pytest
import torch
from relational_transformers import RTJModel
from relational_transformers.checkpoints import save_checkpoint

from relativedb.model import NormalizationMode
from relativedb.relql.ast import TaskType
from relativedb_engine.backend import FineTunedHead, RtNativeBackend
from relativedb_engine.models import RT_DEVICE_CPU
from relativedb_engine.scorer import RelationalScorer, TextEncoder

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


def _backend_with_stub_embeddings(classes):
    backend = object.__new__(RtNativeBackend)
    scorer = RelationalScorer(device=RT_DEVICE_CPU, inference_backend="torch")
    rng = np.random.default_rng(1)
    table = {str(c): rng.normal(size=D_TEXT).astype(np.float32) for c in classes}
    scorer.embedder.install_precomputed(table, strict=True)
    scorer.embedder._cache_norm.update(
        {k: v / np.linalg.norm(v) for k, v in table.items()})
    backend.scorer = scorer
    backend.column_stats = None
    return backend


def test_multiclass_head_fits_saves_and_reloads(checkpoint, tmp_path):
    classes = ["yes", "no", "maybe"]
    backend = _backend_with_stub_embeddings(classes)

    rng = np.random.default_rng(3)
    labels = np.asarray([0, 1, 2] * 15, np.float32)
    features = rng.normal(size=(45, D_MODEL)).astype(np.float32)
    for k in range(3):
        features[labels == k, k] += 1.5

    model = backend.scorer._relational_model_for(
        str(checkpoint), "torch", RT_DEVICE_CPU)
    head = RtNativeBackend.fit_head(
        backend, model, TaskType.MULTICLASS_CLASSIFICATION,
        features, labels, np.zeros(1, np.int32), 0,
        epochs=120, classes=classes,
        normalization_mode=NormalizationMode.ZERO_SHOT)
    assert head.n_outputs == 3
    assert head.initial_loss >= head.final_loss

    path = tmp_path / "head.safetensors"
    head.save(str(path))
    reloaded = FineTunedHead.load(str(path))
    assert reloaded.classes == ("yes", "no", "maybe")
    logits = reloaded.predict(features)
    assert logits.shape == (45, 3)
    assert (logits.argmax(1) == labels).mean() > 0.8


def test_head_load_requires_sidecar(checkpoint, tmp_path):
    classes = ["a", "b"]
    backend = _backend_with_stub_embeddings(classes)
    rng = np.random.default_rng(7)
    labels = np.asarray([0, 1] * 10, np.float32)
    features = rng.normal(size=(20, D_MODEL)).astype(np.float32)
    model = backend.scorer._relational_model_for(
        str(checkpoint), "torch", RT_DEVICE_CPU)
    head = RtNativeBackend.fit_head(
        backend, model, TaskType.MULTICLASS_CLASSIFICATION,
        features, labels, np.zeros(1, np.int32), 0, epochs=5, classes=classes)
    path = tmp_path / "head.safetensors"
    head.save(str(path))
    (tmp_path / "head.safetensors.preproc.json").unlink()
    with pytest.raises(Exception, match="preproc"):
        FineTunedHead.load(str(path))


def test_ranking_head_learns_group_ordering(checkpoint):
    backend = _backend_with_stub_embeddings([])
    rng = np.random.default_rng(11)
    groups = 12
    per_group = 4
    features = rng.normal(size=(groups * per_group, D_MODEL)).astype(np.float32)
    labels = np.zeros(groups * per_group, np.float32)
    for g in range(groups):
        labels[g * per_group] = 1.0            # first candidate is relevant
        features[g * per_group] += 0.6
    offsets = np.arange(0, groups * per_group + 1, per_group, dtype=np.int32)

    model = backend.scorer._relational_model_for(
        str(checkpoint), "torch", RT_DEVICE_CPU)
    head = RtNativeBackend.fit_head(
        backend, model, TaskType.MULTILABEL_RANKING,
        features, labels, offsets, groups, epochs=80)
    logits = head.predict(features)[:, 0]
    winners = [int(np.argmax(logits[s:s + per_group])) == 0
               for s in range(0, groups * per_group, per_group)]
    assert sum(winners) >= groups - 2
