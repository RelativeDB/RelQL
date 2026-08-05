"""Checkpoint URI resolution and the quantized-sibling convention."""

from __future__ import annotations

import numpy as np
import pytest
from safetensors.numpy import save_file

from relativedb.rt.scorer import (
    EngineUnavailableError,
    resolve_model_path,
    torch_device_name,
)


def _write(path, name="model.safetensors"):
    path.mkdir(parents=True, exist_ok=True)
    save_file({"w": np.zeros(2, np.float32)}, str(path / name))
    return path / name


def test_explicit_file_paths_are_used_as_given(tmp_path, monkeypatch):
    target = _write(tmp_path)
    monkeypatch.setenv("RELATIVEDB_RT_QUANTIZED", "q8")
    _write(tmp_path, "model.q8.safetensors")
    # explicit file: the env never redirects it
    assert resolve_model_path(str(target)) == str(target)


def test_directory_resolution_prefers_the_selected_sibling(tmp_path, monkeypatch):
    _write(tmp_path)
    monkeypatch.delenv("RELATIVEDB_RT_QUANTIZED", raising=False)
    assert resolve_model_path(str(tmp_path)).endswith("model.safetensors")

    _write(tmp_path, "model.int8.safetensors")
    monkeypatch.setenv("RELATIVEDB_RT_QUANTIZED", "q8")
    # both spellings select the int8 sibling; rt-quantize writes .int8
    assert resolve_model_path(str(tmp_path)).endswith("model.int8.safetensors")
    monkeypatch.setenv("RELATIVEDB_RT_QUANTIZED", "int8")
    assert resolve_model_path(str(tmp_path)).endswith("model.int8.safetensors")

    monkeypatch.setenv("RELATIVEDB_RT_QUANTIZED", "q4")
    # the selected variant is absent, so resolution falls back to the base
    assert resolve_model_path(str(tmp_path)).endswith("model.safetensors")


def test_unresolvable_uris_fail_precisely(tmp_path):
    with pytest.raises(EngineUnavailableError, match="no model.safetensors"):
        resolve_model_path(str(tmp_path))
    with pytest.raises(EngineUnavailableError, match="malformed hf://"):
        resolve_model_path("hf://only-one-part")
    with pytest.raises(EngineUnavailableError, match="cannot resolve"):
        resolve_model_path("s3://nope/nope")


def test_device_constants_map_to_torch_names():
    assert [torch_device_name(d) for d in (0, 1, 2)] == ["cpu", "mps", "cuda"]
    assert torch_device_name(None) is None
    with pytest.raises(Exception, match="unknown device"):
        torch_device_name(7)
