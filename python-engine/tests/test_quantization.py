from __future__ import annotations

import pytest
import torch
from relativedb_engine.quantization import quantize_checkpoint_fp8, quantize_tensor_fp8
from safetensors import safe_open
from safetensors.torch import save_file


def test_fp8_quantizes_matrix_weights_but_retains_vectors():
    matrix = torch.linspace(-1, 1, 32).reshape(4, 8)
    vector = torch.linspace(-1, 1, 4)

    assert quantize_tensor_fp8("projection.weight", matrix).dtype == torch.float8_e4m3fn
    assert quantize_tensor_fp8("projection.bias", vector) is vector


def test_fp8_checkpoint_records_format_and_stays_numerically_close(tmp_path):
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "model.fp8.safetensors"
    weight = torch.randn(8, 8)
    save_file({"projection.weight": weight, "projection.bias": torch.ones(8)}, source)

    quantize_checkpoint_fp8(source, destination)

    with safe_open(destination, framework="pt", device="cpu") as handle:
        assert handle.metadata()["quantization"] == "fp8_e4m3fn"
        actual = handle.get_tensor("projection.weight").float()
    torch.testing.assert_close(actual, weight, atol=0.07, rtol=0.07)


def test_fp8_requires_safetensors_input(tmp_path):
    with pytest.raises(ValueError, match="requires a safetensors"):
        quantize_checkpoint_fp8(tmp_path / "model.pt", tmp_path / "model.fp8.safetensors")
