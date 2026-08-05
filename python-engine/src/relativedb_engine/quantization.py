"""Checkpoint quantization, delegated to relational-transformers-utils.

The quantizers live in :mod:`relational_transformers_utils.quantization`,
which produces FP8, row-wise int8, and packed int4 checkpoints that the
relational-transformers loader accepts. This module keeps the engine's
FP8-flavored names and the ``relativedb-quantize-fp8`` entry point.
"""

from __future__ import annotations

from pathlib import Path

from relational_transformers_utils.quantization import (FP8_DTYPE, FP8_MAX,
                                                        main,
                                                        quantize_checkpoint,
                                                        quantize_model,
                                                        quantize_state,
                                                        quantize_tensor_fp8)

__all__ = [
    "FP8_DTYPE", "FP8_MAX", "main",
    "quantize_checkpoint", "quantize_checkpoint_fp8",
    "quantize_model", "quantize_model_fp8",
    "quantize_state", "quantize_tensor_fp8",
]


def quantize_checkpoint_fp8(source: str | Path, destination: str | Path) -> Path:
    """Quantize one safetensors checkpoint to native E4M3 FP8 matrix weights."""
    return quantize_checkpoint(source, destination, fmt="fp8")


def quantize_model_fp8(
    model_name_or_path: str | Path,
    output_directory: str | Path,
    *,
    revision: str | None = None,
    tasks: tuple[str, ...] = ("classification", "regression"),
) -> Path:
    """Resolve and quantize every requested task subfolder of an RT-J model."""
    return quantize_model(model_name_or_path, output_directory, fmt="fp8",
                          revision=revision, tasks=tasks)


if __name__ == "__main__":
    raise SystemExit(main())
