"""Checkpoint quantization tools owned by the RelativeDB engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from relational_transformers.checkpoints import resolve_checkpoint
from safetensors.torch import load_file, save_file

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


def quantize_tensor_fp8(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Convert matrix weights to native E4M3 FP8 and retain other tensors."""

    if not tensor.is_floating_point() or tensor.ndim != 2 or not name.endswith(".weight"):
        return tensor
    return tensor.float().clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def quantize_checkpoint_fp8(source: str | Path, destination: str | Path) -> Path:
    """Quantize one safetensors checkpoint to native E4M3 FP8 matrix weights."""

    source = Path(source)
    destination = Path(destination)
    if source.suffix != ".safetensors":
        raise ValueError("FP8 quantization requires a safetensors checkpoint")
    state = load_file(str(source), device="cpu")
    quantized = {name: quantize_tensor_fp8(name, tensor) for name, tensor in state.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        quantized,
        str(destination),
        metadata={"format": "pt", "quantization": "fp8_e4m3fn"},
    )
    return destination


def quantize_model_fp8(
    model_name_or_path: str | Path,
    output_directory: str | Path,
    *,
    revision: str | None = None,
    tasks: tuple[str, ...] = ("classification", "regression"),
) -> Path:
    """Resolve and quantize every requested task subfolder of a model."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        config, checkpoint = resolve_checkpoint(
            model_name_or_path,
            task=task,
            revision=revision,
        )
        task_directory = output_directory / task
        task_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_name = "model.fp8.safetensors"
        quantize_checkpoint_fp8(checkpoint, task_directory / checkpoint_name)
        output_config = {
            **config,
            "checkpoint_file": checkpoint_name,
            "quantization": {
                "format": "fp8_e4m3fn",
                "matrix_weights_only": True,
            },
        }
        (task_directory / "config.json").write_text(json.dumps(output_config, indent=2) + "\n")
    return output_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize a relational model to E4M3 FP8")
    parser.add_argument("model_name_or_path")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--revision")
    parser.add_argument(
        "--task",
        action="append",
        choices=("classification", "regression"),
        dest="tasks",
        help="quantize only this task (repeatable; defaults to both)",
    )
    args = parser.parse_args()
    quantize_model_fp8(
        args.model_name_or_path,
        args.output_directory,
        revision=args.revision,
        tasks=tuple(args.tasks) if args.tasks else ("classification", "regression"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
