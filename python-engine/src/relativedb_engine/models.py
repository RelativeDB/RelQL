"""Checkpoint URI resolution and shared engine types.

The model runtime is the ``relational-transformers`` package; this module owns
the pieces that survived the native engine's removal: the ``hf://`` URI
scheme, the quantized-sibling file convention, the device constants, and the
fine-tuning result record.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from relativedb.model import NormalizationMode
from relativedb.scoring import ColumnStats

__all__ = [
    "RT_DEVICE_CPU", "RT_DEVICE_MPS", "RT_DEVICE_CUDA",
    "EngineError", "EngineUnavailableError",
    "RtNativeError", "RtNativeUnavailableError",
    "FineTunedCheckpoint", "resolve_model_path", "torch_device_name",
]

RT_DEVICE_CPU, RT_DEVICE_MPS, RT_DEVICE_CUDA = 0, 1, 2

_TORCH_DEVICE_NAMES = {
    RT_DEVICE_CPU: "cpu",
    RT_DEVICE_MPS: "mps",
    RT_DEVICE_CUDA: "cuda",
}


def torch_device_name(device: Optional[int]) -> Optional[str]:
    """Map a legacy integer device constant to a torch device string."""
    if device is None:
        return None
    try:
        return _TORCH_DEVICE_NAMES[int(device)]
    except (KeyError, ValueError) as e:
        raise EngineError(f"unknown device constant {device!r}") from e


class EngineError(RuntimeError):
    """A model-engine call failed."""


class EngineUnavailableError(EngineError):
    """A required model artifact or dependency is missing."""


# Historical names, kept so existing handlers keep catching.
RtNativeError = EngineError
RtNativeUnavailableError = EngineUnavailableError


@dataclass
class FineTunedCheckpoint:
    """Complete RT-J checkpoint produced by full-model fine-tuning."""

    path: Path
    losses: tuple[float, ...]
    grad_norms: tuple[float, ...]
    seconds: float
    examples: int
    steps: int
    column_stats: Optional[ColumnStats] = None
    normalization_mode: NormalizationMode = NormalizationMode.REFERENCE

    @property
    def model_uri(self) -> str:
        return str(self.path)


def _quantized_variant() -> tuple[str, ...]:
    """RELATIVEDB_RT_QUANTIZED selects a quantized sibling checkpoint.

    ``1``/``true``/``q8``/``int8`` prefer the int8 sibling, ``q4``/``int4``
    the packed int4 one, ``f16``/``fp16`` the half-precision one, and ``fp8``
    the float8 one. Both the rt-quantize spellings (``model.int8.safetensors``)
    and the historical ones (``model.q8.safetensors``) are accepted.
    """
    v = os.environ.get("RELATIVEDB_RT_QUANTIZED", "").lower()
    if v in ("1", "true", "q8", "int8"):
        return ("int8", "q8")
    if v in ("q4", "int4"):
        return ("int4", "q4")
    if v in ("f16", "fp16"):
        return ("fp16", "f16")
    if v == "fp8":
        return ("fp8",)
    return ()


def _pick_model(dirpath: str) -> str:
    """dir -> model.<variant>.safetensors when RELATIVEDB_RT_QUANTIZED selects
    a variant and it is present, else model.safetensors."""
    for v in _quantized_variant():
        q = os.path.join(dirpath, f"model.{v}.safetensors")
        if os.path.isfile(q):
            return q
    return os.path.join(dirpath, "model.safetensors")


def resolve_model_path(uri: str) -> str:
    """Resolve a checkpoint URI to a local ``model.safetensors`` path.

    Accepts a filesystem path (file or directory containing
    ``model.safetensors``) or ``hf://org/repo/subdir`` (resolved through
    huggingface_hub, cache-first). With env ``RELATIVEDB_RT_QUANTIZED=1``,
    an int8 ``model.q8.safetensors`` sibling is preferred when present
    (explicit file paths are always used as given)."""
    if os.path.isfile(uri):
        return uri
    if os.path.isdir(uri):
        p = _pick_model(uri)
        if os.path.isfile(p):
            return p
        raise EngineUnavailableError(
            f"directory {uri!r} has no model.safetensors")
    if uri.startswith("hf://"):
        rest = uri[len("hf://"):].strip("/")
        parts = rest.split("/")
        if len(parts) < 2:
            raise EngineUnavailableError(f"malformed hf:// URI: {uri!r}")
        repo_id = "/".join(parts[:2])
        sub = "/".join(parts[2:])
        prefix = (sub + "/" if sub else "")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise EngineUnavailableError(
                f"resolving {uri!r} requires huggingface_hub: "
                f"pip install huggingface_hub") from e

        def download(filename: str) -> str:
            try:  # cache-first: never hit the network when already downloaded
                return hf_hub_download(repo_id, filename, local_files_only=True)
            except Exception:
                return hf_hub_download(repo_id, filename)

        checkpoint_file = "model.safetensors"
        try:
            config_path = download(prefix + "config.json")
            with open(config_path) as config_file:
                checkpoint_file = json.load(config_file).get(
                    "checkpoint_file", checkpoint_file)
        except Exception:
            # Older repositories did not publish a config; retain the legacy
            # model.safetensors convention for them.
            pass
        path = download(prefix + checkpoint_file)
        # a quantized sibling lives beside the snapshot file, not in the repo
        picked = _pick_model(os.path.dirname(path))
        return picked if os.path.isfile(picked) else path
    raise EngineUnavailableError(
        f"cannot resolve model uri {uri!r} (not a path, not hf://)")
