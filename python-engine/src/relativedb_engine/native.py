"""ctypes binding to the C ABI in ``cpp/src/rt_c.h`` (``rt_model_load`` /
``rt_forward`` / fine-tuning), plus checkpoint URI resolution.

The library is lazy-loaded from ``RELATIVEDB_RT_LIB``, the in-package bundled
copy, or the sibling ``cpp/build`` tree of a monorepo checkout; a clear
:class:`RtNativeUnavailableError` is raised when missing.
"""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from relativedb.model import NormalizationMode
from relativedb.scoring import D_MODEL, D_TEXT, MAX_F2P, ColumnStats

__all__ = ["RtNativeUnavailableError", "RtNativeError", "RtLib", "RtModel",
           "FineTunedCheckpoint", "load_lib", "resolve_model_path",
           "RT_DEVICE_CPU", "RT_DEVICE_MPS", "RT_DEVICE_CUDA",
           "FT_BINARY", "FT_REGRESSION", "FT_MULTICLASS", "FT_RANKING"]

# Compute devices (rt_c.h)
RT_DEVICE_CPU, RT_DEVICE_MPS, RT_DEVICE_CUDA = 0, 1, 2

# Fine-tune task codes (rt_c.h) — re-exported from relativedb.scoring so the
# wire values live in exactly one place.
from relativedb.scoring import (FT_BINARY, FT_MULTICLASS,  # noqa: E402
                                FT_RANKING, FT_REGRESSION)


class RtNativeUnavailableError(RuntimeError):
    """librt_c (or a runtime dependency) could not be located/loaded."""


class RtNativeError(RuntimeError):
    """An error reported by the native RT engine."""


def _lib_filename() -> str:
    if sys.platform == "darwin":
        return "librt_c.dylib"
    if sys.platform.startswith("win"):
        return "rt_c.dll"
    return "librt_c.so"


def _warn_if_stale_bundled(chosen, candidates) -> None:
    """A bundled library that shadows a fresher monorepo build is silent poison.

    The in-package copy exists so a wheel carries its engine, and it is
    searched first for that reason. In a monorepo checkout that ordering means
    a stale artifact wins over cpp/build -- so a whole session of C++ work can
    be rebuilt, tested and benchmarked while none of it is ever loaded. That
    happened: every C++ change on 2026-07-24 was invisible to the benchmarks
    because a dylib built at 12:10 shadowed one built at 17:21.
    """
    import warnings
    try:
        picked = Path(chosen)
        if "cpp/build" in str(picked):
            return                                # already the build tree
        for cand in candidates:
            cand = Path(cand)
            if "cpp/build" not in str(cand) or not cand.exists():
                continue
            if cand.stat().st_mtime > picked.stat().st_mtime:
                warnings.warn(
                    f"loaded the bundled {picked.name} ({picked}) but {cand} "
                    f"is NEWER: the bundled copy shadows it, so a fresh C++ "
                    f"build is not what is running. Delete the bundled copy, "
                    f"or set RELATIVEDB_RT_LIB to the build you mean.",
                    RuntimeWarning, stacklevel=3)
                return
    except OSError:
        return


def _candidate_paths() -> list[str]:
    cands = []
    env = os.environ.get("RELATIVEDB_RT_LIB")
    if env:
        cands.append(env)
    fname = _lib_filename()
    here = os.path.dirname(os.path.abspath(__file__))
    # in-package drop-in, then the sibling C++ build tree of the monorepo
    cands.append(os.path.join(here, fname))
    cands.append(os.path.abspath(os.path.join(
        here, "..", "..", "..", "cpp", "build", fname)))
    return cands


class RtLib:
    """A loaded librt_c with bound signatures (see ``cpp/src/rt_c.h``)."""

    def __init__(self, cdll: ctypes.CDLL, path: str):
        self._lib = cdll
        self.path = path
        lib = self._lib
        lib.rt_model_load.restype = ctypes.c_void_p
        lib.rt_model_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.c_size_t]
        lib.rt_model_free.restype = None
        lib.rt_model_free.argtypes = [ctypes.c_void_p]
        lib.rt_model_num_params.restype = ctypes.c_int64
        lib.rt_model_num_params.argtypes = [ctypes.c_void_p]
        f64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
        i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
        f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
        lib.rt_forward.restype = ctypes.c_int
        lib.rt_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p,          # node, f2p, col, table
            u8p, f64p, u8p,                  # is_padding, sem_types, is_target
            f32p, f32p, f32p, f32p, f32p,    # number, datetime, boolean, text, col_name
            ctypes.c_int32, f32p,            # n_threads, out_target_scores
            ctypes.c_char_p, ctypes.c_size_t]
        # rt_forward_ex: rt_forward + a trailing B*384 out_target_text buffer
        # (the dec_dict.text head output at each row's target cell).
        lib.rt_forward_ex.restype = ctypes.c_int
        lib.rt_forward_ex.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p,
            u8p, f64p, u8p,
            f32p, f32p, f32p, f32p, f32p,
            ctypes.c_int32, f32p, f32p,      # n_threads, out_scores, out_target_text
            ctypes.c_char_p, ctypes.c_size_t]
        # rt_forward on an explicit device. rt_forward is the CPU entry point;
        # without this binding every score and every encode ran on CPU while
        # Metal sat idle, which is most of the per-row inference cost.
        lib.rt_forward_device.restype = ctypes.c_int
        lib.rt_forward_device.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p,
            u8p, f64p, u8p,
            f32p, f32p, f32p, f32p, f32p,
            ctypes.c_int32, ctypes.c_int32,  # n_threads, device
            f32p,                            # out_target_scores
            ctypes.c_char_p, ctypes.c_size_t]
        # Per-token number-head output (multi-target scoring); absent from
        # older libraries, in which case shared-context scoring is refused.
        if hasattr(lib, "rt_forward_tokens_device"):
            lib.rt_forward_tokens_device.restype = ctypes.c_int
            lib.rt_forward_tokens_device.argtypes = [
                ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
                f64p, f64p, f64p, f64p,
                u8p, f64p, u8p,
                f32p, f32p, f32p, f32p, f32p,
                ctypes.c_int32, ctypes.c_int32,
                f32p,                        # out_token_scores [B*S]
                ctypes.c_char_p, ctypes.c_size_t]

        # ---- frozen-backbone fine-tuning (see rt_c.h) ----------------------
        lib.rt_encode_targets_device.restype = ctypes.c_int
        lib.rt_encode_targets_device.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p,
            u8p, f64p, u8p,
            f32p, f32p, f32p, f32p, f32p,
            ctypes.c_int32, ctypes.c_int32,
            f32p,                            # out_target_features [B, 512]
            ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_create.restype = ctypes.c_void_p
        lib.rt_finetune_head_create.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_void_p,                 # class_embeddings or NULL
            ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_load.restype = ctypes.c_void_p
        lib.rt_finetune_head_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                              ctypes.c_size_t]
        lib.rt_finetune_head_free.restype = None
        lib.rt_finetune_head_free.argtypes = [ctypes.c_void_p]
        lib.rt_finetune_head_save.restype = ctypes.c_int
        lib.rt_finetune_head_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                              ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_reparameterize_standardized.restype = ctypes.c_int
        lib.rt_finetune_head_reparameterize_standardized.argtypes = [
            ctypes.c_void_p, f32p, f32p, ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_fit_metal.restype = ctypes.c_int
        lib.rt_finetune_head_fit_metal.argtypes = [
            ctypes.c_void_p, ctypes.c_int32,
            f32p, f32p,                      # features [N,512], labels [N]
            i32p, ctypes.c_int32,            # group_offsets, n_groups
            ctypes.c_int32, ctypes.c_float, ctypes.c_float,   # epochs, lr, wd
            ctypes.POINTER(ctypes.c_float),  # out_initial_loss
            ctypes.POINTER(ctypes.c_float),  # out_final_loss
            ctypes.POINTER(ctypes.c_double), # out_seconds
            ctypes.c_char_p, ctypes.c_size_t]
        if hasattr(lib, "rt_finetune_head_fit_device"):
            lib.rt_finetune_head_fit_device.restype = ctypes.c_int
            lib.rt_finetune_head_fit_device.argtypes = [
                ctypes.c_void_p, ctypes.c_int32,
                f32p, f32p,
                i32p, ctypes.c_int32,
                ctypes.c_int32, ctypes.c_float, ctypes.c_float,
                ctypes.c_int32,              # device
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_predict.restype = ctypes.c_int
        lib.rt_finetune_head_predict.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, f32p, f32p,
            ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_finetune_head_outputs.restype = ctypes.c_int32
        lib.rt_finetune_head_outputs.argtypes = [ctypes.c_void_p]
        lib.rt_finetune_head_task.restype = ctypes.c_int32
        lib.rt_finetune_head_task.argtypes = [ctypes.c_void_p]
        lib.rt_model_finetune_step_metal.restype = ctypes.c_int
        lib.rt_model_finetune_step_metal.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p, u8p, f64p, u8p,
            f32p, f32p, f32p, f32p, f32p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_model_finetune_microbatch_metal.restype = ctypes.c_int
        lib.rt_model_finetune_microbatch_metal.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p, u8p, f64p, u8p,
            f32p, f32p, f32p, f32p, f32p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_model_finetune_optimizer_save.restype = ctypes.c_int
        lib.rt_model_finetune_optimizer_save.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_model_finetune_optimizer_load.restype = ctypes.c_int
        lib.rt_model_finetune_optimizer_load.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_model_finetune_gradient_check_metal.restype = ctypes.c_int
        lib.rt_model_finetune_gradient_check_metal.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
            f64p, f64p, f64p, f64p, u8p, f64p, u8p,
            f32p, f32p, f32p, f32p, f32p, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_model_save.restype = ctypes.c_int
        lib.rt_model_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.c_char_p, ctypes.c_size_t]
        lib.rt_model_finetune_reset_optimizer.restype = None
        lib.rt_model_finetune_reset_optimizer.argtypes = [ctypes.c_void_p]
        lib.rt_device_available.restype = ctypes.c_int
        lib.rt_device_available.argtypes = [ctypes.c_int32]
        lib.rt_full_finetune_available.restype = ctypes.c_int
        lib.rt_full_finetune_available.argtypes = []

    def device_available(self, device: int) -> bool:
        return bool(self._lib.rt_device_available(device))

    def full_finetune_available(self) -> bool:
        """Whether full-model fine-tuning can run on this machine's GPU.

        Stricter than ``device_available(RT_DEVICE_MPS)``. Inference runs on a
        paravirtualised GPU; the full-train kernels need the Metal 3 feature
        set and would return NaN without it, so they refuse instead.
        """
        return bool(self._lib.rt_full_finetune_available())

    def load_model(self, safetensors_path: str) -> "RtModel":
        err = ctypes.create_string_buffer(512)
        handle = self._lib.rt_model_load(
            safetensors_path.encode("utf-8"), err, len(err))
        if not handle:
            raise RtNativeError(
                f"rt_model_load({safetensors_path!r}) failed: "
                f"{err.value.decode('utf-8', 'replace')}")
        return RtModel(self, handle, safetensors_path)


@dataclass(frozen=True)
class FineTunedCheckpoint:
    """Complete RT-J checkpoint produced by native GPU fine-tuning."""

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


class RtModel:
    """A loaded RT-J checkpoint living in the native engine."""

    def __init__(self, lib: RtLib, handle: int, path: str):
        self._native = lib
        self._handle = handle
        self.path = path

    @property
    def num_params(self) -> int:
        return int(self._native._lib.rt_model_num_params(self._handle))

    def save(self, path: str) -> None:
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_model_save(
            self._handle, os.fspath(path).encode("utf-8"), err, len(err))
        if rc:
            raise RtNativeError(
                f"rt_model_save failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")

    def reset_finetune_optimizer(self) -> None:
        self._native._lib.rt_model_finetune_reset_optimizer(self._handle)

    def save_finetune_optimizer(self, path: str | os.PathLike) -> None:
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_model_finetune_optimizer_save(
            self._handle, os.fspath(path).encode("utf-8"), err, len(err))
        if rc:
            raise RtNativeError(
                f"optimizer save failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")

    def load_finetune_optimizer(self, path: str | os.PathLike) -> None:
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_model_finetune_optimizer_load(
            self._handle, os.fspath(path).encode("utf-8"), err, len(err))
        if rc:
            raise RtNativeError(
                f"optimizer load failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")

    def finetune_step(self, *, node_idxs, f2p, col_idxs, table_idxs,
                      is_padding, sem_types, is_target, number_v,
                      datetime_v, boolean_v, text_v, col_name_v,
                      learning_rate: float = 1e-5,
                      weight_decay: float = 1e-2,
                      grad_clip_norm: float = 1.0,
                      apply_update: bool = True) -> dict[str, float | int | bool]:
        """Accumulate one native GPU microbatch and optionally update."""
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        loss = ctypes.c_float()
        grad = ctypes.c_float()
        step = ctypes.c_uint64()
        accumulated = ctypes.c_uint32()
        updated = ctypes.c_int32()
        seconds = ctypes.c_double()
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_model_finetune_microbatch_metal(
            self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
            is_padding, sem_types, is_target, number_v, datetime_v,
            boolean_v, text_v, col_name_v, float(learning_rate),
            float(weight_decay), float(grad_clip_norm), int(apply_update),
            ctypes.byref(loss), ctypes.byref(grad), ctypes.byref(step),
            ctypes.byref(accumulated), ctypes.byref(updated), ctypes.byref(seconds),
            err, len(err))
        if rc:
            raise RtNativeError(
                f"native full-model step failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return {"loss": float(loss.value), "grad_norm": float(grad.value),
                "step": int(step.value), "seconds": float(seconds.value),
                "accumulated_microbatches": int(accumulated.value),
                "updated": bool(updated.value)}

    def gradient_check(self, *, node_idxs, f2p, col_idxs, table_idxs,
                       is_padding, sem_types, is_target, number_v,
                       datetime_v, boolean_v, text_v, col_name_v,
                       epsilon: float = 1e-3) -> dict[str, float | int]:
        """Compare representative native gradients with finite differences."""
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        max_abs, max_rel = ctypes.c_float(), ctypes.c_float()
        checked = ctypes.c_int32()
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_model_finetune_gradient_check_metal(
            self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
            is_padding, sem_types, is_target, number_v, datetime_v,
            boolean_v, text_v, col_name_v, float(epsilon),
            ctypes.byref(max_abs), ctypes.byref(max_rel), ctypes.byref(checked),
            err, len(err))
        if rc:
            raise RtNativeError(
                f"native gradient check failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return {"max_absolute_error": float(max_abs.value),
                "max_relative_error": float(max_rel.value),
                "checked": int(checked.value)}

    @staticmethod
    def _prep(node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
              is_target, number_v, datetime_v, boolean_v, text_v, col_name_v):
        node_idxs = np.ascontiguousarray(node_idxs, np.int64)
        B, S = node_idxs.shape
        return (B, S, node_idxs,
                np.ascontiguousarray(f2p, np.int64).reshape(B, S, MAX_F2P),
                np.ascontiguousarray(col_idxs, np.int64).reshape(B, S),
                np.ascontiguousarray(table_idxs, np.int64).reshape(B, S),
                np.ascontiguousarray(is_padding, np.uint8).reshape(B, S),
                np.ascontiguousarray(sem_types, np.int64).reshape(B, S),
                np.ascontiguousarray(is_target, np.uint8).reshape(B, S),
                np.ascontiguousarray(number_v, np.float32).reshape(B, S),
                np.ascontiguousarray(datetime_v, np.float32).reshape(B, S),
                np.ascontiguousarray(boolean_v, np.float32).reshape(B, S),
                np.ascontiguousarray(text_v, np.float32).reshape(B, S, D_TEXT),
                np.ascontiguousarray(col_name_v, np.float32).reshape(B, S, D_TEXT))

    def forward(self, *, node_idxs, f2p, col_idxs, table_idxs, is_padding,
                sem_types, is_target, number_v, datetime_v, boolean_v,
                text_v, col_name_v, n_threads: int = 0,
                device: int = RT_DEVICE_CPU) -> np.ndarray:
        """Raw PRE-sort arrays in (see rt_c.h) -> per-row target score [B]."""
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        out = np.zeros(B, np.float32)
        err = ctypes.create_string_buffer(512)
        if device == RT_DEVICE_CPU:
            rc = self._native._lib.rt_forward(
                self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
                is_padding, sem_types, is_target, number_v, datetime_v,
                boolean_v, text_v, col_name_v, int(n_threads), out,
                err, len(err))
        else:
            rc = self._native._lib.rt_forward_device(
                self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
                is_padding, sem_types, is_target, number_v, datetime_v,
                boolean_v, text_v, col_name_v, int(n_threads), int(device),
                out, err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"rt_forward failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return out

    def forward_tokens(self, *, node_idxs, f2p, col_idxs, table_idxs,
                       is_padding, sem_types, is_target, number_v, datetime_v,
                       boolean_v, text_v, col_name_v, n_threads: int = 0,
                       device: int = RT_DEVICE_CPU) -> np.ndarray:
        """Number-head output for every token ``[B, S]`` in pre-sort order.

        Lets one sequence carry several masked targets, each read at its own
        cell (shared-context multi-target scoring)."""
        if not hasattr(self._native._lib, "rt_forward_tokens_device"):
            raise RtNativeUnavailableError(
                "librt_c is too old for shared-context scoring "
                "(rt_forward_tokens_device missing); rebuild the native library")
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        out = np.zeros(B * S, np.float32)
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_forward_tokens_device(
            self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
            is_padding, sem_types, is_target, number_v, datetime_v,
            boolean_v, text_v, col_name_v, int(n_threads), int(device),
            out, err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"rt_forward_tokens_device failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return out.reshape(B, S)

    def encode_targets(self, *, node_idxs, f2p, col_idxs, table_idxs,
                       is_padding, sem_types, is_target, number_v, datetime_v,
                       boolean_v, text_v, col_name_v, n_threads: int = 0,
                       device: int = RT_DEVICE_CPU) -> np.ndarray:
        """Frozen-backbone features: the final target-cell state ``[B, 512]``.

        This is what task-head fitting trains on — the transformer is not
        updated, so every example need only be encoded once."""
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        out = np.zeros(B * D_MODEL, np.float32)
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_encode_targets_device(
            self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
            is_padding, sem_types, is_target, number_v, datetime_v,
            boolean_v, text_v, col_name_v, int(n_threads), int(device),
            out, err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"rt_encode_targets_device failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return out.reshape(B, D_MODEL)

    def forward_ex(self, *, node_idxs, f2p, col_idxs, table_idxs, is_padding,
                   sem_types, is_target, number_v, datetime_v, boolean_v,
                   text_v, col_name_v, n_threads: int = 0
                   ) -> tuple[np.ndarray, np.ndarray]:
        """As :meth:`forward`, but also returns the TEXT decoder-head output at
        each row's target cell: ``(scores[B], target_text[B, 384])`` (rt_c.h
        ``rt_forward_ex``). ``target_text`` is NOT L2-normalized."""
        (B, S, node_idxs, f2p, col_idxs, table_idxs, is_padding, sem_types,
         is_target, number_v, datetime_v, boolean_v, text_v, col_name_v
         ) = self._prep(node_idxs, f2p, col_idxs, table_idxs, is_padding,
                        sem_types, is_target, number_v, datetime_v, boolean_v,
                        text_v, col_name_v)
        out = np.zeros(B, np.float32)
        out_text = np.zeros((B, D_TEXT), np.float32)
        err = ctypes.create_string_buffer(512)
        rc = self._native._lib.rt_forward_ex(
            self._handle, B, S, node_idxs, f2p, col_idxs, table_idxs,
            is_padding, sem_types, is_target, number_v, datetime_v,
            boolean_v, text_v, col_name_v, int(n_threads), out, out_text,
            err, len(err))
        if rc != 0:
            raise RtNativeError(
                f"rt_forward_ex failed ({rc}): "
                f"{err.value.decode('utf-8', 'replace')}")
        return out, out_text

    def close(self) -> None:
        if self._handle:
            self._native._lib.rt_model_free(self._handle)
            self._handle = 0

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


_cached_lib: Optional[RtLib] = None


def load_lib(path: Optional[str] = None) -> RtLib:
    """Lazy-load librt_c; raises :class:`RtNativeUnavailableError` listing the
    searched paths when it cannot be found."""
    global _cached_lib
    if _cached_lib is not None and path is None:
        return _cached_lib
    candidates = [path] if path else _candidate_paths()
    tried = []
    for cand in candidates:
        if not cand:
            continue
        tried.append(cand)
        if not os.path.exists(cand):
            continue
        try:
            lib = RtLib(ctypes.CDLL(cand), cand)
        except (OSError, AttributeError) as e:
            raise RtNativeUnavailableError(
                f"found {cand} but could not bind the rt_c ABI: {e}") from e
        if path is None:
            _cached_lib = lib
        _warn_if_stale_bundled(cand, candidates)
        return lib
    raise RtNativeUnavailableError(
        "librt_c was not found (build cpp/ with cmake, or set RELATIVEDB_RT_LIB "
        "to the built library). Searched: " + ", ".join(tried))


# ---------------------------------------------------------------------------
# checkpoint URI resolution
# ---------------------------------------------------------------------------

def _quantized_variant() -> str | None:
    """RELATIVEDB_RT_QUANTIZED: 1/true/q8 -> q8, q4 -> q4, f16 -> f16."""
    v = os.environ.get("RELATIVEDB_RT_QUANTIZED", "").lower()
    if v in ("1", "true", "q8"):
        return "q8"
    if v in ("q4", "f16"):
        return v
    return None


def _pick_model(dirpath: str) -> str:
    """dir -> model.<variant>.safetensors (from cpp/rt_quantize) when
    RELATIVEDB_RT_QUANTIZED selects a variant and it is present, else
    model.safetensors."""
    v = _quantized_variant()
    if v:
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
        raise RtNativeUnavailableError(
            f"directory {uri!r} has no model.safetensors")
    if uri.startswith("hf://"):
        rest = uri[len("hf://"):].strip("/")
        parts = rest.split("/")
        if len(parts) < 2:
            raise RtNativeUnavailableError(f"malformed hf:// URI: {uri!r}")
        repo_id = "/".join(parts[:2])
        sub = "/".join(parts[2:])
        filename = (sub + "/" if sub else "") + "model.safetensors"
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RtNativeUnavailableError(
                f"resolving {uri!r} requires huggingface_hub: "
                f"pip install huggingface_hub") from e
        try:  # cache-first: never hit the network when already downloaded
            path = hf_hub_download(repo_id, filename, local_files_only=True)
        except Exception:
            path = hf_hub_download(repo_id, filename)
        # a quantized sibling lives beside the snapshot file, not in the repo
        picked = _pick_model(os.path.dirname(path))
        return picked if os.path.isfile(picked) else path
    raise RtNativeUnavailableError(
        f"cannot resolve model uri {uri!r} (not a path, not hf://)")
