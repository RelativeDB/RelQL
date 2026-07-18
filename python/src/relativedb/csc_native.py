"""Binding to the shared C++ CSC adjacency (``csc_build`` / ``csc_children``
in ``librt_c``).

The time-bounded "latest <= anchor" children query is the CSC index hot path;
it lives once in ``cpp/src/csc.*`` and is shared by all three language
bindings. This module wraps the C ABI with ctypes and is used by
:mod:`relativedb.csc`, which keeps the Python-side id<->dense mapping and row
storage. ``librt_c`` is a hard dependency (:class:`NativeCscUnavailable` is
raised if it cannot be loaded). A conformance test asserts equivalence with a
brute-force reference.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Sequence

__all__ = ["native_available", "NativeCsc", "NativeCscUnavailable"]

_ERR = 1024


class NativeCscUnavailable(RuntimeError):
    pass


def _candidate_paths() -> list[Path]:
    env = os.environ.get("RELATIVEDB_RT_LIB")
    here = Path(__file__).resolve()
    # repo root: .../python/src/relativedb/csc_native.py -> parents[3]
    root = here.parents[3]
    names = ["librt_c.dylib", "librt_c.so", "librt_c.dll", "rt_c.dll"]
    out = [Path(env)] if env else []
    out += [root / "cpp" / "build" / n for n in names]
    return out


_lib: Optional[ctypes.CDLL] = None
_load_failed: Optional[str] = None


def _load() -> Optional[ctypes.CDLL]:
    global _lib, _load_failed
    if _lib is not None or _load_failed is not None:
        return _lib
    for p in _candidate_paths():
        if p and p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                lib.csc_build.restype = ctypes.c_void_p
                lib.csc_build.argtypes = [
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_char_p, ctypes.c_size_t,
                ]
                lib.csc_free.restype = None
                lib.csc_free.argtypes = [ctypes.c_void_p]
                lib.csc_children.restype = ctypes.c_int
                lib.csc_children.argtypes = [
                    ctypes.c_void_p, ctypes.c_int64, ctypes.c_double,
                    ctypes.c_int32, ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int32),
                    ctypes.c_char_p, ctypes.c_size_t,
                ]
                _lib = lib
                return _lib
            except (OSError, AttributeError) as e:
                _load_failed = f"{p}: {e}"
    if _load_failed is None:
        _load_failed = "librt_c not found (build cpp/ with cmake)"
    return None


def native_available() -> bool:
    return _load() is not None


class NativeCsc:
    """Wraps a native ``csc_index``. Build once from edge arrays, then answer
    many :meth:`children` queries. Frees the native handle on GC."""

    def __init__(self, n_parents: int,
                 edge_parent: Sequence[int], edge_child: Sequence[int],
                 edge_ts: Sequence[float]) -> None:
        lib = _load()
        if lib is None:
            raise NativeCscUnavailable(_load_failed or "librt_c unavailable")
        self._lib = lib
        self._handle: Optional[int] = None
        n_edges = len(edge_parent)
        if len(edge_child) != n_edges or len(edge_ts) != n_edges:
            raise ValueError("edge arrays must have equal length")
        ep = (ctypes.c_int64 * n_edges)(*edge_parent)
        ec = (ctypes.c_int64 * n_edges)(*edge_child)
        et = (ctypes.c_double * n_edges)(*edge_ts)
        err = ctypes.create_string_buffer(_ERR)
        handle = lib.csc_build(
            ctypes.c_int64(n_parents), ctypes.c_int64(n_edges),
            ep if n_edges else None, ec if n_edges else None,
            et if n_edges else None, err, _ERR)
        if not handle:
            raise NativeCscUnavailable(
                err.value.decode("utf-8", "replace") or "csc_build failed")
        self._handle = handle

    def children(self, parent_dense: int, anchor_ts: float,
                 limit: int) -> list[int]:
        """Up to ``limit`` dense child ids with ts <= ``anchor_ts``,
        newest-first. ``limit <= 0`` returns ``[]``."""
        if self._handle is None:
            raise NativeCscUnavailable("csc_index already freed")
        if limit <= 0:
            return []
        out = (ctypes.c_int64 * limit)()
        n = ctypes.c_int32(0)
        err = ctypes.create_string_buffer(_ERR)
        rc = self._lib.csc_children(
            ctypes.c_void_p(self._handle), ctypes.c_int64(parent_dense),
            ctypes.c_double(anchor_ts), ctypes.c_int32(limit), out,
            ctypes.byref(n), err, _ERR)
        if rc != 0:
            raise RuntimeError(
                err.value.decode("utf-8", "replace") or "csc_children failed")
        return [out[i] for i in range(n.value)]

    def __del__(self) -> None:
        h = getattr(self, "_handle", None)
        if h:
            try:
                self._lib.csc_free(ctypes.c_void_p(h))
            except Exception:
                pass
            self._handle = None
