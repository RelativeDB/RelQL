"""Binding to the native array-backed context assembler (``rel_graph_*``).

The engine's only real demand on storage is context population: hand back the
few thousand rows each prediction needs. Doing that from Python means building
a :class:`~relativedb.Row` for every database row up front, which scales with
the database rather than the context -- rel-event took minutes per task that
way and seconds when the graph stayed as arrays.

This module owns none of that logic. It hands flat numpy arrays to
``cpp/src/graph.cpp`` and gets back the ordered node ids a context emits; the
caller materializes only those.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["NativeGraph", "NativeGraphUnavailable", "ContextTruncated",
           "native_available"]

_ERR = 1024


class NativeGraphUnavailable(RuntimeError):
    pass


class ContextTruncated(RuntimeError):
    """The emitted-node buffer bound, so part of the context was dropped.

    Nodes and cells are different quantities -- a row whose feature columns are
    all null costs zero cells but still occupies a node slot -- so a buffer
    sized from the cell budget can bind on a real graph. That silently drops
    the tail of a context before the model ever sees it, which is invisible in
    every metric except accuracy. It is an error, never a truncation.
    """


def _candidate_paths() -> list[Path]:
    env = os.environ.get("RELATIVEDB_RT_LIB")
    here = Path(__file__).resolve()
    names = ["librt_c.dylib", "librt_c.so", "librt_c.dll", "rt_c.dll"]
    out = [Path(env)] if env else []
    out += [here.parent / n for n in names]
    if len(here.parents) > 3:
        out += [here.parents[3] / "cpp" / "build" / n for n in names]
    return out


_lib: Optional[ctypes.CDLL] = None
_load_failed: Optional[str] = None


def _warn_if_stale_bundled(chosen, candidates) -> None:
    """A bundled library that shadows a fresher monorepo build is silent poison.

    The in-package copy exists so a wheel carries its engine, and it is
    searched first for that reason. In a monorepo checkout that ordering means
    a stale artifact wins over `cpp/build`, so a whole session of C++ work can
    be rebuilt, tested and benchmarked while none of it is actually loaded --
    which is exactly what happened on 2026-07-24. Say so rather than let a
    build silently not take effect.
    """
    import warnings
    try:
        chosen = Path(chosen)
        if "cpp/build" in str(chosen):
            return                       # already using the build tree
        for cand in candidates:
            cand = Path(cand)
            if "cpp/build" not in str(cand) or not cand.exists():
                continue
            if cand.stat().st_mtime > chosen.stat().st_mtime:
                warnings.warn(
                    f"loaded the bundled {chosen.name} ({chosen}) but "
                    f"{cand} is NEWER -- the bundled copy shadows it, so a "
                    f"fresh C++ build is not being used. Delete the bundled "
                    f"copy or set RELATIVEDB_RT_LIB to the build you mean.",
                    RuntimeWarning, stacklevel=3)
                return
    except OSError:
        return


def _load() -> Optional[ctypes.CDLL]:
    global _lib, _load_failed
    if _lib is not None or _load_failed is not None:
        return _lib
    i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
    i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    for p in _candidate_paths():
        if p and p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                lib.rel_graph_build.restype = ctypes.c_void_p
                lib.rel_graph_build.argtypes = [
                    ctypes.c_int64, f64p, i32p, i32p, u8p, ctypes.c_int64,
                    i64p, i64p, ctypes.c_char_p, ctypes.c_size_t]
                lib.rel_graph_free.restype = None
                lib.rel_graph_free.argtypes = [ctypes.c_void_p]
                lib.rel_graph_n_nodes.restype = ctypes.c_int64
                lib.rel_graph_n_nodes.argtypes = [ctypes.c_void_p]
                lib.rel_graph_n_edges.restype = ctypes.c_int64
                lib.rel_graph_n_edges.argtypes = [ctypes.c_void_p]
                lib.rel_graph_adjacency.restype = ctypes.c_int
                lib.rel_graph_adjacency.argtypes = [
                    ctypes.c_void_p, ctypes.c_int, i64p, i64p,
                    ctypes.c_char_p, ctypes.c_size_t]
                lib.rel_graph_assemble.restype = ctypes.c_int
                lib.rel_graph_assemble.argtypes = [
                    ctypes.c_void_p, ctypes.c_int64, ctypes.c_double,
                    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    ctypes.c_uint64, ctypes.c_uint8, ctypes.c_int64,
                    ctypes.c_int64, i64p, u8p, ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p,
                    ctypes.c_size_t]
                _warn_if_stale_bundled(p, _candidate_paths())
                _lib = lib
                return _lib
            except (OSError, AttributeError) as e:
                _load_failed = f"{p}: {e}"
    if _load_failed is None:
        _load_failed = "librt_c not found (build cpp/ with cmake)"
    return None


def native_available() -> bool:
    return _load() is not None


class NativeGraph:
    """A built graph. Construct once per snapshot, assemble many contexts."""

    def __init__(self, node_ts, node_cells, node_table, node_is_task,
                 edge_parent, edge_child) -> None:
        lib = _load()
        if lib is None:
            raise NativeGraphUnavailable(_load_failed or "librt_c unavailable")
        self._lib = lib
        self._handle: Optional[int] = None
        self._ts = np.ascontiguousarray(node_ts, dtype=np.float64)
        cells = np.ascontiguousarray(node_cells, dtype=np.int32)
        table = np.ascontiguousarray(node_table, dtype=np.int32)
        is_task = np.ascontiguousarray(node_is_task, dtype=np.uint8)
        ep = np.ascontiguousarray(edge_parent, dtype=np.int64)
        ec = np.ascontiguousarray(edge_child, dtype=np.int64)
        n_nodes = len(self._ts)
        if not (len(cells) == len(table) == len(is_task) == n_nodes):
            raise ValueError("node arrays must have equal length")
        if len(ep) != len(ec):
            raise ValueError("edge arrays must have equal length")
        err = ctypes.create_string_buffer(_ERR)
        handle = lib.rel_graph_build(n_nodes, self._ts, cells, table, is_task,
                                     len(ep), ep, ec, err, _ERR)
        if not handle:
            raise NativeGraphUnavailable(
                err.value.decode("utf-8", "replace") or "rel_graph_build failed")
        self._handle = handle
        self.n_nodes = n_nodes

    def adjacency(self, children: bool = True):
        """The ordered CSR the graph built: (offsets, values).

        Read back rather than rebuilt on this side. The order children are
        visited in decides which rows a context sees, so a binding that
        recomputed it would be a second implementation of that rule.
        """
        if self._handle is None:
            raise NativeGraphUnavailable("graph already freed")
        n_nodes = self._lib.rel_graph_n_nodes(ctypes.c_void_p(self._handle))
        n_edges = self._lib.rel_graph_n_edges(ctypes.c_void_p(self._handle))
        offsets = np.empty(n_nodes + 1, dtype=np.int64)
        values = np.empty(max(int(n_edges), 1), dtype=np.int64)
        err = ctypes.create_string_buffer(_ERR)
        rc = self._lib.rel_graph_adjacency(
            ctypes.c_void_p(self._handle), 1 if children else 0,
            offsets, values, err, _ERR)
        if rc != 0:
            raise RuntimeError(
                err.value.decode("utf-8", "replace") or "adjacency failed")
        return offsets, values[:int(n_edges)]

    def assemble(self, target: int, cutoff_ts: float, eligible, policy,
                 fallback_base: int = 0, fallback_n: int = 0,
                 max_nodes: int = 1 << 16):
        """Ordered emitted node ids and their focal flags.

        ``eligible`` is a uint8 mask turning on the peer-ranking walk, or None
        for the target's own neighbourhood alone.
        """
        if self._handle is None:
            raise NativeGraphUnavailable("graph already freed")
        out = np.empty(max_nodes, dtype=np.int64)
        focal = np.empty(max_nodes, dtype=np.uint8)
        count = ctypes.c_int32(0)
        err = ctypes.create_string_buffer(_ERR)
        mask = None
        if eligible is not None:
            mask = np.ascontiguousarray(eligible, dtype=np.uint8)
        rc = self._lib.rel_graph_assemble(
            ctypes.c_void_p(self._handle), ctypes.c_int64(int(target)),
            ctypes.c_double(float(cutoff_ts)),
            mask.ctypes.data_as(ctypes.c_void_p) if mask is not None else None,
            policy.max_context_cells, policy.local_context_cells,
            policy.bfs_width, policy.num_walks, policy.walk_length,
            ctypes.c_uint64(policy.seed & 0xFFFF_FFFF_FFFF_FFFF),
            1 if policy.prefer_latest else 0,
            ctypes.c_int64(int(fallback_base)), ctypes.c_int64(int(fallback_n)),
            out, focal, max_nodes, ctypes.byref(count), err, _ERR)
        if rc != 0:
            raise RuntimeError(
                err.value.decode("utf-8", "replace") or "assemble failed")
        n = count.value
        if n >= max_nodes:
            raise ContextTruncated(
                f"context for node {target} filled the emitted-node buffer "
                f"({n} of {max_nodes}); the rest of the context was dropped. "
                f"Size max_nodes from the graph, not from max_context_cells: "
                f"a zero-cell row still occupies a node slot.")
        return out[:n], focal[:n]

    def __del__(self) -> None:
        h = getattr(self, "_handle", None)
        if h:
            try:
                self._lib.rel_graph_free(ctypes.c_void_p(h))
            except Exception:      # pragma: no cover - interpreter teardown
                pass
            self._handle = None
