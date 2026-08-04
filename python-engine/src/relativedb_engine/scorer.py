"""In-process scorer: Triton on CUDA, native engine on MPS/CPU and for legacy.

This is the "optional engine dependency" half of the scoring split: the base
``relativedb`` package assembles token batches (text still as raw strings);
this scorer embeds the strings with the NATIVE MiniLM encoder compiled into
librt_c (``cpp/src/minilm_c.h``) — text embedding never runs in Python — and
runs the transformer through Triton by default on CUDA. The golden-verified
C++ transformer remains the MPS/CPU implementation, the training backend, and
an explicitly selected CUDA compatibility path.
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional, Sequence

import numpy as np

from relativedb.scoring import (D_TEXT, ForwardResult, ScoringError,
                                TokenBatch, bf16_as_f32)

from .native import (RT_DEVICE_CPU, RT_DEVICE_CUDA, RT_DEVICE_MPS,
                     RtNativeError, RtModel, load_lib, resolve_model_path)

__all__ = ["NativeScorer", "NativeTextEncoder", "resolve_minilm_snapshot"]

_MINILM_REPO = "sentence-transformers/all-MiniLM-L12-v2"


def resolve_minilm_snapshot() -> Optional[str]:
    """Resolve (and if needed download) the pinned MiniLM snapshot directory.

    The C++ encoder can auto-locate an existing HF-cache snapshot itself; this
    helper exists for the cold-cache case, where only huggingface_hub knows how
    to fetch one. Returns None when huggingface_hub is unavailable — the native
    loader then reports precisely what is missing."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:  # cache-first: never hit the network when already downloaded
        return snapshot_download(_MINILM_REPO, local_files_only=True)
    except Exception:
        return snapshot_download(_MINILM_REPO)


class NativeTextEncoder:
    """ctypes binding to the native MiniLM encoder in librt_c
    (``cpp/src/minilm_c.h``), with a per-process embedding cache.

    Embedding is deliberately NOT implemented in Python: RT-J is frozen
    against vectors this exact encoder produced, and the encoder ships with
    the engine (this package's librt_c) or with the serving backend — never
    with context creation.
    """

    def __init__(self, *, lib_path: Optional[str] = None,
                 snapshot_dir: Optional[str] = None):
        self._lib_path = lib_path
        self._snapshot_dir = snapshot_dir
        self._handle: Optional[int] = None
        self._lib = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_norm: dict[str, np.ndarray] = {}
        self._strict_precomputed = False

    def install_precomputed(self, values: dict[str, np.ndarray], *,
                            strict: bool = True) -> None:
        """Install preprocessing-time embeddings.

        With ``strict=True`` an unknown string is an error, never an implicit
        switch to a newly computed embedding distribution.
        """
        self._cache.update({str(key): np.asarray(value, np.float32)
                            for key, value in values.items()})
        self._strict_precomputed = strict

    def _load(self):
        if self._handle is None:
            lib = load_lib(self._lib_path)._lib
            if not hasattr(lib, "minilm_load"):
                raise RtNativeError(
                    "this librt_c was built without the native MiniLM text "
                    "encoder (minilm_load missing); rebuild cpp/ or point "
                    "RELATIVEDB_RT_LIB at a full build")
            lib.minilm_load.restype = ctypes.c_void_p
            lib.minilm_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                        ctypes.c_size_t]
            lib.minilm_free.restype = None
            lib.minilm_free.argtypes = [ctypes.c_void_p]
            lib.minilm_encode.restype = ctypes.c_int
            lib.minilm_encode.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int32, ctypes.c_int32,
                np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
                ctypes.c_char_p, ctypes.c_size_t]
            snap = self._snapshot_dir or os.environ.get(
                "RELATIVEDB_MINILM_DIR") or resolve_minilm_snapshot() or ""
            err = ctypes.create_string_buffer(512)
            handle = lib.minilm_load(snap.encode("utf-8"), err, len(err))
            if not handle:
                raise RtNativeError(
                    f"minilm_load({snap!r}) failed: "
                    f"{err.value.decode('utf-8', 'replace')}")
            self._lib = lib
            self._handle = handle
        return self._lib

    def encode(self, texts: Sequence[str], *,
               normalize: bool = False) -> list[np.ndarray]:
        """Mean-pooled MiniLM embeddings, cached. ``normalize=True`` returns
        L2-normalized (unit) vectors (a separate cache) — used for multiclass
        class-label embeddings; the default (un-normalized) matches training
        for text CELL values."""
        cache = self._cache_norm if normalize else self._cache
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            if self._strict_precomputed:
                raise RtNativeError(
                    "precomputed embedding table has no value for "
                    + ", ".join(repr(value) for value in missing[:3]))
            lib = self._load()
            arr = (ctypes.c_char_p * len(missing))(
                *[t.encode("utf-8") for t in missing])
            out = np.zeros((len(missing), D_TEXT), np.float32)
            err = ctypes.create_string_buffer(512)
            rc = lib.minilm_encode(self._handle, arr, len(missing),
                                   1 if normalize else 0, out, err, len(err))
            if rc != 0:
                raise RtNativeError(
                    f"minilm_encode failed ({rc}): "
                    f"{err.value.decode('utf-8', 'replace')}")
            for t, e in zip(missing, out):
                cache[t] = e.copy()
        return [cache[t] for t in texts]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def __del__(self):  # pragma: no cover
        try:
            if self._handle and self._lib is not None:
                self._lib.minilm_free(self._handle)
                self._handle = None
        except Exception:
            pass


class NativeScorer:
    """Embeds text and forwards through librt_c, all in native code.

    ``device=None`` resolves once to the best available backend (MPS, then
    CUDA, then CPU). Loaded checkpoints are cached per resolved path so cohort
    chunks and multi-task queries do not reload 342 MB of weights.
    """

    def __init__(self, *, lib_path: Optional[str] = None,
                 embedder: Optional[NativeTextEncoder] = None,
                 n_threads: int = 0,
                 device: Optional[int] = None,
                 cuda_backend: str = "triton"):
        if cuda_backend not in ("triton", "native"):
            raise ValueError("cuda_backend must be 'triton' or 'native'")
        self._lib_path = lib_path
        self.embedder = embedder or NativeTextEncoder(lib_path=lib_path)
        self.n_threads = n_threads
        self.device = device
        self.cuda_backend = cuda_backend
        self._models: dict[str, RtModel] = {}
        self._triton_models: dict[str, object] = {}

    # -- Scorer protocol ----------------------------------------------------
    def embed(self, texts: Sequence[str], *,
              normalize: bool = False) -> np.ndarray:
        vecs = self.embedder.encode(list(texts), normalize=normalize)
        return np.asarray(vecs, np.float32).reshape(len(vecs), D_TEXT)

    def forward(self, batch: TokenBatch, *, model_uri: str,
                output: str = "target_scores") -> ForwardResult:
        kw = self._materialize(batch)
        device = self._resolve_device()
        if (output == "target_scores" and device == RT_DEVICE_CUDA
                and self.cuda_backend == "triton"):
            model = self._triton_model_for(model_uri)
            raw = {
                "node_idxs": kw["node_idxs"],
                "f2p_nbr_idxs": kw["f2p"],
                "col_name_idxs": kw["col_idxs"],
                "table_name_idxs": kw["table_idxs"],
                "is_padding": kw["is_padding"],
                "sem_types": kw["sem_types"],
                "is_targets": kw["is_target"],
                "number_values": kw["number_v"],
                "datetime_values": kw["datetime_v"],
                "boolean_values": kw["boolean_v"],
                "text_values": kw["text_v"],
                "col_name_values": kw["col_name_v"],
            }
            return ForwardResult(scores=model.predict(raw))

        # Compatibility/training outputs remain behind the native engine.
        model = self._model_for(model_uri)
        kw["n_threads"] = self.n_threads
        if output == "target_scores":
            return ForwardResult(scores=model.forward(device=device, **kw))
        if output == "token_scores":
            return ForwardResult(scores=model.forward_tokens(
                device=self._resolve_device(), **kw))
        if output == "target_scores_and_text":
            scores, text = model.forward_ex(**kw)   # CPU path (rt_forward_ex)
            return ForwardResult(scores=scores, target_text=text)
        if output == "target_features":
            return ForwardResult(features=model.encode_targets(
                device=self._resolve_device(), **kw))
        raise ScoringError(f"unknown forward output kind {output!r}")

    # -- internals ------------------------------------------------------------
    def _model_for(self, model_uri: str) -> RtModel:
        path = resolve_model_path(model_uri)
        if path not in self._models:
            self._models[path] = load_lib(self._lib_path).load_model(path)
        return self._models[path]

    def _triton_model_for(self, model_uri: str):
        path = resolve_model_path(model_uri)
        if path not in self._triton_models:
            try:
                from .triton_model import RTJTriton
            except ImportError as exc:
                raise ScoringError(
                    "CUDA scoring uses Triton by default; install "
                    "'relativedb-engine[triton]' or explicitly construct "
                    "RtNativeBackend(cuda_backend='native') for the legacy "
                    "CUDA implementation"
                ) from exc
            self._triton_models[path] = RTJTriton(path)
        return self._triton_models[path]

    def _resolve_device(self) -> int:
        if self.device is None:
            lib = load_lib(self._lib_path)
            if lib.device_available(RT_DEVICE_MPS):
                self.device = RT_DEVICE_MPS
            elif lib.device_available(RT_DEVICE_CUDA):
                self.device = RT_DEVICE_CUDA
            else:
                self.device = RT_DEVICE_CPU
        return self.device

    def _materialize(self, batch: TokenBatch) -> dict:
        """Fill the two 384-d channels from the batch's raw strings.

        Schema phrases and text cells embed in one native call: the batch
        already carries each set deduplicated, so this is one encoder
        invocation per collate, then pure scatters. The bf16 rounding mirrors
        the numeric channels (rustler persists every model-valued channel as
        bfloat16)."""
        B, S = batch.b, batch.s
        n_phrases = len(batch.col_phrases)
        embedded = self.embed(list(batch.col_phrases) + list(batch.texts))
        phrase_emb = embedded[:n_phrases]
        text_emb = embedded[n_phrases:]

        col_name_v = np.zeros((B, S, D_TEXT), np.float32)
        text_v = np.zeros((B, S, D_TEXT), np.float32)
        real = batch.is_padding == 0
        if n_phrases:
            col_name_v[real] = phrase_emb[batch.col_idxs[real]]
        if batch.text_idx is not None and len(batch.texts):
            has_text = (batch.text_idx >= 0) & real
            text_v[has_text] = text_emb[batch.text_idx[has_text]]
        return dict(
            node_idxs=batch.node_idxs, f2p=batch.f2p,
            col_idxs=batch.col_idxs, table_idxs=batch.table_idxs,
            is_padding=batch.is_padding, sem_types=batch.sem_types,
            is_target=batch.is_target, number_v=batch.number_v,
            datetime_v=batch.datetime_v,
            # booleans route through the number channel (bool_as_num, F52);
            # the boolean channel is always zero.
            boolean_v=np.zeros((B, S), np.float32),
            text_v=bf16_as_f32(text_v),
            col_name_v=bf16_as_f32(col_name_v))
