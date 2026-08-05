"""In-process scorer over the shared relational-transformers runtime.

The base ``relativedb`` package assembles token batches with text still as raw
strings; this scorer embeds the strings with MiniLM
(``sentence-transformers/all-MiniLM-L12-v2``, mean-pooled and un-normalized,
exactly the space RT-J is frozen against) and forwards through
:class:`relational_transformers.RelationalTransformer`. Backend selection:
Triton serves classification target scores on CUDA, torch serves everything
else everywhere, and ONNX is available for exported graphs.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from relativedb.scoring import (D_TEXT, ForwardResult, ScoringError,
                                TokenBatch, bf16_as_f32)

from .models import (RT_DEVICE_CPU, RT_DEVICE_CUDA, RT_DEVICE_MPS,
                     EngineError, resolve_model_path, torch_device_name)

__all__ = ["RelationalScorer", "NativeScorer", "TextEncoder",
           "NativeTextEncoder", "resolve_minilm_snapshot"]

_MINILM_REPO = "sentence-transformers/all-MiniLM-L12-v2"


def resolve_minilm_snapshot() -> Optional[str]:
    """Resolve (and if needed download) the pinned MiniLM snapshot directory.

    Returns None when huggingface_hub is unavailable; the encoder then reports
    precisely what is missing when it loads."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:  # cache-first: never hit the network when already downloaded
        return snapshot_download(_MINILM_REPO, local_files_only=True)
    except Exception:
        return snapshot_download(_MINILM_REPO)


class TextEncoder:
    """MiniLM embedding with a per-process cache and precomputed-table support.

    Wraps :class:`~relativedb.rt.torch_minilm.TorchTextEncoder`, which
    mean-pools over the attention mask and rounds through bfloat16, matching
    the reference preprocessor. ``normalize=True`` returns L2-normalized
    vectors (a separate cache) — used for multiclass class-label embeddings;
    the default (un-normalized) matches training for text CELL values.
    """

    def __init__(self, *, snapshot_dir: Optional[str] = None,
                 device: Optional[str] = None):
        self._snapshot_dir = snapshot_dir
        self._device = device
        self._encoder = None
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
        if self._encoder is None:
            import torch

            from .torch_minilm import TorchTextEncoder

            device = self._device
            if device is None:
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            snapshot = (self._snapshot_dir
                        or os.environ.get("RELATIVEDB_MINILM_DIR")
                        or resolve_minilm_snapshot())
            self._encoder = TorchTextEncoder(model_dir=snapshot, device=device)
        return self._encoder

    def encode(self, texts: Sequence[str], *,
               normalize: bool = False) -> list[np.ndarray]:
        cache = self._cache_norm if normalize else self._cache
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            if self._strict_precomputed:
                raise EngineError(
                    "precomputed embedding table has no value for "
                    + ", ".join(repr(value) for value in missing[:3]))
            encoded = self._load().encode(missing, normalize=normalize)
            for t, e in zip(missing, encoded):
                cache[t] = np.asarray(e, np.float32)
        return [cache[t] for t in texts]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# Historical name: text embedding used to bind to librt_c's MiniLM.
NativeTextEncoder = TextEncoder


class RelationalScorer:
    """Embeds text and forwards through relational-transformers backends.

    ``device=None`` resolves once to the best available device (MPS, then
    CUDA, then CPU). Loaded checkpoints are cached per resolved path so cohort
    chunks and multi-task queries do not reload 342 MB of weights.
    """

    def __init__(self, *, lib_path: Optional[str] = None,
                 embedder: Optional[TextEncoder] = None,
                 n_threads: int = 0,
                 device: Optional[int] = None,
                 cuda_backend: str = "triton",
                 inference_backend: str = "auto",
                 onnx_model_path: Optional[str] = None):
        if cuda_backend not in ("triton", "torch"):
            raise ValueError("cuda_backend must be 'triton' or 'torch'")
        if inference_backend not in ("auto", "torch", "triton", "onnx"):
            raise ValueError(
                "inference_backend must be 'auto', 'torch', 'triton', or 'onnx'"
            )
        del lib_path  # retained in the signature for source compatibility
        self.embedder = embedder or TextEncoder()
        self.n_threads = n_threads
        self.device = device
        self.cuda_backend = cuda_backend
        self.inference_backend = inference_backend
        self.onnx_model_path = onnx_model_path
        self._relational_models: dict[tuple[str, str, str], object] = {}

    # -- Scorer protocol ----------------------------------------------------
    def embed(self, texts: Sequence[str], *,
              normalize: bool = False) -> np.ndarray:
        vecs = self.embedder.encode(list(texts), normalize=normalize)
        return np.asarray(vecs, np.float32).reshape(len(vecs), D_TEXT)

    def forward(self, batch: TokenBatch, *, model_uri: str,
                output: str = "target_scores") -> ForwardResult:
        if output not in ("target_scores", "token_scores", "target_features",
                          "target_scores_and_text"):
            raise ScoringError(f"unknown forward output kind {output!r}")
        kw = self._materialize(batch)
        device = self._resolve_device()
        selected = self._selected_backend(device)
        if selected in ("triton", "onnx") and output != "target_scores":
            # Those backends serve target scores; richer outputs run torch.
            selected = "torch"
        relational_batch = self._relational_batch(kw)
        model = self._relational_model_for(model_uri, selected, device)
        result = model.forward(relational_batch, output=output)
        if output == "target_scores":
            return ForwardResult(scores=result.scores.detach().cpu().numpy())
        if output == "token_scores":
            return ForwardResult(scores=result.token_scores.detach().cpu().numpy())
        if output == "target_features":
            return ForwardResult(features=result.features.detach().cpu().numpy())
        return ForwardResult(
            scores=result.scores.detach().cpu().numpy(),
            target_text=result.target_text.detach().cpu().numpy(),
        )

    # -- internals ------------------------------------------------------------
    @staticmethod
    def _relational_batch(kw):
        try:
            from relational_transformers import RelationalBatch
        except ImportError as exc:
            raise ScoringError(
                "install relational-transformers to use the shared model runtime"
            ) from exc
        return RelationalBatch.from_mapping({
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
        })

    def _selected_backend(self, device: int) -> str:
        inference_backend = getattr(self, "inference_backend", "auto")
        if inference_backend != "auto":
            return inference_backend
        if device == RT_DEVICE_CUDA:
            return self.cuda_backend
        return "torch"

    def _relational_model_for(self, model_uri: str, backend: str, device: int):
        path = resolve_model_path(model_uri)
        if backend == "onnx":
            if not self.onnx_model_path:
                raise ScoringError(
                    "inference_backend='onnx' requires onnx_model_path pointing "
                    "to an exported RT-J ONNX model"
                )
            path = os.fspath(self.onnx_model_path)
        torch_device = torch_device_name(device)
        key = (path, backend, torch_device)
        if key not in self._relational_models:
            try:
                from relational_transformers import RelationalTransformer
            except ImportError as exc:
                raise ScoringError(
                    "install relational-transformers to use the shared model runtime"
                ) from exc
            self._relational_models[key] = RelationalTransformer(
                path, backend=backend, device=torch_device
            )
        return self._relational_models[key]

    def torch_model(self, model_uri: str):
        """The loaded torch :class:`relational_transformers.RTJModel` for
        ``model_uri`` — the handle training paths differentiate through."""
        device = self._resolve_device()
        return self._relational_model_for(model_uri, "torch", device)

    def _resolve_device(self) -> int:
        if self.device is None:
            import torch
            if torch.backends.mps.is_available():
                self.device = RT_DEVICE_MPS
            elif torch.cuda.is_available():
                self.device = RT_DEVICE_CUDA
            else:
                self.device = RT_DEVICE_CPU
        return self.device

    def _materialize(self, batch: TokenBatch) -> dict:
        """Fill the two 384-d channels from the batch's raw strings.

        Schema phrases and text cells embed in one call: the batch already
        carries each set deduplicated, so this is one encoder invocation per
        collate, then pure scatters. The bf16 rounding mirrors the numeric
        channels (the reference persists every model-valued channel as
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


# Historical name from the librt_c era.
NativeScorer = RelationalScorer
