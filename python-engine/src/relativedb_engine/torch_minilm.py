"""MiniLM text encoding in Python, on the same GPU as the model.

The native encoder in ``librt_c`` exists so that a C++ serving process needs no
Python at all. The Triton worker already *is* Python and already holds a CUDA
context, so borrowing a C++ library to embed a few dozen strings meant building
and shipping ``librt_c`` for no reason other than the encoder.

Same weights (``sentence-transformers/all-MiniLM-L12-v2``), same contract as
:class:`NativeTextEncoder`: ``encode(texts, normalize=...) -> [n, 384]``, mean
pooling over the attention mask, which is what that checkpoint was trained to
be read with.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

D_TEXT = 384
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


class TorchTextEncoder:
    """The MiniLM encoder, loaded once and kept on the device."""

    def __init__(self, model_dir: Optional[str] = None,
                 device: str = "cuda", batch: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        name = (model_dir or os.environ.get("RELATIVEDB_MINILM_DIR")
                or DEFAULT_MODEL)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        # bf16 on GPU, exactly as the reference loads it (rt/embed.py), so the
        # embeddings the model sees at inference are rounded the same way as
        # the ones it was trained on.
        dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
        self.model = (AutoModel.from_pretrained(name, dtype=dtype)
                      .to(device).eval())
        self.device = device
        self.batch = batch

    def encode(self, texts: Sequence[str], *,
               normalize: bool = False) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, D_TEXT), np.float32)
        torch = self._torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch):
                chunk = texts[i:i + self.batch]
                enc = self.tokenizer(chunk, padding=True, truncation=True,
                                     return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                # Mean pool over real tokens only: padding must not drag a
                # short phrase toward zero.
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                # The reference stores these as bf16; round through it so a
                # value here is bit-for-bit the value training saw, then widen
                # for the float32 wire.
                out.append(pooled.to(torch.bfloat16).float().cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32, copy=False)
