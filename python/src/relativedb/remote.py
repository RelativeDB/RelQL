"""Remote scoring: a :class:`~relativedb.scoring.Scorer` over HTTP.

Context creation — retrieval, assembly, token sequence building,
normalization — happens entirely in this (pure Python) package, next to the
data. What crosses the wire is the prepared token batch with text cells still
as RAW STRINGS, plus the query text and scoring metadata; the service (the
C++ ``rt_serve`` backend, or anything speaking the same protocol) embeds the
strings with the pinned MiniLM encoder it carries and runs the transformer
forward. Model weights, the text encoder, and the GPU live in one process
while any number of query processes stay light — and no embedding model ever
runs client-side.

    backend = RemoteBackend("http://localhost:8500", schema=schema)
    engine = Engine(schema, wiring, model_backend=backend)

Endpoints (JSON over HTTP):

``POST /v1/forward``
    ``{"model_uri", "output", "query", "task_type", "batch": {...}}`` ->
    ``{"scores": [...]}"`` / ``{"scores": [[...]]}`` (token_scores) /
    ``+ "target_text": [[384]]`` / ``{"features": [[512]]}``.
    ``output="text_embeddings"`` carries ``texts`` instead of ``batch`` and
    returns ``{"embeddings": [[384]...]}`` for multiclass label decoding.
    ``batch`` is the :class:`~relativedb.scoring.TokenBatch` encoding below;
    ``query``/``task_type`` ride along for validation and logging only — the
    service never parses RelQL.

``GET /health`` -> ``{"status": "ok", ...}``

Errors from the service are raised as :class:`RemoteScoringError`; the caller
sees the same failure surface as a local backend that could not score.
"""
from __future__ import annotations

import json
import contextvars
import threading
import urllib.error
import urllib.request
from typing import Any, Optional, Sequence

import numpy as np

from .relql import ParsedQuery, TaskType
from .retrieve import RetrieverWiring
from relational_transformers_utils.schema import Schema
from .scoring import (MAX_F2P, D_TEXT, ForwardResult,
                      SequenceBackend, TokenBatch)

__all__ = ["RemoteBackend", "RemoteScorer", "RemoteScoringError",
           "encode_batch", "decode_batch"]


class RemoteScoringError(RuntimeError):
    """The scoring service refused, failed, or returned an unusable body."""


# ---------------------------------------------------------------------------
# wire encoding
# ---------------------------------------------------------------------------

def encode_batch(batch: TokenBatch) -> dict:
    """:class:`TokenBatch` -> JSON-able dict.

    Integer channels ship as nested lists; the two float channels are already
    bfloat16-rounded, so the JSON float round-trip is exact (every bf16 value
    is a short decimal). Text is the deduplicated string tables plus per-token
    indices — never embeddings.
    """
    return {
        "b": batch.b, "s": batch.s,
        "node_idxs": batch.node_idxs.tolist(),
        "f2p": batch.f2p.tolist(),
        "col_idxs": batch.col_idxs.tolist(),
        "table_idxs": batch.table_idxs.tolist(),
        "is_padding": batch.is_padding.tolist(),
        "sem_types": batch.sem_types.tolist(),
        "is_target": batch.is_target.tolist(),
        "number_v": batch.number_v.tolist(),
        "datetime_v": batch.datetime_v.tolist(),
        "col_phrases": list(batch.col_phrases),
        "texts": list(batch.texts),
        "text_idx": (batch.text_idx.tolist()
                     if batch.text_idx is not None else None),
    }


def decode_batch(d: dict) -> TokenBatch:
    b, s = int(d["b"]), int(d["s"])
    def arr(key, dtype, shape):
        return np.asarray(d[key], dtype=dtype).reshape(shape)
    text_idx = d.get("text_idx")
    return TokenBatch(
        node_idxs=arr("node_idxs", np.int64, (b, s)),
        f2p=arr("f2p", np.int64, (b, s, MAX_F2P)),
        col_idxs=arr("col_idxs", np.int64, (b, s)),
        table_idxs=arr("table_idxs", np.int64, (b, s)),
        is_padding=arr("is_padding", np.uint8, (b, s)),
        sem_types=arr("sem_types", np.int64, (b, s)),
        is_target=arr("is_target", np.uint8, (b, s)),
        number_v=arr("number_v", np.float32, (b, s)),
        datetime_v=arr("datetime_v", np.float32, (b, s)),
        col_phrases=list(d.get("col_phrases") or []),
        texts=list(d.get("texts") or []),
        text_idx=(None if text_idx is None
                  else np.asarray(text_idx, np.int32).reshape(b, s)))


# ---------------------------------------------------------------------------
# the scorer + backend
# ---------------------------------------------------------------------------

class RemoteScorer:
    """Ships token batches to an inference service and text to its embedder.

    ``session`` lets a service cache its loaded checkpoint and any derived
    state across calls for one project. ``context`` metadata (the query text
    and task type) is attached per forward by :class:`RemoteBackend` for
    validation/observability on the service side.
    """

    def __init__(self, url: str, *, session: Optional[str] = None,
                 timeout: float = 120.0,
                 headers: Optional[dict[str, str]] = None):
        self.url = url.rstrip("/")
        self.session = session
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.last_stats: dict[str, Any] = {}
        # A remote backend can issue several independent forwards at once
        # (notably EXPLAIN ABLATE). Query metadata must follow the calling
        # thread/task rather than live in one shared mutable slot.
        self._query_meta = contextvars.ContextVar(
            f"remote_query_meta_{id(self)}", default=None)
        # Class-label embeddings are tiny and stable; cache per (text, norm).
        self._embed_cache: dict[tuple[str, bool], np.ndarray] = {}
        self._embed_lock = threading.RLock()

    def health(self) -> dict:
        return self._get("/health")

    # -- Scorer protocol ----------------------------------------------------
    def forward(self, batch: TokenBatch, *, model_uri: str,
                output: str = "target_scores") -> ForwardResult:
        meta: dict[str, Any] = {"model_uri": model_uri, "output": output}
        query_meta = self._query_meta.get()
        if query_meta:
            meta.update(query_meta)
        if self.session:
            meta["session"] = self.session
        body = dict(meta)
        body["batch"] = encode_batch(batch)
        out = self._post("/v1/forward", body)
        self.last_stats = {k: v for k, v in out.items()
                           if k not in ("scores", "target_text", "features")}
        if output == "target_features":
            feats = out.get("features")
            if feats is None:
                raise RemoteScoringError(
                    "service returned no 'features' for target_features")
            return ForwardResult(features=np.asarray(feats, np.float32))
        scores = out.get("scores")
        if scores is None:
            raise RemoteScoringError("service returned no 'scores'")
        scores = np.asarray(scores, np.float32)
        if output == "token_scores":
            scores = scores.reshape(batch.b, batch.s)
        else:
            scores = scores.reshape(batch.b)
        target_text = None
        if output == "target_scores_and_text":
            tt = out.get("target_text")
            if tt is None:
                raise RemoteScoringError(
                    "service returned no 'target_text' for "
                    "target_scores_and_text")
            target_text = np.asarray(tt, np.float32).reshape(batch.b, D_TEXT)
        return ForwardResult(scores=scores, target_text=target_text)

    def embed(self, texts: Sequence[str], *,
              normalize: bool = False) -> np.ndarray:
        texts = list(texts)
        with self._embed_lock:
            missing = [t for t in dict.fromkeys(texts)
                       if (t, normalize) not in self._embed_cache]
            if missing:
                # Text encoding is part of the model forward contract.  Keep
                # one authenticated/metered worker endpoint rather than a
                # second embedding API that can drift from model execution.
                out = self._post("/v1/forward", {
                    "output": "text_embeddings",
                    "texts": missing,
                    "normalize": bool(normalize),
                })
                embs = out.get("embeddings")
                if embs is None or len(embs) != len(missing):
                    raise RemoteScoringError(
                        f"service returned {0 if embs is None else len(embs)} "
                        f"embeddings for {len(missing)} texts")
                for t, e in zip(missing, embs):
                    self._embed_cache[(t, normalize)] = np.asarray(e, np.float32)
            return np.stack([self._embed_cache[(t, normalize)] for t in texts]) \
                if texts else np.zeros((0, D_TEXT), np.float32)

    # -- transport --------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body, default=_json_default).encode()
        req = urllib.request.Request(
            self.url + path, data=data, method="POST",
            headers={"Content-Type": "application/json", **self.headers})
        return self._send(req)

    def _post_raw(self, path: str, data: bytes) -> bytes:
        req = urllib.request.Request(
            self.url + path, data=data, method="POST",
            headers={"Content-Type": "application/octet-stream",
                     **self.headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:2000]
            raise RemoteScoringError(
                f"inference service {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RemoteScoringError(
                f"inference service unreachable at {self.url}: "
                f"{e.reason}") from e

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(self.url + path, method="GET",
                                     headers=self.headers)
        return self._send(req)

    def _send(self, req) -> dict:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:2000]
            raise RemoteScoringError(
                f"inference service {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RemoteScoringError(
                f"inference service unreachable at {self.url}: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise RemoteScoringError(
                f"inference service returned a non-JSON body: {e}") from e


class RemoteBackend(SequenceBackend):
    """The cloud-backend :class:`~relativedb.engine.ModelBackend`: sequence
    assembly here, embeddings and the transformer forward on the service.

    Accepts the same assembly options as any :class:`SequenceBackend`; the
    query text and task type are attached to each forward so the service can
    validate and log without ever parsing RelQL itself.
    """

    def __init__(self, url: str, *, schema: Optional[Schema] = None,
                 wiring: Optional[RetrieverWiring] = None,
                 session: Optional[str] = None, timeout: float = 120.0,
                 headers: Optional[dict[str, str]] = None,
                 **assembly_options):
        scorer = RemoteScorer(url, session=session, timeout=timeout,
                              headers=headers)
        super().__init__(scorer, schema=schema, wiring=wiring,
                         **assembly_options)

    @property
    def url(self) -> str:
        return self.scorer.url

    @property
    def last_stats(self) -> dict:
        return self.scorer.last_stats

    def health(self) -> dict:
        return self.scorer.health()

    def score(self, query: ParsedQuery, task_type: TaskType, contexts,
              model_uri: str, config) -> list:
        token = self.scorer._query_meta.set({
            "query": query.text,
            "task_type": (task_type.value if hasattr(task_type, "value")
                           else str(task_type)),
        })
        try:
            return super().score(query, task_type, contexts, model_uri, config)
        finally:
            self.scorer._query_meta.reset(token)


def _json_default(o):
    from datetime import date, datetime
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return list(o)
    if hasattr(o, "item"):          # numpy scalars
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
