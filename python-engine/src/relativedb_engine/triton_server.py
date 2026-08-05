"""HTTP worker for the Triton CUDA inference path.

The wire protocol matches :class:`relativedb.remote.RemoteScorer`. Python
owns HTTP, decoding, text embedding, and scheduling; model math stays in the
Triton kernels used by :class:`RTJTriton`.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


from relativedb.remote import decode_batch
from relativedb.scoring import TokenBatch

from .models import RT_DEVICE_CUDA
from .scorer import NativeScorer
from .torch_minilm import TorchTextEncoder


class TritonService:
    def __init__(self, checkpoint: str):
        self.checkpoint = checkpoint
        # Text embedding in Python, on the GPU this process already owns.
        # The native encoder would drag librt_c into a worker that otherwise
        # needs no C++ at all.
        self.scorer = NativeScorer(device=RT_DEVICE_CUDA,
                                   cuda_backend="triton",
                                   embedder=TorchTextEncoder())
        # RTJTriton reuses device buffers, so one model invocation owns them
        # at a time. Threaded HTTP clients queue here without blocking health.
        self.forward_lock = threading.Lock()
        self.started = time.monotonic()
        self.requests = 0

    def forward(self, batch: TokenBatch, output: str):
        if output != "target_scores":
            raise ValueError(
                "the Triton worker currently serves target_scores only")
        with self.forward_lock:
            result = self.scorer.forward(
                batch, model_uri=self.checkpoint, output=output)
            self.requests += 1
        return result


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    service: TritonService
    max_body: int

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        self._send(status, json.dumps(value, separators=(",", ":")).encode(),
                   "application/json")

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._json(404, {"error": "not found"})
            return
        self._json(200, {
            "status": "ok",
            "device": "cuda",
            "backend": "triton",
            "dtype": "float16",
            "requests": self.service.requests,
            "uptime_seconds": round(time.monotonic() - self.service.started, 3),
        })

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > self.max_body:
                raise ValueError("invalid request body size")
            body = self.rfile.read(length)
            path = self.path.split("?", 1)[0]
            if path != "/v1/forward":
                self._json(404, {"error": "not found"})
                return
            request = json.loads(body)
            batch = decode_batch(request["batch"])
            output = request.get("output", "target_scores")
            result = self.service.forward(batch, output)
            self._json(200, {"scores": result.scores.tolist(),
                             "device": "cuda", "backend": "triton"})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: object) -> None:
        print(json.dumps({"remote": self.client_address[0],
                          "message": fmt % args}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default=os.environ.get("RELATIVEDB_MODEL"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8500")))
    parser.add_argument("--max-body", type=int,
                        default=int(os.environ.get(
                            "MAX_BODY_BYTES", str(6 * 1024 * 1024))))
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint or RELATIVEDB_MODEL is required")
    Handler.service = TritonService(args.checkpoint)
    Handler.max_body = args.max_body
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(json.dumps({"status": "listening", "port": args.port,
                      "backend": "triton", "dtype": "float16"}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
