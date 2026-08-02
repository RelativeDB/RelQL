"""Small persistent HTTP adapter for the Lambda-shaped gateway handler."""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import app


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _handle(self) -> None:
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if length > app.MAX_BODY:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0]
        headers = {str(k).lower(): str(v) for k, v in self.headers.items()}
        if path in app.LLM_PATHS and self.command in {"GET", "POST"}:
            self._handle_llm(path, headers, body)
            return
        event = {
            "httpMethod": self.command,
            "path": path,
            "headers": dict(self.headers.items()),
            "body": base64.b64encode(body).decode(),
            "isBase64Encoded": True,
        }
        response = app.handler(event, None)
        raw = (base64.b64decode(response["body"])
               if response.get("isBase64Encoded")
               else response.get("body", "").encode())
        self.send_response(int(response["statusCode"]))
        for name, value in response.get("headers", {}).items():
            self.send_header(name, value)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def _handle_llm(self, path: str, headers: dict[str, str],
                    body: bytes) -> None:
        started = time.monotonic()
        customer = app._authenticate(headers)
        if customer is None:
            self._send_result(app._error(401, "missing or invalid API key"))
            return
        request_id = headers.get("x-request-id") or str(uuid.uuid4())
        try:
            upstream_body, streaming = app._llm_body(path, body)
            if not streaming:
                event = {
                    "httpMethod": self.command, "path": path,
                    "headers": headers,
                    "body": base64.b64encode(body).decode(),
                    "isBase64Encoded": True,
                }
                self._send_result(app.handler(event, None))
                return
            request = app._openai_request(self.command, path, upstream_body)
            response = urllib.request.urlopen(
                request, timeout=app.UPSTREAM_TIMEOUT)
        except ValueError as exc:
            self._send_result(app._error(400, str(exc)))
            return
        except urllib.error.HTTPError as exc:
            self._send_result(app._response(
                exc.code, exc.read(), exc.headers.get_content_type()))
            return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            self._send_result(app._error(503, str(exc)))
            return

        self.send_response(response.status)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache, no-store")
        self.send_header("x-request-id", request_id)
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        connected = True
        data_out = 0
        input_tokens = output_tokens = total_tokens = 0
        try:
            while True:
                line = response.readline()
                if not line:
                    break
                data_out += len(line)
                if line.startswith(b"data: "):
                    payload = line[6:].strip()
                    if payload and payload != b"[DONE]":
                        try:
                            usage = app._usage(json.loads(payload))
                            if any(usage):
                                input_tokens, output_tokens, total_tokens = usage
                        except (json.JSONDecodeError, ValueError, TypeError):
                            pass
                if connected:
                    try:
                        self.wfile.write(
                            f"{len(line):X}\r\n".encode() + line + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        # Finish consuming upstream so the final usage event is
                        # still metered after a browser disconnect.
                        connected = False
        finally:
            response.close()
        latency_ms = round((time.monotonic() - started) * 1000)
        try:
            app._meter(customer, request_id, path, response.status, len(body),
                       data_out, total_tokens, latency_ms,
                       input_tokens, output_tokens)
        except Exception as exc:
            print(json.dumps({"metering_error": str(exc),
                              "request_id": request_id}), flush=True)
        if connected:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def _send_result(self, response: dict) -> None:
        raw = (base64.b64decode(response["body"])
               if response.get("isBase64Encoded")
               else response.get("body", "").encode())
        self.send_response(int(response["statusCode"]))
        for name, value in response.get("headers", {}).items():
            self.send_header(name, value)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = _handle
    do_POST = _handle

    def log_message(self, fmt: str, *args: object) -> None:
        print(json.dumps({"remote": self.client_address[0],
                          "message": fmt % args}), flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    print(f"gateway listening on :{port}", flush=True)
    server.serve_forever()
