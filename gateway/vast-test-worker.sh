#!/usr/bin/env bash
set -euo pipefail

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3

python3 - <<'PY'
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        try:
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, timeout=5).strip()
        except Exception as exc:
            self.send_error(503, str(exc))
            return
        body = json.dumps(
            {"status": "ok", "device": "cuda", "gpu": gpu}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


ThreadingHTTPServer(("0.0.0.0", 8500), Handler).serve_forever()
PY
