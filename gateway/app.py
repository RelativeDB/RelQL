"""Lambda inference gateway for RelativeDB workers registered in Cloud Map."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

import boto3

_discovery = boto3.client("servicediscovery")
_ddb = boto3.client("dynamodb")

NAMESPACE = os.environ["CLOUDMAP_NAMESPACE"]
SERVICE = os.environ["CLOUDMAP_SERVICE"]
KEYS_TABLE = os.environ["KEYS_TABLE"]
USAGE_TABLE = os.environ["USAGE_TABLE"]
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT_SECONDS", "25"))
MAX_BODY = int(os.environ.get("MAX_BODY_BYTES", str(10 * 1024 * 1024)))
ALLOWED_PATHS = {"/health", "/v1/forward", "/v1/flat_features"}
LLM_PATHS = {"/v1/chat/completions", "/v1/responses", "/v1/models"}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
OPENAI_ALLOWED_MODELS = {
    model.strip() for model in os.environ.get(
        "OPENAI_ALLOWED_MODELS",
        "gpt-5.6-sol,gpt-5.6-terra,gpt-5.6-luna").split(",") if model.strip()
}
DISCOVERY_CACHE_SECONDS = float(os.environ.get("DISCOVERY_CACHE_SECONDS", "5"))
_worker_cache: tuple[float, list[str]] = (0.0, [])
_worker_lock = threading.Lock()


def _response(status: int, body: bytes | str, content_type="application/json",
              headers: dict[str, str] | None = None) -> dict[str, Any]:
    raw = body.encode() if isinstance(body, str) else body
    is_text = content_type.startswith("application/json") or content_type.startswith("text/")
    out_headers = {
        "content-type": content_type,
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
    }
    out_headers.update(headers or {})
    return {
        "statusCode": status,
        "headers": out_headers,
        "body": raw.decode("utf-8") if is_text else base64.b64encode(raw).decode(),
        "isBase64Encoded": not is_text,
    }


def _error(status: int, message: str) -> dict[str, Any]:
    return _response(status, json.dumps({"error": message}, separators=(",", ":")))


def _request_parts(event: dict[str, Any]) -> tuple[str, str, dict[str, str], bytes]:
    method = event.get("requestContext", {}).get("http", {}).get("method")
    if not method:  # REST API v1
        method = event.get("httpMethod", "GET")
    path = event.get("rawPath") or event.get("path") or "/"
    headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}
    body = event.get("body") or ""
    raw = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode()
    return method.upper(), path, headers, raw


def _authenticate(headers: dict[str, str]) -> str | None:
    supplied = headers.get("x-api-key", "")
    auth = headers.get("authorization", "")
    if not supplied and auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    if not supplied:
        return None
    digest = hashlib.sha256(supplied.encode()).hexdigest()
    item = _ddb.get_item(
        TableName=KEYS_TABLE,
        Key={"key_hash": {"S": digest}},
        ConsistentRead=True,
        ProjectionExpression="customer_id, enabled",
    ).get("Item")
    if not item or not item.get("enabled", {}).get("BOOL", False):
        return None
    return item["customer_id"]["S"]


def _workers() -> list[str]:
    global _worker_cache
    now = time.monotonic()
    if now - _worker_cache[0] < DISCOVERY_CACHE_SECONDS:
        result = list(_worker_cache[1])
        random.shuffle(result)
        return result
    with _worker_lock:
        now = time.monotonic()
        if now - _worker_cache[0] < DISCOVERY_CACHE_SECONDS:
            result = list(_worker_cache[1])
            random.shuffle(result)
            return result
        result = _discover_workers()
        _worker_cache = (now, result)
    result = list(result)
    random.shuffle(result)
    return result


def _discover_workers() -> list[str]:
    result = _discovery.discover_instances(
        NamespaceName=NAMESPACE,
        ServiceName=SERVICE,
        HealthStatus="HEALTHY",
        MaxResults=100,
    )
    urls = []
    for instance in result.get("Instances", []):
        attrs = instance.get("Attributes", {})
        url = attrs.get("INSTANCE_URL")
        if not url:
            host = attrs.get("AWS_INSTANCE_IPV4")
            port = attrs.get("AWS_INSTANCE_PORT", "8500")
            if host:
                url = f"http://{host}:{port}"
        if url:
            urls.append(url.rstrip("/"))
    return urls


def _model_tokens(path: str, body: bytes) -> int:
    try:
        if path == "/v1/forward":
            batch = json.loads(body).get("batch", {})
            return int(batch.get("b", 0)) * int(batch.get("s", 0))
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    return 0


def _llm_body(path: str, body: bytes) -> tuple[bytes, bool]:
    if path == "/v1/models":
        return body, False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM request body must be valid JSON") from exc
    model = payload.get("model")
    if not isinstance(model, str) or model not in OPENAI_ALLOWED_MODELS:
        raise ValueError(
            f"model must be one of {', '.join(sorted(OPENAI_ALLOWED_MODELS))}")
    streaming = payload.get("stream") is True
    if streaming and path == "/v1/chat/completions":
        options = dict(payload.get("stream_options") or {})
        options["include_usage"] = True
        payload["stream_options"] = options
    return json.dumps(payload, separators=(",", ":")).encode(), streaming


def _usage(payload: dict[str, Any]) -> tuple[int, int, int]:
    usage = payload.get("usage")
    if not usage and isinstance(payload.get("response"), dict):
        usage = payload["response"].get("usage")
    usage = usage or {}
    input_tokens = int(usage.get("input_tokens",
                                 usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens",
                                  usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens",
                                 input_tokens + output_tokens) or 0)
    return input_tokens, output_tokens, total_tokens


def _openai_request(method: str, path: str, body: bytes) -> urllib.request.Request:
    if not OPENAI_API_KEY:
        raise RuntimeError("LLM provider is not configured")
    return urllib.request.Request(
        OPENAI_BASE_URL + path,
        data=body if method != "GET" else None,
        method=method,
        headers={
            "authorization": f"Bearer {OPENAI_API_KEY}",
            "content-type": "application/json",
            "accept": "text/event-stream, application/json",
        },
    )


def _proxy(method: str, path: str, headers: dict[str, str],
           body: bytes) -> tuple[int, bytes, str, str]:
    workers = _workers()
    if not workers:
        raise RuntimeError("no healthy inference workers are registered")
    content_type = headers.get("content-type", "application/json")
    last_error = "all workers failed"
    for worker in workers[:3]:
        request = urllib.request.Request(
            worker + path,
            data=body if method != "GET" else None,
            method=method,
            headers={
                "content-type": content_type,
                "accept": headers.get("accept", "application/json"),
                "x-request-id": headers.get("x-request-id", str(uuid.uuid4())),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT) as response:
                return response.status, response.read(), (
                    response.headers.get_content_type()), worker
        except urllib.error.HTTPError as exc:
            # Application errors are authoritative; retrying could duplicate work.
            return exc.code, exc.read(), exc.headers.get_content_type(), worker
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
    raise RuntimeError(last_error)


def _meter(customer_id: str, request_id: str, path: str, status: int,
           data_in: int, data_out: int, tokens: int, latency_ms: int,
           input_tokens: int = 0, output_tokens: int = 0) -> None:
    now = int(time.time())
    period = time.strftime("%Y-%m", time.gmtime(now))
    _ddb.transact_write_items(TransactItems=[
        {"Put": {
            "TableName": USAGE_TABLE,
            "Item": {
                "pk": {"S": f"REQUEST#{request_id}"},
                "sk": {"S": str(now)},
                "customer_id": {"S": customer_id},
                "path": {"S": path},
                "status": {"N": str(status)},
                "data_in_bytes": {"N": str(data_in)},
                "data_out_bytes": {"N": str(data_out)},
                "model_tokens": {"N": str(tokens)},
                "input_tokens": {"N": str(input_tokens)},
                "output_tokens": {"N": str(output_tokens)},
                "latency_ms": {"N": str(latency_ms)},
                "expires_at": {"N": str(now + 90 * 86400)},
            },
            "ConditionExpression": "attribute_not_exists(pk)",
        }},
        {"Update": {
            "TableName": USAGE_TABLE,
            "Key": {"pk": {"S": f"CUSTOMER#{customer_id}"}, "sk": {"S": period}},
            "UpdateExpression": (
                "ADD request_count :one, data_in_bytes :din, "
                "data_out_bytes :dout, model_tokens :tok, "
                "input_tokens :itok, output_tokens :otok"
            ),
            "ExpressionAttributeValues": {
                ":one": {"N": "1"}, ":din": {"N": str(data_in)},
                ":dout": {"N": str(data_out)}, ":tok": {"N": str(tokens)},
                ":itok": {"N": str(input_tokens)},
                ":otok": {"N": str(output_tokens)},
            },
        }},
    ])


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    started = time.monotonic()
    try:
        method, path, headers, body = _request_parts(event)
    except (ValueError, TypeError):
        return _error(400, "invalid request encoding")
    if path == "/gateway/health" and method == "GET":
        return _response(200, '{"status":"ok"}')
    if path not in ALLOWED_PATHS | LLM_PATHS or method not in {"GET", "POST"}:
        return _error(404, "not found")
    if len(body) > MAX_BODY:
        return _error(413, "request body too large")
    customer = _authenticate(headers)
    if customer is None:
        return _error(401, "missing or invalid API key")
    request_id = headers.get("x-request-id") or str(uuid.uuid4())
    if path in LLM_PATHS:
        try:
            upstream_body, streaming = _llm_body(path, body)
            if streaming:
                return _error(500, "streaming requires the persistent gateway")
            request = _openai_request(method, path, upstream_body)
            with urllib.request.urlopen(
                    request, timeout=UPSTREAM_TIMEOUT) as response:
                status, response_body = response.status, response.read()
                content_type = response.headers.get_content_type()
        except ValueError as exc:
            return _error(400, str(exc))
        except urllib.error.HTTPError as exc:
            status, response_body = exc.code, exc.read()
            content_type = exc.headers.get_content_type()
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            return _error(503, str(exc))
        latency_ms = round((time.monotonic() - started) * 1000)
        input_tokens = output_tokens = total_tokens = 0
        try:
            input_tokens, output_tokens, total_tokens = _usage(
                json.loads(response_body))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        try:
            _meter(customer, request_id, path, status, len(body),
                   len(response_body), total_tokens, latency_ms,
                   input_tokens, output_tokens)
        except Exception:
            return _error(503, "usage metering unavailable")
        return _response(status, response_body, content_type, {
            "x-request-id": request_id,
            "server-timing": f"gateway;dur={latency_ms}",
        })
    try:
        status, response_body, content_type, _worker = _proxy(
            method, path, headers, body)
    except RuntimeError as exc:
        return _error(503, str(exc))
    latency_ms = round((time.monotonic() - started) * 1000)
    try:
        _meter(customer, request_id, path, status, len(body),
               len(response_body), _model_tokens(path, body), latency_ms)
    except _ddb.exceptions.TransactionCanceledException:
        return _error(409, "duplicate request id")
    except Exception:
        # Do not release an unmetered result.
        return _error(503, "usage metering unavailable")
    return _response(status, response_body, content_type, {
        "x-request-id": request_id,
        "server-timing": f"gateway;dur={latency_ms}",
    })
