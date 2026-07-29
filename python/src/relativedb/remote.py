"""Remote scoring: a :class:`~relativedb.engine.ModelBackend` over HTTP.

The engine assembles contexts locally — retrievers stay next to the data — and
ships only the assembled context to a service that holds the checkpoint. That
keeps model weights, the native library, and the GPU in one process while any
number of query processes stay light.

The wire format is the same JSON shape :meth:`relativedb.retrieve.Row.to_json_dict`
already defines for the C ABI, so a row costs nothing extra to serialize.

    backend = RemoteBackend("http://localhost:8500", schema=schema)
    engine = Engine(schema, wiring, model_backend=backend)

Errors from the service are raised as :class:`RemoteScoringError`; the caller
sees the same failure surface as a local backend that could not score.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Optional

from .engine import EntityContext, EntityPrediction
from .model import ModelConfig
from .relql import ParsedQuery, TaskType
from .schema import Schema

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = ["RemoteBackend", "RemoteScoringError", "encode_contexts",
           "decode_contexts", "encode_schema", "decode_schema"]


class RemoteScoringError(RuntimeError):
    """The scoring service refused, failed, or returned an unusable body."""


# ---------------------------------------------------------------------------
# wire encoding
# ---------------------------------------------------------------------------

def encode_schema(schema: Schema) -> dict:
    return {
        "tables": [{
            "name": t.name,
            "primary_key": t.primary_key,
            "time_column": t.time_column,
            "columns": [{"name": c.name, "type": c.type.value} for c in t.columns],
        } for t in schema.tables],
        "links": [{
            "from_table": l.from_table, "fk_column": l.fk_column,
            "to_table": l.to_table,
            "feature_type": l.feature_type.value if l.feature_type else None,
        } for l in schema.links],
    }


def decode_schema(d: dict) -> Schema:
    from .schema import ColumnDef, LinkDef, TableDef, ValueType
    tables = []
    for t in d.get("tables", []):
        b = TableDef.new_table(t["name"])
        for c in t.get("columns", []):
            b = b.column(ColumnDef(c["name"], ValueType(c["type"])))
        if t.get("primary_key"):
            b = b.primary_key(t["primary_key"])
        if t.get("time_column"):
            b = b.time_column(t["time_column"])
        tables.append(b.build())
    links = [LinkDef(l["from_table"], l["fk_column"], l["to_table"],
                     ValueType(l["feature_type"]) if l.get("feature_type") else None)
             for l in d.get("links", [])]
    return Schema(tables=tuple(tables), links=tuple(links))


def _encode_row(row) -> dict:
    """``Row.to_json_dict`` plus the cell keys that were datetimes.

    ``to_json_dict`` renders datetime cells as ISO strings and
    ``from_json_dict`` leaves them as strings, so a naive round trip silently
    retypes temporal features as text. The C ABI never round-trips, so it does
    not notice; this transport does. Carrying the key list keeps the decode
    exact without guessing which strings look like timestamps.
    """
    from datetime import date, datetime
    d = row.to_json_dict()
    dt_cells = [k for k, v in row.cells.items() if isinstance(v, (datetime, date))]
    if dt_cells:
        d["dt_cells"] = dt_cells
    return d


def _decode_row(d: dict):
    from datetime import datetime

    from .retrieve import Row
    row = Row.from_json_dict(d)
    keys = d.get("dt_cells")
    if not keys:
        return row
    cells = dict(row.cells)
    for k in keys:
        v = cells.get(k)
        if isinstance(v, str):
            try:
                cells[k] = datetime.fromisoformat(v)
            except ValueError:
                pass
    return Row(table=row.table, id=row.id, cells=cells,
               timestamp=row.timestamp, parents=dict(row.parents))


def encode_contexts(contexts: list[EntityContext]) -> list[dict]:
    out = []
    for c in contexts:
        out.append({
            "entity_id": c.entity_id,
            "anchor": c.anchor.isoformat() if c.anchor else None,
            "rows": [_encode_row(r) for r in c.rows],
            "truncated_children": c.truncated_children,
            "hit_cell_budget": c.hit_cell_budget,
            "focal_row_keys": [[t, i] for (t, i) in c.focal_row_keys],
            "node_ids": [[[t, i], n] for (t, i), n in c.node_ids.items()],
        })
    return out


def decode_contexts(payload: list[dict]) -> list[EntityContext]:
    from datetime import datetime
    out = []
    for d in payload:
        anchor = d.get("anchor")
        ctx = EntityContext(
            entity_id=d["entity_id"],
            anchor=datetime.fromisoformat(anchor) if anchor else None,
            rows=[_decode_row(r) for r in d.get("rows", [])],
            truncated_children=int(d.get("truncated_children") or 0),
            hit_cell_budget=bool(d.get("hit_cell_budget")),
            focal_row_keys=frozenset((t, i) for t, i in d.get("focal_row_keys", [])),
            node_ids={(t, i): n for (t, i), n in
                      ((tuple(k), v) for k, v in d.get("node_ids", []))},
        )
        out.append(ctx)
    return out


def encode_predictions(preds: list[EntityPrediction]) -> list[dict]:
    return [{
        "id": p.id, "value": p.value, "probability": p.probability,
        "class_probs": dict(p.class_probs), "ranked": list(p.ranked),
        "forecast": list(p.forecast), "predicted_class": p.predicted_class,
    } for p in preds]


def decode_predictions(payload: list[dict]) -> list[EntityPrediction]:
    return [EntityPrediction(
        id=p["id"], value=p.get("value"), probability=p.get("probability"),
        class_probs=dict(p.get("class_probs") or {}),
        ranked=tuple(tuple(x) if isinstance(x, list) else x
                     for x in (p.get("ranked") or ())),
        forecast=tuple(p.get("forecast") or ()),
        predicted_class=p.get("predicted_class"),
    ) for p in payload]


def encode_config(config: ModelConfig) -> dict:
    mode = config.normalization_mode
    return {
        "classification_model_uri": config.classification_model_uri,
        "regression_model_uri": config.regression_model_uri,
        "embedding_model": config.embedding_model,
        "allow_embedding_mismatch": config.allow_embedding_mismatch,
        "normalization_mode": getattr(mode, "value", str(mode)),
    }


def decode_config(d: dict) -> ModelConfig:
    return ModelConfig(
        classification_model_uri=d.get(
            "classification_model_uri", ModelConfig.defaults().classification_model_uri),
        regression_model_uri=d.get(
            "regression_model_uri", ModelConfig.defaults().regression_model_uri),
        embedding_model=d.get("embedding_model",
                              ModelConfig.defaults().embedding_model),
        allow_embedding_mismatch=bool(d.get("allow_embedding_mismatch", False)),
        normalization_mode=d.get("normalization_mode", "zero_shot"),
    )


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------

class RemoteBackend:
    """Scores assembled contexts through an HTTP inference service.

    ``schema`` is sent with each request so the service can validate the query
    the same way the local engine did; pass the schema the engine was built
    with. ``session`` lets a service cache its loaded checkpoint and any
    schema-derived state across calls for one project.
    """

    def __init__(self, url: str, *, schema: Optional[Schema] = None,
                 session: Optional[str] = None, timeout: float = 120.0,
                 headers: Optional[dict[str, str]] = None):
        self.url = url.rstrip("/")
        self.schema = schema
        self.session = session
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.last_stats: dict[str, Any] = {}

    def health(self) -> dict:
        return self._get("/health")

    def score(self, query: ParsedQuery, task_type: TaskType,
              contexts: list[EntityContext], model_uri: str,
              config: ModelConfig) -> list[EntityPrediction]:
        if not contexts:
            return []
        body = {
            "query": query.text,
            "task_type": task_type.value if hasattr(task_type, "value") else str(task_type),
            "model_uri": model_uri,
            "config": encode_config(config),
            "contexts": encode_contexts(contexts),
        }
        if self.schema is not None:
            body["schema"] = encode_schema(self.schema)
        if self.session:
            body["session"] = self.session
        out = self._post("/score", body)
        preds = decode_predictions(out.get("predictions") or [])
        self.last_stats = {k: v for k, v in out.items() if k != "predictions"}
        if len(preds) != len(contexts):
            raise RemoteScoringError(
                f"scoring service returned {len(preds)} predictions for "
                f"{len(contexts)} contexts")
        return preds

    # -- transport --------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body, default=_json_default).encode()
        req = urllib.request.Request(
            self.url + path, data=data, method="POST",
            headers={"Content-Type": "application/json", **self.headers})
        return self._send(req)

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


def _json_default(o):
    from datetime import date, datetime
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return list(o)
    if hasattr(o, "item"):          # numpy scalars
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
