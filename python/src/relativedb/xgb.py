"""XGBoost backend: flat-feature execution of eligible RelQL queries.

The C++ layer decides everything that is a decision (``cpp/src/flat.*``):
whether a query can run as flat features at all, which feature columns the
schema yields, and the numeric evaluation of those columns over an assembled
context. This module only moves matrices — it encodes contexts to the shared
JSON row shape, receives a dense ``float32`` matrix back through the C ABI,
and hands it to XGBoost (>= 3.3).

Trees cannot see graph structure, so eligibility is narrow — scalar
regression / binary targets with one horizon, no RANK, no ASSUMING, no
ABLATE — and :func:`analyze_flat` answers "can this query run here" without
raising. Everything else belongs to the sequence model.

    analysis = analyze_flat(query, schema)          # eligible? which columns?
    backend = XgboostBackend(schema)
    backend.fit(engine, query, anchors)             # label logic reused from
    engine.model_backend = backend                  # the training module
    engine.execute(query, anchor_time=t0)

XGBoost carries its own CUDA implementation: when the installed build has
CUDA support and a CUDA device is visible, fitting and scoring run with
``device="cuda"`` (pass ``device=`` to override).
"""
from __future__ import annotations

import ctypes
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from .engine import EntityContext, EntityPrediction
from .errors import ExecutionError
from .model import ModelConfig
from .relql.ast import ParsedQuery, TaskType
from .relql.parser import parse, validate
from .schema import Schema

__all__ = ["XgboostBackend", "XgboostUnavailableError", "FlatAnalysis",
           "analyze_flat", "fit_xgboost"]

_ERR = 1024
_OUT = 1 << 16
_OUT_MAX = 1 << 26

# The val-tuned RelBench reference hyperparameters (evaluation/xgboost_worker
# at commit eece048): the starting point unless the caller overrides.
_TUNED_PARAMS: dict[str, Any] = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    tree_method="hist",
    # n_jobs=1 is deliberate, not a speed knob left at a bad default: on
    # macOS, XGBoost's OpenMP threading next to another OpenMP runtime
    # (torch / the reference stack) segfaults in DMatrix construction —
    # the same clash evaluation/run_xgboost_reference.py isolates a whole
    # subprocess to avoid. Override via xgb_params on platforms where the
    # process holds no second OpenMP runtime.
    n_jobs=1,
    verbosity=0,
)


class XgboostUnavailableError(RuntimeError):
    """librt_c or the xgboost package cannot be loaded."""


# ---------------------------------------------------------------------------
# C ABI binding (relql_flat_analyze / relql_flat_features in librt_c)
# ---------------------------------------------------------------------------

def _candidate_paths() -> list[Path]:
    env = os.environ.get("RELATIVEDB_RT_LIB")
    here = Path(__file__).resolve()
    names = ["librt_c.dylib", "librt_c.so", "librt_c.dll", "rt_c.dll"]
    out = [Path(env)] if env else []
    out += [here.parent / n for n in names]
    if len(here.parents) > 3:
        root = here.parents[3]
        out += [root / "cpp" / "build" / n for n in names]
    return out


_lib: Optional[ctypes.CDLL] = None
_load_failed: Optional[str] = None


def _load() -> ctypes.CDLL:
    global _lib, _load_failed
    if _lib is not None:
        return _lib
    if _load_failed is not None:
        raise XgboostUnavailableError(_load_failed)
    for p in _candidate_paths():
        if p and p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                lib.relql_flat_analyze.restype = ctypes.c_int
                lib.relql_flat_analyze.argtypes = [
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                    ctypes.c_char_p, ctypes.c_size_t,
                    ctypes.c_char_p, ctypes.c_size_t,
                ]
                lib.relql_flat_features.restype = ctypes.c_int
                lib.relql_flat_features.argtypes = [
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                    ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int, ctypes.c_int,
                    ctypes.c_char_p, ctypes.c_size_t,
                ]
                _lib = lib
                return _lib
            except (OSError, AttributeError) as e:
                _load_failed = (
                    f"{p}: {e} (an older librt_c without the flat-feature "
                    f"ABI? rebuild cpp/ with cmake)")
    if _load_failed is None:
        _load_failed = "librt_c not found (build cpp/ with cmake)"
    raise XgboostUnavailableError(_load_failed)


def _schema_json(schema: Schema) -> bytes:
    return json.dumps(schema.to_json_dict()).encode()


def _params_json(params: Optional[dict[str, Any]]) -> bytes:
    def default(o):
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"not JSON serializable: {type(o).__name__}")
    return json.dumps(params or {}, default=default).encode()


@dataclass(frozen=True)
class FlatAnalysis:
    """The C++ layer's answer to "can this query run as flat features"."""

    eligible: bool
    reason: str                       # why not, when ineligible
    task_type: str
    entity_table: str
    feature_names: tuple[str, ...] = ()


def analyze_flat(query: Union[str, ParsedQuery], schema: Schema,
                 params: Optional[dict[str, Any]] = None) -> FlatAnalysis:
    """Ask the C++ planner whether ``query`` runs as flat features, and with
    which feature columns. Ineligibility is an answer, not an exception;
    a query that does not parse/validate at all still raises."""
    lib = _load()
    text = query if isinstance(query, str) else query.text
    if not text:
        raise ExecutionError(
            "flat analysis needs the query text; this ParsedQuery carries "
            "none — pass the RelQL string")
    err = ctypes.create_string_buffer(_ERR)
    outlen = _OUT
    while True:
        out = ctypes.create_string_buffer(outlen)
        rc = lib.relql_flat_analyze(
            text.encode(), _schema_json(schema), _params_json(params),
            out, outlen, err, _ERR)
        if rc != 2:
            break
        outlen *= 4
        if outlen > _OUT_MAX:
            raise ExecutionError("flat spec JSON exceeds the output ceiling")
    if rc != 0:
        raise ExecutionError(err.value.decode(errors="replace"))
    d = json.loads(out.value.decode())
    return FlatAnalysis(
        eligible=bool(d["eligible"]), reason=d.get("reason", ""),
        task_type=d.get("task_type", ""),
        entity_table=d.get("entity_table", ""),
        feature_names=tuple(d.get("features", ())))


# ---------------------------------------------------------------------------
# context encoding: focal rows only, datetimes as epoch seconds
# ---------------------------------------------------------------------------

def _epoch(dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _encode_cells(cells: dict[str, Any]) -> dict[str, Any]:
    return {k: (_epoch(v) if isinstance(v, datetime) else v)
            for k, v in cells.items()}


def _encode_context(ctx: EntityContext) -> dict:
    """Focal rows only: the evaluator aggregates whatever it is given, and a
    context's peer (global) rows would count into this entity's features."""
    rows = ([r for r in ctx.rows if r.key in ctx.focal_row_keys]
            if ctx.focal_row_keys else ctx.rows)
    return {
        "entity_id": ctx.entity_id,
        "anchor": _epoch(ctx.anchor),
        "rows": [{
            "table": r.table,
            "id": r.id,
            "ts": _epoch(r.timestamp),
            "cells": _encode_cells(r.cells),
            "parents": dict(r.parents),
        } for r in rows],
    }


def _features(query_text: str, schema: Schema, params: Optional[dict],
              contexts: Sequence[EntityContext],
              n_features: int) -> np.ndarray:
    lib = _load()
    payload = json.dumps([_encode_context(c) for c in contexts],
                         default=str).encode()
    out = np.full((len(contexts), n_features), np.nan, dtype=np.float32)
    err = ctypes.create_string_buffer(_ERR)
    rc = lib.relql_flat_features(
        query_text.encode(), _schema_json(schema), _params_json(params),
        payload, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(contexts), n_features, err, _ERR)
    if rc != 0:
        raise ExecutionError(err.value.decode(errors="replace"))
    return out


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------

def _xgboost():
    try:
        import xgboost
    except ImportError as e:                     # pragma: no cover
        raise XgboostUnavailableError(
            "the xgboost backend needs the xgboost package "
            "(pip install 'relativedb[xgboost]')") from e
    return xgboost


def _detect_device() -> str:
    """XGBoost's own CUDA path when the build has it and a device is visible;
    otherwise CPU. Explicit ``device=`` on the backend overrides."""
    try:
        xgboost = _xgboost()
    except XgboostUnavailableError:
        return "cpu"
    try:
        if not xgboost.build_info().get("USE_CUDA", False):
            return "cpu"
    except Exception:                            # noqa: BLE001 - old builds
        return "cpu"
    import shutil
    if not (shutil.which("nvidia-smi") or os.environ.get("CUDA_VISIBLE_DEVICES")):
        return "cpu"
    return "cuda"


@dataclass
class XgboostFitResult:
    n_examples: int
    n_features: int
    dropped: int
    task_type: TaskType
    device: str
    feature_names: tuple[str, ...] = ()
    save_path: Optional[str] = None


def fit_xgboost(engine, query, anchors, *, device=None, xgb_params=None,
                **fit_kwargs) -> "XgboostBackend":
    """Fit an XGBoost model for ``query`` on labelled anchors — the training
    entry point next to ``Engine.fit_head`` / ``Engine.finetune`` (exposed as
    ``Engine.fit_xgboost``). Same supervision contract: contexts bounded at
    each anchor, labels read from what actually happened after it.

    Returns the fitted :class:`XgboostBackend` with the fit summary on
    ``fit_result``; assign it to ``engine.model_backend`` (or pass
    ``model_backend=`` to a new Engine) to serve the query."""
    backend = XgboostBackend(engine.schema, device=device,
                             xgb_params=xgb_params)
    backend.fit_result = backend.fit(engine, query, anchors, **fit_kwargs)
    return backend


class XgboostBackend:
    """A :class:`~relativedb.engine.ModelBackend` that scores flat-eligible
    queries with a fitted XGBoost model.

    Unlike the RT backend there is no zero-shot path: a tree model knows
    nothing until :meth:`fit` has seen labelled anchors (or :meth:`load` is
    given a directory :meth:`save` wrote). ``score`` raises for queries the
    C++ planner declines — callers that want a fallback should consult
    :func:`analyze_flat` first and route ineligible queries to the sequence
    model."""

    def __init__(self, schema: Schema, *, device: Optional[str] = None,
                 xgb_params: Optional[dict[str, Any]] = None):
        self.schema = schema
        self.device = device or _detect_device()
        self.xgb_params = dict(_TUNED_PARAMS, **(xgb_params or {}))
        self._model = None                     # fitted sklearn-API estimator
        self._fitted_query: Optional[str] = None
        self._analysis: Optional[FlatAnalysis] = None
        self.fit_result: Optional[XgboostFitResult] = None
        self.last_stats: dict[str, Any] = {}

    # -- training ---------------------------------------------------------
    def fit(self, engine, query: Union[str, ParsedQuery],
            anchors: Sequence[datetime], *,
            entity_ids: Optional[Sequence[Any]] = None,
            params: Optional[dict[str, Any]] = None,
            labels: Optional[dict] = None,
            save_path: Optional[Union[str, os.PathLike]] = None
            ) -> XgboostFitResult:
        """Fit on labelled anchors, exactly the ``fit_head`` contract: for
        each anchor the context is bounded at the anchor — as at prediction
        time — while the label is a fact fetched from the database
        (``engine._label_rows``), never read off the context sample."""
        from .training import _scalar_label
        xgboost = _xgboost()
        if not anchors:
            raise ExecutionError("xgboost fitting needs at least one anchor")
        anchors = [engine._coerce_anchor(a) for a in anchors]
        pq = parse(query) if isinstance(query, str) else query
        text = pq.text
        pq = validate(pq, engine.schema).query.bind_params(params)
        task_type = pq.task_type(engine.schema)
        analysis = analyze_flat(text, engine.schema, params)
        if not analysis.eligible:
            raise ExecutionError(
                f"query is not flat-eligible: {analysis.reason}")

        from .engine import ExecutionInput
        entity_table = pq.entity_key.table
        contexts: list[EntityContext] = []
        ys: list[float] = []
        dropped = 0
        for t in anchors:
            ids = (list(entity_ids) if entity_ids is not None
                   else engine._resolve_entity_ids(
                       pq, ExecutionInput(query=pq, anchor_time=t)))
            for eid in ids:
                if labels is not None and (eid, t) not in labels:
                    continue
                ctx = engine.assemble_context(entity_table, eid, t, query=pq)
                y = _scalar_label(engine, pq, task_type, t, labels, eid, [])
                if y is None:
                    dropped += 1
                    continue
                contexts.append(ctx)
                ys.append(float(y))
        if dropped:
            warnings.warn(
                f"xgboost fitting dropped {dropped} example(s) whose derived "
                f"label was NULL (empty target window or non-numeric outcome)",
                UserWarning, stacklevel=2)
        if not contexts:
            raise ExecutionError(
                "xgboost fitting produced no training examples — check the "
                "anchors and that the cohort resolves at them")

        x = _features(text, engine.schema, params, contexts,
                      len(analysis.feature_names))
        y = np.asarray(ys, dtype=np.float32)
        common = dict(self.xgb_params, device=self.device)
        if task_type is TaskType.BINARY_CLASSIFICATION:
            y_int = (y > 0).astype(np.int32)
            if len(np.unique(y_int)) < 2:
                raise ExecutionError(
                    "binary xgboost fitting saw a single class; widen the "
                    "anchors or cohort so both outcomes occur")
            n_pos = int(y_int.sum())
            n_neg = int(len(y_int) - n_pos)
            model = xgboost.XGBClassifier(
                **common, objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0)
            model.fit(x, y_int)
        else:
            model = xgboost.XGBRegressor(
                **common, objective="reg:squarederror", eval_metric="mae")
            model.fit(x, y)

        self._model = model
        self._fitted_query = text
        self._analysis = analysis
        saved = None
        if save_path is not None:
            saved = str(self.save(save_path))
        return XgboostFitResult(
            n_examples=len(contexts), n_features=x.shape[1], dropped=dropped,
            task_type=task_type, device=self.device,
            feature_names=analysis.feature_names, save_path=saved)

    # -- persistence ------------------------------------------------------
    def save(self, path: Union[str, os.PathLike]) -> Path:
        """Write ``model.ubj`` + ``flat.json`` (query, features, task)."""
        if self._model is None or self._analysis is None:
            raise ExecutionError("nothing to save: fit (or load) first")
        out = Path(path).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(out / "model.ubj"))
        (out / "flat.json").write_text(json.dumps({
            "query": self._fitted_query,
            "task_type": self._analysis.task_type,
            "entity_table": self._analysis.entity_table,
            "feature_names": list(self._analysis.feature_names),
            "device": self.device,
        }, indent=2) + "\n")
        return out

    def load(self, path: Union[str, os.PathLike]) -> "XgboostBackend":
        xgboost = _xgboost()
        d = Path(path).expanduser().resolve()
        meta = json.loads((d / "flat.json").read_text())
        cls = (xgboost.XGBClassifier
               if meta["task_type"] == "binary_classification"
               else xgboost.XGBRegressor)
        model = cls(device=self.device)
        model.load_model(str(d / "model.ubj"))
        self._model = model
        self._fitted_query = meta["query"]
        self._analysis = FlatAnalysis(
            eligible=True, reason="", task_type=meta["task_type"],
            entity_table=meta.get("entity_table", ""),
            feature_names=tuple(meta["feature_names"]))
        return self

    # -- scoring (the ModelBackend protocol) ------------------------------
    def score(self, query: ParsedQuery, task_type: TaskType,
              contexts: list[EntityContext], model_uri: str,
              config: ModelConfig) -> list[EntityPrediction]:
        if not contexts:
            return []
        if self._model is None:
            # model_uri routing: an engine configured with a saved directory
            # can lazy-load it on first score.
            if model_uri and Path(model_uri).expanduser().exists():
                self.load(model_uri)
            else:
                raise ExecutionError(
                    "the xgboost backend has no fitted model: call fit() or "
                    "load() (or point the model URI at a saved directory)")
        assert self._analysis is not None
        text = query.text
        if not text:
            raise ExecutionError(
                "xgboost scoring needs the query text on the ParsedQuery")
        analysis = analyze_flat(text, self.schema)
        if not analysis.eligible:
            raise ExecutionError(
                f"query is not flat-eligible: {analysis.reason}")
        if analysis.feature_names != self._analysis.feature_names:
            raise ExecutionError(
                "this query derives different feature columns than the "
                "fitted one; fit the backend on the query it will serve")
        x = _features(text, self.schema, None, contexts,
                      len(analysis.feature_names))
        binary = task_type is TaskType.BINARY_CLASSIFICATION
        self.last_stats = {"n_contexts": len(contexts),
                           "n_features": x.shape[1], "device": self.device}
        if binary:
            probs = self._model.predict_proba(x)[:, 1]
            return [EntityPrediction(c.entity_id, probability=float(p))
                    for c, p in zip(contexts, probs)]
        values = self._model.predict(x)
        return [EntityPrediction(c.entity_id, value=float(v))
                for c, v in zip(contexts, values)]
