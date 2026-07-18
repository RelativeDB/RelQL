"""Binding to the shared C++ PQL parser (``pql_parse`` in ``librt_c``).

This is *the* PQL parser for the Python binding — grammar and lexing live once
in the C++ layer, shared with the Java and Rust bindings. The C ABI returns a
JSON AST (see ``cpp/src/pql.cpp``) which we deserialize into the
:mod:`relativedb.pql.ast` dataclasses. ``librt_c`` is a hard dependency (the
same library the RT-J model requires); :func:`parse_native` raises
:class:`NativeParserUnavailable` if it cannot be loaded.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .ast import (AggFunc, Aggregation, BoolOp, ColumnRef, Condition,
                  LogicalOp, Not, Operator, ParsedQuery, RankKind, TargetExpr,
                  TimeUnit, Window)
from .parser import PqlSyntaxError

__all__ = ["parse_native", "native_available", "NativeParserUnavailable"]

_OUT = 1 << 16   # 64 KiB JSON buffer — far beyond any real query's AST
_ERR = 1024


class NativeParserUnavailable(RuntimeError):
    pass


def _candidate_paths() -> list[Path]:
    env = os.environ.get("RELATIVEDB_RT_LIB")
    here = Path(__file__).resolve()
    # repo root: .../python/src/relativedb/pql/native.py -> parents[4]
    root = here.parents[4]
    names = ["librt_c.dylib", "librt_c.so", "librt_c.dll", "rt_c.dll"]
    out = [Path(env)] if env else []
    out += [root / "cpp" / "build" / n for n in names]
    return out


_lib: Optional[ctypes.CDLL] = None
_load_failed: Optional[str] = None


def _load() -> Optional[ctypes.CDLL]:
    global _lib, _load_failed
    if _lib is not None or _load_failed is not None:
        return _lib
    for p in _candidate_paths():
        if p and p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                lib.pql_parse.restype = ctypes.c_int
                lib.pql_parse.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                          ctypes.c_size_t, ctypes.c_char_p,
                                          ctypes.c_size_t]
                _lib = lib
                return _lib
            except (OSError, AttributeError) as e:
                _load_failed = f"{p}: {e}"
    if _load_failed is None:
        _load_failed = "librt_c not found (build cpp/ with cmake)"
    return None


def native_available() -> bool:
    return _load() is not None


def parse_native(query: str) -> ParsedQuery:
    """Parse ``query`` with the shared C++ parser; returns the same AST the
    pure-Python :func:`~relativedb.pql.parser.parse` returns."""
    lib = _load()
    if lib is None:
        raise NativeParserUnavailable(_load_failed or "librt_c unavailable")
    if not isinstance(query, str) or not query.strip():
        raise PqlSyntaxError("empty query")
    out = ctypes.create_string_buffer(_OUT)
    err = ctypes.create_string_buffer(_ERR)
    rc = lib.pql_parse(query.encode("utf-8"), out, _OUT, err, _ERR)
    if rc != 0:
        raise PqlSyntaxError(err.value.decode("utf-8", "replace") or "parse failed")
    obj = json.loads(out.value.decode("utf-8"))
    return _query_from_json(obj, query)


# ---------------------------------------------------------------------------
# JSON -> AST (must reproduce the pure-Python parser's values exactly)
# ---------------------------------------------------------------------------
def _bound(x: Any) -> float:
    if x == "inf":
        return math.inf
    if x == "-inf":
        return -math.inf
    return float(x)


def _lit(x: Any) -> Any:
    if isinstance(x, dict) and "date" in x:            # DATE literal
        s = x["date"]
        fmt = "%Y-%m-%d %H:%M:%S" if " " in s else "%Y-%m-%d"
        return datetime.strptime(s, fmt)
    if isinstance(x, list):                            # IN / NOT IN list
        return tuple(_lit(v) for v in x)
    return x                                           # str / int / float / bool / None


def _expr(o: Optional[dict]) -> Optional[TargetExpr]:
    if o is None:
        return None
    kind = o["kind"]
    if kind == "col":
        return ColumnRef(o["table"], o["column"])
    if kind == "agg":
        w = o["window"]
        window = (Window(_bound(w["start"]), _bound(w["end"]),
                         TimeUnit[w["unit"].upper()]) if w else None)
        return Aggregation(AggFunc[o["func"]], ColumnRef(o["column"]["table"],
                           o["column"]["column"]), _expr(o["filter"]), window)
    if kind == "cond":
        return Condition(_expr(o["left"]), Operator[o["op"]], _lit(o["right"]))
    if kind == "logic":
        return LogicalOp(_expr(o["left"]), BoolOp[o["op"]], _expr(o["right"]))
    if kind == "not":
        return Not(_expr(o["expr"]))
    raise ValueError(f"unknown expr kind {kind!r}")


def _query_from_json(o: dict, text: str) -> ParsedQuery:
    ek = o["entity_key"]
    return ParsedQuery(
        target=_expr(o["target"]),
        entity_key=ColumnRef(ek["table"], ek["column"]),
        entity_ids=tuple(_lit(v) for v in o.get("entity_ids", ())),
        where=_expr(o.get("where")),
        assuming=_expr(o.get("assuming")),
        rank=RankKind[o["rank"]] if o.get("rank") else None,
        top_k=o.get("top_k"),
        num_forecasts=o.get("num_forecasts"),
        text=text,
    )
