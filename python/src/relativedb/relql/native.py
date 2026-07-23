"""Binding to the shared C++ RelQL parser (``relql_parse`` in ``librt_c``).

This is *the* RelQL parser for the Python binding — grammar and lexing live once
in the C++ layer, shared with the Java and Rust bindings. The C ABI returns a
JSON AST (see ``cpp/src/relql.cpp``) which we deserialize into the
:mod:`relativedb.relql.ast` dataclasses. ``librt_c`` is a hard dependency (the
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

from .ast import (Ablation, AggFunc, Aggregation, Arith, AsOf, BoolOp, Case,
                  ColumnRef, Condition, Explain, Func, Lit, LogicalOp, Not,
                  Operator, Param, ParsedQuery, RankKind, ReturnSpec,
                  TargetExpr, TimeUnit, Window)
from .parser import RelqlSyntaxError

__all__ = ["parse_native", "native_available", "NativeParserUnavailable"]

_OUT = 1 << 16   # 64 KiB JSON buffer — far beyond any real query's AST
_ERR = 1024


class NativeParserUnavailable(RuntimeError):
    pass


def _candidate_paths() -> list[Path]:
    env = os.environ.get("RELATIVEDB_RT_LIB")
    here = Path(__file__).resolve()
    names = ["librt_c.dylib", "librt_c.so", "librt_c.dll", "rt_c.dll"]
    out = [Path(env)] if env else []
    # in-package drop-in: installed wheels ship the library in the parent
    # relativedb package, next to rt_native.py
    out += [here.parents[1] / n for n in names]
    # sibling C++ build tree of a monorepo checkout
    # (.../python/src/relativedb/relql/native.py -> repo root is parents[4])
    if len(here.parents) > 4:
        root = here.parents[4]
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
                lib.relql_parse.restype = ctypes.c_int
                lib.relql_parse.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
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
    pure-Python :func:`~relativedb.relql.parser.parse` returns."""
    lib = _load()
    if lib is None:
        raise NativeParserUnavailable(_load_failed or "librt_c unavailable")
    if not isinstance(query, str) or not query.strip():
        raise RelqlSyntaxError("empty query")
    out = ctypes.create_string_buffer(_OUT)
    err = ctypes.create_string_buffer(_ERR)
    rc = lib.relql_parse(query.encode("utf-8"), out, _OUT, err, _ERR)
    if rc != 0:
        raise RelqlSyntaxError(err.value.decode("utf-8", "replace") or "parse failed")
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


def _window(w: Optional[dict]) -> Optional[Window]:
    if not w:
        return None
    step = w.get("step")
    return Window(
        _bound(w["start"]), _bound(w["end"]),
        TimeUnit[w["unit"].upper()],
        horizons=int(w.get("horizons", 1)),
        step=(_bound(step) if step is not None else None),
        top_k=w.get("top_k"),
        implied=bool(w.get("implied", False)),
    )


def _expr(o: Optional[dict]) -> Optional[TargetExpr]:
    if o is None:
        return None
    kind = o["kind"]
    if kind == "col":
        return ColumnRef(o["table"], o["column"])
    if kind == "agg":
        return Aggregation(AggFunc[o["func"]], ColumnRef(o["column"]["table"],
                           o["column"]["column"]), _expr(o.get("filter")),
                           _window(o.get("window")))
    if kind == "cond":
        rexpr = _expr(o.get("right_expr"))
        right = None if rexpr is not None else _lit(o.get("right"))
        return Condition(_expr(o["left"]), Operator[o["op"]], right, rexpr)
    if kind == "logic":
        return LogicalOp(_expr(o["left"]), BoolOp[o["op"]], _expr(o["right"]))
    if kind == "not":
        return Not(_expr(o["expr"]))
    if kind == "arith":
        return Arith(o["op"], _expr(o["left"]), _expr(o["right"]))
    if kind == "func":
        return Func(o["name"], tuple(_expr(a) for a in o.get("args", ())))
    if kind == "case":
        whens = tuple((_expr(w["cond"]), _expr(w["then"]))
                      for w in o.get("whens", ()))
        return Case(whens, _expr(o.get("else")))
    if kind == "lit":
        return Lit(_lit(o.get("value")))
    if kind == "param":
        return Param(o["name"])
    raise ValueError(f"unknown expr kind {kind!r}")


def _explain(o: Optional[dict]) -> Optional[Explain]:
    if not o:
        return None
    return Explain(o.get("mode", "PLAN"), o.get("format", "TEXT"))


def _as_of(o: Optional[dict]) -> Optional[AsOf]:
    if not o:
        return None
    return AsOf(o["kind"], o.get("value"))


def _ret(o: Optional[dict]) -> Optional[ReturnSpec]:
    if not o:
        return None
    return ReturnSpec(o["kind"])


def _query_from_json(o: dict, text: str) -> ParsedQuery:
    ek = o["entity_key"]
    return ParsedQuery(
        target=_expr(o["target"]),
        entity_key=ColumnRef(ek["table"], ek.get("column")),
        entity_inferred=bool(ek.get("inferred", False)),
        where=_expr(o.get("where")),
        assuming=_expr(o.get("assuming")),
        rank=RankKind[o["rank"]] if o.get("rank") else None,
        top_k=o.get("top_k"),
        num_forecasts=o.get("num_forecasts"),
        explain=_explain(o.get("explain")),
        as_of=_as_of(o.get("as_of")),
        ablations=tuple(Ablation(a.get("kind", "table"), a.get("name", ""))
                        for a in o.get("ablations", ())),
        ret=_ret(o.get("ret")),
        windows={name: _window(w) for name, w in o.get("windows", {}).items()},
        text=text,
    )
