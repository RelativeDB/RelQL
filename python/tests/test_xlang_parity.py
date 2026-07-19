"""Cross-language ranking-parity test (Python side).

Loads the shared fixture in ``benchmarks/xlang_fixture/`` — a fixed real
MovieLens (ml-latest-small) Top-5 ranking scenario — runs it through the
native RT-J backend, and asserts the ranking is correct AND non-degenerate.

It guards against the two ranking bugs found on 2026-07-18 (Python emitting no
candidate cells; Java emitting no target token for cell-less entity tables),
both of which produced the degenerate candidate-enumeration order
``[1, 2, 3, 50, 260]``. See ``benchmarks/xlang_fixture/README.md``.

Run:
    cd python && SSL_CERT_FILE=/etc/ssl/cert.pem \
      RELATIVEDB_RT_LIB=/Users/henneberger/getasterisk/cpp/build/librt_c.dylib \
      .venv/bin/python -m pytest tests/test_xlang_parity.py -q

Skips cleanly (like ``tests/test_rt_native.py``) if the native lib / checkpoint
is unavailable.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from relativedb import (ContextPolicy, Engine, ExecutionInput, LinkDef,
                        RetrieverWiring, Row, SamplerMode, Schema, TableDef,
                        TemporalBound, ValueType)
from relativedb.rt_native import (RtNativeBackend, RtNativeUnavailableError,
                               load_lib, resolve_model_path)

FIXTURE = (Path(__file__).resolve().parent.parent.parent
           / "benchmarks" / "xlang_fixture")

# Mirror benchmarks/harness/datasets.WIDE_POLICY: one hop entity->events, a
# generous budget so the full per-user history + all candidates enter context.
WIDE_POLICY = ContextPolicy(max_context_cells=5_000_000, bfs_width=20_000,
                            max_hops=1)

_TYPES = {"TEXT": ValueType.TEXT, "NUMBER": ValueType.NUMBER,
          "DATETIME": ValueType.DATETIME, "BOOLEAN": ValueType.BOOLEAN}


def _lib_or_skip():
    try:
        return load_lib()
    except RtNativeUnavailableError as e:
        pytest.skip(f"librt_c not available: {e}")


def _checkpoint_or_skip(variant: str = "classification") -> str:
    try:
        return resolve_model_path(f"hf://stanford-star/rt-j/{variant}")
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"rt-j {variant} checkpoint not available: {e}")


def _golden() -> dict:
    if not FIXTURE.is_dir():
        pytest.skip(f"xlang fixture not found at {FIXTURE}")
    return json.loads((FIXTURE / "golden.json").read_text())


def _build_schema(spec: dict) -> Schema:
    b = Schema.new_schema()
    for name, t in spec.items():
        if name == "links":
            continue
        tb = TableDef.new_table(name)
        for col, ty in t["columns"]:
            tb = tb.column(col, _TYPES[ty])
        tb = tb.primary_key(t["primary_key"])
        if t.get("time_column"):
            tb = tb.time_column(t["time_column"])
        b = b.table(tb.build())
    for frm, fk, to in spec["links"]:
        b = b.link(LinkDef(frm, fk, to))
    return b.build()


def _load_rows() -> dict[str, list[Row]]:
    movies: list[Row] = []
    for line in (FIXTURE / "movies.tsv").read_text().splitlines():
        if not line:
            continue
        mid, title, genres = line.split("\t")
        movies.append(Row("movies", int(mid),
                          {"title": title, "genres": genres}))

    ratings: list[Row] = []
    user_ids: set[int] = set()
    for line in (FIXTURE / "ratings.tsv").read_text().splitlines():
        if not line:
            continue
        rid, uid, mid, rating, ts_epoch = line.split("\t")
        ts = datetime.fromtimestamp(int(ts_epoch), tz=timezone.utc)
        user_ids.add(int(uid))
        ratings.append(Row("ratings", int(rid),
                           {"rating": float(rating), "ts": ts},
                           timestamp=ts,
                           parents={"user_id": int(uid),
                                    "movie_id": int(mid)}))

    users = [Row("users", u) for u in sorted(user_ids)]
    return {"users": users, "movies": movies, "ratings": ratings}


def _wire(rows: dict[str, list[Row]]) -> RetrieverWiring:
    by_id = {t: {r.id: r for r in rs} for t, rs in rows.items()}

    def entities(table, ids, bound: TemporalBound):
        out = []
        for i in ids:
            r = by_id[table].get(i)
            if r is not None and bound.admits_row(r):
                out.append(r)
        return out

    def links(link, parent_id, bound: TemporalBound, limit):
        kids = [r for r in rows[link.from_table]
                if r.parents.get(link.fk_column) == parent_id
                and bound.admits_row(r)]
        kids.sort(key=lambda r: r.timestamp.timestamp() if r.timestamp
                  else float("-inf"), reverse=True)
        return kids[:limit]

    def make_scanner(table):
        def scan(t, bound: TemporalBound):
            return (r for r in rows[table] if bound.admits_row(r))
        return scan

    wb = RetrieverWiring.new_wiring().default_links(links)
    for t in rows:
        wb.entities(t, entities).scanner(t, make_scanner(t))
    return wb.build()


def _engine(schema: Schema, wiring: RetrieverWiring) -> Engine:
    backend = RtNativeBackend(schema=schema, wiring=wiring)
    return Engine(schema, wiring, model_backend=backend,
                  sampler_mode=SamplerMode.CSC, context_policy=WIDE_POLICY)


def _rank(engine: Engine, query: str, anchor: datetime,
          users: list[int]) -> dict[int, list[int]]:
    """entity_id -> ranked movie_id list (ints)."""
    res = engine.execute(ExecutionInput(
        query=query, anchor_time=anchor, entity_ids=users))
    assert res.task_type.name == "MULTILABEL_RANKING", res.task_type
    return {int(p.id): [int(x) for x in p.ranked] for p in res.predictions}


def test_xlang_ranking_parity():
    """Non-degenerate ranking + top1==593 for both users + reproduce the
    Python per-binding golden top-5, and (scores not exposed on the ranking
    API) prove candidate discrimination via a full RANK TOP 10 whose order is
    not the sorted candidate-id order."""
    pytest.importorskip("sentence_transformers")
    _lib_or_skip()
    _checkpoint_or_skip()

    g = _golden()
    inv = g["invariants"]
    users = [int(u) for u in g["users"]]
    anchor = datetime.utcfromtimestamp(g["anchor_epoch"])
    degenerate = list(inv["must_not_equal_degenerate_order"])
    expected_top1 = {int(k): int(v) for k, v in inv["expected_top1"].items()}
    golden_py = {int(k): [int(x) for x in v]
                 for k, v in g["per_binding_golden"]["python"].items()}
    candidates_sorted = sorted(int(c) for c in g["candidate_ids"])

    schema = _build_schema(g["schema"])
    wiring = _wire(_load_rows())
    engine = _engine(schema, wiring)

    # --- The golden RANK TOP 5 query --------------------------------------
    top5 = _rank(engine, g["query"], anchor, users)
    assert set(top5) == set(users), top5
    for u in users:
        got = top5[u]
        # (1) not the degenerate candidate-enumeration order
        assert got != degenerate, (
            f"user {u} returned the degenerate order {got} — ranking bug")
        # (2) top1 is Silence of the Lambs (the strong, tie-free signal)
        assert got[0] == expected_top1[u], (
            f"user {u} top1={got[0]} expected {expected_top1[u]}")
        # (4) reproduce the captured Python per-binding golden top-5
        assert got == golden_py[u], (
            f"user {u} top-5 {got} != python golden {golden_py[u]}")

    # --- (3) candidate discrimination via full RANK TOP 10 ----------------
    # The ranking API exposes only the ordered ids (no per-candidate scores),
    # so we rank ALL candidates and assert the order is not the sorted
    # candidate-id order (which is what the degenerate bug produced) and that
    # top1 is still 593 — proving the candidates are genuinely discriminated.
    full_query = g["query"].replace(
        f"RANK TOP {g['top_k']}", f"RANK TOP {len(candidates_sorted)}")
    full = _rank(engine, full_query, anchor, users)
    for u in users:
        order = full[u]
        assert sorted(order) == candidates_sorted, (
            f"user {u} full ranking {order} is not a permutation of the "
            f"candidate set {candidates_sorted}")
        assert order != candidates_sorted, (
            f"user {u} full ranking equals the sorted candidate-id order "
            f"{order} — candidates are not discriminated (ranking bug)")
        assert order[0] == expected_top1[u], (
            f"user {u} full-rank top1={order[0]} expected {expected_top1[u]}")
        assert len(set(order)) >= inv["min_distinct_scores"], (
            f"user {u} produced fewer than {inv['min_distinct_scores']} "
            f"distinct ranked candidates")
