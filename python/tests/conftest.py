"""Shared fixtures: the worked churn example from kb/example.md as a toy graph.

Also the single place where the unit/integration tier split is implemented:

* ``pytest -m "not integration"`` — the unit tier. No model checkpoint, no
  network, no GPU. Sub-second.
* ``pytest -m integration`` — the real thing: the native compute kernels and
  the rt-j checkpoint end to end.

Sub-markers refine the integration tier for CI scheduling:

* ``native``     — needs ``librt_c`` compute kernels (no HF download)
* ``checkpoint`` — needs an HF checkpoint (rt-j weights and/or MiniLM)

So ``-m "integration and native and not checkpoint"`` is a cheap CI job that
needs only a built ``cpp/``, and ``-m "integration and checkpoint"`` is the
one that needs the warmed model cache.

WHAT THE UNIT TIER DOES *NOT* GUARANTEE
---------------------------------------
It does not run without ``librt_c``. That is by design, not an oversight:
``relativedb.relql.parser.parse`` delegates to ``relql_parse`` in the shared
C++ library only through the optional relativedb-engine package, so any
test that parses a RelQL string needs the library. Measured: with ``librt_c``
made unloadable, 186 of the 214 unit tests fail with
``NativeParserUnavailable``. Making the unit tier library-free would mean
re-introducing a second, pure-Python RelQL grammar — exactly the duplication
the C++ single-sourcing exists to prevent.

``librt_c`` is a build artifact, not a download: CI builds it for the wheel
regardless, and a missing one is already a red build. The property this tier
*does* guarantee, and which is verified, is the expensive/flaky one: no model
checkpoint, no network, no GPU.

Strict mode: with ``RELATIVEDB_REQUIRE_NATIVE=1`` every skip that would fire
for a missing library or an unresolvable checkpoint becomes a hard FAILURE, so
a broken CI cache or a failed download turns the build red instead of quietly
reporting "0 tests ran, all green". Route every such decision through
:func:`require_native` / :func:`require_checkpoint`; do not call
``pytest.skip`` for these conditions directly.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from relativedb import (ColumnDef, EntityPrediction, LinkDef, RetrieverWiring,
                      Row, Schema, TableDef, TaskType, TemporalBound, ValueType)

# ---------------------------------------------------------------------------
# tier machinery: markers, strict mode, empty-selection guard
# ---------------------------------------------------------------------------

STRICT_ENV = "RELATIVEDB_REQUIRE_NATIVE"


def strict_native() -> bool:
    """True when missing checkpoint prerequisites must FAIL, not skip."""
    return os.environ.get(STRICT_ENV, "").strip().lower() in (
        "1", "true", "yes", "on")


def _unavailable(what: str, detail: str):
    """Skip, or under strict mode fail loudly. Never returns."""
    if strict_native():
        pytest.fail(
            f"{STRICT_ENV}=1 but {what} is unavailable: {detail}\n"
            "This is the integration tier; a missing prerequisite here means a "
            "broken build/cache, not a test that may be skipped.",
            pytrace=False)
    pytest.skip(f"{what} unavailable: {detail}")


def require_checkpoint(variant: str) -> str:
    """Resolve ``hf://RelativeDB/rt-j-fp16/<variant>``, or skip/fail per strict
    mode. Cache-first; on CI the HF cache is pre-warmed, so a miss here means
    the cache key is wrong or the download failed."""
    from relativedb.rt import resolve_model_path
    uri = f"hf://RelativeDB/rt-j-fp16/{variant}"
    try:
        return resolve_model_path(uri)
    except Exception as e:                       # noqa: BLE001 - report anything
        _unavailable(f"rt-j {variant} checkpoint ({uri})", f"{type(e).__name__}: {e}")


def require_text_embedder():
    """The pinned MiniLM encoder, served by transformers in torch. Skips/fails
    like the other prerequisites — it needs a model snapshot, so it never
    belongs in the unit tier."""
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        _unavailable("transformers (pip install relativedb)", str(e))
    from relativedb.rt import resolve_minilm_snapshot
    try:
        if resolve_minilm_snapshot() is None:
            _unavailable("MiniLM snapshot", "huggingface_hub is unavailable")
    except Exception as e:                       # noqa: BLE001
        _unavailable("MiniLM snapshot", f"{type(e).__name__}: {e}")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: needs a real model checkpoint; excluded from the unit "
        "tier")
    config.addinivalue_line(
        "markers", "checkpoint: needs an HF model checkpoint (rt-j / MiniLM)")


@pytest.hookimpl(trylast=True)     # after pytest's own marker deselection
def pytest_collection_modifyitems(config, items):
    """A marker expression that selects nothing is an error, not a pass.

    Without this, a typo'd ``-m integration`` or a suite that lost its markers
    exits 0 with "no tests ran" and CI goes green having verified nothing.
    """
    markexpr = getattr(config.option, "markexpr", "")
    if markexpr and not items:
        raise pytest.UsageError(
            f"marker expression {markexpr!r} selected 0 tests. Refusing to "
            "report success for an empty run.")


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


class StubScorer:
    """Scorer double for assembly-only tests: sequence building, validation
    and error paths never reach the model, so forward/embed must not be hit
    unless a test explicitly wants deterministic outputs."""

    def __init__(self, score_value: float = 0.0):
        self.score_value = score_value
        self.forwards = 0

    def forward(self, batch, *, model_uri, output="target_scores"):
        import numpy as np

        from relativedb.scoring import D_MODEL, D_TEXT, ForwardResult
        self.forwards += 1
        B, S = batch.b, batch.s
        if output == "token_scores":
            return ForwardResult(scores=np.full((B, S), self.score_value,
                                                np.float32))
        if output == "target_features":
            return ForwardResult(features=np.zeros((B, D_MODEL), np.float32))
        if output == "target_scores_and_text":
            return ForwardResult(scores=np.full(B, self.score_value, np.float32),
                                 target_text=np.zeros((B, D_TEXT), np.float32))
        return ForwardResult(scores=np.full(B, self.score_value, np.float32))

    def embed(self, texts, *, normalize=False):
        import numpy as np

        from relativedb.scoring import D_TEXT
        return np.zeros((len(list(texts)), D_TEXT), np.float32)


class StubBackend:
    """Tiny deterministic test-only ModelBackend. The engine ships no scorer;
    plumbing tests (routing, AS OF, CSC execute, EXPLAIN ANALYZE) use this so
    they stay fast and offline without a real checkpoint. RETURN output-shaping
    is a native-backend concern and is exercised in test_rt_native.py."""

    def score(self, query, task_type, contexts, model_uri, config):
        binary = task_type is TaskType.BINARY_CLASSIFICATION
        return [EntityPrediction(c.entity_id,
                                 probability=0.5 if binary else None,
                                 value=None if binary else 1.0)
                for c in contexts]


@pytest.fixture
def stub_backend() -> StubBackend:
    return StubBackend()


@pytest.fixture
def churn_schema() -> Schema:
    return (Schema.new_schema()
            .table(TableDef.new_table("customers")
                   .column("age", ValueType.NUMBER)
                   .column("signup_date", ValueType.DATETIME)
                   .primary_key("customer_id").build())
            .table(TableDef.new_table("products")
                   .column("price", ValueType.NUMBER)
                   .column("name", ValueType.TEXT)
                   .primary_key("product_id").build())
            .table(TableDef.new_table("orders")
                   .column("qty", ValueType.NUMBER)
                   .column("order_date", ValueType.DATETIME)
                   .primary_key("order_id")
                   .time_column("order_date").build())
            .link(LinkDef("orders", "customer_id", "customers"))
            .link(LinkDef("orders", "product_id", "products"))
            .build())


def churn_rows() -> dict[str, list[Row]]:
    """The kb/example.md database. O4 (2026-07-05) is AFTER the anchor t0 =
    2026-07-01 and must never enter context."""
    customers = [
        Row("customers", "C1", {"age": 34.0, "signup_date": dt("2026-02-10")}),
        Row("customers", "C7", {"age": 52.0, "signup_date": dt("2026-01-20")}),
        Row("customers", "C9", {"age": 27.0, "signup_date": dt("2026-03-05")}),
    ]
    products = [
        Row("products", "P1", {"price": 25.0, "name": "running shoes"}),
        Row("products", "P2", {"price": 90.0, "name": "espresso machine"}),
        Row("products", "P3", {"price": 35.0, "name": "yoga mat"}),
    ]
    orders = [
        Row("orders", "O1", {"qty": 1.0, "order_date": dt("2026-03-10")},
            timestamp=dt("2026-03-10"),
            parents={"customer_id": "C7", "product_id": "P2"}),
        Row("orders", "O2", {"qty": 2.0, "order_date": dt("2026-05-02")},
            timestamp=dt("2026-05-02"),
            parents={"customer_id": "C7", "product_id": "P1"}),
        Row("orders", "O3", {"qty": 1.0, "order_date": dt("2026-06-20")},
            timestamp=dt("2026-06-20"),
            parents={"customer_id": "C1", "product_id": "P3"}),
        Row("orders", "O4", {"qty": 1.0, "order_date": dt("2026-07-05")},
            timestamp=dt("2026-07-05"),  # future of t0!
            parents={"customer_id": "C7", "product_id": "P3"}),
    ]
    return {"customers": customers, "products": products, "orders": orders}


def in_memory_wiring(rows: dict[str, list[Row]], *,
                     honor_bound: bool = True) -> RetrieverWiring:
    """Well-behaved (or, with honor_bound=False, deliberately leaky)
    retrievers + scanners over an in-memory row dict."""
    by_id = {t: {r.id: r for r in rs} for t, rs in rows.items()}

    def entity(table, ids, bound: TemporalBound):
        out = []
        for i in ids:
            r = by_id[table].get(i)
            if r is None:
                continue
            if honor_bound and not bound.admits_row(r):
                continue
            out.append(r)
        return out

    def links(link, parent_id, bound: TemporalBound, limit):
        kids = [r for r in rows[link.from_table]
                if r.parents.get(link.fk_column) == parent_id]
        if honor_bound:
            kids = [r for r in kids if bound.admits_row(r)]
        kids.sort(key=lambda r: (r.timestamp is None,
                                 -(r.timestamp.timestamp() if r.timestamp
                                   else 0.0)))
        return kids[:limit] if honor_bound else kids

    def make_scanner(table):
        def scan(t, bound: TemporalBound):
            for r in rows[table]:
                if not honor_bound or bound.admits_row(r):
                    yield r
        return scan

    wb = RetrieverWiring.new_wiring().default_links(links)
    for t in rows:
        wb.entities(t, entity)
        wb.scanner(t, make_scanner(t))
    return wb.build()


@pytest.fixture
def churn_wiring() -> RetrieverWiring:
    return in_memory_wiring(churn_rows())
