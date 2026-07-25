"""The snapshot, presented to RelativeDB as a seven-table relational database.

    events ──< markets ──< price_ticks
       │           │
       │           └──< market_tags >── tags
       │
       └──< news_mentions >── news_articles

Nothing here is feature engineering: no rolling means, no sentiment scores, no
TF-IDF between headlines and questions. The tables are the raw records, the
links say how they relate, and RT-J discovers the subgraph it wants. The one
derived column is ``price_ticks.ret`` (this hour's close minus last hour's),
which exists because it is the *target* — the thing being aggregated over the
future window — not because it is an input feature.

Time is the whole game. Every fact carries the timestamp at which it became
true, so the engine's temporal bound can cut the database at the anchor and
guarantee that a prediction never reads a candle or a headline from its own
future.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from relativedb import (LinkDef, RetrieverWiring, Row, Schema, TableDef,
                        TemporalBound, ValueType)

SNAPSHOT = Path(__file__).parent / "data" / "snapshot.json"

NEWS_TABLES = ("news_articles", "news_mentions")


def load(path: Path = SNAPSHOT) -> dict:
    if not path.exists():
        raise SystemExit(f"no snapshot at {path} — run `python fetch.py` first")
    return json.loads(path.read_text())


def ts(value) -> datetime:
    """Milliseconds since epoch (pmxt candles) or an ISO string (GDELT)."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    return datetime.fromisoformat(value)


def floor_hour(when: datetime) -> datetime:
    return when.replace(minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# labels: the realized move over the held-out window
# ---------------------------------------------------------------------------
def price_at(bars, when: datetime):
    """Last close at or before ``when`` — the price a trader could see then."""
    last = None
    for bar in bars:
        if ts(bar["ts"]) <= when:
            last = bar["close"]
        else:
            break
    return last


def labels(snapshot: dict, anchor: datetime, horizon_hours: int) -> dict:
    """market_id -> (price at anchor, price at anchor+horizon, move, went_up).

    Markets without a quote on both sides of the window are dropped rather
    than filled: an imputed label is a fabricated one."""
    out = {}
    for market_id, bars in snapshot["candles"].items():
        bars = sorted(bars, key=lambda b: b["ts"])
        before = price_at(bars, anchor)
        after = price_at(bars, anchor + timedelta(hours=horizon_hours))
        if before is None or after is None:
            continue
        if ts(bars[-1]["ts"]) < anchor + timedelta(hours=horizon_hours - 1):
            continue                      # window not fully observed yet
        out[market_id] = (before, after, after - before, after > before)
    return out


# ---------------------------------------------------------------------------
# schema + rows
# ---------------------------------------------------------------------------
def build(snapshot: dict, *, with_news: bool = True):
    """Return (schema, wiring, rows). ``with_news=False`` drops the two news
    tables entirely — the ablation that answers "did the headlines matter?"."""
    tables = [
        TableDef.new_table("events")
        .column("title", ValueType.TEXT)
        .column("category", ValueType.TEXT)
        .primary_key("event_id").build(),

        TableDef.new_table("markets")
        .column("question", ValueType.TEXT)
        .column("outcome_label", ValueType.TEXT)
        .column("category", ValueType.TEXT)
        .column("resolution_date", ValueType.DATETIME)
        .column("tick_size", ValueType.NUMBER)
        .primary_key("market_id").build(),

        TableDef.new_table("price_ticks")
        .column("ts", ValueType.DATETIME)
        .column("close", ValueType.NUMBER)
        .column("ret", ValueType.NUMBER)
        .column("high", ValueType.NUMBER)
        .column("low", ValueType.NUMBER)
        .column("volume", ValueType.NUMBER)
        .primary_key("tick_id").time_column("ts").build(),

        TableDef.new_table("tags")
        .column("name", ValueType.TEXT)
        .primary_key("tag_id").build(),

        TableDef.new_table("market_tags")
        .primary_key("market_tag_id").build(),
    ]
    links = [
        LinkDef("markets", "event_id", "events"),
        LinkDef("price_ticks", "market_id", "markets"),
        LinkDef("market_tags", "market_id", "markets"),
        LinkDef("market_tags", "tag_id", "tags"),
    ]
    if with_news:
        tables += [
            TableDef.new_table("news_articles")
            .column("published_at", ValueType.DATETIME)
            .column("headline", ValueType.TEXT)
            .column("domain", ValueType.TEXT)
            .column("country", ValueType.TEXT)
            .primary_key("article_id").time_column("published_at").build(),

            TableDef.new_table("news_mentions")
            .column("observed_at", ValueType.DATETIME)
            .column("matched_query", ValueType.TEXT)
            .primary_key("mention_id").time_column("observed_at").build(),
        ]
        links += [
            LinkDef("news_mentions", "article_id", "news_articles"),
            LinkDef("news_mentions", "event_id", "events"),
        ]

    rows: dict[str, list[Row]] = {t.name: [] for t in tables}

    for e in snapshot["events"]:
        rows["events"].append(Row("events", e["event_id"], {
            "title": e["title"], "category": e["category"]}))

    tag_ids: dict[str, str] = {}
    for m in snapshot["markets"]:
        # Deliberately absent: volume / liquidity / current price. Those come
        # from the fetch, which happens *after* the anchor, and a backtest may
        # not read anything stamped after its own cutoff.
        rows["markets"].append(Row("markets", m["market_id"], {
            "question": m["question"],
            "outcome_label": m["outcome_label"],
            "category": m["category"],
            "resolution_date": (datetime.fromisoformat(m["resolution_date"])
                                if m["resolution_date"] else None),
            "tick_size": m["tick_size"],
        }, None, {"event_id": m["event_id"]}))
        for name in m["tags"]:
            tag_id = tag_ids.setdefault(name, str(len(tag_ids)))
            rows["market_tags"].append(Row(
                "market_tags", f'{m["market_id"]}:{tag_id}', {}, None,
                {"market_id": m["market_id"], "tag_id": tag_id}))
    for name, tag_id in tag_ids.items():
        rows["tags"].append(Row("tags", tag_id, {"name": name}))

    for market_id, bars in snapshot["candles"].items():
        previous = None
        for bar in sorted(bars, key=lambda b: b["ts"]):
            when = ts(bar["ts"])
            rows["price_ticks"].append(Row(
                "price_ticks", f"{market_id}:{bar['ts']}", {
                    "ts": when,
                    "close": bar["close"],
                    "ret": (None if previous is None
                            else round(bar["close"] - previous, 6)),
                    "high": bar["high"], "low": bar["low"],
                    "volume": bar["volume"],
                }, when, {"market_id": market_id}))
            previous = bar["close"]

    if with_news:
        for a in snapshot["articles"]:
            when = ts(a["published_at"])
            rows["news_articles"].append(Row("news_articles", a["article_id"], {
                "published_at": when, "headline": a["headline"],
                "domain": a["domain"], "country": a["country"]}, when))
        for mention in snapshot["mentions"]:
            when = ts(mention["observed_at"])
            rows["news_mentions"].append(Row(
                "news_mentions", mention["mention_id"], {
                    "observed_at": when,
                    "matched_query": mention["matched_query"]}, when,
                {"article_id": mention["article_id"],
                 "event_id": mention["event_id"]}))

    schema = Schema(tuple(tables), tuple(links))
    by_id = {name: {row.id: row for row in table_rows}
             for name, table_rows in rows.items()}
    children: dict[tuple[str, str], dict] = {}
    for link in links:
        index: dict = {}
        for row in rows[link.from_table]:
            parent = row.parents.get(link.fk_column)
            if parent is not None:
                index.setdefault(parent, []).append(row)
        for bucket in index.values():                 # newest first
            bucket.sort(key=lambda r: (r.timestamp is None,
                                       -(r.timestamp.timestamp()
                                         if r.timestamp else 0.0)))
        children[(link.from_table, link.fk_column)] = index

    def entities(table, ids, bound: TemporalBound):
        return [row for i in ids
                if (row := by_id[table].get(i)) is not None
                and bound.admits_row(row)]

    def link_rows(link, parent_id, bound: TemporalBound, limit: int):
        index = children[(link.from_table, link.fk_column)]
        found = [row for row in index.get(parent_id, ())
                 if bound.admits_row(row)]
        return found[:limit]

    def scanner(table, bound: TemporalBound):
        return (row for row in rows[table] if bound.admits_row(row))

    wiring = RetrieverWiring.new_wiring().default_links(link_rows)
    for table in rows:
        wiring.entities(table, entities)
        wiring.scanner(table, scanner)
    return schema, wiring.build(), rows
