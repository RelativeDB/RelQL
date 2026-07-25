"""Two exchanges and a month of world news, as one relational database.

    events ──< markets ──< price_ticks
       │
       └──< news_mentions >── news_articles

`markets` is the population: one row per binary contract from either venue,
carrying the question text and the settled outcome. Everything else is
evidence — the hourly tape (price, dollar flow, distinct takers) and the
headlines that mentioned the market's subject before the anchor.

Two rules keep the backtest honest, both enforced in the retrievers rather
than by convention:

  1. Every row is stamped with the moment it became knowable, and the engine's
     temporal bound drops anything later.
  2. A *peer* market's outcome is masked unless that market had already
     settled at the anchor. Without this, the model reads tomorrow's results
     off yesterday's context — the subtlest leak in the whole design, and the
     one a timestamp filter alone does not catch, because a market row exists
     long before its outcome does.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq

from relativedb import (LinkDef, RetrieverWiring, Row, Schema, TableDef,
                        TemporalBound, ValueType)

SCALE = Path(__file__).resolve().parent.parent / "data" / "scale"

# Which pulled tables this module reads. `use_sources` swaps in a wider window
# without duplicating the builder: the enlarged study is the same code over a
# bigger universe, so any difference in the result is the population, not the
# pipeline.
SOURCES = {"pm": "pm", "kalshi": "kalshi",
           "news": ["news_articles.parquet"],
           "semantic": "semantic_mentions.parquet"}


def use_sources(*, pm: str = None, kalshi: str = None, news: list = None,
                semantic: str = None):
    if pm:
        SOURCES["pm"] = pm
    if kalshi:
        SOURCES["kalshi"] = kalshi
    if news:
        SOURCES["news"] = list(news)
    if semantic:
        SOURCES["semantic"] = semantic
    return SOURCES


WIDE = dict(pm="pm_wide", kalshi="kalshi_wide",
            news=["news_articles.parquet", "news_aug_oct.parquet"],
            semantic="semantic_mentions_wide.parquet")

STOP = {
    "will", "the", "and", "for", "with", "from", "that", "this", "have", "has",
    "before", "after", "than", "more", "less", "above", "below", "between",
    "what", "when", "which", "who", "whom", "whose", "there", "their", "them",
    "into", "over", "under", "about", "any", "all", "how", "many", "much",
    "does", "did", "was", "were", "been", "being", "are", "his", "her", "its",
    "2025", "2026", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december", "yes", "no",
    "market", "markets", "price", "prices", "win", "wins", "won", "new",
    "next", "day", "days", "week", "weeks", "month", "year", "end", "top",
}

TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")


def tokens(text: str) -> set[str]:
    return {w.lower() for w in TOKEN.findall(text or "")} - STOP


def stamp(value) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@dataclass
class Corpus:
    rows: dict[str, list[Row]]
    markets: list[dict]          # market_id, question, close_at, anchor_at, yes
    counts: dict[str, int]


# ---------------------------------------------------------------------------
# venues
# ---------------------------------------------------------------------------
def load_venues(venues: tuple[str, ...]):
    """Both exchanges, normalized to one shape: markets, event titles, and
    hourly bars keyed by a venue-prefixed id."""
    market_meta: list[dict] = []
    event_titles: dict[str, str] = {}
    ticks: dict[str, list] = {}

    if "polymarket" in venues:
        for t in pq.read_table(SCALE / f'{SOURCES["pm"]}_price_ticks.parquet').to_pylist():
            ticks.setdefault(f"pm:{t['market_id']}", []).append({
                "hour": stamp(t["hour"]), "vwap": t["vwap"],
                "close": t["close"], "high": t["high"], "low": t["low"],
                "usd": t["usd"], "buy_usd": t["buy_usd"],
                "sell_usd": t["sell_usd"], "takers": t["takers"],
                "fills": t["fills"]})
        for r in pq.read_table(SCALE / f'{SOURCES["pm"]}_markets.parquet').to_pylist():
            event_id = f"pm:{r['event_id']}"
            event_titles.setdefault(event_id, r["event_title"] or r["question"])
            market_meta.append({
                "market_id": f"pm:{r['id']}", "event_id": event_id,
                "venue": "polymarket", "question": r["question"],
                "outcome_label": r["answer1"],
                "opened_at": stamp(r["created_at"]),
                "closes_at": stamp(r["end_date"]),
                "resolved_yes": bool(r["resolved_yes"])})

    if "kalshi" in venues:
        for t in pq.read_table(SCALE / f'{SOURCES["kalshi"]}_price_ticks.parquet').to_pylist():
            ticks.setdefault(f"ks:{t['ticker']}", []).append({
                "hour": stamp(t["hour"]), "vwap": t["vwap"],
                "close": t["close"], "high": t["high"], "low": t["low"],
                "usd": None, "buy_usd": None, "sell_usd": None,
                "takers": None, "fills": t["fills"]})
        for r in pq.read_table(SCALE / f'{SOURCES["kalshi"]}_markets.parquet').to_pylist():
            event_id = f"ks:{r['event_ticker']}"
            event_titles.setdefault(event_id, r["title"])
            market_meta.append({
                "market_id": f"ks:{r['ticker']}", "event_id": event_id,
                "venue": "kalshi", "question": r["title"],
                "outcome_label": r["yes_sub_title"],
                "opened_at": stamp(r["open_time"]),
                "closes_at": stamp(r["close_time"]),
                "resolved_yes": bool(r["resolved_yes"])})
    return market_meta, event_titles, ticks


def live_universe(*, lead_hours: int = 48, min_ticks: int = 6,
                  venues: tuple[str, ...] = ("polymarket", "kalshi"),
                  tick_history_hours: int = 336):
    """Markets that have a tape before their anchor — the ones actually
    predictable — with their event titles. Shared by the builder and the
    semantic linker so both see exactly the same universe."""
    market_meta, event_titles, ticks = load_venues(venues)
    lead, history = timedelta(hours=lead_hours), timedelta(hours=tick_history_hours)
    kept = []
    for m in market_meta:
        # Floored to midnight UTC: markets that close at 23:59 and at 12:00
        # then share an anchor, and the engine scores a whole day's cohort in
        # one pass instead of one pass per market. Lead time becomes 48-72h
        # rather than exactly 48, and is recorded per market.
        anchor = (m["closes_at"] - lead).replace(hour=0, minute=0, second=0,
                                                 microsecond=0)
        bars = [b for b in ticks.get(m["market_id"], ())
                if anchor - history <= b["hour"] <= anchor]
        if len(bars) < min_ticks:
            continue
        m["anchor_at"] = anchor
        m["bars"] = sorted(bars, key=lambda b: b["hour"])
        m["price_at_anchor"] = m["bars"][-1]["vwap"]
        kept.append(m)
    titles = {m["event_id"]: event_titles[m["event_id"]] for m in kept}
    return kept, titles


def build_events(**kwargs) -> dict[str, str]:
    return live_universe(**kwargs)[1]


def read_articles(columns=("article_id", "published_at", "headline", "domain",
                           "tone", "persons", "orgs")) -> list[dict]:
    out = []
    for name in SOURCES["news"]:
        out.extend(pq.read_table(SCALE / name,
                                 columns=list(columns)).to_pylist())
    return out


def candidate_articles(events: dict[str, str], *, min_overlap: int = 1,
                       limit: int = 400_000, max_df: float = 0.02):
    """Cheap prefilter before the expensive part: an article is a candidate if
    it shares at least one distinctive word with some event title. Embedding
    900k headlines to find 20k relevant ones is a waste of an afternoon."""
    index: dict[str, set[str]] = {}
    for event_id, title in events.items():
        for token in tokens(title):
            index.setdefault(token, set()).add(event_id)
    cutoff = max(2, int(len(events) * max_df))
    vocabulary = {t for t, e in index.items() if len(e) <= cutoff}
    out = []
    for article in read_articles():
        text = f"{article['headline']} {article['persons']} {article['orgs']}"
        if len(tokens(text) & vocabulary) >= min_overlap:
            out.append(article)
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# news -> event linking
# ---------------------------------------------------------------------------
def link_news(events: dict[str, str], articles, *, min_overlap: int,
              max_per_event: int, max_df: float):
    """Attach headlines to events by distinctive shared vocabulary.

    A token that appears in more than ``max_df`` of event titles ("bitcoin"
    across 300 strike markets) carries no information about *which* event an
    article is about, so it is dropped from the matching vocabulary. What is
    left is names, places and specifics."""
    index: dict[str, set[str]] = {}
    for event_id, title in events.items():
        for token in tokens(title):
            index.setdefault(token, set()).add(event_id)
    cutoff = max(2, int(len(events) * max_df))
    index = {t: e for t, e in index.items() if len(e) <= cutoff}

    mentions, keep = [], {}
    for article in articles:
        hits: dict[str, int] = {}
        text = f"{article['headline']} {article['persons']} {article['orgs']}"
        for token in tokens(text):
            for event_id in index.get(token, ()):
                hits[event_id] = hits.get(event_id, 0) + 1
        for event_id, overlap in hits.items():
            if overlap < min_overlap:
                continue
            keep[article["article_id"]] = article
            mentions.append((event_id, article, overlap))
    return keep, mentions


def cap_uniformly(mentions, *, per_bucket: int, bucket_days: int):
    """Trim mentions per event *per time bucket*, keeping the best matches.

    A single global cap per event is not time-neutral: articles arrive in
    chronological order, so a first-come cap fills up with the oldest matches
    and starves the end of the window. Measured on the first version of this
    study, a 40-per-event cap left 96% of linked articles before December 10
    and 4% after, out of a corpus that was 71/29 — the holdout period was
    nearly newsless, which is precisely the period the ablation is scored on.

    Bucketing by week and capping inside each bucket makes coverage flat in
    time, and ranking within a bucket keeps the *strongest* matches rather
    than the earliest.

    ``mentions`` is a list of (event_id, article, overlap, similarity).
    """
    buckets: dict[tuple[str, int], list] = {}
    for event_id, article, overlap, similarity in mentions:
        when = stamp(article["published_at"])
        key = (event_id, when.toordinal() // bucket_days)
        buckets.setdefault(key, []).append(
            (event_id, article, overlap, similarity, when))
    out = []
    for group in buckets.values():
        # strongest first: semantic similarity when present, else word overlap
        group.sort(key=lambda m: (-(m[3] if m[3] is not None else 0.0),
                                  -m[2], m[4]))
        out.extend(group[:per_bucket])
    return [(event_id, article, overlap, similarity)
            for event_id, article, overlap, similarity, _ in out]


# ---------------------------------------------------------------------------
def build(*, lead_hours: int = 48, min_ticks: int = 6,
          venues: tuple[str, ...] = ("polymarket", "kalshi"),
          min_overlap: int = 2, max_per_event: int = 40,
          max_df: float = 0.02, with_news: bool = True,
          tick_history_hours: int = 336,
          per_bucket: int = 12, bucket_days: int = 7):
    tables = [
        TableDef.new_table("events")
        .column("title", ValueType.TEXT)
        .column("venue", ValueType.TEXT)
        .primary_key("event_id").build(),

        TableDef.new_table("markets")
        .column("question", ValueType.TEXT)
        .column("outcome_label", ValueType.TEXT)
        .column("venue", ValueType.TEXT)
        .column("opened_at", ValueType.DATETIME)
        .column("closes_at", ValueType.DATETIME)
        .column("known_at", ValueType.DATETIME)         # the anchor itself
        .column("resolved_yes", ValueType.BOOLEAN)      # the target
        .primary_key("market_id").time_column("known_at").build(),

        TableDef.new_table("price_ticks")
        .column("hour", ValueType.DATETIME)
        .column("vwap", ValueType.NUMBER)
        .column("close", ValueType.NUMBER)
        .column("high", ValueType.NUMBER)
        .column("low", ValueType.NUMBER)
        .column("usd", ValueType.NUMBER)
        .column("buy_usd", ValueType.NUMBER)
        .column("sell_usd", ValueType.NUMBER)
        .column("takers", ValueType.NUMBER)
        .column("fills", ValueType.NUMBER)
        .primary_key("tick_id").time_column("hour").build(),
    ]
    links = [LinkDef("markets", "event_id", "events"),
             LinkDef("price_ticks", "market_id", "markets")]
    if with_news:
        tables += [
            TableDef.new_table("news_articles")
            .column("published_at", ValueType.DATETIME)
            .column("headline", ValueType.TEXT)
            .column("domain", ValueType.TEXT)
            .column("tone", ValueType.NUMBER)
            .column("persons", ValueType.TEXT)
            .column("orgs", ValueType.TEXT)
            .primary_key("article_id").time_column("published_at").build(),

            TableDef.new_table("news_mentions")
            .column("observed_at", ValueType.DATETIME)
            .column("overlap", ValueType.NUMBER)
            .column("similarity", ValueType.NUMBER)
            .primary_key("mention_id").time_column("observed_at").build(),
        ]
        links += [LinkDef("news_mentions", "article_id", "news_articles"),
                  LinkDef("news_mentions", "event_id", "events")]

    rows: dict[str, list[Row]] = {t.name: [] for t in tables}
    kept, event_titles = live_universe(lead_hours=lead_hours,
                                       min_ticks=min_ticks, venues=venues,
                                       tick_history_hours=tick_history_hours)

    # -- assemble -----------------------------------------------------------
    for m in kept:
        anchor, bars = m["anchor_at"], m["bars"]
        rows["markets"].append(Row("markets", m["market_id"], {
            "question": m["question"], "outcome_label": m["outcome_label"],
            "venue": m["venue"], "opened_at": m["opened_at"],
            "closes_at": m["closes_at"], "known_at": anchor,
            "resolved_yes": m["resolved_yes"],
        }, anchor, {"event_id": m["event_id"]}))
        for b in bars:
            rows["price_ticks"].append(Row(
                "price_ticks", f'{m["market_id"]}:{int(b["hour"].timestamp())}',
                {k: v for k, v in b.items() if v is not None},
                b["hour"], {"market_id": m["market_id"]}))

    live_events = {m["event_id"] for m in kept}
    for event_id in live_events:
        rows["events"].append(Row("events", event_id, {
            "title": event_titles[event_id],
            "venue": "polymarket" if event_id.startswith("pm:") else "kalshi"}))

    if with_news:
        articles = read_articles()
        titles = {e: event_titles[e] for e in live_events}
        keep, mentions = link_news(titles, articles, min_overlap=min_overlap,
                                   max_per_event=max_per_event, max_df=max_df)
        candidates = [(event_id, a, float(overlap), None)
                      for event_id, a, overlap in mentions]

        # Semantic links from scale/semantic.py, if it has been run. They are
        # merged rather than replacing the lexical ones: exact-name matches and
        # paraphrase matches fail in different directions.
        semantic = SCALE / SOURCES["semantic"]
        if semantic.exists():
            by_id = {a["article_id"]: a for a in articles}
            for row in pq.read_table(semantic).to_pylist():
                article = by_id.get(row["article_id"])
                if article is None or row["event_id"] not in live_events:
                    continue
                candidates.append((row["event_id"], article, 0.0,
                                   float(row["similarity"])))

        # Both sources are capped the same way, and only here: the semantic
        # pass takes a global top-k per event, which is no more time-neutral
        # than a first-come lexical cap.
        best: dict[tuple[str, str], tuple] = {}
        for event_id, article, overlap, similarity in candidates:
            key = (event_id, article["article_id"])
            previous = best.get(key)
            if previous is None:
                best[key] = (event_id, article, overlap, similarity)
            else:
                best[key] = (event_id, article,
                             max(overlap, previous[2]),
                             similarity if similarity is not None else previous[3])
        capped = cap_uniformly(list(best.values()),
                               per_bucket=per_bucket, bucket_days=bucket_days)
        linked = {(event_id, a["article_id"]): (a, overlap, similarity)
                  for event_id, a, overlap, similarity in capped}
        keep = {a["article_id"]: a for _, a, _, _ in capped}
        weeks = {}
        for _, a, _, _ in capped:
            weeks[stamp(a["published_at"]).strftime("%Y-W%V")] = \
                weeks.get(stamp(a["published_at"]).strftime("%Y-W%V"), 0) + 1
        print(f"   news: {len(candidates):,} candidate mentions -> "
              f"{len(capped):,} kept, by week: "
              + " ".join(f"{w}:{n}" for w, n in sorted(weeks.items())))

        for a in keep.values():
            when = stamp(a["published_at"])
            rows["news_articles"].append(Row("news_articles", a["article_id"], {
                "published_at": when, "headline": a["headline"],
                "domain": a["domain"], "tone": a["tone"],
                "persons": a["persons"][:200], "orgs": a["orgs"][:200]}, when))
        for (event_id, article_id), (a, overlap, similarity) in linked.items():
            when = stamp(a["published_at"])
            cells = {"observed_at": when, "overlap": overlap}
            if similarity is not None:
                cells["similarity"] = similarity
            rows["news_mentions"].append(Row(
                "news_mentions", f"{event_id}:{article_id}", cells, when,
                {"article_id": article_id, "event_id": event_id}))

    schema = Schema(tuple(tables), tuple(links))
    counts = {name: len(table_rows) for name, table_rows in rows.items()}
    return schema, wiring(schema, rows, kept), Corpus(rows, kept, counts)


# ---------------------------------------------------------------------------
def wiring(schema: Schema, rows: dict[str, list[Row]],
           markets: list[dict]) -> RetrieverWiring:
    settled_at = {m["market_id"]: m["closes_at"] for m in markets}
    by_id = {name: {r.id: r for r in table_rows}
             for name, table_rows in rows.items()}
    children: dict[tuple[str, str], dict] = {}
    for link in schema.links:
        index: dict = {}
        for row in rows[link.from_table]:
            parent = row.parents.get(link.fk_column)
            if parent is not None:
                index.setdefault(parent, []).append(row)
        for bucket in index.values():
            bucket.sort(key=lambda r: (r.timestamp is None,
                                       -(r.timestamp.timestamp()
                                         if r.timestamp else 0.0)))
        children[(link.from_table, link.fk_column)] = index

    def mask_unsettled(row: Row, bound: TemporalBound) -> Row:
        """A market's outcome exists only after it settles."""
        closes = settled_at.get(row.id)
        if closes is None or bound.as_of is None or closes <= bound.as_of:
            return row
        cells = dict(row.cells)
        cells.pop("resolved_yes", None)
        return Row(row.table, row.id, cells, row.timestamp, row.parents)

    def entities(table, ids, bound: TemporalBound):
        found = [row for i in ids
                 if (row := by_id[table].get(i)) is not None
                 and bound.admits_row(row)]
        return ([mask_unsettled(r, bound) for r in found]
                if table == "markets" else found)

    def link_rows(link, parent_id, bound: TemporalBound, limit: int):
        found = [row for row in children[(link.from_table, link.fk_column)]
                 .get(parent_id, ()) if bound.admits_row(row)]
        found = found[:limit]
        return ([mask_unsettled(r, bound) for r in found]
                if link.from_table == "markets" else found)

    def scanner(table, bound: TemporalBound):
        for row in rows[table]:
            if not bound.admits_row(row):
                continue
            yield mask_unsettled(row, bound) if table == "markets" else row

    builder = RetrieverWiring.new_wiring().default_links(link_rows)
    for table in rows:
        builder.entities(table, entities)
        builder.scanner(table, scanner)
    return builder.build()
