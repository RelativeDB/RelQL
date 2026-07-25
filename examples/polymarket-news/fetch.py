"""Snapshot live prediction-market + news data into ``data/snapshot.json``.

Two sources, both live and both free:

  markets   pmxt (https://pmxt.dev) in self-hosted mode — one unified client
            over Polymarket/Kalshi/... Here: Polymarket events, markets and
            hourly OHLCV candles per outcome.
  news      A dozen newsroom RSS feeds (BBC, Guardian, Al Jazeera, CNBC,
            MarketWatch, Politico, NPR, oilprice, the Fed's press wire),
            each item stamped with its own publication time. GDELT is
            available behind --gdelt, but it throttles an unauthenticated
            caller hard enough to stall a fetch.

The snapshot is a plain JSON file so a run is reproducible after the fact:
prices and headlines both move, and a backtest that re-fetches its own
universe every run cannot be compared against itself.

Usage:
    python fetch.py                 # default universe, 96h of candles
    python fetch.py --markets 60 --news-hours 12 --gdelt
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path(__file__).parent / "data"
SNAPSHOT = DATA / "snapshot.json"

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_INTERVAL = 20.0         # the endpoint asks for <= 1 request / 5s; it
                                  # throttles harder than that in practice
AGENT = "relativedb-example/0.2 (+https://relql.com)"
_last_gdelt_call = 0.0

# One query every 5+ seconds makes an interrupted fetch expensive to redo, so
# responses are cached on disk. The cache key includes the time window, and
# `timespan=24h` means something different an hour from now — delete
# data/gdelt_cache.json to force fresh news.
CACHE = DATA / "gdelt_cache.json"
_CACHE: dict[str, list] = (json.loads(CACHE.read_text())
                           if CACHE.exists() else {})


def _save_cache() -> None:
    DATA.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(_CACHE))

STOPWORDS = {
    "will", "the", "a", "an", "be", "is", "are", "in", "on", "of", "to", "by",
    "for", "and", "or", "at", "before", "after", "any", "more", "than", "who",
    "what", "when", "which", "there", "this", "that", "it", "its", "with",
    "up", "down", "out", "next", "new", "how", "many", "much", "do", "does",
    "did", "win", "wins", "won", "vs", "2026", "2027", "2028", "january",
    "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

# Too broad to be a subject: every article in the window matches, and none of
# them tells you anything about a particular market.
GENERIC = {"politics", "world", "news", "economy", "business", "breaking",
           "global", "united states", "world elections", "hit price",
           "recurring", "weekly", "monthly", "daily"} | STOPWORDS


# ---------------------------------------------------------------------------
# markets (pmxt)
# ---------------------------------------------------------------------------
def fetch_markets(client, *, want: int, pool: int, min_volume_24h: float,
                  price_band: tuple[float, float], exclude_tags: set[str]):
    """Liquid, still-open, still-uncertain markets — the ones news can move.

    A market pinned at 0.99 cannot teach anything in a six-hour window, and an
    illiquid one moves on a single fill rather than on information."""
    # pmxt-core returns markets already ordered by 24h volume, descending; the
    # `sort`/`status` params are rejected (422) by the self-hosted build, so
    # the filtering happens here.
    raw = client.fetch_markets(limit=pool)
    lo, hi = price_band
    now = datetime.now(timezone.utc)
    picked = []
    for m in raw:
        if m.status != "active":
            continue
        if not m.outcomes or (m.volume_24h or 0) < min_volume_24h:
            continue
        yes = m.outcomes[0]
        if yes.price is None or not (lo <= yes.price <= hi):
            continue
        # Markets resolving inside the prediction horizon are excluded: their
        # price collapses to 0/1 on resolution, which is a different process
        # from the news-driven repricing we are asking about.
        if m.resolution_date and m.resolution_date <= now + timedelta(hours=24):
            continue
        # A tennis match reprices on the scoreboard, not on the newswire; the
        # question here is whether news explains price, so games are out.
        labels = {t.lower() for t in (m.tags or [])} | {(m.category or "").lower()}
        if labels & exclude_tags:
            continue
        picked.append(m)
        if len(picked) >= want:
            break
    return picked


def fetch_candles(client, outcome, *, hours: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    try:
        return client.fetch_ohlcv(outcome, resolution="1h", start=start, end=end)
    except Exception as exc:                       # one dead outcome != no run
        print(f"    ! candles failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# news (GDELT, optional)
# ---------------------------------------------------------------------------
def keywords(title: str, limit: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'&.-]+", title)
    seen, out = set(), []
    for w in words:
        low = w.lower()
        # GDELT rejects three-character phrases outright, so "Fed", "WTI" and
        # "NBA" can never be search terms here — but "Iran" can, and it is
        # the single most load-bearing word in this universe.
        if low in STOPWORDS or len(low) < 4 or low in seen:
            continue
        seen.add(low)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def gdelt(query: str, *, hours: int, maxrecords: int = 12) -> list[dict]:
    global _last_gdelt_call
    cached = _CACHE.get(f"{hours}h|{maxrecords}|{query}")
    if cached is not None:
        return cached
    wait = GDELT_MIN_INTERVAL - (time.monotonic() - _last_gdelt_call)
    if wait > 0:
        time.sleep(wait)
    # A quoted phrase must be five characters or more ("Iran" is rejected),
    # but the same word unquoted is fine — so only multi-word subjects are
    # quoted, and single words go bare.
    phrase = f'"{query}"' if " " in query else query
    url = GDELT + "?" + urllib.parse.urlencode({
        "query": f"{phrase} sourcelang:english",
        "mode": "artlist",
        "maxrecords": maxrecords,
        "timespan": f"{hours}h",
        "sort": "datedesc",
        "format": "json",
    })
    # GDELT answers a burst with 429s for a while afterwards, so a refused
    # request is backed off and retried rather than silently dropped — an
    # event with no news is a real observation and must not be faked by one.
    body = ""
    for attempt, backoff in enumerate((0, 30, 75), 1):
        if backoff:
            print(f"    . gdelt backing off {backoff}s", file=sys.stderr)
            time.sleep(backoff)
        _last_gdelt_call = time.monotonic()
        try:
            # GDELT 429s the default urllib agent string outright.
            request = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8", "replace")
        except Exception as exc:
            print(f"    ! gdelt failed ({attempt}/3): "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            body = ""
            continue
        if body.lstrip().startswith("{"):
            break
        # Plain prose comes back for two very different reasons: a throttle
        # (worth waiting out) and a malformed query (never worth retrying).
        print(f"    ! gdelt said: {body.strip()[:100]}", file=sys.stderr)
        throttled = "limit requests" in body
        body = ""
        if not throttled:
            return []
    if not body:
        return []
    try:
        articles = json.loads(body).get("articles", []) or []
    except json.JSONDecodeError:
        return []
    _CACHE[f"{hours}h|{maxrecords}|{query}"] = articles
    _save_cache()
    return articles


# ---------------------------------------------------------------------------
# news (newsroom RSS)
# ---------------------------------------------------------------------------
# Live newsroom feeds, refreshed continuously, covering the subjects this
# universe trades on: war and diplomacy, macro, energy, politics, crypto.
# Each item carries its own publication time, which is what makes the anchor
# cut meaningful — GDELT is an optional extra (--gdelt) because it throttles
# an unauthenticated caller hard enough to stall a fetch.
FEEDS = (
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.timesofisrael.com/feed/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://rss.politico.com/politics-news.xml",
    "https://feeds.npr.org/1004/rss.xml",
    "https://oilprice.com/rss/main",
    "https://www.federalreserve.gov/feeds/press_all.xml",
)

RSS_ITEM = re.compile(r"<item[ >].*?</item>|<entry[ >].*?</entry>", re.S)
RSS_FIELD = {
    "title": re.compile(r"<title[^>]*>(.*?)</title>", re.S),
    "link": re.compile(r"<link[^>]*>(.*?)</link>", re.S),
    "date": re.compile(r"<(?:pubDate|published|updated|dc:date)[^>]*>(.*?)"
                       r"</(?:pubDate|published|updated|dc:date)>", re.S),
}
RSS_DATE_FORMATS = ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                    "%a, %d %b %Y %H:%M %Z", "%a, %d %b %Y %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ")


def untag(text: str) -> str:
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    # Feeds mix named, decimal and hex entities; the headline is model input,
    # so "I&#x2019;m" has to arrive as "I'm".
    return " ".join(html.unescape(text).split())


def rss_time(raw: str) -> datetime | None:
    raw = untag(raw)
    for fmt in RSS_DATE_FORMATS:
        try:
            when = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return None


def fetch_feed(url: str, *, hours: int) -> list[dict]:
    """One feed's items, dropped to those published inside the window.

    Feeds are parsed with regexes rather than a dependency: an example that
    needs a parser library to read four fields out of RSS is an example about
    the parser library."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception as exc:
        print(f"    ! feed failed: {type(exc).__name__}: {exc} — {url}",
              file=sys.stderr)
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    domain = urllib.parse.urlparse(url).netloc.removeprefix("www.")
    out = []
    for chunk in RSS_ITEM.findall(body):
        headline = untag(RSS_FIELD["title"].search(chunk).group(1)) \
            if RSS_FIELD["title"].search(chunk) else ""
        date_match = RSS_FIELD["date"].search(chunk)
        link_match = RSS_FIELD["link"].search(chunk)
        published = rss_time(date_match.group(1)) if date_match else None
        if not headline or published is None or published < cutoff:
            continue
        link = untag(link_match.group(1)) if link_match else ""
        out.append({"headline": headline, "published_at": published.isoformat(),
                    "domain": domain, "url": link or url, "country": None,
                    "language": "English"})
    return out


def topics(events: list[dict], *, limit: int) -> list[str]:
    """A handful of broad subjects covering the universe, most common first.

    One query per event would be precise, but GDELT throttles a burst of
    thirty into oblivion. A dozen topic queries pull a much larger article
    pool for a fraction of the requests, and the article-to-event linking
    then happens locally, where it can be inspected."""
    counts: dict[str, int] = {}
    for e in events:
        seen = set()
        for term in list(e["tags"] or []) + keywords(e["title"], limit=3):
            term = term.strip()
            # Polymarket tags carry promo and shape junk ("Earn 4%", "by...");
            # a subject has to be words. Very broad ones return the whole
            # newswire and link to nothing, so they go too.
            if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", term):
                continue
            if len(term) < 4 or term.lower() in GENERIC or term in seen:
                continue
            seen.add(term)
            counts[term] = counts.get(term, 0) + 1
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    return ranked[:limit]


def tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text or "")
            if w.lower() not in STOPWORDS}


def link(events: list[dict], articles: dict[str, dict], *, min_overlap: int):
    """Attach an article to an event when their wording genuinely overlaps.

    Two shared content words is a low bar, and it will let some noise through
    — an article about a different Iran story still lands on the Iran event.
    That is the honest version of the signal: nobody hands you a labeled
    article-to-market mapping, and pretending otherwise would be the
    interesting part of the problem quietly solved by hand."""
    mentions = []
    for e in events:
        subject = tokens(e["title"]) | {t.lower() for t in (e["tags"] or [])}
        # "Politics" is a tag on half the universe; matching on it links every
        # market to every article and tells the model nothing.
        subject -= GENERIC
        for article in articles.values():
            shared = subject & tokens(article["headline"])
            if len(shared) < min_overlap:
                continue
            mentions.append({
                "mention_id": f'{e["event_id"]}:{article["article_id"]}',
                "event_id": e["event_id"],
                "article_id": article["article_id"],
                "matched_query": " ".join(sorted(shared)),
                "observed_at": article["published_at"],
            })
    return mentions


def parse_seendate(value: str) -> str | None:
    try:
        return (datetime.strptime(value, "%Y%m%dT%H%M%SZ")
                .replace(tzinfo=timezone.utc).isoformat())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=40,
                    help="markets to keep in the universe")
    ap.add_argument("--pool", type=int, default=600,
                    help="markets to pull before filtering")
    ap.add_argument("--min-volume", type=float, default=25_000.0,
                    help="minimum 24h volume, USD")
    ap.add_argument("--candle-hours", type=int, default=96)
    ap.add_argument("--news-hours", type=int, default=24)
    ap.add_argument("--gdelt", action="store_true",
                    help="also query GDELT (slow: it throttles hard)")
    ap.add_argument("--news-topics", type=int, default=14,
                    help="how many subjects to query GDELT for")
    ap.add_argument("--news-per-topic", type=int, default=60,
                    help="articles per topic")
    ap.add_argument("--min-overlap", type=int, default=2,
                    help="shared content words needed to link article -> event")
    ap.add_argument("--price-band", type=float, nargs=2, default=(0.06, 0.94))
    ap.add_argument("--exclude-tags", nargs="*",
                    default=["sports", "esports", "games", "mlb", "nba", "nfl",
                             "tennis", "soccer", "epl", "chess"],
                    help="drop markets carrying any of these tags/categories")
    args = ap.parse_args()

    import pmxt                                    # self-hosted: no API key
    client = pmxt.Polymarket()

    print(f">> pmxt {pmxt.__version__}: fetching markets")
    markets = fetch_markets(client, want=args.markets, pool=args.pool,
                            min_volume_24h=args.min_volume,
                            price_band=tuple(args.price_band),
                            exclude_tags={t.lower() for t in args.exclude_tags})
    print(f"   kept {len(markets)} markets")

    market_rows, candles, event_ids = [], {}, {}
    for i, m in enumerate(markets, 1):
        yes = m.outcomes[0]
        bars = fetch_candles(client, yes, hours=args.candle_hours)
        print(f"   [{i:>3}/{len(markets)}] {len(bars):>3} candles  "
              f"{m.title[:64]}")
        if len(bars) < 24:                          # too thin to backtest
            continue
        candles[m.market_id] = [
            {"ts": c.timestamp, "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume}
            for c in bars]
        market_rows.append({
            "market_id": m.market_id,
            "event_id": m.event_id,
            "question": m.title,
            "outcome_label": yes.label,
            "outcome_id": yes.outcome_id,
            "category": m.category,
            "tags": list(m.tags or []),
            "volume": m.volume,
            "volume_24h": m.volume_24h,
            "liquidity": m.liquidity,
            "tick_size": m.tick_size,
            "resolution_date": (m.resolution_date.isoformat()
                                if m.resolution_date else None),
            "url": m.url,
        })
        if m.event_id:
            event_ids.setdefault(m.event_id, m)

    print(f">> events: {len(event_ids)}")
    event_rows = []
    for event_id, m in event_ids.items():
        try:
            e = client.fetch_event({"event_id": event_id})
        except Exception:
            e = None                                # market fields stand in
        meta = getattr(e, "source_metadata", None) or {}
        event_rows.append({
            "event_id": event_id,
            "title": getattr(e, "title", None) or m.title,
            "category": getattr(e, "category", None) or m.category,
            "tags": list(getattr(e, "tags", None) or m.tags or []),
            "volume_24h": getattr(e, "volume_24h", None),
            "volume": getattr(e, "volume", None),
            "start_date": meta.get("startDate"),
            "end_date": meta.get("endDate"),
        })

    def keep(article: dict) -> None:
        article_id = hashlib.sha1(article["url"].encode()).hexdigest()[:12]
        articles.setdefault(article_id, dict(article, article_id=article_id))

    articles: dict[str, dict] = {}
    print(f">> news: last {args.news_hours}h from {len(FEEDS)} newsroom feeds")
    for i, feed in enumerate(FEEDS, 1):
        found = fetch_feed(feed, hours=args.news_hours)
        print(f"   [{i:>3}/{len(FEEDS)}] {len(found):>3} in window  "
              f"{urllib.parse.urlparse(feed).netloc}")
        for a in found:
            keep(a)

    if args.gdelt:
        subjects = topics(event_rows, limit=args.news_topics)
        print(f">> gdelt: {len(subjects)} topics "
              f"(~{GDELT_MIN_INTERVAL:.0f}s apart): {', '.join(subjects)}")
        for i, subject in enumerate(subjects, 1):
            found = gdelt(subject, hours=args.news_hours,
                          maxrecords=args.news_per_topic)
            print(f"   [{i:>3}/{len(subjects)}] {len(found):>3} articles  "
                  f"{subject}")
            for a in found:
                url, seen = a.get("url"), parse_seendate(a.get("seendate", ""))
                if not url or not seen:
                    continue
                keep({"published_at": seen, "url": url,
                      "headline": " ".join((a.get("title") or "").split()),
                      "domain": a.get("domain"),
                      "country": a.get("sourcecountry"),
                      "language": a.get("language")})

    mentions = link(event_rows, articles, min_overlap=args.min_overlap)
    linked = len({m["event_id"] for m in mentions})
    print(f"   linked {len(mentions)} mentions across {linked}/"
          f"{len(event_rows)} events")

    DATA.mkdir(exist_ok=True)
    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": {"markets": f"pmxt {pmxt.__version__} / polymarket",
                   "news": ("newsroom rss" + (" + gdelt 2.0 doc api"
                                              if args.gdelt else "")),
                   "feeds": list(FEEDS),
                   "news_hours": args.news_hours,
                   "candle_hours": args.candle_hours},
        "events": event_rows,
        "markets": market_rows,
        "candles": candles,
        "articles": sorted(articles.values(), key=lambda a: a["published_at"]),
        "mentions": mentions,
    }
    SNAPSHOT.write_text(json.dumps(snapshot, indent=1))
    ticks = sum(len(v) for v in candles.values())
    print(f">> wrote {SNAPSHOT} — {len(market_rows)} markets, {ticks} candles, "
          f"{len(articles)} articles, {len(mentions)} mentions")


if __name__ == "__main__":
    main()
