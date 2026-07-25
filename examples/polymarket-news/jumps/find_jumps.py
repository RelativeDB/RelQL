"""Find the minutes where a market repriced, and see what dropped just before.

This is an event study, not a model: minute bars for liquid markets, headlines
stamped to the quarter hour, and one question — when a market jumps, was there
news in the half hour before it?

The number that matters is not "how many jumps had news nearby". News is
constant; with 900k headlines a month you can find something near anything. It
is the **lift over a control**: the same match test run at random minutes in
the same markets. If jumps and random minutes both show news 60% of the time,
news does not mark jumps.

    python -m jumps.find_jumps --days 3 --markets 150
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent.parent / "data" / "jumps"
AGENT_UA = "relativedb-example/0.2 (+https://relql.com)"
SKIP_TAGS = {"sports", "esports", "games", "mlb", "nba", "nfl", "nhl", "cfb",
             "tennis", "soccer", "epl", "ucl", "chess", "f1", "lol", "cs2"}


# ---------------------------------------------------------------------------
class GammaMarket:
    """The few fields the study needs, from gamma's JSON."""

    __slots__ = ("market_id", "title", "volume_24h", "price", "token_id",
                 "tags", "status", "outcomes")

    def __init__(self, raw: dict):
        import json as _json
        self.market_id = str(raw["id"])
        self.title = raw.get("question") or ""
        self.volume_24h = float(raw.get("volume24hr") or 0.0)
        try:
            self.price = float(_json.loads(raw.get("outcomePrices") or "[]")[0])
        except Exception:
            self.price = None
        try:
            self.token_id = _json.loads(raw.get("clobTokenIds") or "[]")[0]
        except Exception:
            self.token_id = None
        self.tags = []
        self.status = "active"
        self.outcomes = [self.token_id]


def gamma_markets(want: int, *, page: int = 100):
    """Active markets, busiest first, paged. `closed=false&active=true` keeps
    the universe to things that can still move."""
    import json as _json
    import urllib.request as _req
    base = ("https://gamma-api.polymarket.com/markets?closed=false&active=true"
            f"&order=volume24hr&ascending=false&limit={page}")
    out, offset = [], 0
    while len(out) < want:
        url = f"{base}&offset={offset}"
        request = _req.Request(url, headers={"User-Agent": AGENT_UA,
                                             "Accept": "application/json"})
        try:
            with _req.urlopen(request, timeout=45) as response:
                rows = _json.loads(response.read())
        except Exception as exc:
            print(f"   gamma page {offset} failed: {type(exc).__name__}")
            break
        if not rows:
            break
        out.extend(GammaMarket(r) for r in rows)
        offset += page
    return [m for m in out if m.token_id][:want]


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------
def minute_bars(client, outcome, *, days: int):
    end = datetime.now(timezone.utc)
    try:
        bars = client.fetch_ohlcv(outcome, resolution="1m",
                                  start=end - timedelta(days=days), end=end)
    except Exception:
        return []
    return [{"at": datetime.fromtimestamp(b.timestamp / 1000, tz=timezone.utc),
             "close": b.close, "volume": b.volume} for b in bars]


def jumps_in(bars, *, window: int, min_move: float, sigmas: float,
             cooldown: int):
    """Minutes where the price moved more in `window` minutes than it usually
    does — `sigmas` times the market's own median absolute move, and at least
    `min_move` in absolute terms so a quiet market's noise cannot qualify."""
    if len(bars) < window * 3:
        return []
    moves = [bars[i + window]["close"] - bars[i]["close"]
             for i in range(len(bars) - window)]
    scale = statistics.median([abs(m) for m in moves]) or 0.001
    threshold = max(min_move, sigmas * scale)
    found = []
    last = -10 ** 9
    order = sorted(range(len(moves)), key=lambda i: -abs(moves[i]))
    for i in order:
        if abs(moves[i]) < threshold:
            break
        if any(abs(i - j) < cooldown for j, _ in found):
            continue
        found.append((i, moves[i]))
    return [{"at": bars[i]["at"], "move": move,
             "from": bars[i]["close"], "to": bars[i + window]["close"],
             "typical": scale} for i, move in sorted(found)]


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------
def quarter_hours(start: datetime, end: datetime):
    when = start.replace(minute=(start.minute // 15) * 15, second=0,
                         microsecond=0)
    while when < end:
        yield when
        when += timedelta(minutes=15)


def pull_news(start: datetime, end: datetime, *, workers: int = 8):
    """GDELT's 15-minute GKG files over the window, titles only."""
    from scale.pull_gdelt import fetch
    slots = list(quarter_hours(start, end))
    print(f">> gdelt: {len(slots)} quarter-hour files", flush=True)
    articles, downloaded = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (when, rows, size) in enumerate(pool.map(fetch, slots), 1):
            downloaded += size
            articles.extend(rows)
            if done % 40 == 0 or done == len(slots):
                print(f"   [{done:>4}/{len(slots)}] {len(articles):>8,} articles"
                      f"  {downloaded / 1e9:5.2f} GB", flush=True)
    return articles


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def match_news(questions: dict, articles: list[dict], *, lookback: int,
               floor: float):
    """For a (market, minute) pair, the best-matching headline published in the
    `lookback` minutes before it. Lexical prefilter, then MiniLM cosine."""
    from scale.build import tokens
    from scale.semantic import embed
    from collections import Counter, defaultdict
    import numpy as np

    frequency = Counter()
    per_market = {k: tokens(v) for k, v in questions.items()}
    for ts in per_market.values():
        frequency.update(ts)
    cutoff = max(2, int(len(questions) * 0.05))
    vocabulary = {t for t, n in frequency.items() if n <= cutoff and len(t) >= 5}

    keep = [a for a in articles if len(tokens(a["headline"]) & vocabulary) >= 1]
    print(f">> {len(articles):,} headlines -> {len(keep):,} plausibly on-topic")
    if not keep:
        return {}, {}
    order = sorted(questions)
    market_vectors = embed([questions[k] for k in order], tag="jump_markets")
    news_vectors = embed([a["headline"] for a in keep], tag="jump_news")
    index = {k: i for i, k in enumerate(order)}

    by_minute = defaultdict(list)
    for i, a in enumerate(keep):
        by_minute[a["published_at"].replace(second=0, microsecond=0)].append(i)

    def best(market_id: str, when: datetime):
        rows = []
        for offset in range(lookback + 1):
            rows.extend(by_minute.get(when - timedelta(minutes=offset), ()))
        if not rows:
            return None
        vector = market_vectors[index[market_id]]
        scores = news_vectors[rows] @ vector
        top = int(scores.argmax())
        if scores[top] < floor:
            return None
        article = keep[rows[top]]
        return {"headline": article["headline"], "domain": article["domain"],
                "published_at": article["published_at"],
                "similarity": float(scores[top]),
                "lead_minutes": int((when - article["published_at"]).total_seconds() // 60)}

    return best, keep


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--markets", type=int, default=150)
    ap.add_argument("--pool", type=int, default=1200)
    ap.add_argument("--min-volume", type=float, default=25_000.0)
    ap.add_argument("--window", type=int, default=15, help="jump window, minutes")
    ap.add_argument("--min-move", type=float, default=0.03)
    ap.add_argument("--sigmas", type=float, default=6.0)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--lookback", type=int, default=30, help="news lookback, minutes")
    ap.add_argument("--floor", type=float, default=0.40)
    ap.add_argument("--reuse-news", action="store_true",
                    help="reuse data/jumps/news.parquet instead of refetching")
    ap.add_argument("--controls", type=int, default=4,
                    help="random minutes per jump, for the control rate")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import pmxt
    client = pmxt.Polymarket()
    print(f">> markets: scanning {args.pool}")
    picked = []
    # pmxt's build accepts neither `offset` nor `page` (both 422) and caps a
    # single request near 1,200, so the universe comes from Polymarket's gamma
    # endpoint, which does paginate. Candles still come through pmxt, which
    # accepts a raw CLOB token id.
    scanned = gamma_markets(args.pool)
    print(f"   {len(scanned)} markets returned")
    for m in scanned:
        if not m.token_id or m.price is None:
            continue
        if (m.volume_24h or 0) < args.min_volume:
            continue
        if not (0.03 <= m.price <= 0.97):
            continue
        if {t.lower() for t in (m.tags or [])} & SKIP_TAGS:
            continue
        picked.append(m)
        if len(picked) >= args.markets:
            break
    print(f"   {len(picked)} markets")

    all_jumps, questions, series = [], {}, {}
    for i, m in enumerate(picked, 1):
        bars = minute_bars(client, m.token_id, days=args.days)
        if len(bars) < 200:
            continue
        questions[m.market_id] = m.title
        series[m.market_id] = bars
        found = jumps_in(bars, window=args.window, min_move=args.min_move,
                         sigmas=args.sigmas, cooldown=args.cooldown)
        for j in found:
            j["market_id"] = m.market_id
        all_jumps.extend(found)
        if i % 25 == 0:
            print(f"   [{i:>3}/{len(picked)}] {len(all_jumps)} jumps so far",
                  flush=True)
    print(f">> {len(all_jumps)} jumps across {len(series)} markets "
          f"({args.window}-minute moves >= {args.min_move} and "
          f"{args.sigmas}x typical)")

    start = min(b["at"] for bars in series.values() for b in bars[:1])
    end = datetime.now(timezone.utc)
    cached_news = OUT / "news.parquet"
    if args.reuse_news and cached_news.exists():
        # Prices are free (API); GDELT bytes are not. Widening the market
        # universe over the same days must not re-download the same news.
        articles = pq.read_table(cached_news).to_pylist()
        for a in articles:
            a["published_at"] = a["published_at"].replace(tzinfo=timezone.utc)
        print(f">> reusing {len(articles):,} cached headlines "
              f"({cached_news.name})")
    else:
        articles = pull_news(start - timedelta(minutes=args.lookback), end)
    best, kept = match_news(questions, articles, lookback=args.lookback,
                            floor=args.floor)
    if not best:
        raise SystemExit("no usable headlines")

    hits = []
    for j in all_jumps:
        hit = best(j["market_id"], j["at"])
        if hit:
            j["news"] = hit
            hits.append(j)
    rate = len(hits) / max(1, len(all_jumps))

    rng = random.Random(0)
    control_hits = 0, 0
    tries = 0
    matched = 0
    for j in all_jumps:
        bars = series[j["market_id"]]
        for _ in range(args.controls):
            other = rng.choice(bars)["at"]
            tries += 1
            if best(j["market_id"], other):
                matched += 1
    control_rate = matched / max(1, tries)

    print(f"\n>> news within {args.lookback} min of a jump: "
          f"{len(hits)}/{len(all_jumps)} = {rate:.1%}")
    print(f">> same test at random minutes:            "
          f"{matched}/{tries} = {control_rate:.1%}")
    print(f">> lift: {rate - control_rate:+.1%}")

    hits.sort(key=lambda j: -abs(j["move"]))
    print("\n== biggest jumps with a matching headline ==")
    for j in hits[:15]:
        print(f"  {j['at']:%m-%d %H:%M}  {j['from']:.3f} -> {j['to']:.3f} "
              f"({j['move']:+.3f}, typical {j['typical']:.3f})")
        print(f"     market: {questions[j['market_id']][:70]}")
        print(f"     news  : [{j['news']['lead_minutes']:>2} min before, "
              f"cos {j['news']['similarity']:.2f}] "
              f"{j['news']['headline'][:78]}")

    # Persist prices and headlines, not just the jump list: the model task in
    # `jumps.impact` needs the same bars and the same news, and refetching
    # would give it a different world than the one measured here.
    pq.write_table(pa.Table.from_pylist(
        [{"market_id": mid, "at": b["at"], "close": b["close"],
          "volume": b["volume"]} for mid, bars in series.items() for b in bars]),
        OUT / "bars.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(
        [{k: a[k] for k in ("article_id", "published_at", "headline", "domain",
                            "tone", "persons", "orgs")} for a in kept]),
        OUT / "news.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(
        [{"market_id": k, "question": v} for k, v in questions.items()]),
        OUT / "markets.parquet")
    print(f">> cached {sum(len(b) for b in series.values()):,} minute bars, "
          f"{len(kept):,} on-topic headlines")

    payload = {"generated_at": end.isoformat(), "window": args.window,
               "lookback": args.lookback, "jumps": len(all_jumps),
               "matched": len(hits), "match_rate": rate,
               "control_rate": control_rate,
               "rows": [{**{k: (v.isoformat() if isinstance(v, datetime) else v)
                            for k, v in j.items() if k != "news"},
                         "question": questions[j["market_id"]],
                         "news": {**j["news"],
                                  "published_at": j["news"]["published_at"].isoformat()}}
                        for j in hits]}
    (OUT / "jumps.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'jumps.json'}")


if __name__ == "__main__":
    main()
