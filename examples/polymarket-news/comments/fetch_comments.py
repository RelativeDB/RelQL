"""Snapshot Polymarket comments + the hourly tape they sit on.

One question, one population: **when somebody posts a comment on a market,
does what they wrote predict where the price goes next?**

Comments are attached to events, so each event contributes its highest-volume
open market as the price series. Polymarket caps an hourly candle request at
seven days, so the tape is fetched in weekly chunks and stitched.

    python -m comments.fetch_comments --events 160 --days 28 --horizon 6
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "comments"
SNAPSHOT = DATA / "snapshot.json"
GAMMA = "https://gamma-api.polymarket.com/comments"
AGENT = "relativedb-example/0.2 (+https://relql.com)"

SKIP_TAGS = {"sports", "esports", "games", "mlb", "nba", "nfl", "nhl", "cfb",
             "tennis", "soccer", "epl", "ucl", "chess", "f1"}


def get(url: str):
    request = urllib.request.Request(
        url, headers={"User-Agent": AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read())


def comments_for(event_id: str) -> list[dict]:
    out, offset = [], 0
    while True:
        try:
            page = get(f"{GAMMA}?parent_entity_type=Event&"
                       f"parent_entity_id={event_id}&limit=500&offset={offset}")
        except Exception as exc:
            print(f"    ! comments {event_id}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            break
        out.extend(page)
        if len(page) < 500:
            break
        offset += 500
    return out


def tape(client, outcome, *, days: int) -> list[dict]:
    """Hourly bars, stitched from seven-day requests (the API's cap)."""
    end = datetime.now(timezone.utc)
    bars: dict[int, object] = {}
    for week in range((days + 6) // 7):
        hi = end - timedelta(days=7 * week)
        lo = hi - timedelta(days=7)
        try:
            for b in client.fetch_ohlcv(outcome, resolution="1h",
                                        start=lo, end=hi):
                bars[b.timestamp] = b
        except Exception as exc:
            print(f"    ! candles: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            break
    return [{"ts": t, "open": b.open, "high": b.high, "low": b.low,
             "close": b.close, "volume": b.volume}
            for t, b in sorted(bars.items())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=160)
    ap.add_argument("--pool", type=int, default=600)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--min-volume", type=float, default=25_000.0)
    ap.add_argument("--price-band", type=float, nargs=2, default=(0.04, 0.96))
    args = ap.parse_args()

    import pmxt
    client = pmxt.Polymarket()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    print(f">> markets: pulling {args.pool}, keeping one per event")
    best: dict[str, object] = {}
    for m in client.fetch_markets(limit=args.pool):
        lo, hi = args.price_band
        if m.status != "active" or not m.outcomes or not m.event_id:
            continue
        if (m.volume_24h or 0) < args.min_volume:
            continue
        if m.outcomes[0].price is None or not (lo <= m.outcomes[0].price <= hi):
            continue
        if {t.lower() for t in (m.tags or [])} & SKIP_TAGS:
            continue
        if (m.event_id not in best
                or (m.volume_24h or 0) > (best[m.event_id].volume_24h or 0)):
            best[m.event_id] = m
    chosen = sorted(best.values(), key=lambda m: -(m.volume_24h or 0))
    chosen = chosen[:args.events]
    print(f"   {len(chosen)} events")

    events, markets, candles, comments, authors = [], [], {}, [], {}
    for i, m in enumerate(chosen, 1):
        posts = [c for c in comments_for(m.event_id)
                 if c.get("createdAt", "") >= cutoff.isoformat()]
        if not posts:
            print(f"   [{i:>3}/{len(chosen)}]   0 comments  {m.title[:52]}")
            continue
        bars = tape(client, m.outcomes[0], days=args.days)
        print(f"   [{i:>3}/{len(chosen)}] {len(posts):>4} comments, "
              f"{len(bars):>3} bars  {m.title[:52]}", flush=True)
        if len(bars) < 48:
            continue
        candles[m.market_id] = bars
        events.append({"event_id": m.event_id, "title": m.title,
                       "category": m.category, "tags": list(m.tags or [])})
        markets.append({"market_id": m.market_id, "event_id": m.event_id,
                        "question": m.title, "outcome_label": m.outcomes[0].label,
                        "tick_size": m.tick_size, "url": m.url,
                        "resolution_date": (m.resolution_date.isoformat()
                                            if m.resolution_date else None)})
        for c in posts:
            profile = c.get("profile") or {}
            address = c.get("userAddress") or "unknown"
            authors.setdefault(address, {
                "author_id": address,
                "name": profile.get("pseudonym") or profile.get("name")})
            comments.append({
                "comment_id": str(c["id"]),
                "event_id": m.event_id,
                "market_id": m.market_id,
                "author_id": address,
                "created_at": c["createdAt"],
                "body": " ".join((c.get("body") or "").split())[:600],
                "reactions": c.get("reactionCount") or 0,
                "reports": c.get("reportCount") or 0,
            })

    DATA.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "events": events, "markets": markets, "candles": candles,
        "comments": comments, "authors": list(authors.values())}, indent=1))
    print(f">> wrote {SNAPSHOT} — {len(markets)} markets, "
          f"{sum(len(v) for v in candles.values())} bars, "
          f"{len(comments)} comments, {len(authors)} authors")


if __name__ == "__main__":
    main()
