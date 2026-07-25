"""Pull everything needed to price the AAA daily gas ladder — and to check it.

  ladders    Kalshi KXAAAGASD, one strike ladder per day. Settled ladders
             bracket the true AAA average (highest YES strike < value <=
             lowest NO strike), reconstructing the daily series.
  book       the last trade on each settled strike before it closed — the
             benchmark that matters. A model that cannot beat the closing
             price on days we already know has no business quoting a live one.
  wholesale  daily WTI / Brent / RBOB settles from Stooq. Retail follows
             wholesale with a lag; the first attempt had no idea crude existed.

    python gas/pull.py --days 70
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "gas"
API = "https://api.elections.kalshi.com/trade-api/v2"
AGENT = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}


def get(url: str, tries: int = 5):
    import time
    for attempt in range(tries):
        if attempt:
            time.sleep(0.6 * attempt)
        try:
            request = urllib.request.Request(url, headers=AGENT)
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.loads(response.read())
        except Exception:
            if attempt == tries - 1:
                return None
    return None


def ladder(day: dt.date):
    event = f"KXAAAGASD-{day.strftime('%y%b%d').upper()}"
    payload = get(f"{API}/markets?event_ticker={event}&limit=60")
    return day, (payload or {}).get("markets", [])


def last_trade(ticker: str):
    payload = get(f"{API}/markets/trades?ticker={ticker}&limit=100")
    trades = (payload or {}).get("trades", [])
    if not trades:
        return ticker, None, 0
    trades.sort(key=lambda t: t["created_time"])
    final = trades[-1]
    yes = final.get("yes_price_dollars")
    if yes is None and final.get("no_price_dollars") is not None:
        yes = 1.0 - float(final["no_price_dollars"])
    return ticker, (None if yes is None else float(yes)), len(trades)


def daily_closes(symbol: str, months: str = "6mo"):
    """Daily settles from Yahoo's chart endpoint. Stooq now sits behind a
    JavaScript proof-of-work wall, which a plain fetch cannot pass."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range={months}&interval=1d")
    payload = get(url)
    try:
        result = payload["chart"]["result"][0]
        stamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (TypeError, KeyError, IndexError):
        print(f"   ! {symbol}: no data")
        return {}
    out = {}
    for stamp, close in zip(stamps, closes):
        if close is None:
            continue
        out[dt.datetime.fromtimestamp(stamp, dt.timezone.utc).date()] = close
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=70)
    ap.add_argument("--today", default="2026-07-25")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    today = dt.date.fromisoformat(args.today)

    wanted = [today + dt.timedelta(days=1)] + [
        today - dt.timedelta(days=b) for b in range(0, args.days)]
    print(f">> ladders: {len(wanted)} days")
    ladders = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for day, markets in pool.map(ladder, wanted):
            if markets:
                ladders[str(day)] = [
                    {k: m.get(k) for k in ("ticker", "title", "result",
                                           "status", "close_time")}
                    for m in markets]
    print(f"   {len(ladders)} ladders, "
          f"{sum(len(v) for v in ladders.values())} strikes")

    levels = {}
    for day, markets in ladders.items():
        yes = [float(m["ticker"].rsplit("-", 1)[1]) for m in markets
               if m.get("result") == "yes"]
        no = [float(m["ticker"].rsplit("-", 1)[1]) for m in markets
              if m.get("result") == "no"]
        if yes and no and min(no) > max(yes):
            levels[day] = round((max(yes) + min(no)) / 2, 4)
    print(f">> reconstructed {len(levels)} daily averages")

    settled = [m["ticker"] for markets in ladders.values() for m in markets
               if m.get("result") in ("yes", "no")]
    print(f">> closing prices for {len(settled)} settled strikes", flush=True)
    book = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (ticker, price, n) in enumerate(pool.map(last_trade, settled), 1):
            if price is not None:
                book[ticker] = {"close_price": price, "trades": n}
            if i % 250 == 0:
                print(f"   [{i}/{len(settled)}] {len(book)} priced", flush=True)
    print(f"   {len(book)} strikes have a traded price")

    print(">> wholesale (stooq)")
    wholesale = {}
    for name, symbol in (("wti", "CL=F"), ("brent", "BZ=F"), ("rbob", "RB=F")):
        series = daily_closes(symbol)
        print(f"   {name}: {len(series)} days"
              + (f", latest {max(series)} = {series[max(series)]}" if series else ""))
        for day, close in series.items():
            wholesale.setdefault(str(day), {})[name] = close

    (DATA / "ladders.json").write_text(json.dumps(ladders, indent=1))
    (DATA / "levels.json").write_text(json.dumps(dict(sorted(levels.items())), indent=1))
    (DATA / "book.json").write_text(json.dumps(book, indent=1))
    (DATA / "wholesale.json").write_text(json.dumps(dict(sorted(wholesale.items())), indent=1))
    print(f">> wrote {DATA}/")


if __name__ == "__main__":
    main()
