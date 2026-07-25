"""Timestamped public chatter for the Nov–Dec 2025 market window, from HF.

Two sources, both already on the Hub — nothing is scraped:

  reddit   mitanshugoel/reddit-2025, monthly comment dumps. One month is 37 GB
           of zstd, which no budget here can hold, but the dump is written in
           timestamp order and the Hub serves HTTP range requests: fetching a
           prefix of N bytes yields a contiguous slice starting at the first of
           the month (~870 MB per day of all-of-Reddit). Filtering to topical
           subreddits happens while the stream decodes, so what lands on disk
           is a thousandth of what crosses the wire.

  hackernews  open-index/hacker-news, monthly parquet. A whole month is 50 MB,
           so both months come down in full.

    python -m comments.pull_chatter --reddit-gb 2 --month 2025-11
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent.parent / "data" / "chatter"
REDDIT = ("https://huggingface.co/datasets/mitanshugoel/reddit-2025/"
          "resolve/main/2025/RC_{month}.zst")
AGENT = "relativedb-example/0.2 (+https://relql.com)"

# Subreddits where people argue about things prediction markets list. Sports
# and hobby subs dominate raw volume and say nothing about a Fed decision.
SUBREDDITS = {
    "politics", "worldnews", "news", "geopolitics", "PoliticalDiscussion",
    "Economics", "economy", "finance", "investing", "stocks", "StockMarket",
    "wallstreetbets", "CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets",
    "Conservative", "democrats", "Republican", "moderatepolitics",
    "anime_titties", "UkrainianConflict", "IsraelPalestine", "syriancivilwar",
    "energy", "oil", "inflation", "Superstonk", "options", "Forex",
    "PredictionMarkets", "Polymarket", "Kalshi", "elections", "fivethirtyeight",
}

MIN_WORDS = 6


def pull_reddit(month: str, gigabytes: float) -> pa.Table:
    url = REDDIT.format(month=month)
    limit = int(gigabytes * 1e9)
    request = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{limit - 1}", "User-Agent": AGENT})
    print(f">> reddit {month}: requesting first {gigabytes:.1f} GB", flush=True)
    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as response:
        blob = response.read()
    print(f"   downloaded {len(blob) / 1e9:.2f} GB in {time.time() - started:.0f}s")

    import zstandard as zstd
    reader = zstd.ZstdDecompressor().stream_reader(io.BytesIO(blob))
    stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    rows, seen, first, last = [], 0, None, None
    try:
        for line in stream:
            try:
                c = json.loads(line)
            except Exception:
                continue
            seen += 1
            if c.get("subreddit") not in SUBREDDITS:
                continue
            body = (c.get("body") or "").strip()
            if len(body.split()) < MIN_WORDS or body in ("[deleted]", "[removed]"):
                continue
            when = datetime.fromtimestamp(int(c["created_utc"]), tz=timezone.utc)
            first = first or when
            last = when
            rows.append({"chatter_id": f"rd:{c['id']}",
                         "source": "reddit",
                         "channel": c["subreddit"],
                         "author": c.get("author") or "unknown",
                         "posted_at": when,
                         "body": " ".join(body.split())[:600],
                         "score": int(c.get("score") or 0)})
    except Exception as exc:                 # truncated final frame: expected
        print(f"   stream ended at the byte limit ({type(exc).__name__})")
    print(f"   scanned {seen:,} comments -> kept {len(rows):,} "
          f"from {len(SUBREDDITS)} subreddits, {first} .. {last}")
    return pa.Table.from_pylist(rows)


def pull_hackernews(months: list[str]) -> pa.Table:
    from huggingface_hub import hf_hub_download
    rows = []
    for month in months:
        path = hf_hub_download("open-index/hacker-news",
                               f"data/{month[:4]}/{month}.parquet",
                               repo_type="dataset",
                               local_dir=str(OUT.parent / "hf" / "hn"))
        t = pq.read_table(path, columns=["id", "type", "by", "time", "text",
                                         "title", "score", "dead", "deleted"])
        # `dead`/`deleted` come back as uint8 in some months and as string in
        # others; casting once keeps the filter working either way.
        flag = lambda name: pc.equal(pc.cast(t[name], pa.string()), "0")
        t = t.filter(pc.and_(flag("dead"), flag("deleted")))
        for r in t.to_pylist():
            body = " ".join(((r["title"] or "") + " " + (r["text"] or "")).split())
            if len(body.split()) < MIN_WORDS:
                continue
            rows.append({"chatter_id": f"hn:{r['id']}",
                         "source": "hackernews",
                         "channel": "news.ycombinator.com",
                         "author": r["by"] or "unknown",
                         "posted_at": r["time"],
                         "body": body[:600],
                         "score": int(r["score"] or 0)})
        print(f"   {month}: {t.num_rows:,} items -> {len(rows):,} cumulative")
    return pa.Table.from_pylist(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2025-11")
    ap.add_argument("--reddit-gb", type=float, default=2.0,
                    help="bytes of the monthly dump to fetch (0 to skip)")
    ap.add_argument("--hn-months", nargs="*", default=["2025-11", "2025-12"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    parts = []
    if args.hn_months:
        print(">> hacker news")
        parts.append(pull_hackernews(args.hn_months))
    if args.reddit_gb > 0:
        parts.append(pull_reddit(args.month, args.reddit_gb))

    table = pa.concat_tables(parts).combine_chunks()
    table = table.sort_by([("posted_at", "ascending")])
    pq.write_table(table, OUT / "chatter.parquet", compression="zstd")
    span = (table["posted_at"][0].as_py(), table["posted_at"][-1].as_py())
    print(f">> wrote {OUT / 'chatter.parquet'} — {table.num_rows:,} posts, "
          f"{span[0]:%Y-%m-%d %H:%M} .. {span[1]:%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
