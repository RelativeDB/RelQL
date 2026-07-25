"""News at scale: GDELT 2.0 GKG, pulled from the bulk archive.

The GDELT *API* throttles an unauthenticated caller into 429s within a few
requests. The *file* archive behind it does not: every 15 minutes since 2015,
GDELT publishes a zipped CSV of every article it saw, and those files download
at full speed. One file is ~6 MB zipped and holds ~10k articles with title,
outlet, themes, named people and organizations, and tone.

    python -m scale.pull_gdelt --start 2025-10-25 --end 2026-01-01 --every 2

``--every`` is the sampling cadence in hours: every 15-minute file for two
months is 25 GB, one file every two hours is ~4 GB for the same coverage of
*stories* (a story that matters appears in many consecutive files).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent.parent / "data" / "scale"
BASE = "http://data.gdeltproject.org/gdeltv2"
AGENT = "relativedb-example/0.2 (+https://relql.com)"
TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)

# GKG 2.1 is 27 tab-separated fields with no header.
DATE, DOMAIN, URL, THEMES, PERSONS, ORGS, TONE, EXTRAS = 1, 3, 4, 7, 11, 13, 15, 26


def slots(start: datetime, end: datetime, every: int):
    when = start
    while when < end:
        yield when
        when += timedelta(hours=every)


def parse(body: bytes, when: datetime) -> list[dict]:
    rows = []
    for line in body.decode("utf-8", "replace").split("\n"):
        f = line.split("\t")
        if len(f) < 27:
            continue
        title = TITLE.search(f[EXTRAS])
        if not title:
            continue                     # no headline, no text signal
        headline = " ".join(title.group(1).split())
        if not headline:
            continue
        tone = f[TONE].split(",")
        try:
            stamp = datetime.strptime(f[DATE], "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc)
        except ValueError:
            stamp = when
        rows.append({
            "article_id": hashlib.sha1(f[URL].encode()).hexdigest()[:12],
            "published_at": stamp,
            "domain": f[DOMAIN],
            "headline": headline,
            "tone": float(tone[0]) if tone and tone[0] else None,
            "themes": ";".join(f[THEMES].split(";")[:12]),
            "persons": ";".join(f[PERSONS].split(";")[:10]),
            "orgs": ";".join(f[ORGS].split(";")[:10]),
            "url": f[URL],
        })
    return rows


def fetch(when: datetime) -> tuple[datetime, list[dict], int]:
    url = f"{BASE}/{when:%Y%m%d%H%M%S}.gkg.csv.zip"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            blob = response.read()
    except Exception as exc:
        print(f"   ! {when:%Y-%m-%d %H:%M} {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return when, [], 0
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        body = z.read(z.namelist()[0])
    return when, parse(body, when), len(blob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-25")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--every", type=int, default=2, help="hours between files")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="news_articles.parquet")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    want = list(slots(start, end, args.every))
    print(f">> gdelt gkg: {len(want)} files, {args.start}..{args.end} "
          f"every {args.every}h")

    OUT.mkdir(parents=True, exist_ok=True)
    writer, total, downloaded, done = None, 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for when, rows, size in pool.map(fetch, want):
            done += 1
            downloaded += size
            if not rows:
                continue
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(OUT / args.out, table.schema,
                                          compression="zstd")
            writer.write_table(table)
            total += len(rows)
            if done % 25 == 0 or done == len(want):
                print(f"   [{done:>4}/{len(want)}] {when:%Y-%m-%d %H:%M}  "
                      f"{total:>9,} articles  {downloaded / 1e9:5.2f} GB",
                      flush=True)
    if writer is not None:
        writer.close()
    print(f">> wrote {OUT / args.out} — {total:,} articles, "
          f"{downloaded / 1e9:.2f} GB fetched")


if __name__ == "__main__":
    main()
