"""Polymarket at scale: market metadata + a time-slice of the trade tape.

Source: SII-WANGZJ/Polymarket_data on Hugging Face — 1.8M markets (0.3 GB) and
1.03B on-chain fills (37.5 GB). The fills are sorted by time, and every parquet
row group carries min/max statistics, so a two-month window is 92 of 1,027 row
groups: ~3.6 GB fetched instead of 37.5.

What lands on disk is much smaller than what is fetched — each row group is
filtered to the target markets and folded into hourly bars as it arrives, then
dropped.

    python -m scale.pull_polymarket --start 2025-10-15 --end 2026-01-01
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPO = "SII-WANGZJ/Polymarket_data"
DATA = Path(__file__).resolve().parent.parent / "data"
HF = DATA / "hf"
OUT = DATA / "scale"

# Machine-made markets: thousands of 15-minute crypto candles priced by a bot,
# resolving on a price feed. They swamp the universe by count, carry no news,
# and their "resolution" is a coin flip on a tick — a different problem.
MACHINE = ("up or down", "up or down?", "higher or lower")

# Games resolve on a scoreboard. Nothing in a news corpus or an order tape
# forecasts an NBA fourth quarter, and by count they would be half the
# universe — so they are out, by the league prefix Polymarket puts on the
# event slug.
LEAGUES = ("nba", "nhl", "nfl", "mlb", "cfb", "cbb", "epl", "ucl", "uel",
           "uef", "atp", "wta", "lol", "cs2", "csgo", "dota", "val", "ufc",
           "f1", "mls", "laliga", "seriea", "bundesliga", "ligue", "acn",
           "afc", "nascar", "pga", "golf", "rugby", "cricket", "boxing",
           "eredivisie", "primeira", "brasileirao", "liga", "copa", "euro")


def row_group_index(force: bool = False) -> list[dict]:
    """Per-row-group statistics for the fills file, read from its footer."""
    cache = HF / "trades_rowgroups.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    with HfFileSystem().open(f"datasets/{REPO}/trades.parquet") as handle:
        md = pq.ParquetFile(handle).metadata
    names = [md.row_group(0).column(i).path_in_schema
             for i in range(md.row_group(0).num_columns)]
    time_col = names.index("timestamp")
    groups = []
    for g in range(md.num_row_groups):
        stats = md.row_group(g).column(time_col).statistics
        groups.append({"g": g, "rows": md.row_group(g).num_rows,
                       "bytes": md.row_group(g).total_byte_size,
                       "tmin": stats.min, "tmax": stats.max})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(groups))
    return groups


def markets(start: datetime, end: datetime, *, min_volume: float,
            keep_machine: bool) -> pa.Table:
    """Markets resolving inside the window, with a clean binary outcome.

    ``outcome_prices`` is the settled pair: "['1', '0']" is YES, "['0', '1']"
    is NO. Anything else (unresolved, refunded, multi-outcome) is dropped
    rather than guessed at — a label you have to infer is not a label."""
    path = hf_hub_download(REPO, "markets.parquet", repo_type="dataset",
                           local_dir=str(HF))
    t = pq.read_table(path)
    t = t.filter(pc.and_(pc.greater_equal(t["end_date"], start),
                         pc.less(t["end_date"], end)))
    t = t.filter(pc.greater_equal(t["volume"], min_volume))
    t = t.filter(pc.is_in(t["outcome_prices"],
                          value_set=pa.array(["['1', '0']", "['0', '1']"])))
    # A market can only be predicted from history it actually has.
    t = t.filter(pc.greater(pc.subtract(t["end_date"], t["created_at"]),
                            pa.scalar(timedelta(days=2))))
    if not keep_machine:
        lowered = pc.utf8_lower(t["question"])
        keep = pc.invert(pc.or_(pc.match_substring(lowered, MACHINE[0]),
                                pc.match_substring(lowered, MACHINE[2])))
        t = t.filter(keep)
        league = pa.array([s.split("-")[0] in LEAGUES
                           for s in t["event_slug"].to_pylist()])
        t = t.filter(pc.invert(league))
    resolved_yes = pc.equal(t["outcome_prices"], "['1', '0']")
    return t.append_column("resolved_yes", resolved_yes)


def hourly_bars(fills: pa.Table) -> pa.Table:
    """Fold raw fills into one row per market-hour.

    VWAP rather than last trade: a single 5-dollar print at a stale price is
    not where the market was. The flow columns (buy/sell dollars, distinct
    takers) are the part an OHLCV feed cannot give you — they come from the
    on-chain tape."""
    hour = pc.multiply(pc.divide(fills["timestamp"], pa.scalar(3600)),
                       pa.scalar(3600))
    fills = fills.append_column("hour", hour)
    buy = pc.if_else(pc.equal(fills["taker_direction"], "BUY"),
                     fills["usd_amount"], pa.scalar(0.0))
    sell = pc.if_else(pc.equal(fills["taker_direction"], "SELL"),
                      fills["usd_amount"], pa.scalar(0.0))
    fills = fills.append_column("buy_usd", buy).append_column("sell_usd", sell)
    notional = pc.multiply(fills["price"], fills["token_amount"])
    fills = fills.append_column("notional", notional)
    grouped = fills.group_by(["market_id", "hour"], use_threads=False).aggregate([
        ("notional", "sum"), ("token_amount", "sum"), ("usd_amount", "sum"),
        ("price", "min"), ("price", "max"), ("price", "last"),
        ("buy_usd", "sum"), ("sell_usd", "sum"),
        ("taker", "count_distinct"), ("price", "count"),
    ])
    vwap = pc.divide(grouped["notional_sum"],
                     pc.max_element_wise(grouped["token_amount_sum"],
                                         pa.scalar(1e-9)))
    return pa.table({
        "market_id": grouped["market_id"],
        "hour": grouped["hour"],
        "vwap": vwap,
        "low": grouped["price_min"],
        "high": grouped["price_max"],
        "close": grouped["price_last"],
        "usd": grouped["usd_amount_sum"],
        "buy_usd": grouped["buy_usd_sum"],
        "sell_usd": grouped["sell_usd_sum"],
        "takers": grouped["taker_count_distinct"],
        "fills": grouped["price_count"],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-15")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--resolve-from", default="2025-11-01",
                    help="keep markets resolving on/after this date")
    ap.add_argument("--min-volume", type=float, default=250_000.0)
    ap.add_argument("--out-prefix", default="pm",
                    help="filename prefix, so a wider window can be pulled "
                         "without clobbering a table a run is reading")
    ap.add_argument("--keep-machine", action="store_true",
                    help="keep the 15-minute crypto up/down markets")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    resolve_from = datetime.fromisoformat(args.resolve_from).replace(
        tzinfo=timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)

    universe = markets(resolve_from, end, min_volume=args.min_volume,
                       keep_machine=args.keep_machine)
    print(f">> universe: {universe.num_rows} resolved markets "
          f"({pc.sum(universe['resolved_yes']).as_py()} YES) "
          f"ending {args.resolve_from}..{args.end}")
    pq.write_table(universe, OUT / f"{args.out_prefix}_markets.parquet")

    wanted = pa.array(sorted(set(universe["id"].to_pylist())))
    # token1 is the YES side; every bar is built from YES fills only.
    yes_tokens = pa.array(sorted({t for t in universe["token1"].to_pylist() if t}))
    groups = [g for g in row_group_index()
              if g["tmax"] >= start.timestamp() and g["tmin"] < end.timestamp()]
    fetched = sum(g["bytes"] for g in groups) / 1e9
    print(f">> fills: {len(groups)} row groups, ~{fetched * 0.29:.1f} GB "
          f"on the wire ({sum(g['rows'] for g in groups) / 1e6:.0f}M fills)")

    # asset_id says WHICH outcome token traded. Without it, a market's bars
    # mix YES fills at 0.01 with NO fills at 0.99 and the series flips between
    # p and 1-p — the defect that made "Will Jesus Christ return in 2025?"
    # appear to swing from 0.009 to 0.993 and back, 1,600 times.
    columns = ["timestamp", "market_id", "asset_id", "price", "usd_amount",
               "token_amount", "taker", "maker", "taker_direction"]
    bars, kept, seen = [], 0, 0
    with HfFileSystem().open(f"datasets/{REPO}/trades.parquet") as handle:
        pf = pq.ParquetFile(handle)
        for i, g in enumerate(groups, 1):
            chunk = pf.read_row_group(g["g"], columns=columns)
            mine = chunk.filter(pc.and_(
                pc.is_in(chunk["market_id"], value_set=wanted),
                pc.is_in(chunk["asset_id"], value_set=yes_tokens)))
            seen += chunk.num_rows
            kept += mine.num_rows
            if mine.num_rows:
                bars.append(hourly_bars(mine))
            print(f"   [{i:>3}/{len(groups)}] group {g['g']:>4}  "
                  f"{chunk.num_rows / 1e6:5.2f}M fills -> {mine.num_rows:>8,} "
                  f"in universe", flush=True)
    ticks = pa.concat_tables(bars).combine_chunks()
    # Row groups are cut by time, so one market-hour can span two of them.
    ticks = ticks.group_by(["market_id", "hour"], use_threads=False).aggregate([
        ("vwap", "mean"), ("low", "min"), ("high", "max"), ("close", "last"),
        ("usd", "sum"), ("buy_usd", "sum"), ("sell_usd", "sum"),
        ("takers", "sum"), ("fills", "sum")])
    ticks = ticks.rename_columns([c.removesuffix("_mean").removesuffix("_min")
                                  .removesuffix("_max").removesuffix("_last")
                                  .removesuffix("_sum")
                                  for c in ticks.column_names])
    pq.write_table(ticks, OUT / f"{args.out_prefix}_price_ticks.parquet")
    print(f">> wrote {ticks.num_rows:,} market-hours from {kept:,} fills "
          f"(scanned {seen / 1e6:.0f}M)")


if __name__ == "__main__":
    main()
