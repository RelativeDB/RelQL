"""Kalshi: the same task on a second venue, from TrevorJS/kalshi-trades.

5.7 GB of markets and fills covering 2023-03 to 2026-01. Kalshi settles in
cents and publishes `result` ("yes"/"no") plus `close_time`, so a market that
closed inside the window is a labeled example — the same label as Polymarket's
`outcome_prices`, from an entirely separate exchange with separate traders.

    python -m scale.pull_kalshi --start 2025-10-15 --end 2026-01-01
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent.parent / "data"
SRC = DATA / "hf" / "kalshi"
OUT = DATA / "scale"

# Same reasoning as Polymarket: games resolve on a scoreboard.
SPORT_PREFIXES = ("KXNBA", "KXNHL", "KXNFL", "KXMLB", "KXNCAA", "KXEPL",
                  "KXUCL", "KXATP", "KXWTA", "KXUFC", "KXF1", "KXPGA",
                  "KXMLS", "KXLALIGA", "KXSERIEA", "KXBUNDES", "KXCFB",
                  "KXCBB", "KXTENNIS", "KXGOLF", "KXBOX", "KXCRICKET")


def markets(start: datetime, end: datetime, *, min_volume: int) -> pa.Table:
    t = ds.dataset(sorted(SRC.glob("markets-*.parquet"))).to_table()
    t = t.filter(pc.and_(pc.greater_equal(t["close_time"], start),
                         pc.less(t["close_time"], end)))
    t = t.filter(pc.is_in(t["result"], value_set=pa.array(["yes", "no"])))
    t = t.filter(pc.greater_equal(t["volume"], min_volume))
    sporty = pa.array([any(tk.startswith(p) for p in SPORT_PREFIXES)
                       for tk in t["ticker"].to_pylist()])
    t = t.filter(pc.invert(sporty))
    # The dump carries repeated snapshots of the same ticker; the last one is
    # the settled state, which is where `result` is trustworthy.
    t = t.sort_by([("ticker", "ascending"), ("close_time", "ascending")])
    seen, keep = set(), []
    for i, ticker in enumerate(t["ticker"].to_pylist()):
        if ticker in seen:
            keep.append(False)
        else:
            seen.add(ticker)
            keep.append(True)
    t = t.filter(pa.array(keep))
    return t.append_column("resolved_yes", pc.equal(t["result"], "yes"))


def hourly_bars(tickers: set[str], start: datetime, end: datetime) -> pa.Table:
    """Kalshi fills -> one row per ticker-hour. Prices are cents; the tape
    quotes the YES side, so `yes_price / 100` lines up with a Polymarket
    probability without any further translation."""
    wanted = pa.array(sorted(tickers))
    bars = []
    for path in sorted(SRC.glob("trades-*.parquet")):
        table = pq.read_table(path)
        table = table.filter(pc.and_(
            pc.and_(pc.greater_equal(table["created_time"], start),
                    pc.less(table["created_time"], end)),
            pc.is_in(table["ticker"], value_set=wanted)))
        if not table.num_rows:
            continue
        seconds = pc.divide(pc.cast(table["created_time"], pa.int64()),
                            pa.scalar(1_000_000))
        hour = pc.multiply(pc.divide(seconds, pa.scalar(3600)), pa.scalar(3600))
        price = pc.divide(pc.cast(table["yes_price"], pa.float64()),
                          pa.scalar(100.0))
        notional = pc.multiply(price, pc.cast(table["count"], pa.float64()))
        table = (table.append_column("hour", hour)
                      .append_column("yes", price)
                      .append_column("notional", notional))
        g = table.group_by(["ticker", "hour"], use_threads=False).aggregate([
            ("notional", "sum"), ("count", "sum"), ("yes", "min"),
            ("yes", "max"), ("yes", "last"), ("yes", "count")])
        bars.append(pa.table({
            "ticker": g["ticker"], "hour": g["hour"],
            "vwap": pc.divide(g["notional_sum"],
                              pc.max_element_wise(
                                  pc.cast(g["count_sum"], pa.float64()),
                                  pa.scalar(1e-9))),
            "low": g["yes_min"], "high": g["yes_max"], "close": g["yes_last"],
            "contracts": g["count_sum"], "fills": g["yes_count"]}))
        print(f"   {path.name}: {table.num_rows:>9,} fills in window",
              flush=True)
    ticks = pa.concat_tables(bars).combine_chunks()
    ticks = ticks.group_by(["ticker", "hour"], use_threads=False).aggregate([
        ("vwap", "mean"), ("low", "min"), ("high", "max"), ("close", "last"),
        ("contracts", "sum"), ("fills", "sum")])
    return ticks.rename_columns([c.removesuffix("_mean").removesuffix("_min")
                                 .removesuffix("_max").removesuffix("_last")
                                 .removesuffix("_sum")
                                 for c in ticks.column_names])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-15")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--resolve-from", default="2025-11-01")
    ap.add_argument("--min-volume", type=int, default=5_000)
    ap.add_argument("--out-prefix", default="kalshi",
                    help="filename prefix, so a wider pull can sit beside the "
                         "one study 2 was built on")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    resolve_from = datetime.fromisoformat(args.resolve_from).replace(
        tzinfo=timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)

    universe = markets(resolve_from, end, min_volume=args.min_volume)
    print(f">> kalshi universe: {universe.num_rows} markets "
          f"({pc.sum(universe['resolved_yes']).as_py()} YES)")
    pq.write_table(universe, OUT / f"{args.out_prefix}_markets.parquet")

    ticks = hourly_bars(set(universe["ticker"].to_pylist()), start, end)
    pq.write_table(ticks, OUT / f"{args.out_prefix}_price_ticks.parquet")
    print(f">> wrote {ticks.num_rows:,} ticker-hours")


if __name__ == "__main__":
    main()
