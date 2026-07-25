"""Given a headline, will this market move in the next 15 minutes?

`jumps.find_jumps` established the hard part: 70.6% of jumps have a matching
headline within 30 minutes, and so do 71.9% of *random* minutes. The presence
of news carries no information at all — lift −1.4%. Any skill here has to come
from reading the headline and judging whether *this* one matters.

That makes it the cleanest possible test of a sentence encoder:

    PREDICT news_events.market_moves
    FROM news_events
    WHERE news_events.event_id IN :ids
    RETURN PROBABILITY

Population: (headline, market) pairs, matched by embedding similarity. Label:
did the market's price move by at least the jump threshold in the 15 minutes
after the headline appeared. Two arms, identical but for one column — the
headline text.

    python -m jumps.impact --context-cells 512
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from relativedb import (ContextPolicy, Engine, ExecutionInput, LinkDef,
                        RetrieverWiring, Row, RtNativeBackend, Schema,
                        TableDef, TemporalBound, ValueType)
from relativedb.rt_native import ContextTruncationWarning

from scale.analyze import paired_bootstrap
from scale.resolve import accuracy, auroc, brier, logloss
from scale.semantic import embed

DATA = Path(__file__).resolve().parent.parent / "data" / "jumps"
QUERY = ("PREDICT news_events.market_moves FROM news_events "
         "WHERE news_events.event_id IN :ids RETURN PROBABILITY")


def minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


# ---------------------------------------------------------------------------
def events(*, floor: float, window: int, min_move: float, sigmas: float):
    """(headline, market) pairs with the label the model has to predict."""
    markets = {r["market_id"]: r["question"]
               for r in pq.read_table(DATA / "markets.parquet").to_pylist()}
    news = pq.read_table(DATA / "news.parquet").to_pylist()
    bars_rows = pq.read_table(DATA / "bars.parquet").to_pylist()

    series: dict[str, list] = {}
    for r in bars_rows:
        series.setdefault(r["market_id"], []).append(
            {"at": r["at"].replace(tzinfo=timezone.utc), "close": r["close"],
             "volume": r["volume"]})
    for bars in series.values():
        bars.sort(key=lambda b: b["at"])
    lookup = {mid: {b["at"]: b["close"] for b in bars}
              for mid, bars in series.items()}
    typical = {}
    for mid, bars in series.items():
        moves = [abs(bars[i + window]["close"] - bars[i]["close"])
                 for i in range(0, max(0, len(bars) - window), 5)]
        typical[mid] = statistics.median(moves) if moves else 0.001

    order = sorted(markets)
    market_vectors = embed([markets[k] for k in order], tag="jump_markets")
    news_vectors = embed([a["headline"] for a in news], tag="jump_news")
    print(f">> matching {len(news):,} headlines against {len(order)} markets")

    out = []
    for start in range(0, len(news), 4096):
        block = news_vectors[start:start + 4096]
        sims = block @ market_vectors.T
        for row in range(block.shape[0]):
            best = int(sims[row].argmax())
            score = float(sims[row][best])
            if score < floor:
                continue
            article = news[start + row]
            market_id = order[best]
            at = minute(article["published_at"].replace(tzinfo=timezone.utc))
            prices = lookup.get(market_id, {})
            before = prices.get(at)
            after = prices.get(at + timedelta(minutes=window))
            if before is None or after is None:
                continue
            move = after - before
            threshold = max(min_move, sigmas * typical[market_id])
            out.append({
                "event_id": f'{article["article_id"]}:{market_id}',
                "article_id": article["article_id"], "market_id": market_id,
                "at": at, "headline": article["headline"],
                "domain": article["domain"], "tone": article["tone"],
                "similarity": round(score, 4), "before": before, "after": after,
                "move": move, "market_moves": abs(move) >= threshold,
                "threshold": threshold})

    # GDELT republishes the same story across outlets and consecutive files, so
    # one headline can land on one market at one minute a dozen times. Keeping
    # every copy lets one loud story dominate the sample and puts near-copies of
    # an event into its own context.
    unique: dict[tuple, dict] = {}
    for e in out:
        key = (e["market_id"], e["at"], e["headline"][:120])
        if key not in unique or e["similarity"] > unique[key]["similarity"]:
            unique[key] = e
    print(f">> deduped {len(out):,} -> {len(unique):,} distinct events")
    return list(unique.values()), markets, series


def build(events_all, markets, series, population, *, muted: bool,
          bar_every: int = 5, history: int = 720):
    """Four tables. Minute bars are thinned to every `bar_every` minutes: the
    walk cannot read 350k rows through a 512-cell context, and a five-minute
    grid carries the same shape of the tape."""
    tables = [
        TableDef.new_table("markets")
        .column("question", ValueType.TEXT)
        .primary_key("market_id").build(),

        TableDef.new_table("price_ticks")
        .column("at", ValueType.DATETIME)
        .column("close", ValueType.NUMBER)
        .column("ret", ValueType.NUMBER)
        .column("volume", ValueType.NUMBER)
        .primary_key("tick_id").time_column("at").build(),

        TableDef.new_table("outlets")
        .column("domain", ValueType.TEXT)
        .primary_key("outlet_id").build(),

        TableDef.new_table("news_events")
        .column("headline", ValueType.TEXT)
        .column("published_at", ValueType.DATETIME)
        .column("tone", ValueType.NUMBER)
        .column("similarity", ValueType.NUMBER)
        .column("market_moves", ValueType.BOOLEAN)       # the target
        .primary_key("event_id").time_column("published_at").build(),
    ]
    links = [LinkDef("price_ticks", "market_id", "markets"),
             LinkDef("news_events", "market_id", "markets"),
             LinkDef("news_events", "outlet_id", "outlets")]

    rows: dict[str, list[Row]] = {t.name: [] for t in tables}
    for market_id, question in markets.items():
        rows["markets"].append(Row("markets", market_id, {"question": question}))
    for market_id, bars in series.items():
        previous = None
        for i, b in enumerate(bars):
            if i % bar_every:
                continue
            rows["price_ticks"].append(Row(
                "price_ticks", f'{market_id}:{int(b["at"].timestamp())}',
                {"at": b["at"], "close": b["close"], "volume": b["volume"],
                 **({} if previous is None else {"ret": b["close"] - previous})},
                b["at"], {"market_id": market_id}))
            previous = b["close"]

    seen = set()
    scored = {e["event_id"] for e in population}
    for e in events_all:
        outlet = e["domain"] or "unknown"
        if outlet not in seen:
            seen.add(outlet)
            rows["outlets"].append(Row("outlets", outlet, {"domain": outlet}))
        cells = {"headline": "" if muted else e["headline"],
                 "published_at": e["at"], "similarity": e["similarity"]}
        if e["tone"] is not None:
            cells["tone"] = e["tone"]
        # An event's own outcome is knowable only after its window closes; a
        # peer keeps its label only if it resolved before this anchor.
        cells["market_moves"] = e["market_moves"]
        rows["news_events"].append(Row(
            "news_events", e["event_id"], cells, e["at"],
            {"market_id": e["market_id"], "outlet_id": outlet}))

    schema = Schema(tuple(tables), tuple(links))
    window = 15
    by_id = {name: {r.id: r for r in table_rows}
             for name, table_rows in rows.items()}
    children: dict[tuple[str, str], dict] = {}
    for link in links:
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

    def unlabel(row: Row, bound: TemporalBound) -> Row:
        if bound.as_of is None or row.timestamp is None:
            return row
        if row.timestamp + timedelta(minutes=window) <= bound.as_of:
            return row
        cells = dict(row.cells)
        cells.pop("market_moves", None)
        return Row(row.table, row.id, cells, row.timestamp, row.parents)

    def entities(table, ids, bound):
        found = [r for i in ids if (r := by_id[table].get(i)) is not None
                 and bound.admits_row(r)]
        return ([unlabel(r, bound) for r in found]
                if table == "news_events" else found)

    def link_rows(link, parent_id, bound, limit):
        found = [r for r in children[(link.from_table, link.fk_column)]
                 .get(parent_id, ()) if bound.admits_row(r)][:limit]
        return ([unlabel(r, bound) for r in found]
                if link.from_table == "news_events" else found)

    def scanner(table, bound):
        for row in rows[table]:
            if bound.admits_row(row):
                yield unlabel(row, bound) if table == "news_events" else row

    wiring = RetrieverWiring.new_wiring().default_links(link_rows)
    for table in rows:
        wiring.entities(table, entities)
        wiring.scanner(table, scanner)
    return schema, wiring.build(), {k: len(v) for k, v in rows.items()}


def score(schema, wiring, ids, *, cells, batch):
    engine = Engine(schema, wiring,
                    model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                                  max_seq_len=cells,
                                                  batch_size=batch),
                    context_policy=ContextPolicy(max_context_cells=cells,
                                                 local_context_cells=cells // 2,
                                                 bfs_width=24, max_hops=3,
                                                 seed=0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = engine.execute(ExecutionInput(query=QUERY,
                                              per_entity_anchor=True,
                                              params={"ids": list(ids)}))
    truncated = sum(1 for w in caught
                    if isinstance(w.message, ContextTruncationWarning))
    return {p.id: float(p.probability) for p in result.predictions}, truncated


def report(name, scores, truth):
    area = auroc(scores, truth)
    print(f"  {name:<28} acc {accuracy(scores, truth):.3f}   "
          f"auroc {'n/a' if area is None else f'{area:.3f}'}   "
          f"brier {brier(scores, truth):.4f}   logloss {logloss(scores, truth):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.45)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--min-move", type=float, default=0.03)
    ap.add_argument("--sigmas", type=float, default=6.0)
    ap.add_argument("--per-class", type=int, default=450,
                    help="movers and non-movers to score in each split")
    ap.add_argument("--context-cells", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    all_events, markets, series = events(floor=args.floor, window=args.window,
                                        min_move=args.min_move,
                                        sigmas=args.sigmas)
    movers = [e for e in all_events if e["market_moves"]]
    quiet = [e for e in all_events if not e["market_moves"]]
    print(f">> {len(all_events):,} (headline, market) events — "
          f"{len(movers):,} followed by a move ({len(movers)/max(1,len(all_events)):.1%})")

    # Balanced by design: the natural rate is a few percent, and a balanced
    # panel makes the AUROC comparison between arms readable at this n. The
    # split is by time, so the holdout is genuinely later news.
    rng = random.Random(0)
    movers.sort(key=lambda e: e["at"])
    quiet.sort(key=lambda e: e["at"])
    cut = movers[len(movers) // 2]["at"] if movers else None
    population = []
    for name in ("dev", "holdout"):
        take = lambda pool: [e for e in pool
                             if (e["at"] < cut) == (name == "dev")]
        pos, neg = take(movers), take(quiet)
        pos = rng.sample(pos, min(len(pos), args.per_class))
        neg = rng.sample(neg, min(len(neg), args.per_class))
        for e in pos + neg:
            e["split"] = name
        population.extend(pos + neg)
    print(f">> scoring {len(population)} events, split at {cut:%m-%d %H:%M}")

    ids = [e["event_id"] for e in population]
    arms = {}
    for arm, muted in (("full", False), ("muted", True)):
        path = DATA / f"impact_{arm}_{args.context_cells}.parquet"
        if path.exists() and not args.force:
            arms[arm] = {r["event_id"]: r["p"]
                         for r in pq.read_table(path).to_pylist()}
            print(f"   reusing {path.name}")
            continue
        schema, wiring, counts = build(all_events, markets, series, population,
                                       muted=muted)
        if arm == "full":
            print("   tables: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        partial = path.with_suffix(".partial.parquet")
        scored = ({r["event_id"]: r["p"]
                   for r in pq.read_table(partial).to_pylist()}
                  if partial.exists() and not args.force else {})
        todo = [i for i in ids if i not in scored]
        print(f"   scoring {arm}: {len(todo)} to go", flush=True)
        for start in range(0, len(todo), args.chunk):
            got, truncated = score(schema, wiring, todo[start:start + args.chunk],
                                   cells=args.context_cells,
                                   batch=args.batch_size)
            scored.update(got)
            pq.write_table(pa.table({"event_id": list(scored),
                                     "p": [scored[i] for i in scored]}), partial)
            print(f"     {min(start + args.chunk, len(todo)):>5}/{len(todo)}",
                  flush=True)
        pq.write_table(pa.table({"event_id": list(scored),
                                 "p": [scored[i] for i in scored]}), path)
        partial.unlink(missing_ok=True)
        arms[arm] = scored

    for name in ("dev", "holdout"):
        group = [e for e in population if e["split"] == name]
        if len(group) < 40:
            continue
        gids = [e["event_id"] for e in group]
        truth = [e["market_moves"] for e in group]
        full = [arms["full"][i] for i in gids]
        muted = [arms["muted"][i] for i in gids]
        print(f"\n== {name}: {len(group)} events, "
              f"{sum(truth) / len(truth):.1%} followed by a move ==")
        report("always moves (p=0.5)", [0.5] * len(group), truth)
        report("similarity to market", [e["similarity"] for e in group], truth)
        report("RT-J, headline text", full, truth)
        report("RT-J, headline muted", muted, truth)
        lo, hi = paired_bootstrap(full, muted, truth)
        print(f"  AUROC gain from the headline: "
              f"{auroc(full, truth) - auroc(muted, truth):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    best = sorted([e for e in population if e["market_moves"]],
                  key=lambda e: -arms["full"][e["event_id"]])[:8]
    print("\n== headlines the model flagged as high-impact, that did move ==")
    for e in best:
        print(f"  p={arms['full'][e['event_id']]:.2f} (muted "
              f"{arms['muted'][e['event_id']]:.2f})  "
              f"{e['before']:.3f}->{e['after']:.3f}  {e['at']:%m-%d %H:%M}")
        print(f"     {markets[e['market_id']][:66]}")
        print(f"     {e['headline'][:76]}")


if __name__ == "__main__":
    main()
