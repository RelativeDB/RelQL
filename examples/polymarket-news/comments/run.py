"""Does what a commenter *says* predict the next few hours of price?

The population is comments, not markets. Each comment is one row with a
masked boolean:

    PREDICT comments.price_up
    FROM comments
    WHERE comments.comment_id IN :ids
    RETURN PROBABILITY

`price_up` is the sign of the market's move over the H hours after the comment
was posted. The database is cut at the comment's timestamp, so the model sees
the thread so far, the tape so far, and the commenter's own history — and
nothing after.

The experiment is the ablation. Two arms differ by one column:

    full     the comment body is present
    muted    the body is blanked, everything else identical

Same rows, same tape, same author history, same timestamps. The gap between
them is what the *sentence* was worth. Baselines (momentum, reaction count,
a constant) say whether either arm beat the price's own recent behaviour.

    python -m comments.run --horizon 6
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from relativedb import (ContextPolicy, Engine, ExecutionInput, LinkDef,
                        RetrieverWiring, Row, RtNativeBackend, Schema,
                        TableDef, TemporalBound, ValueType)
from relativedb.rt_native import ContextTruncationWarning

from scale.resolve import accuracy, auroc, brier, logloss
from scale.analyze import paired_bootstrap

DATA = Path(__file__).resolve().parent.parent / "data" / "comments"
QUERY = ("PREDICT comments.price_up FROM comments "
         "WHERE comments.comment_id IN :ids RETURN PROBABILITY")


def hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def price_at(bars, when: datetime):
    last = None
    for b in bars:
        if b["at"] <= when:
            last = b["close"]
        else:
            break
    return last


def build(snapshot: dict, *, horizon: int, muted: bool, history: int = 168):
    tables = [
        TableDef.new_table("markets")
        .column("question", ValueType.TEXT)
        .column("outcome_label", ValueType.TEXT)
        .primary_key("market_id").build(),

        TableDef.new_table("price_ticks")
        .column("at", ValueType.DATETIME)
        .column("close", ValueType.NUMBER)
        .column("ret", ValueType.NUMBER)
        .column("volume", ValueType.NUMBER)
        .primary_key("tick_id").time_column("at").build(),

        TableDef.new_table("authors")
        .column("name", ValueType.TEXT)
        .primary_key("author_id").build(),

        TableDef.new_table("comments")
        .column("body", ValueType.TEXT)
        .column("posted_at", ValueType.DATETIME)
        .column("reactions", ValueType.NUMBER)
        .column("price_up", ValueType.BOOLEAN)          # the target
        .primary_key("comment_id").time_column("posted_at").build(),
    ]
    links = [LinkDef("price_ticks", "market_id", "markets"),
             LinkDef("comments", "market_id", "markets"),
             LinkDef("comments", "author_id", "authors")]

    bars_by_market = {}
    for market_id, bars in snapshot["candles"].items():
        series, previous = [], None
        for b in sorted(bars, key=lambda b: b["ts"]):
            at = datetime.fromtimestamp(b["ts"] / 1000, tz=timezone.utc)
            series.append({"at": at, "close": b["close"],
                           "ret": None if previous is None else b["close"] - previous,
                           "volume": b["volume"]})
            previous = b["close"]
        bars_by_market[market_id] = series

    rows = {t.name: [] for t in tables}
    for m in snapshot["markets"]:
        rows["markets"].append(Row("markets", m["market_id"], {
            "question": m["question"], "outcome_label": m["outcome_label"]}))
    for a in snapshot["authors"]:
        rows["authors"].append(Row("authors", a["author_id"],
                                   {"name": a["name"] or "anonymous"}))
    for market_id, series in bars_by_market.items():
        for b in series:
            rows["price_ticks"].append(Row(
                "price_ticks", f"{market_id}:{int(b['at'].timestamp())}",
                {k: v for k, v in b.items() if v is not None}, b["at"],
                {"market_id": market_id}))

    kept = []
    for c in snapshot["comments"]:
        series = bars_by_market.get(c["market_id"])
        if not series:
            continue
        posted = hour(datetime.fromisoformat(c["created_at"].replace("Z", "+00:00")))
        before = price_at(series, posted)
        after = price_at(series, posted + timedelta(hours=horizon))
        if before is None or after is None or after == before:
            continue                    # no quote either side, or a flat window
        if series[-1]["at"] < posted + timedelta(hours=horizon):
            continue                    # the window is not fully observed yet
        recent = price_at(series, posted - timedelta(hours=horizon))
        c = dict(c, posted_at=posted, before=before, after=after,
                 price_up=after > before,
                 momentum=0.0 if recent is None else before - recent)
        kept.append(c)
        rows["comments"].append(Row("comments", c["comment_id"], {
            "body": "" if muted else c["body"],
            "posted_at": posted, "reactions": c["reactions"],
            "price_up": c["price_up"]}, posted,
            {"market_id": c["market_id"], "author_id": c["author_id"]}))

    schema = Schema(tuple(tables), tuple(links))
    by_id = {name: {r.id: r for r in table_rows} for name, table_rows in rows.items()}
    children = {}
    for link in links:
        index = {}
        for row in rows[link.from_table]:
            parent = row.parents.get(link.fk_column)
            if parent is not None:
                index.setdefault(parent, []).append(row)
        for bucket in index.values():
            bucket.sort(key=lambda r: (r.timestamp is None,
                                       -(r.timestamp.timestamp() if r.timestamp else 0)))
        children[(link.from_table, link.fk_column)] = index

    def unlabel(row: Row, bound: TemporalBound) -> Row:
        """A comment's own outcome is only known H hours after it was posted;
        peers in context may only show a label once their window has closed."""
        if bound.as_of is None or row.timestamp is None:
            return row
        if row.timestamp + timedelta(hours=horizon) <= bound.as_of:
            return row
        cells = dict(row.cells)
        cells.pop("price_up", None)
        return Row(row.table, row.id, cells, row.timestamp, row.parents)

    def entities(table, ids, bound):
        found = [r for i in ids if (r := by_id[table].get(i)) is not None
                 and bound.admits_row(r)]
        return [unlabel(r, bound) for r in found] if table == "comments" else found

    def link_rows(link, parent_id, bound, limit):
        found = [r for r in children[(link.from_table, link.fk_column)].get(parent_id, ())
                 if bound.admits_row(r)][:limit]
        return ([unlabel(r, bound) for r in found]
                if link.from_table == "comments" else found)

    def scanner(table, bound):
        for row in rows[table]:
            if bound.admits_row(row):
                yield unlabel(row, bound) if table == "comments" else row

    wiring = RetrieverWiring.new_wiring().default_links(link_rows)
    for table in rows:
        wiring.entities(table, entities)
        wiring.scanner(table, scanner)
    return schema, wiring.build(), kept, {k: len(v) for k, v in rows.items()}


def score(schema, wiring, ids, *, cells, batch):
    engine = Engine(schema, wiring,
                    model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                                  max_seq_len=cells,
                                                  batch_size=batch),
                    context_policy=ContextPolicy(max_context_cells=cells,
                                                 local_context_cells=cells // 2,
                                                 bfs_width=32, max_hops=3, seed=0))
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
    return {"name": name, "acc": accuracy(scores, truth), "auroc": area,
            "brier": brier(scores, truth), "logloss": logloss(scores, truth)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6, help="hours ahead")
    ap.add_argument("--context-cells", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--holdout-days", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    snapshot = json.loads((DATA / "snapshot.json").read_text())
    schema, wiring, kept, counts = build(snapshot, horizon=args.horizon,
                                         muted=False)
    if not kept:
        raise SystemExit("no labelled comments — widen the fetch")
    split_at = max(c["posted_at"] for c in kept) - timedelta(days=args.holdout_days)
    print(f"tables: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"labelled comments: {len(kept)}  "
          f"({sum(c['price_up'] for c in kept) / len(kept):.1%} up)  "
          f"horizon {args.horizon}h  holdout after {split_at:%Y-%m-%d}")

    arms = {}
    for arm, muted in (("full", False), ("muted", True)):
        path = DATA / f"pred_{arm}_{args.horizon}h_{args.context_cells}.parquet"
        if path.exists() and not args.force:
            arms[arm] = {r["comment_id"]: r["p"]
                         for r in pq.read_table(path).to_pylist()}
            continue
        s, w, _, _ = build(snapshot, horizon=args.horizon, muted=muted)
        ids = [c["comment_id"] for c in kept]
        print(f"   scoring {arm} ...", flush=True)
        scored, truncated = score(s, w, ids, cells=args.context_cells,
                                  batch=args.batch_size)
        pq.write_table(pa.table({"comment_id": list(scored),
                                 "p": [scored[i] for i in scored]}), path)
        print(f"   wrote {path.name} ({truncated} truncated)")
        arms[arm] = scored

    for split in ("dev", "holdout"):
        group = [c for c in kept
                 if (c["posted_at"] < split_at) == (split == "dev")]
        if len(group) < 30:
            continue
        ids = [c["comment_id"] for c in group]
        truth = [c["price_up"] for c in group]
        full = [arms["full"][i] for i in ids]
        muted = [arms["muted"][i] for i in ids]
        print(f"\n== {split}: {len(group)} comments, "
              f"{sum(truth) / len(truth):.1%} followed by an up move ==")
        report("momentum (prior window)",
               [1 / (1 + math.exp(-c["momentum"] / 0.02)) for c in group], truth)
        report("reaction count", [min(0.99, 0.5 + 0.05 * c["reactions"])
                                  for c in group], truth)
        report("always up (p=0.5)", [0.5] * len(group), truth)
        report("RT-J, comment text", full, truth)
        report("RT-J, text muted", muted, truth)
        lo, hi = paired_bootstrap(full, muted, truth)
        print(f"  AUROC gain from the text: "
              f"{auroc(full, truth) - auroc(muted, truth):+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
