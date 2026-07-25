"""Fine-tune on Trump / war / oil markets, then test on later ones.

Every result so far has been zero-shot: a frozen checkpoint, no gradient ever
taken. This asks the obvious next question — if the model is *trained* on the
resolutions of one family of markets, does it get better at that family's
future?

    train   in-domain markets settling 2025-08-01 .. 2025-12-05
    test    in-domain markets settling 2025-12-06 .. 2025-12-31   (later)
    control out-of-domain markets in the same test window

The control is the point. A fine-tune that lifts the domain *and* everything
else has probably just learned the task; one that lifts the domain and leaves
the control flat has learned the subject matter. One that lifts nothing is a
null, and one that hurts the control is a warning.

Kalshi only. Its tape quotes the YES side explicitly, so the outcome-token
defect that corrupted the Polymarket price series cannot apply here, and its
full history is already on disk — the training window costs nothing to widen.

    python -m finetune.domain --epochs 2 --train-cap 700
"""
from __future__ import annotations

import argparse
import json
import random
import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from relativedb import (ContextPolicy, Engine, ExecutionInput, LinkDef,
                        ModelConfig, RetrieverWiring, Row, RtNativeBackend,
                        Schema, TableDef, TemporalBound, ValueType)
from relativedb.rt_native import ContextTruncationWarning

from scale.analyze import paired_bootstrap
from scale.resolve import accuracy, auroc, brier, logloss

SCALE = Path(__file__).resolve().parent.parent / "data" / "scale"
OUT = Path(__file__).resolve().parent.parent / "data" / "finetune"
SPLIT_AT = datetime(2025, 12, 6, tzinfo=timezone.utc)
QUERY = ("PREDICT markets.resolved_yes FROM markets "
         "WHERE markets.market_id IN :ids RETURN PROBABILITY")

DOMAIN = {
    "trump": r"\btrump\b|white house|potus",
    "war": (r"\biran\b|israel|ukrain|russia|\bwar\b|militar|strike|ceasefire|"
            r"nato|gaza|hormuz|invad|troops|missile|nuclear"),
    "oil": r"\boil\b|crude|\bwti\b|brent|opec|gasoline|barrel",
}


def categories(question: str) -> set[str]:
    low = question.lower()
    return {name for name, pattern in DOMAIN.items()
            if re.search(pattern, low)}


def stamp(value) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
def load(*, lead_hours: int, min_ticks: int, history_hours: int):
    ticks: dict[str, list] = {}
    for t in pq.read_table(SCALE / "kalshi_ft_price_ticks.parquet").to_pylist():
        ticks.setdefault(t["ticker"], []).append(
            {"at": stamp(t["hour"]), "close": t["close"], "vwap": t["vwap"],
             "high": t["high"], "low": t["low"], "fills": t["fills"]})
    lead, history = timedelta(hours=lead_hours), timedelta(hours=history_hours)
    markets = []
    for r in pq.read_table(SCALE / "kalshi_ft_markets.parquet").to_pylist():
        closes = stamp(r["close_time"])
        anchor = (closes - lead).replace(hour=0, minute=0, second=0,
                                         microsecond=0)
        bars = sorted([b for b in ticks.get(r["ticker"], ())
                       if anchor - history <= b["at"] <= anchor],
                      key=lambda b: b["at"])
        if len(bars) < min_ticks:
            continue
        markets.append({
            "market_id": r["ticker"], "event_id": r["event_ticker"],
            "question": r["title"], "outcome_label": r["yes_sub_title"],
            "opened_at": stamp(r["open_time"]), "closes_at": closes,
            "anchor_at": anchor, "bars": bars,
            "price_at_anchor": sum(b["vwap"] for b in bars[-24:]) / len(bars[-24:]),
            "resolved_yes": bool(r["resolved_yes"]),
            "categories": categories(r["title"])})
    return markets


def database(markets):
    tables = [
        TableDef.new_table("events")
        .column("ticker", ValueType.TEXT)
        .primary_key("event_id").build(),

        TableDef.new_table("markets")
        .column("question", ValueType.TEXT)
        .column("outcome_label", ValueType.TEXT)
        .column("opened_at", ValueType.DATETIME)
        .column("closes_at", ValueType.DATETIME)
        .column("known_at", ValueType.DATETIME)
        .column("resolved_yes", ValueType.BOOLEAN)
        .primary_key("market_id").time_column("known_at").build(),

        TableDef.new_table("price_ticks")
        .column("at", ValueType.DATETIME)
        .column("vwap", ValueType.NUMBER)
        .column("close", ValueType.NUMBER)
        .column("high", ValueType.NUMBER)
        .column("low", ValueType.NUMBER)
        .column("fills", ValueType.NUMBER)
        .primary_key("tick_id").time_column("at").build(),
    ]
    links = [LinkDef("markets", "event_id", "events"),
             LinkDef("price_ticks", "market_id", "markets")]
    rows: dict[str, list[Row]] = {t.name: [] for t in tables}
    seen = set()
    for m in markets:
        if m["event_id"] not in seen:
            seen.add(m["event_id"])
            rows["events"].append(Row("events", m["event_id"],
                                      {"ticker": m["event_id"]}))
        rows["markets"].append(Row("markets", m["market_id"], {
            "question": m["question"], "outcome_label": m["outcome_label"],
            "opened_at": m["opened_at"], "closes_at": m["closes_at"],
            "known_at": m["anchor_at"], "resolved_yes": m["resolved_yes"]},
            m["anchor_at"], {"event_id": m["event_id"]}))
        for b in m["bars"]:
            rows["price_ticks"].append(Row(
                "price_ticks", f'{m["market_id"]}:{int(b["at"].timestamp())}',
                {k: v for k, v in b.items() if v is not None}, b["at"],
                {"market_id": m["market_id"]}))

    schema = Schema(tuple(tables), tuple(links))
    settles = {m["market_id"]: m["closes_at"] for m in markets}
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

    def mask(row: Row, bound: TemporalBound) -> Row:
        closes = settles.get(row.id)
        if closes is None or bound.as_of is None or closes <= bound.as_of:
            return row
        cells = dict(row.cells)
        cells.pop("resolved_yes", None)
        return Row(row.table, row.id, cells, row.timestamp, row.parents)

    def entities(table, ids, bound):
        found = [r for i in ids if (r := by_id[table].get(i)) is not None
                 and bound.admits_row(r)]
        return [mask(r, bound) for r in found] if table == "markets" else found

    def link_rows(link, parent_id, bound, limit):
        found = [r for r in children[(link.from_table, link.fk_column)]
                 .get(parent_id, ()) if bound.admits_row(r)][:limit]
        return ([mask(r, bound) for r in found]
                if link.from_table == "markets" else found)

    def scanner(table, bound):
        for row in rows[table]:
            if bound.admits_row(row):
                yield mask(row, bound) if table == "markets" else row

    wiring = RetrieverWiring.new_wiring().default_links(link_rows)
    for table in rows:
        wiring.entities(table, entities)
        wiring.scanner(table, scanner)
    return schema, wiring.build(), {k: len(v) for k, v in rows.items()}


def engine_for(schema, wiring, *, cells, batch, model_uri=None):
    config = ModelConfig()
    if model_uri:
        config = config.with_model_uri(str(model_uri))
    # Checkpoint routing lives on ModelConfig, not the backend: the backend
    # asks the engine's config which checkpoint serves the task type.
    return Engine(schema, wiring, model_config=config,
                  model_backend=RtNativeBackend(
                      schema=schema, wiring=wiring, max_seq_len=cells,
                      batch_size=batch),
                  context_policy=ContextPolicy(
                      max_context_cells=cells, local_context_cells=cells // 2,
                      bfs_width=32, max_hops=3, seed=0))


def score(engine, ids, chunk: int, tag: str):
    out = {}
    for start in range(0, len(ids), chunk):
        piece = ids[start:start + chunk]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = engine.execute(ExecutionInput(
                query=QUERY, per_entity_anchor=True, params={"ids": piece}))
        out.update({p.id: float(p.probability) for p in result.predictions})
        print(f"     {tag} {min(start + chunk, len(ids)):>5}/{len(ids)}",
              flush=True)
    return out


def report(name, scores, truth):
    area = auroc(scores, truth)
    print(f"  {name:<34} acc {accuracy(scores, truth):.3f}   "
          f"auroc {'n/a' if area is None else f'{area:.3f}'}   "
          f"brier {brier(scores, truth):.4f}   logloss {logloss(scores, truth):.4f}")
    return area


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-hours", type=int, default=48)
    ap.add_argument("--min-ticks", type=int, default=6)
    ap.add_argument("--history-hours", type=int, default=336)
    ap.add_argument("--context-cells", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--train-cap", type=int, default=700)
    ap.add_argument("--test-cap", type=int, default=600)
    ap.add_argument("--control-cap", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--ft-batch", type=int, default=16,
                    help="fine-tune batch. Small batches destabilize the "
                         "per-batch label normalization: with 4 binary "
                         "targets, an all-YES or all-NO batch has ~zero label "
                         "variance, so the normalized target explodes and the "
                         "reported loss rises even as the weights improve. "
                         "16 makes a degenerate batch far less likely.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    markets = load(lead_hours=args.lead_hours, min_ticks=args.min_ticks,
                   history_hours=args.history_hours)
    schema, wiring, counts = database(markets)
    print(">> tables: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    rng = random.Random(args.seed)
    in_domain = [m for m in markets if m["categories"]]
    train = [m for m in in_domain if m["closes_at"] < SPLIT_AT]
    test = [m for m in in_domain if m["closes_at"] >= SPLIT_AT]
    control = [m for m in markets
               if not m["categories"] and m["closes_at"] >= SPLIT_AT]
    if len(train) > args.train_cap:
        train = rng.sample(train, args.train_cap)
    if len(test) > args.test_cap:
        test = rng.sample(test, args.test_cap)
    if len(control) > args.control_cap:
        control = rng.sample(control, args.control_cap)
    print(f">> train {len(train)} in-domain (settle < {SPLIT_AT:%m-%d}), "
          f"test {len(test)} in-domain, control {len(control)} out-of-domain")

    test_ids = [m["market_id"] for m in test]
    control_ids = [m["market_id"] for m in control]
    truth = {m["market_id"]: m["resolved_yes"] for m in test + control}
    price = {m["market_id"]: min(0.99, max(0.01, m["price_at_anchor"]))
             for m in test + control}

    zero_path = OUT / f"zeroshot_{args.context_cells}.json"
    if zero_path.exists():
        zero = json.loads(zero_path.read_text())
        print(">> reusing cached zero-shot scores")
    else:
        base = engine_for(schema, wiring, cells=args.context_cells,
                          batch=args.batch_size)
        print(">> scoring zero-shot")
        zero = score(base, test_ids + control_ids, args.chunk, "zero")
        zero_path.write_text(json.dumps(zero))

    print(f">> fine-tuning on {len(train)} markets, {args.epochs} epoch(s)")
    trainer = engine_for(schema, wiring, cells=args.context_cells,
                         batch=args.batch_size)
    labels = {(m["market_id"], m["anchor_at"]): float(m["resolved_yes"])
              for m in train}
    checkpoint = trainer.finetune(
        QUERY, anchors=sorted({m["anchor_at"] for m in train}),
        entity_ids=[m["market_id"] for m in train], labels=labels,
        params={"ids": [m["market_id"] for m in train]},
        epochs=args.epochs, batch_size=args.ft_batch,
        learning_rate=args.learning_rate,
        output_dir=str(OUT / "checkpoint"))
    print(f"   {checkpoint.examples} examples, {checkpoint.steps} steps, "
          f"{checkpoint.seconds:.0f}s, final loss "
          f"{checkpoint.losses[-1]:.4f} (first {checkpoint.losses[0]:.4f})")

    tuned_engine = engine_for(schema, wiring, cells=args.context_cells,
                              batch=args.batch_size,
                              model_uri=checkpoint.path)
    print(">> scoring fine-tuned")
    tuned = score(tuned_engine, test_ids + control_ids, args.chunk, "tuned")
    (OUT / f"tuned_{args.context_cells}.json").write_text(json.dumps(tuned))

    for label, ids in (("in-domain test (Trump / war / oil)", test_ids),
                       ("control: everything else", control_ids)):
        y = [truth[i] for i in ids]
        if not (0 < sum(y) < len(y)):
            continue
        print(f"\n== {label}: {len(ids)} markets, {sum(y)/len(y):.1%} YES ==")
        report("market price (24h vwap)", [price[i] for i in ids], y)
        a_zero = report("RT-J zero-shot", [zero[i] for i in ids], y)
        a_tuned = report("RT-J fine-tuned", [tuned[i] for i in ids], y)
        lo, hi = paired_bootstrap([tuned[i] for i in ids],
                                  [zero[i] for i in ids], y)
        print(f"  AUROC gain from fine-tuning: {a_tuned - a_zero:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
