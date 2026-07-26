"""Score the WTI ladder with RT-J, judged against a martingale baseline.

    PREDICT strikes.above
    FROM strikes
    WHERE strikes.strike_id IN :ids
    RETURN PROBABILITY

Population is one row per (day, threshold). Each strike hangs off its day,
which carries the state that preceded it: prior close, 1- and 5-day returns,
30-day realized vol, the Brent spread and the RBOB crack. Days that have
already settled keep their outcomes; the day being priced has its 30 masked.

Baseline: a normal centred on the prior close with sigma from trailing
realized vol — i.e. "futures are a martingale". Beating that is the bar.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import warnings
from pathlib import Path

from relativedb import (ContextPolicy, Engine, ExecutionInput, LinkDef,
                        RetrieverWiring, Row, RtNativeBackend, Schema,
                        TableDef, TemporalBound, ValueType)

DATA = Path(__file__).resolve().parent.parent / "data" / "wti"
QUERY = ("PREDICT strikes.above FROM strikes "
         "WHERE strikes.strike_id IN :ids RETURN PROBABILITY")


def stamp(day: str, hour: int = 18) -> dt.datetime:
    return dt.datetime.combine(dt.date.fromisoformat(day), dt.time(hour),
                               tzinfo=dt.timezone.utc)


def build(days, strikes, *, mask_after: dt.datetime):
    tables = [
        TableDef.new_table("days")
        .column("settles_at", ValueType.DATETIME)
        .column("weekday", ValueType.TEXT)
        .column("prior_close", ValueType.NUMBER)
        .column("ret_1d", ValueType.NUMBER)
        .column("ret_5d", ValueType.NUMBER)
        .column("vol_30d", ValueType.NUMBER)
        .column("brent_spread", ValueType.NUMBER)
        .column("rbob_crack", ValueType.NUMBER)
        .primary_key("day").time_column("settles_at").build(),

        TableDef.new_table("strikes")
        .column("threshold", ValueType.NUMBER)
        .column("pct_from_prior", ValueType.NUMBER)
        .column("known_at", ValueType.DATETIME)
        .column("above", ValueType.BOOLEAN)          # the target
        .primary_key("strike_id").time_column("known_at").build(),
    ]
    links = [LinkDef("strikes", "day", "days")]
    rows = {"days": [], "strikes": []}
    # a day's state is knowable the evening before it settles
    for d in days:
        known = stamp(d["day"], 0)
        rows["days"].append(Row("days", d["day"], {
            "settles_at": stamp(d["day"]), "weekday": d["weekday"],
            "prior_close": d["prior_close"], "ret_1d": d["ret_1d"],
            "ret_5d": d["ret_5d"], "vol_30d": d["vol_30d"],
            **({} if d["brent_spread"] is None else {"brent_spread": d["brent_spread"]}),
            **({} if d["rbob_crack"] is None else {"rbob_crack": d["rbob_crack"]}),
        }, known))
    for s in strikes:
        known = stamp(s["day"], 0)
        cells = {"threshold": s["threshold"],
                 "pct_from_prior": s["pct_from_prior"], "known_at": known}
        if stamp(s["day"]) <= mask_after:
            cells["above"] = s["above"]
        rows["strikes"].append(Row("strikes", s["strike_id"], cells, known,
                                   {"day": s["day"]}))

    schema = Schema(tuple(tables), tuple(links))
    by_id = {n: {r.id: r for r in rs} for n, rs in rows.items()}
    kids = {}
    for row in rows["strikes"]:
        kids.setdefault(row.parents["day"], []).append(row)
    settle = {s["strike_id"]: stamp(s["day"]) for s in strikes}

    def mask(row, bound):
        """A strike's outcome exists only after its day has settled."""
        if bound.as_of is None or settle.get(row.id, dt.datetime.max) <= bound.as_of:
            return row
        cells = dict(row.cells); cells.pop("above", None)
        return Row(row.table, row.id, cells, row.timestamp, row.parents)

    def entities(table, ids, bound):
        found = [r for i in ids if (r := by_id[table].get(i)) is not None
                 and bound.admits_row(r)]
        return [mask(r, bound) for r in found] if table == "strikes" else found

    def link_rows(link, parent_id, bound, limit):
        found = [r for r in kids.get(parent_id, ()) if bound.admits_row(r)][:limit]
        return [mask(r, bound) for r in found]

    def scanner(table, bound):
        for r in rows[table]:
            if bound.admits_row(r):
                yield mask(r, bound) if table == "strikes" else r

    w = RetrieverWiring.new_wiring().default_links(link_rows)
    for t in rows:
        w.entities(t, entities); w.scanner(t, scanner)
    return schema, w.build(), {k: len(v) for k, v in rows.items()}


def martingale(day, strike):
    """Normal centred on the prior close, sigma = trailing realized vol."""
    sigma = day["prior_close"] * day["vol_30d"] / 100
    z = (strike["threshold"] - day["prior_close"]) / sigma
    return 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))


def brier(p, y): return (p - y) ** 2
def logloss(p, y):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -(math.log(p) if y else math.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-days", type=int, default=40)
    ap.add_argument("--context-cells", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=300)
    args = ap.parse_args()

    days = json.loads((DATA / "days.json").read_text())
    strikes = json.loads((DATA / "strikes.json").read_text())
    by_day = {d["day"]: d for d in days}
    eval_days = [d["day"] for d in days[-args.eval_days:]]
    targets = [s for s in strikes if s["day"] in set(eval_days)]
    print(f"evaluating {len(eval_days)} days, {len(targets)} strikes")

    # every eval day is masked; earlier days keep their labels
    cutoff = stamp(eval_days[0], 0)
    schema, wiring, counts = build(days, strikes, mask_after=cutoff)
    print("tables: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    engine = Engine(schema, wiring,
                    model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                                  max_seq_len=args.context_cells,
                                                  batch_size=args.batch_size),
                    context_policy=ContextPolicy(
                        max_context_cells=args.context_cells,
                        local_context_cells=args.context_cells // 2,
                        bfs_width=24, max_hops=3, seed=0))
    ids = [s["strike_id"] for s in targets]
    scored, cache = {}, DATA / f"pred_{args.context_cells}.json"
    if cache.exists():
        scored = json.loads(cache.read_text())
    todo = [i for i in ids if i not in scored]
    print(f"scoring {len(todo)} (cached {len(scored)})", flush=True)
    for start in range(0, len(todo), args.chunk):
        piece = todo[start:start + args.chunk]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = engine.execute(ExecutionInput(query=QUERY, per_entity_anchor=True,
                                              params={"ids": piece}))
        scored.update({p.id: float(p.probability) for p in r.predictions})
        cache.write_text(json.dumps(scored))
        print(f"   {min(start+args.chunk, len(todo))}/{len(todo)}", flush=True)

    rows = [(scored[s["strike_id"]], martingale(by_day[s["day"]], s), s["above"])
            for s in targets if s["strike_id"] in scored]
    n = len(rows)
    print(f"\n== {n} strikes over {len(eval_days)} held-out days ==")
    print(f"  {'':<14}{'brier':>9}{'logloss':>10}{'acc':>8}")
    for name, idx in (("RT-J", 0), ("martingale", 1)):
        b = sum(brier(r[idx], r[2]) for r in rows) / n
        l = sum(logloss(r[idx], r[2]) for r in rows) / n
        a = sum((r[idx] >= 0.5) == r[2] for r in rows) / n
        print(f"  {name:<14}{b:>9.4f}{l:>10.4f}{a:>8.3f}")
    base = sum(r[2] for r in rows) / n
    print(f"  {'base rate':<14}{sum(brier(base, r[2]) for r in rows)/n:>9.4f}")


if __name__ == "__main__":
    main()
