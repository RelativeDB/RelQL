"""Real-data loaders → (schema, CSC engine, precomputed truth arrays).

Two public domains, deliberately different in their re-purchase structure so
the ranking task is meaningful in one and a known dead-end in the other:

* **MovieLens** (grouplens ml-latest-small) — users rate movies once; there
  are *zero* re-rated movies, so any "recommend what you saw before" ranker
  has a recall ceiling near zero. Good for churn / activity-count tasks.
* **Online Retail II** (UCI) — customers re-buy the same SKUs constantly, so
  buy-it-again ranking is genuinely learnable.

Truth is precomputed as per-entity time-sorted numpy arrays (epoch seconds)
plus parallel value/item arrays, so windowed ground-truth is a pair of
``searchsorted`` calls — never a re-run of the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from relativedb import (ContextPolicy, Engine, LinkDef, RetrieverWiring, Row,
                        SamplerMode, Schema, TableDef, TemporalBound, ValueType)

# Star schemas here need one hop (entity -> its events). The default policy
# caps children at bfs_width=32/hop, which would silently truncate COUNT/SUM
# over any busy window; give the engine a generous budget so metrics reflect
# the model, not the cap. (The default-cap effect is measured separately.)
WIDE_POLICY = ContextPolicy(max_context_cells=5_000_000, bfs_width=20_000,
                            max_hops=1)

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
DAY = 86400.0
UNIT_SECONDS = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0,
                "days": DAY, "weeks": 7 * DAY, "months": 30 * DAY}
_EPOCH0 = pd.Timestamp("1970-01-01")


def to_epoch(ts) -> float:
    """Naive-UTC epoch seconds. Deliberately *not* ``datetime.timestamp()``,
    which reinterprets a naive datetime in local time and would offset the
    anchor from the event times by the machine's UTC offset."""
    return (pd.Timestamp(ts) - _EPOCH0).total_seconds()


def _epoch_array(series: pd.Series) -> np.ndarray:
    """Robust to the column's datetime resolution (ns/us/s)."""
    return ((series - _EPOCH0) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


@dataclass
class EntityEvents:
    """Per-entity time-sorted event arrays for O(log n) windowed truth."""
    times: dict[Any, np.ndarray] = field(default_factory=dict)   # epoch secs, asc
    values: dict[Any, np.ndarray] = field(default_factory=dict)  # parallel amounts
    items: dict[Any, np.ndarray] = field(default_factory=dict)   # parallel item ids

    def _slice(self, eid, t0: float, t1: float):
        ts = self.times.get(eid)
        if ts is None or ts.size == 0:
            return slice(0, 0), None
        lo = int(np.searchsorted(ts, t0, side="right"))   # start EXCLUDED
        hi = int(np.searchsorted(ts, t1, side="right"))   # end INCLUDED
        return slice(lo, hi), ts

    def count(self, eid, anchor: datetime, start, end, unit) -> float:
        u = UNIT_SECONDS[unit]
        a = to_epoch(anchor)
        sl, ts = self._slice(eid, a + start * u, a + end * u)
        return float(sl.stop - sl.start)

    def sum_value(self, eid, anchor, start, end, unit) -> float:
        u = UNIT_SECONDS[unit]
        a = to_epoch(anchor)
        sl, ts = self._slice(eid, a + start * u, a + end * u)
        v = self.values.get(eid)
        return float(v[sl].sum()) if v is not None else 0.0

    def item_set(self, eid, anchor, start, end, unit) -> set:
        u = UNIT_SECONDS[unit]
        a = to_epoch(anchor)
        sl, ts = self._slice(eid, a + start * u, a + end * u)
        it = self.items.get(eid)
        return set(it[sl].tolist()) if it is not None else set()


@dataclass
class Dataset:
    name: str
    schema: Schema
    engine: Engine
    entity_table: str
    entity_ids: list
    events: EntityEvents
    anchors: list[datetime]
    span_days: float                      # full data span, for context
    global_top_items: list = field(default_factory=list)  # popularity prior


def _rows_from_frame(table: TableDef, frame: pd.DataFrame,
                     parent_cols: set[str]) -> list[Row]:
    rows = []
    for rec in frame.to_dict("records"):
        ts = rec.get(table.time_column) if table.time_column else None
        if ts is not None and not pd.isna(ts):
            ts = pd.Timestamp(ts).to_pydatetime()
        else:
            ts = None
        cells = {c.name: rec[c.name] for c in table.columns
                 if rec.get(c.name) is not None and not pd.isna(rec[c.name])}
        parents = {c: rec[c] for c in parent_cols
                   if rec.get(c) is not None and not pd.isna(rec[c])}
        rows.append(Row(table.name, rec[table.primary_key], cells,
                        timestamp=ts, parents=parents))
    return rows


def _wire(schema: Schema, frames: dict[str, pd.DataFrame]) -> RetrieverWiring:
    rows = {t.name: _rows_from_frame(
                t, frames[t.name],
                {l.fk_column for l in schema.links_from(t.name)})
            for t in schema.tables}
    by_id = {t: {r.id: r for r in rs} for t, rs in rows.items()}

    def entities(table, ids, bound: TemporalBound):
        return [by_id[table][i] for i in ids
                if i in by_id[table] and bound.admits_row(by_id[table][i])]

    def children(link, parent_id, bound: TemporalBound, limit):
        m = [r for r in rows[link.from_table]
             if r.parents.get(link.fk_column) == parent_id and bound.admits_row(r)]
        m.sort(key=lambda r: r.timestamp.timestamp() if r.timestamp else float("-inf"),
               reverse=True)
        return m[:limit]

    def scanner(table, bound: TemporalBound):
        return (r for r in rows[table] if bound.admits_row(r))

    b = RetrieverWiring.new_wiring().default_links(children)
    for t in rows:
        b.entities(t, entities).scanner(t, scanner)
    return b.build()


def _events_from(df: pd.DataFrame, entity_col: str, ts_col: str,
                 value_col: Optional[str], item_col: Optional[str]) -> EntityEvents:
    ev = EntityEvents()
    df = df.sort_values(ts_col).reset_index(drop=True)
    epoch = _epoch_array(df[ts_col])
    for eid, pos in df.groupby(entity_col).groups.items():
        pos = np.asarray(pos)
        ev.times[eid] = epoch[pos]
        if value_col is not None:
            ev.values[eid] = df[value_col].to_numpy()[pos].astype(float)
        if item_col is not None:
            ev.items[eid] = df[item_col].to_numpy()[pos]
    return ev


# ---------------------------------------------------------------------------
# MovieLens
# ---------------------------------------------------------------------------
def movielens(max_users: Optional[int] = None) -> Dataset:
    base = CORPUS / "ml-latest-small"
    ratings = pd.read_csv(base / "ratings.csv")
    movies = pd.read_csv(base / "movies.csv")
    ratings["ts"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id",
                                      "rating": "rating"})
    ratings["rating_id"] = np.arange(len(ratings))
    movies = movies.rename(columns={"movieId": "movie_id", "title": "title",
                                    "genres": "genres"})

    if max_users is not None:
        keep = sorted(ratings["user_id"].unique())[:max_users]
        ratings = ratings[ratings["user_id"].isin(keep)]

    users = pd.DataFrame({"user_id": sorted(ratings["user_id"].unique())})

    schema = (Schema.new_schema()
              .table(TableDef.new_table("users").primary_key("user_id").build())
              .table(TableDef.new_table("movies")
                     .column("title", ValueType.TEXT)
                     .column("genres", ValueType.TEXT)
                     .primary_key("movie_id").build())
              .table(TableDef.new_table("ratings")
                     .column("rating", ValueType.NUMBER)
                     .column("ts", ValueType.DATETIME)
                     .primary_key("rating_id").time_column("ts").build())
              .link(LinkDef("ratings", "user_id", "users"))
              .link(LinkDef("ratings", "movie_id", "movies")).build())

    frames = {"users": users, "movies": movies[["movie_id", "title", "genres"]],
              "ratings": ratings[["rating_id", "user_id", "movie_id", "rating", "ts"]]}
    wiring = _wire(schema, frames)
    engine = Engine(schema, wiring, sampler_mode=SamplerMode.CSC, context_policy=WIDE_POLICY)

    events = _events_from(ratings, "user_id", "ts", "rating", "movie_id")
    anchors = _quantile_anchors(ratings["ts"], n=5)
    span = (ratings["ts"].max() - ratings["ts"].min()).days
    top = ratings["movie_id"].value_counts().index[:50].tolist()
    return Dataset("movielens", schema, engine, "users",
                   users["user_id"].tolist(), events, anchors, span, top)


# ---------------------------------------------------------------------------
# Online Retail II
# ---------------------------------------------------------------------------
def online_retail(max_customers: Optional[int] = 1500) -> Dataset:
    csv = CORPUS / "online_retail" / "online_retail_II.csv"
    df = pd.read_csv(csv, parse_dates=["InvoiceDate"])
    df = df.rename(columns={"Customer ID": "customer_id", "Invoice": "invoice",
                            "StockCode": "stock_code", "Quantity": "quantity",
                            "Price": "price", "InvoiceDate": "ts",
                            "Country": "country"})
    # keep genuine sales: real customer, positive qty/price, non-credit invoice
    df = df[df["customer_id"].notna() & (df["quantity"] > 0) & (df["price"] > 0)]
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    df["customer_id"] = df["customer_id"].astype("int64")
    df["amount"] = df["quantity"] * df["price"]
    df["line_id"] = np.arange(len(df))

    if max_customers is not None:
        keep = (df.groupby("customer_id")["ts"].count()
                  .sort_values(ascending=False).index[:max_customers])
        df = df[df["customer_id"].isin(set(keep))]

    customers = pd.DataFrame({"customer_id": sorted(df["customer_id"].unique())})
    schema = (Schema.new_schema()
              .table(TableDef.new_table("customers")
                     .primary_key("customer_id").build())
              .table(TableDef.new_table("purchases")
                     .column("stock_code", ValueType.TEXT)
                     .column("amount", ValueType.NUMBER)
                     .column("quantity", ValueType.NUMBER)
                     .column("ts", ValueType.DATETIME)
                     .primary_key("line_id").time_column("ts").build())
              .link(LinkDef("purchases", "customer_id", "customers")).build())
    frames = {"customers": customers,
              "purchases": df[["line_id", "customer_id", "stock_code",
                               "amount", "quantity", "ts"]]}
    wiring = _wire(schema, frames)
    engine = Engine(schema, wiring, sampler_mode=SamplerMode.CSC, context_policy=WIDE_POLICY)

    events = _events_from(df, "customer_id", "ts", "amount", "stock_code")
    anchors = _quantile_anchors(df["ts"], n=4)
    span = (df["ts"].max() - df["ts"].min()).days
    top = df["stock_code"].value_counts().index[:50].tolist()
    return Dataset("online_retail", schema, engine, "customers",
                   customers["customer_id"].tolist(), events, anchors, span, top)


# ---------------------------------------------------------------------------
# Brightkite check-ins (SNAP) — mobility domain; locations recur
# ---------------------------------------------------------------------------
def brightkite(max_users: Optional[int] = 1500) -> Dataset:
    gz = CORPUS / "brightkite.txt.gz"
    df = pd.read_csv(gz, sep="\t", compression="gzip", header=None,
                     names=["user_id", "ts", "lat", "lon", "location_id"],
                     dtype={"location_id": str})
    df = df[df["location_id"].notna() & (df["location_id"] != "")]
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True).dt.tz_localize(None)
    df = df[df["ts"].notna()]
    df["checkin_id"] = np.arange(len(df))

    if max_users is not None:
        keep = (df.groupby("user_id")["ts"].count()
                  .sort_values(ascending=False).index[:max_users])
        df = df[df["user_id"].isin(set(keep))]

    users = pd.DataFrame({"user_id": sorted(df["user_id"].unique())})
    schema = (Schema.new_schema()
              .table(TableDef.new_table("users").primary_key("user_id").build())
              .table(TableDef.new_table("checkins")
                     .column("location_id", ValueType.TEXT)
                     .column("ts", ValueType.DATETIME)
                     .primary_key("checkin_id").time_column("ts").build())
              .link(LinkDef("checkins", "user_id", "users")).build())
    frames = {"users": users,
              "checkins": df[["checkin_id", "user_id", "location_id", "ts"]]}
    wiring = _wire(schema, frames)
    engine = Engine(schema, wiring, sampler_mode=SamplerMode.CSC, context_policy=WIDE_POLICY)

    events = _events_from(df, "user_id", "ts", None, "location_id")
    anchors = _quantile_anchors(df["ts"], n=4)
    span = (df["ts"].max() - df["ts"].min()).days
    top = df["location_id"].value_counts().index[:50].tolist()
    return Dataset("brightkite", schema, engine, "users",
                   users["user_id"].tolist(), events, anchors, span, top)


def _quantile_anchors(ts: pd.Series, n: int) -> list[datetime]:
    """Anchors spread across the active middle of the timeline, each leaving
    room for a forward horizon before the data ends."""
    lo, hi = ts.quantile(0.35), ts.quantile(0.80)
    return [pd.Timestamp(q).to_pydatetime()
            for q in pd.date_range(lo, hi, periods=n)]
