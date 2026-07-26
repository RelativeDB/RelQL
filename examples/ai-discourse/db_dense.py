"""The same snapshot, reshaped into the schema RT-J actually wants.

``db.py`` hands the model 23,005 individual engagement rows, each carrying a
timestamp and a kind. Measured on the sibling velocity task, a schema of that
shape spends ~69% of its cell budget reading single-event rows and leaves six
accounts visible — the model never reaches the columns that carry signal.

This module models the same facts as **seven tables**, closer to the shape of
``rel-f1`` (drivers / results / races / standings), where every row carries
several meaningful columns and the fan-out per entity is modest:

    accounts ──< posts ──< post_windows          (per-post activity, bucketed)
        │          │
        │          └──< engagements              (kept: the target needs it)
        │
        ├──< author_days                         (per-author daily form)
        └──< follows >── accounts

The two new tables are the point:

``post_windows`` — one row per post per time bucket, holding that bucket's
engagement split by whether it came from an existing follower. Hourly for the
first day of a post's life, daily afterwards, and **empty buckets are not
emitted**, so ~23k events become ~6k rows carrying eight columns each.

``author_days`` — one row per author per active day: how much they posted and
how it did. This is what lets the model normalise a post against its author
without walking to every one of that author's individual engagements.

Both are stamped at the **end** of their window, so the temporal bound admits a
row only once the whole window has elapsed. Stamping at window start would leak
up to a bucket's worth of future into any anchor falling inside it.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from relativedb import (LinkDef, RetrieverWiring, Row, Schema, TableDef,
                        TemporalBound, ValueType)

import db                                   # iso(), load(), SNAPSHOT

ENGAGEMENT_FANOUT = 6                       # raw rows per post reaching context


def _buckets(created: datetime, until: datetime):
    """Hourly for the first 24h, then daily. Yields (start, end)."""
    edge = created
    fine_end = created + timedelta(hours=24)
    while edge < until:
        step = timedelta(hours=1) if edge < fine_end else timedelta(days=1)
        yield edge, edge + step
        edge += step


def build(snapshot: dict, *, with_follows: bool = True, with_flag: bool = True,
          with_windows: bool = True, engagement_fanout: int = ENGAGEMENT_FANOUT,
          fk_features: bool = False, follow_columns: bool = False,
          with_actor_link: bool = True):
    """Return (schema, wiring, rows).

    ``with_windows=False`` reproduces the old sparse-event shape, so the two
    schemas can be compared with the identical query and labels.

    ``fk_features`` sets ``LinkDef.feature_type``, which emits the raw FK value
    as a feature token. Without it a link is *graph structure only* — which is
    why ``follows`` shows up in EXPLAIN CONTEXT with hundreds of rows and
    **zero cells**. Note that ``feature_type=None`` is the reference behaviour
    RT-J was pretrained with, and the values here are opaque DIDs, so this is
    worth measuring rather than assuming.

    ``follow_columns`` is the better-behaved version of the same idea: give the
    edge real attributes (reciprocity, endpoint sizes) so it carries meaning
    without leaning on high-cardinality identifier text.

    ``with_actor_link=False`` drops ``engagements.actor_id -> accounts``. That
    link is what drags hundreds of near-empty engager rows into the context;
    with ``from_follower`` and ``post_windows`` present, the information it
    carried is largely already there.
    """
    members = {m["did"]: m for m in snapshot["members"]}
    flags = snapshot.get("follower_flags", {})
    author_of = {p["post_id"]: p["author_did"] for p in snapshot["posts"]}
    created_of = {p["post_id"]: db.iso(p["created_at"])
                  for p in snapshot["posts"]}

    # ---- tables -------------------------------------------------------
    engagement = (TableDef.new_table("engagements")
                  .column("at", ValueType.DATETIME)
                  .column("kind", ValueType.TEXT))
    if with_flag:
        engagement = engagement.column("from_follower", ValueType.BOOLEAN)

    tables = [
        TableDef.new_table("accounts")
        .column("handle", ValueType.TEXT)
        .column("display_name", ValueType.TEXT)
        .column("description", ValueType.TEXT)
        .column("followers", ValueType.NUMBER)
        .column("following", ValueType.NUMBER)
        .column("posts_total", ValueType.NUMBER)
        .column("in_community", ValueType.BOOLEAN)
        .primary_key("account_id").build(),

        TableDef.new_table("posts")
        .column("created_at", ValueType.DATETIME)
        .column("text", ValueType.TEXT)
        .column("lang", ValueType.TEXT)
        .column("text_length", ValueType.NUMBER)
        .column("has_link", ValueType.BOOLEAN)
        .column("has_image", ValueType.BOOLEAN)
        .column("is_reply", ValueType.BOOLEAN)
        .primary_key("post_id").time_column("created_at").build(),

        engagement.primary_key("engagement_id").time_column("at").build(),
    ]
    fk = ValueType.TEXT if fk_features else None
    links = [LinkDef("posts", "author_id", "accounts", fk),
             LinkDef("engagements", "post_id", "posts", fk)]
    if with_actor_link:
        links.append(LinkDef("engagements", "actor_id", "accounts", fk))

    if with_windows:
        tables += [
            TableDef.new_table("post_windows")
            .column("window_end", ValueType.DATETIME)
            .column("age_hours", ValueType.NUMBER)
            .column("events", ValueType.NUMBER)
            .column("actors", ValueType.NUMBER)
            .column("from_followers", ValueType.NUMBER)
            .column("from_outside", ValueType.NUMBER)
            .column("replies", ValueType.NUMBER)
            .column("cumulative_events", ValueType.NUMBER)
            .column("cumulative_outside", ValueType.NUMBER)
            .primary_key("window_id").time_column("window_end").build(),

            TableDef.new_table("author_days")
            .column("day_end", ValueType.DATETIME)
            .column("posts_made", ValueType.NUMBER)
            .column("events", ValueType.NUMBER)
            .column("actors", ValueType.NUMBER)
            .column("from_outside", ValueType.NUMBER)
            .column("events_per_post", ValueType.NUMBER)
            .primary_key("author_day_id").time_column("day_end").build(),
        ]
        links += [LinkDef("post_windows", "post_id", "posts", fk),
                  LinkDef("author_days", "author_id", "accounts", fk)]

    if with_follows:
        follows_tbl = TableDef.new_table("follows")
        if follow_columns:
            # A follow edge has no natural attributes, so it contributes zero
            # cells and is invisible to the model as anything but connectivity.
            # These give it content that is meaningful rather than opaque.
            follows_tbl = (follows_tbl
                           .column("reciprocal", ValueType.BOOLEAN)
                           .column("src_followers", ValueType.NUMBER)
                           .column("dst_followers", ValueType.NUMBER)
                           .column("both_in_community", ValueType.BOOLEAN))
        tables.append(follows_tbl.primary_key("follow_id").build())
        links += [LinkDef("follows", "src", "accounts", fk),
                  LinkDef("follows", "dst", "accounts", fk)]

    rows: dict[str, list[Row]] = {t.name: [] for t in tables}

    # ---- accounts -----------------------------------------------------
    seen: set[str] = set()
    for did, m in members.items():
        seen.add(did)
        rows["accounts"].append(Row("accounts", did, {
            "handle": m.get("handle"), "display_name": m.get("displayName"),
            "description": m.get("description"),
            "followers": m.get("followers"), "following": m.get("following"),
            "posts_total": m.get("posts_total"), "in_community": True}))
    for a in snapshot.get("actor_profiles", []):
        if a["did"] in seen:
            continue
        seen.add(a["did"])
        rows["accounts"].append(Row("accounts", a["did"], {
            "handle": a.get("handle"), "display_name": a.get("displayName"),
            "description": a.get("description"), "followers": None,
            "following": None, "posts_total": None, "in_community": False}))

    # ---- posts --------------------------------------------------------
    for p in snapshot["posts"]:
        when = created_of[p["post_id"]]
        rows["posts"].append(Row("posts", p["post_id"], {
            "created_at": when, "text": p.get("text") or "",
            "lang": p.get("lang"),
            "text_length": len(p.get("text") or ""),
            "has_link": bool(p.get("has_link")),
            "has_image": bool(p.get("has_image")),
            "is_reply": bool(p.get("reply_to")),
        }, when, {"author_id": p["author_did"]}))

    # ---- engagements (raw, still needed to express the target) --------
    by_post: dict[str, list[dict]] = defaultdict(list)
    for index, e in enumerate(snapshot["engagements"]):
        if not e.get("at"):
            continue                          # undated => would leak, see db.py
        when = db.iso(e.get("at"))
        if when is None:
            continue
        actor = e["actor_did"]
        if actor not in seen:
            seen.add(actor)
            rows["accounts"].append(Row("accounts", actor, {
                "handle": None, "display_name": None, "description": None,
                "followers": None, "following": None, "posts_total": None,
                "in_community": False}))
        author = author_of.get(e["post_id"])
        follower = flags.get(f"{author}|{actor}")
        cells = {"at": when, "kind": e["kind"]}
        if with_flag:
            cells["from_follower"] = follower
        rows["engagements"].append(Row(
            "engagements", f"e{index}", cells, when,
            {"post_id": e["post_id"], "actor_id": actor}))
        by_post[e["post_id"]].append(
            {"at": when, "kind": e["kind"], "actor": actor,
             "follower": bool(follower)})

    # ---- post_windows + author_days -----------------------------------
    if with_windows:
        horizon = max((e["at"] for es in by_post.values() for e in es),
                      default=None)
        widx = 0
        author_day: dict[tuple[str, datetime], dict] = defaultdict(
            lambda: {"posts": 0, "events": 0, "actors": set(),
                     "outside": 0})
        for post_id, events in by_post.items():
            created = created_of.get(post_id)
            if created is None or horizon is None:
                continue
            events.sort(key=lambda e: e["at"])
            last = events[-1]["at"]
            cum = cum_out = 0
            for start, end in _buckets(created, last + timedelta(seconds=1)):
                inside = [e for e in events if start <= e["at"] < end]
                if not inside:
                    continue                  # never emit empty buckets
                actors = {e["actor"] for e in inside}
                outside = sum(1 for e in inside if not e["follower"])
                cum += len(inside)
                cum_out += outside
                rows["post_windows"].append(Row(
                    "post_windows", f"w{widx}", {
                        "window_end": end,
                        "age_hours": round(
                            (end - created).total_seconds() / 3600, 2),
                        "events": len(inside), "actors": len(actors),
                        "from_followers": len(inside) - outside,
                        "from_outside": outside,
                        "replies": sum(1 for e in inside
                                       if e["kind"] == "reply"),
                        "cumulative_events": cum,
                        "cumulative_outside": cum_out,
                    }, end, {"post_id": post_id}))
                widx += 1

            author = author_of.get(post_id)
            for e in events:
                day = e["at"].replace(hour=0, minute=0, second=0,
                                      microsecond=0) + timedelta(days=1)
                bucket = author_day[(author, day)]
                bucket["events"] += 1
                bucket["actors"].add(e["actor"])
                bucket["outside"] += 0 if e["follower"] else 1

        for p in snapshot["posts"]:
            when = created_of[p["post_id"]]
            if when is None:
                continue
            day = when.replace(hour=0, minute=0, second=0,
                               microsecond=0) + timedelta(days=1)
            author_day[(p["author_did"], day)]["posts"] += 1

        for index, ((author, day), agg) in enumerate(sorted(
                author_day.items(), key=lambda kv: kv[0][1])):
            if author is None:
                continue
            rows["author_days"].append(Row("author_days", f"ad{index}", {
                "day_end": day, "posts_made": agg["posts"],
                "events": agg["events"], "actors": len(agg["actors"]),
                "from_outside": agg["outside"],
                "events_per_post": round(
                    agg["events"] / agg["posts"], 2) if agg["posts"] else 0.0,
            }, day, {"author_id": author}))

    # ---- follows ------------------------------------------------------
    if with_follows:
        edges = snapshot.get("follow_edges", [])
        pairs = {(e["src"], e["dst"]) for e in edges} if follow_columns else set()
        for index, edge in enumerate(edges):
            cells = {}
            if follow_columns:
                src, dst = edge["src"], edge["dst"]
                cells = {
                    "reciprocal": (dst, src) in pairs,
                    "src_followers": (members.get(src) or {}).get("followers"),
                    "dst_followers": (members.get(dst) or {}).get("followers"),
                    "both_in_community": src in members and dst in members,
                }
            rows["follows"].append(Row("follows", f"f{index}", cells, None,
                                       {"src": edge["src"],
                                        "dst": edge["dst"]}))

    schema = Schema(tuple(tables), tuple(links))
    return schema, _wire(links, rows, engagement_fanout), rows


def _wire(links, rows, engagement_fanout: int):
    by_id = {t: {r.id: r for r in rs} for t, rs in rows.items()}
    children: dict = {}
    for link in links:
        index: dict = defaultdict(list)
        for row in rows[link.from_table]:
            parent = row.parents.get(link.fk_column)
            if parent is not None:
                index[parent].append(row)
        for bucket in index.values():
            bucket.sort(key=lambda r: (r.timestamp is None,
                                       -(r.timestamp.timestamp()
                                         if r.timestamp else 0.0)))
        children[(link.from_table, link.fk_column)] = index

    def entities(table, ids, bound: TemporalBound):
        return [r for i in ids if (r := by_id[table].get(i)) is not None
                and bound.admits_row(r)]

    def link_rows(link, parent_id, bound: TemporalBound, limit: int):
        found = [r for r in children[(link.from_table, link.fk_column)]
                 .get(parent_id, ()) if bound.admits_row(r)]
        # The raw event table is deliberately throttled: the summaries carry
        # the same information at a fraction of the cell cost, and letting
        # thousands of single-event rows in is what starved the old schema.
        if engagement_fanout and link.from_table == "engagements":
            found = found[:engagement_fanout]
        return found[:limit]

    def scanner(table, bound: TemporalBound):
        return (r for r in rows[table] if bound.admits_row(r))

    wiring = RetrieverWiring.new_wiring().default_links(link_rows)
    for table in rows:
        wiring.entities(table, entities)
        wiring.scanner(table, scanner)
    return wiring.build()
