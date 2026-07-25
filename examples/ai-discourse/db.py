"""The snapshot, presented to RelativeDB as a five-table relational database.

    accounts ──< posts ──< engagements
        │                       │
        │                       └── (actor_id) ──> accounts
        │
        └──< follows >── accounts

``accounts`` deliberately holds *both* sides of every interaction: the people
who post and the people who engage. That is what makes the interesting join
expressible at all — from a post you can reach its author, from the author you
can reach everyone who follows them, and from the post you can reach everyone
who engaged with it. Whether those two sets overlap is the entire question,
and RT-J can walk it without being told to.

There is one engineered column, ``engagements.from_follower``: was this actor
already following this author? It is derived from the follow graph rather than
from the post, it is only ever set on rows the temporal bound admits, and the
same test applied to the held-out window is the label. Everything else is raw:
no rolling means, no sentiment scores, no TF-IDF between post and profile. The
post text goes in as text, because the model is good at text.

Time is the whole game. Posts carry their creation time, engagements carry the
moment they happened, and the engine's temporal bound cuts the database at the
anchor — so a prediction can never see the very engagement it is predicting.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from relativedb import (LinkDef, RetrieverWiring, Row, Schema, TableDef,
                        TemporalBound, ValueType)

SNAPSHOT = Path(__file__).parent / "data" / "snapshot.json"

GRAPH_TABLES = ("follows",)


def load(path: Path = SNAPSHOT) -> dict:
    if not path.exists():
        raise SystemExit(f"no snapshot at {path} — run `python fetch.py` first")
    return json.loads(path.read_text())


def iso(value):
    if not value:
        return None
    try:
        when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def build(snapshot: dict, *, with_follows: bool = True,
          with_flag: bool = True):
    """Return (schema, wiring, rows).

    There are two separate graph ingredients, and they ablate separately:

    * ``with_follows`` — the raw ``follows`` table, the edges RT-J can walk.
    * ``with_flag`` — the derived ``engagements.from_follower`` column.

    Dropping ``follows`` leaves the query text untouched, so that arm is a true
    ablation: same question, same labels, smaller database. Dropping the flag
    is *not*, because the query names that column — a query that cannot be
    written is a different question, not a smaller database. ``predict.py``
    keeps those two cases clearly apart.

    RelQL has an ``ABLATE TABLE`` clause for exactly this; the engine does not
    implement it yet, so the ablation builds a smaller database instead.
    """
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
        .column("has_link", ValueType.BOOLEAN)
        .column("has_image", ValueType.BOOLEAN)
        .column("is_reply", ValueType.BOOLEAN)
        .primary_key("post_id").time_column("created_at").build(),

        engagement.primary_key("engagement_id").time_column("at").build(),
    ]
    links = [
        LinkDef("posts", "author_id", "accounts"),
        LinkDef("engagements", "post_id", "posts"),
        LinkDef("engagements", "actor_id", "accounts"),
    ]
    if with_follows:
        tables.append(TableDef.new_table("follows")
                      .primary_key("follow_id").build())
        links += [LinkDef("follows", "src", "accounts"),
                  LinkDef("follows", "dst", "accounts")]

    rows: dict[str, list[Row]] = {t.name: [] for t in tables}

    # -- accounts ---------------------------------------------------------
    # Community members come with full profiles. Engagers from outside the
    # community are known only by what the like/repost payload carried, so
    # their numeric columns stay NULL rather than being imputed.
    members = {m["did"]: m for m in snapshot["members"]}
    seen: set[str] = set()
    for did, member in members.items():
        seen.add(did)
        rows["accounts"].append(Row("accounts", did, {
            "handle": member.get("handle"),
            "display_name": member.get("displayName"),
            "description": member.get("description"),
            "followers": member.get("followers"),
            "following": member.get("following"),
            "posts_total": member.get("posts_total"),
            "in_community": True,
        }))
    for actor in snapshot.get("actor_profiles", []):
        if actor["did"] in seen:
            continue
        seen.add(actor["did"])
        rows["accounts"].append(Row("accounts", actor["did"], {
            "handle": actor.get("handle"),
            "display_name": actor.get("displayName"),
            "description": actor.get("description"),
            "followers": None, "following": None, "posts_total": None,
            "in_community": False,
        }))

    # -- posts ------------------------------------------------------------
    author_of: dict[str, str] = {}
    for post in snapshot["posts"]:
        when = iso(post["created_at"])
        author_of[post["post_id"]] = post["author_did"]
        rows["posts"].append(Row("posts", post["post_id"], {
            "created_at": when,
            "text": post.get("text") or "",
            "lang": post.get("lang"),
            "has_link": bool(post.get("has_link")),
            "has_image": bool(post.get("has_image")),
            "is_reply": bool(post.get("reply_to")),
        }, when, {"author_id": post["author_did"]}))

    # -- engagements ------------------------------------------------------
    # An undated row is *always* admitted by the temporal bound — that is what
    # "static table" means to the engine. So an engagement with no timestamp
    # would be visible at every anchor, including the ones that happened after
    # it, which is precisely the leak this whole example is built to avoid.
    # ``getRepostedBy`` returns no per-repost time, so reposts are dropped
    # rather than silently treated as static. Likes, quotes and replies all
    # carry their own timestamp and survive.
    flags = snapshot.get("follower_flags", {})
    for index, edge in enumerate(snapshot["engagements"]):
        if not edge.get("at"):
            continue
        actor = edge["actor_did"]
        if actor not in seen:                       # an engager with no profile
            seen.add(actor)
            rows["accounts"].append(Row("accounts", actor, {
                "handle": None, "display_name": None, "description": None,
                "followers": None, "following": None, "posts_total": None,
                "in_community": False}))
        when = iso(edge.get("at"))
        if when is None:                            # unparseable timestamp
            continue
        cells = {"at": when, "kind": edge["kind"]}
        if with_flag:
            author = author_of.get(edge["post_id"])
            cells["from_follower"] = flags.get(f"{author}|{actor}")
        rows["engagements"].append(Row(
            "engagements", f"e{index}", cells, when,
            {"post_id": edge["post_id"], "actor_id": actor}))

    # -- the follow graph -------------------------------------------------
    # Undated: the API does not say when the follow happened. They are admitted
    # at any anchor, which is the same assumption the label makes.
    if with_follows:
        for index, edge in enumerate(snapshot.get("follow_edges", [])):
            rows["follows"].append(Row("follows", f"f{index}", {}, None,
                                       {"src": edge["src"], "dst": edge["dst"]}))

    schema = Schema(tuple(tables), tuple(links))
    return schema, wire(schema, links, rows), rows


def wire(schema: Schema, links, rows: dict[str, list[Row]]):
    """Index the rows and hand the engine callbacks that respect the bound."""
    by_id = {name: {row.id: row for row in table_rows}
             for name, table_rows in rows.items()}
    children: dict[tuple[str, str], dict] = {}
    for link in links:
        index: dict = {}
        for row in rows[link.from_table]:
            parent = row.parents.get(link.fk_column)
            if parent is not None:
                index.setdefault(parent, []).append(row)
        for bucket in index.values():                         # newest first
            bucket.sort(key=lambda r: (r.timestamp is None,
                                       -(r.timestamp.timestamp()
                                         if r.timestamp else 0.0)))
        children[(link.from_table, link.fk_column)] = index

    def entities(table, ids, bound: TemporalBound):
        return [row for i in ids
                if (row := by_id[table].get(i)) is not None
                and bound.admits_row(row)]

    def link_rows(link, parent_id, bound: TemporalBound, limit: int):
        found = [row for row in children[(link.from_table, link.fk_column)]
                 .get(parent_id, ()) if bound.admits_row(row)]
        return found[:limit]

    def scanner(table, bound: TemporalBound):
        return (row for row in rows[table] if bound.admits_row(row))

    wiring = RetrieverWiring.new_wiring().default_links(link_rows)
    for table in rows:
        wiring.entities(table, entities)
        wiring.scanner(table, scanner)
    return wiring.build()
