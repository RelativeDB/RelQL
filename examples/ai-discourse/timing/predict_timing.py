"""The counterfactual: what if this same post had gone out at a different hour?

``analyze_timing.py`` can only ever be correlational. When it says posts made
at 14:00-17:00 do 12% worse, that is consistent with two very different worlds:

1. posting into the afternoon rush genuinely costs you reach, or
2. the posts people write at 15:00 on a Tuesday are just weaker posts.

A residual against the author's own mean does not separate those, because it
controls for *who* posted, not for *what* they posted.

RelQL has the clause that does:

    PREDICT posts.engagement
    FROM posts
    WHERE posts.post_id IN :ids
    ASSUMING posts.hour_bucket = '20-23'
    RETURN EXPECTED VALUE

``ASSUMING`` holds the post — its text, its author, its links, its length —
completely fixed and intervenes on one column. Sweeping the intervention across
every hour bucket and re-scoring the *same* posts gives a within-post curve, so
composition cannot drive it. The remaining caveat is honest and unavoidable: it
is the model's belief about the counterfactual, and it is only as good as the
model.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from relativedb import (ContextPolicy, Engine, ExecutionInput, LinkDef,
                        RetrieverWiring, Row, RtNativeBackend, Schema,
                        TableDef, TemporalBound, ValueType)

sys.path.insert(0, str(Path(__file__).parent))
from tech import is_tech                                       # noqa: E402

DATA = Path(__file__).parent / "data" / "history.json"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BUCKETS = ["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"]


def bucket_of(hour: int) -> str:
    return BUCKETS[hour // 4]


def post_table(name):
    return (TableDef.new_table(name)
            .column("created_at", ValueType.DATETIME)
            .column("text", ValueType.TEXT)
            .column("hour_bucket", ValueType.TEXT)
            .column("day_of_week", ValueType.TEXT)
            .column("is_weekend", ValueType.BOOLEAN)
            .column("is_reply", ValueType.BOOLEAN)
            .column("has_link", ValueType.BOOLEAN)
            .column("has_image", ValueType.BOOLEAN)
            .column("text_length", ValueType.NUMBER)
            .column("is_tech", ValueType.BOOLEAN)
            .column("engagement", ValueType.NUMBER)
            .primary_key(f"{name[:-1]}_id").time_column("created_at").build())


def build(posts, members, zone, focal_ids=frozenset()):
    """Three tables, and the third one exists for a specific reason.

    ``Engine._apply_assumptions`` writes an assumed value onto *every* context
    row of the named table: ``ASSUMING posts.text_length = 300`` sets it on the
    post being scored **and** on every sibling post in the context. The column
    becomes a constant, all of its discriminative content is destroyed, and
    every intervention level returns an identical prediction. That is the right
    semantics for ``ASSUMING customer.plan = 'premium'``, where the context
    holds one customer row; it is the wrong shape for intervening on one row
    among thousands of the same table.

    So the rows being intervened on live in ``focal_posts`` and the rows
    serving as evidence live in ``posts``. ``ASSUMING focal_posts.hour_bucket``
    then reaches exactly one row per context, and the evidence keeps its
    variation. ``--no-split`` reproduces the broken version for comparison.
    """
    tables = [
        TableDef.new_table("accounts")
        .column("handle", ValueType.TEXT)
        .column("display_name", ValueType.TEXT)
        .column("description", ValueType.TEXT)
        .column("followers", ValueType.NUMBER)
        .column("posts_total", ValueType.NUMBER)
        .primary_key("account_id").build(),
        post_table("posts"),
    ]
    links = [LinkDef("posts", "author_id", "accounts")]
    rows = {"accounts": [], "posts": []}
    if focal_ids:
        tables.append(post_table("focal_posts"))
        links.append(LinkDef("focal_posts", "author_id", "accounts"))
        rows["focal_posts"] = []

    for did, member in members.items():
        rows["accounts"].append(Row("accounts", did, {
            "handle": member.get("handle"),
            "display_name": member.get("displayName"),
            "description": member.get("description"),
            "followers": member.get("followers"),
            "posts_total": member.get("posts_total")}))

    for post in posts:
        when = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
        local = when.astimezone(zone)
        table = ("focal_posts" if post["post_id"] in focal_ids else "posts")
        rows[table].append(Row(table, post["post_id"], {
            "created_at": when,
            "text": post.get("text") or "",
            "hour_bucket": bucket_of(local.hour),
            "day_of_week": DAYS[local.weekday()],
            "is_weekend": local.weekday() >= 5,
            "is_reply": bool(post["is_reply"]),
            "has_link": bool(post["has_link"]),
            "has_image": bool(post["has_image"]),
            "text_length": post["text_length"],
            "is_tech": is_tech(post.get("text")),
            "engagement": (post["likes"] + post["reposts"]
                           + post["replies"] + post["quotes"]),
        }, when, {"author_id": post["author_did"]}))

    schema = Schema(tuple(tables), tuple(links))
    by_id = {t: {r.id: r for r in rs} for t, rs in rows.items()}
    children = {}
    for link in links:
        index = {}
        for row in rows[link.from_table]:
            parent = row.parents.get(link.fk_column)
            if parent is not None:
                index.setdefault(parent, []).append(row)
        for bucket in index.values():
            bucket.sort(key=lambda r: (r.timestamp is None,
                                       -(r.timestamp.timestamp()
                                         if r.timestamp else 0.0)))
        children[(link.from_table, link.fk_column)] = index

    def entities(table, ids, bound: TemporalBound):
        return [r for i in ids if (r := by_id[table].get(i)) is not None
                and bound.admits_row(r)]

    def link_rows(link, parent_id, bound: TemporalBound, limit: int):
        return [r for r in children[(link.from_table, link.fk_column)]
                .get(parent_id, ()) if bound.admits_row(r)][:limit]

    def scanner(table, bound: TemporalBound):
        return (r for r in rows[table] if bound.admits_row(r))

    wiring = RetrieverWiring.new_wiring().default_links(link_rows)
    for table in rows:
        wiring.entities(table, entities)
        wiring.scanner(table, scanner)
    return schema, wiring.build(), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--n", type=int, default=60, help="posts to intervene on")
    ap.add_argument("--pool", type=int, default=4000,
                    help="posts loaded into the database as context")
    ap.add_argument("--min-age-days", type=int, default=14)
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--context-cells", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--sweep", default="hour",
                    choices=["hour", "day", "image", "length"],
                    help="'image'/'length' are positive controls: columns the "
                         "descriptive analysis says really do move engagement. "
                         "If ASSUMING cannot move those either, then a flat "
                         "timing curve says nothing about timing.")
    ap.add_argument("--tech-only", action="store_true", default=True)
    ap.add_argument("--no-split", action="store_true",
                    help="put the intervened rows back in `posts` — reproduces "
                         "the collapsed, uninformative counterfactual")
    args = ap.parse_args()

    blob = json.loads(Path(args.data).read_text())
    zone = ZoneInfo(args.tz)
    fetched = datetime.fromisoformat(blob["fetched_at"])
    cutoff = fetched - timedelta(days=args.min_age_days)

    posts = []
    for post in blob["posts"]:
        when = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
        if when > cutoff or post["is_reply"]:
            continue
        if args.tech_only and not is_tech(post.get("text")):
            continue
        posts.append(post)
    posts.sort(key=lambda p: p["created_at"])
    posts = posts[-args.pool:]
    members = {m["did"]: m for m in blob["members"]}

    step = max(1, len(posts) // args.n)
    targets = posts[::step][:args.n]
    print(f"{len(posts)} posts in the database, intervening on {len(targets)}")
    print(f"tz={args.tz}  ctx={args.context_cells}\n")

    focal = frozenset() if args.no_split else frozenset(
        p["post_id"] for p in targets)
    table = "posts" if args.no_split else "focal_posts"
    key = "post_id" if args.no_split else "focal_post_id"
    schema, wiring, _ = build(posts, members, zone, focal_ids=focal)
    engine = Engine(
        schema, wiring,
        model_backend=RtNativeBackend(schema=schema, wiring=wiring,
                                      max_seq_len=args.context_cells,
                                      batch_size=args.batch_size),
        # Reference RT-J evaluation settings.
        context_policy=ContextPolicy(max_context_cells=args.context_cells,
                                     local_context_cells=256, bfs_width=32,
                                     num_walks=10_000, walk_length=20, seed=0))

    ids = [p["post_id"] for p in targets]
    levels, column = {
        "hour": (BUCKETS, "hour_bucket"),
        "day": (DAYS, "day_of_week"),
        "image": (["TRUE", "FALSE"], "has_image"),
        "length": (["20", "120", "300"], "text_length"),
    }[args.sweep]
    anchor = fetched

    baseline_query = (f"PREDICT {table}.engagement FROM {table} "
                      f"WHERE {table}.{key} IN :ids RETURN EXPECTED VALUE")
    print(f"  {baseline_query}\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = engine.execute(ExecutionInput(
            query=baseline_query, anchor_time=anchor, params={"ids": ids}))
    base_scores = {p.id: float(p.value) for p in base.predictions}

    print(f"{'intervention':>14} {'mean pred':>10} {'median':>9} "
          f"{'vs baseline':>12}   (same posts, only the column changed)")
    bl_mean = statistics.mean(base_scores.values())
    bl_med = statistics.median(base_scores.values())
    print(f"{'(as posted)':>14} {bl_mean:>10.1f} {bl_med:>9.1f} "
          f"{'--':>12}")

    curve = {}
    for level in levels:
        # Booleans and numbers must not be quoted, or the parser sees a string.
        literal = level if args.sweep in ("image", "length") else f"'{level}'"
        query = (f"PREDICT {table}.engagement FROM {table} "
                 f"WHERE {table}.{key} IN :ids "
                 f"ASSUMING {table}.{column} = {literal} RETURN EXPECTED VALUE")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engine.execute(ExecutionInput(
                query=query, anchor_time=anchor, params={"ids": ids}))
        scores = {p.id: float(p.value) for p in result.predictions}
        curve[level] = scores
        mean = statistics.mean(scores.values())
        med = statistics.median(scores.values())
        # Paired: every post is its own control, which is the whole point.
        deltas = [scores[i] - base_scores[i] for i in ids if i in scores]
        pct = statistics.mean(deltas) / max(1e-9, bl_mean)
        print(f"{level:>14} {mean:>10.1f} {med:>9.1f} {pct:>+11.1%}",
              flush=True)

    spread = max(statistics.mean(s.values()) for s in curve.values()) - \
        min(statistics.mean(s.values()) for s in curve.values())
    print(f"\nspread across {args.sweep} interventions: {spread:.1f} "
          f"engagements ({spread / max(1e-9, bl_mean):+.1%} of baseline)")
    print("If that spread is small next to the observed differences in "
          "analyze_timing.py,\nthe observed differences were composition, not "
          "timing.")


if __name__ == "__main__":
    main()
