"""Better graph links, so the text arrives somewhere useful.

The first version of this experiment linked a post to a market when their
wording shared two distinctive words, then let a random walk pull ~400 posts
per market into a 1,024-cell context. Muting the text cost nothing — which is
what you would expect if the posts reaching the model were only loosely about
the market. That is a graph problem, not a language problem.

Four changes, each addressing a different failure:

  1. **Post → market by meaning, not tokens.** MiniLM cosine against the
     *market question* (specific: "Bitcoin above $90,000 on Dec 30") instead of
     the event title (vague: "What price will Bitcoin hit?"). The similarity
     rides on the link as a feature, so the model can tell a 0.75 match from a
     0.45 one.

  2. **A subject bridge.** `subjects` are the distinctive words markets and
     posts share ("hormuz", "khamenei", "shutdown"). A post reaches a market
     through the subject it is about, and the model can see *which* subject
     connects them — and reach other posts on that subject two hops away.

  3. **Hourly rollups.** `chatter_hours` is one row per market-hour: how many
     posts, from how many authors, how well matched, how upvoted. A burst is
     then one cell the walk always finds, instead of a pattern it has to
     reconstruct from posts it may never sample.

  4. **Threads.** A Hacker News comment carries its parent; grouping posts
     under a thread lets the walk move between sibling sentences in one
     argument rather than landing on isolated fragments.

Everything stays temporally bound: rollups are stamped at the *end* of their
hour, links inherit the post's timestamp, and a post's own label is masked
until its window closes.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

from relativedb import (LinkDef, RetrieverWiring, Row, Schema, TableDef,
                        TemporalBound, ValueType)

from scale.build import tokens
from scale.semantic import embed


def hour_floor(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
def semantic_links(markets: list[dict], posts: list[dict], *, top_k: int,
                   floor: float, per_post: int = 2):
    """(post, market, similarity) for the best matches in both directions.

    Top-k per market keeps a busy market from starving; top-`per_post` per post
    keeps one loud post from being wired to fifty markets."""
    questions = [m["question"] for m in markets]
    bodies = [p["body"] for p in posts]
    market_vectors = embed(questions, tag="chatter_markets")
    post_vectors = embed(bodies, tag="chatter_posts")

    best: dict[tuple[int, int], float] = {}
    per_market_kept: dict[int, list] = {}
    for start in range(0, len(markets), 64):
        block = market_vectors[start:start + 64]
        sims = block @ post_vectors.T
        for row in range(block.shape[0]):
            scores = sims[row]
            k = min(top_k, len(scores) - 1)
            top = np.argpartition(-scores, k)[:k] if k > 0 else []
            per_market_kept[start + row] = [
                (int(j), float(scores[j])) for j in top if scores[j] >= floor]

    # a post keeps only its strongest few market attachments
    by_post: dict[int, list] = defaultdict(list)
    for market_i, hits in per_market_kept.items():
        for post_i, score in hits:
            by_post[post_i].append((market_i, score))
    for post_i, hits in by_post.items():
        for market_i, score in sorted(hits, key=lambda h: -h[1])[:per_post]:
            best[(post_i, market_i)] = score
    return best


def attach(markets: list[dict], posts: list[dict], *, top_k: int = 60,
           floor: float = 0.35):
    """post index -> (market_id, similarity) for its strongest attachment.

    Separated out so labels can be computed before the database is built:
    a post's target is the move of *the market it is about*, which is only
    known once the attachment is."""
    pairs = semantic_links(markets, posts, top_k=top_k, floor=floor)
    best: dict[int, tuple[str, float]] = {}
    for (post_i, market_i), score in pairs.items():
        market_id = markets[market_i]["market_id"]
        if post_i not in best or score > best[post_i][1]:
            best[post_i] = (market_id, score)
    return best


def subject_index(markets: list[dict], posts: list[dict], *, max_df: float,
                  min_len: int = 4, cap_per_post: int = 6):
    """Distinctive words shared by market questions and posts."""
    market_tokens = [tokens(m["question"]) for m in markets]
    document_frequency = Counter()
    for ts in market_tokens:
        document_frequency.update({t for t in ts if len(t) >= min_len})
    cutoff = max(2, int(len(markets) * max_df))
    vocabulary = {t for t, n in document_frequency.items() if n <= cutoff}

    market_subjects = defaultdict(set)
    for i, ts in enumerate(market_tokens):
        for t in ts & vocabulary:
            market_subjects[i].add(t)
    post_subjects = defaultdict(set)
    for i, post in enumerate(posts):
        hits = tokens(post["body"]) & vocabulary
        for t in sorted(hits)[:cap_per_post]:
            post_subjects[i].add(t)
    live = {t for ts in post_subjects.values() for t in ts}
    live &= {t for ts in market_subjects.values() for t in ts}
    return {i: ts & live for i, ts in market_subjects.items()}, \
           {i: ts & live for i, ts in post_subjects.items()}, live


# ---------------------------------------------------------------------------
def build_rich(markets: list[dict], bars_by_market: dict, posts: list[dict],
               labels: list[dict], *, horizon: int, muted: bool,
               top_k: int = 60, floor: float = 0.35, max_df: float = 0.03,
               attachment: dict | None = None):
    """The nine-table version of the chatter database.

    `labels` are the scored population (posts with an observed window); `posts`
    is every post that may appear in context, labelled or not."""
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
        .column("source", ValueType.TEXT)
        .primary_key("author_id").build(),

        TableDef.new_table("threads")
        .column("channel", ValueType.TEXT)
        .primary_key("thread_id").build(),

        TableDef.new_table("subjects")
        .column("word", ValueType.TEXT)
        .primary_key("subject_id").build(),

        TableDef.new_table("market_subjects")
        .primary_key("market_subject_id").build(),

        TableDef.new_table("comments")
        .column("body", ValueType.TEXT)
        .column("posted_at", ValueType.DATETIME)
        .column("channel", ValueType.TEXT)
        .column("reactions", ValueType.NUMBER)
        .column("similarity", ValueType.NUMBER)
        .column("price_up", ValueType.BOOLEAN)          # the target
        .primary_key("comment_id").time_column("posted_at").build(),

        TableDef.new_table("post_subjects")
        .column("seen_at", ValueType.DATETIME)
        .primary_key("post_subject_id").time_column("seen_at").build(),

        TableDef.new_table("chatter_hours")
        .column("through", ValueType.DATETIME)
        .column("posts", ValueType.NUMBER)
        .column("authors", ValueType.NUMBER)
        .column("mean_similarity", ValueType.NUMBER)
        .column("mean_score", ValueType.NUMBER)
        .primary_key("chatter_hour_id").time_column("through").build(),
    ]
    links = [
        LinkDef("price_ticks", "market_id", "markets"),
        LinkDef("comments", "market_id", "markets"),
        LinkDef("comments", "author_id", "authors"),
        LinkDef("comments", "thread_id", "threads"),
        LinkDef("market_subjects", "market_id", "markets"),
        LinkDef("market_subjects", "subject_id", "subjects"),
        LinkDef("post_subjects", "comment_id", "comments"),
        LinkDef("post_subjects", "subject_id", "subjects"),
        LinkDef("chatter_hours", "market_id", "markets"),
    ]

    rows: dict[str, list[Row]] = {t.name: [] for t in tables}
    market_index = {m["market_id"]: i for i, m in enumerate(markets)}
    for m in markets:
        rows["markets"].append(Row("markets", m["market_id"], {
            "question": m["question"],
            "outcome_label": m.get("outcome_label")}))
    for market_id, bars in bars_by_market.items():
        for b in bars:
            rows["price_ticks"].append(Row(
                "price_ticks", f"{market_id}:{int(b['at'].timestamp())}",
                {k: v for k, v in b.items() if v is not None},
                b["at"], {"market_id": market_id}))

    if attachment is None:
        print("   linking posts to markets by meaning ...", flush=True)
        attachment = attach(markets, posts, top_k=top_k, floor=floor)
    market_subjects, post_subjects, vocabulary = subject_index(
        markets, posts, max_df=max_df)
    print(f"   {len(attachment):,} attached posts, "
          f"{len(vocabulary):,} shared subjects", flush=True)

    for word in sorted(vocabulary):
        rows["subjects"].append(Row("subjects", word, {"word": word}))
    for market_i, words in market_subjects.items():
        market_id = markets[market_i]["market_id"]
        for word in words:
            rows["market_subjects"].append(Row(
                "market_subjects", f"{market_id}:{word}", {}, None,
                {"market_id": market_id, "subject_id": word}))

    seen_authors, seen_threads = set(), set()
    label_by_id = {c["comment_id"]: c for c in labels}
    best_market = attachment

    rollup: dict[tuple[str, datetime], list] = defaultdict(list)
    for post_i, post in enumerate(posts):
        attachment = best_market.get(post_i)
        if attachment is None:
            continue
        market_id, similarity = attachment
        posted = hour_floor(post["posted_at"])
        author = f'{post["source"]}:{post["author"]}'
        if author not in seen_authors:
            seen_authors.add(author)
            rows["authors"].append(Row("authors", author,
                                       {"name": post["author"],
                                        "source": post["source"]}))
        thread_id = post.get("thread_id") or f'{post["source"]}:solo'
        if thread_id not in seen_threads:
            seen_threads.add(thread_id)
            rows["threads"].append(Row("threads", thread_id,
                                       {"channel": post["channel"]}))
        labelled = label_by_id.get(post["chatter_id"])
        cells = {
            "body": "" if muted else post["body"],
            "posted_at": posted, "channel": post["channel"],
            "reactions": post["score"], "similarity": round(similarity, 4),
        }
        if labelled is not None:
            cells["price_up"] = labelled["price_up"]
        rows["comments"].append(Row(
            "comments", post["chatter_id"], cells, posted,
            {"market_id": market_id, "author_id": author,
             "thread_id": thread_id}))
        for word in post_subjects.get(post_i, ()):
            rows["post_subjects"].append(Row(
                "post_subjects", f'{post["chatter_id"]}:{word}',
                {"seen_at": posted}, posted,
                {"comment_id": post["chatter_id"], "subject_id": word}))
        rollup[(market_id, posted)].append((author, similarity, post["score"]))

    for (market_id, when), group in rollup.items():
        through = when + timedelta(hours=1)      # knowable once the hour closes
        rows["chatter_hours"].append(Row(
            "chatter_hours", f"{market_id}:{int(when.timestamp())}", {
                "through": through, "posts": len(group),
                "authors": len({a for a, _, _ in group}),
                "mean_similarity": round(sum(s for _, s, _ in group) / len(group), 4),
                "mean_score": round(sum(v for _, _, v in group) / len(group), 3),
            }, through, {"market_id": market_id}))

    schema = Schema(tuple(tables), tuple(links))
    return schema, wiring(schema, rows, horizon), rows


def wiring(schema: Schema, rows: dict[str, list[Row]], horizon: int):
    by_id = {name: {r.id: r for r in table_rows}
             for name, table_rows in rows.items()}
    children: dict[tuple[str, str], dict] = {}
    for link in schema.links:
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
        """A post's outcome is only known once its window has closed."""
        if bound.as_of is None or row.timestamp is None:
            return row
        if row.timestamp + timedelta(hours=horizon) <= bound.as_of:
            return row
        cells = dict(row.cells)
        cells.pop("price_up", None)
        return Row(row.table, row.id, cells, row.timestamp, row.parents)

    def entities(table, ids, bound: TemporalBound):
        found = [r for i in ids if (r := by_id[table].get(i)) is not None
                 and bound.admits_row(r)]
        return [unlabel(r, bound) for r in found] if table == "comments" else found

    def link_rows(link, parent_id, bound: TemporalBound, limit: int):
        found = [r for r in children[(link.from_table, link.fk_column)]
                 .get(parent_id, ()) if bound.admits_row(r)][:limit]
        return ([unlabel(r, bound) for r in found]
                if link.from_table == "comments" else found)

    def scanner(table, bound: TemporalBound):
        for row in rows[table]:
            if bound.admits_row(row):
                yield unlabel(row, bound) if table == "comments" else row

    builder = RetrieverWiring.new_wiring().default_links(link_rows)
    for table in rows:
        builder.entities(table, entities)
        builder.scanner(table, scanner)
    return builder.build()
