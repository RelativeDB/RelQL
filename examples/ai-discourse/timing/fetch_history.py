"""Deep post history for the posting-time question.

The reach snapshot covers 36 hours. You cannot see a day-of-week effect in 36
hours, so this pulls months of history instead.

It is much cheaper than the reach fetch because it needs no engagement *edges*:
``getAuthorFeed`` already returns ``likeCount`` / ``repostCount`` / ``replyCount``
/ ``quoteCount`` on every post, for free, in the same call that returns the post.
That is ~4 calls per author instead of ~4 calls per post.

What that costs us is honesty about saturation: those counts are read **now**, so
a post from yesterday has not finished accumulating while a post from March has.
``analyze_timing.py`` handles this by discarding anything newer than a cutoff
(default 14 days) rather than by modelling a decay curve — a post that is two
weeks old is done, and the comparison across hours is then like for like.

The community is reused from the reach snapshot rather than rediscovered, so the
two experiments are talking about the same people.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bsky import Bsky                                          # noqa: E402
import db                                                      # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "data" / "history.json"


def iso(value: str):
    try:
        when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=150,
                    help="how far back to walk each author's feed")
    ap.add_argument("--per-author", type=int, default=600,
                    help="cap on posts pulled per author")
    ap.add_argument("--reach-snapshot", default=str(db.SNAPSHOT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()

    reach = db.load(Path(args.reach_snapshot))
    members = {m["did"]: m for m in reach["members"]}
    print(f"{len(members)} community members reused from the reach snapshot")

    api = Bsky()
    now = datetime.now(timezone.utc)
    floor = now - timedelta(days=args.days)
    posts = []

    for index, (did, member) in enumerate(members.items(), 1):
        kept = 0
        # filter=posts_with_replies: a reply is still a posting decision made
        # at a time of day, and excluding them would bias toward broadcasters.
        for item in api.author_feed(did, limit_total=args.per_author,
                                    filter="posts_with_replies"):
            post = item.get("post") or {}
            record = post.get("record") or {}
            if post.get("author", {}).get("did") != did:
                continue                                   # a repost, not a post
            when = iso(record.get("createdAt"))
            if when is None:
                continue
            if when < floor:
                break                                      # feed is newest-first
            embed = record.get("embed") or {}
            posts.append({
                "post_id": post["uri"],
                "author_did": did,
                "created_at": record["createdAt"],
                "text": record.get("text", ""),
                "lang": (record.get("langs") or [None])[0],
                "likes": post.get("likeCount", 0),
                "reposts": post.get("repostCount", 0),
                "replies": post.get("replyCount", 0),
                "quotes": post.get("quoteCount", 0),
                "is_reply": bool(record.get("reply")),
                "has_link": any(
                    f.get("features", [{}])[0].get("$type", "").endswith("#link")
                    for f in (record.get("facets") or [])),
                "has_image": bool(embed.get("images")
                                  or (embed.get("media") or {}).get("images")),
                "has_video": "video" in str(embed.get("$type", "")).lower(),
                "text_length": len(record.get("text", "") or ""),
            })
            kept += 1
        if index % args.progress_every == 0:
            print(f"   {index}/{len(members)} authors, {len(posts)} posts, "
                  f"{api.calls} calls", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fetched_at": now.isoformat(),
        "days": args.days,
        "members": [
            {k: m.get(k) for k in
             ("did", "handle", "displayName", "description", "followers",
              "following", "posts_total", "created_at")}
            for m in members.values()],
        "posts": posts,
    }, indent=1))
    span = sorted(p["created_at"] for p in posts)
    print(f"\n{len(posts)} posts from {len({p['author_did'] for p in posts})} "
          f"authors, {span[0][:10]} .. {span[-1][:10]}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {api.calls} calls)")


if __name__ == "__main__":
    main()
