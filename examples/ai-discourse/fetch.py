"""Snapshot the AI corner of Bluesky, with the graph attached.

Three stages:

1. **Community** — start from a hand-checked seed list of accounts that
   demonstrably post about AI, pull everyone each seed follows, and keep the
   accounts that at least ``--min-votes`` seeds follow. The community is
   defined *by the graph*, not by keyword matching, which is why it comes back
   full of working researchers rather than accounts with "AI" in the bio.

2. **Posts** — every community member's recent posts. Those created in the
   universe window (the hours just before the anchor) are the rows to be
   ranked; older ones are the authors' track record.

3. **Engagement** — for every post, WHO liked / reposted / quoted / replied,
   and WHEN. This is the part that does not exist in any public Twitter dump,
   and the part the whole example rests on.

Nothing is cut at the anchor here. The snapshot holds the future as well as
the past; ``db.py`` decides what the model is allowed to see, and ``predict.py``
computes labels from the remainder. Keeping the split out of the fetch means
you can move the anchor without re-fetching.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bsky import Bsky

DATA = Path(__file__).parent / "data"
SNAPSHOT = DATA / "snapshot.json"

# Accounts checked by hand to be active and about AI, spanning research, RL,
# policy, safety and industry so the graph expansion does not collapse onto
# one clique. They are only a starting point: the community below is whoever
# these accounts collectively follow.
SEEDS = [
    "simonwillison.net",          # independent AI researcher / LLM tooling
    "emilymbender.bsky.social",   # computational linguistics, AI criticism
    "rasbt.bsky.social",          # Sebastian Raschka, LLM research
    "yoshuabengio.bsky.social",   # Yoshua Bengio, AI safety
    "hardmaru.bsky.social",       # David Ha, generative models
    "ai2.bsky.social",            # Allen Institute for AI
    "nlpnoah.bsky.social",        # Noah Smith, NLP
    "stanfordnlp.bsky.social",    # Stanford NLP group
    "natolambert.bsky.social",    # RLHF / post-training
    "soldaini.net",               # Olmo data team
    "milesbrundage.bsky.social",  # AI policy
    "eugenevinitsky.bsky.social", # reinforcement learning
    "lukezettlemoyer.bsky.social",# UW / Meta, language models
    "annarogers.bsky.social",     # NLP, evaluation
    "nsaphra.bsky.social",        # interpretability
    "strubell.bsky.social",       # efficiency, ML policy
]


def iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# stage 1 — the community, discovered through the follow graph
# ---------------------------------------------------------------------------
def build_community(api: Bsky, *, min_votes: int, max_members: int,
                    follows_per_seed: int) -> dict[str, dict]:
    votes: dict[str, int] = {}
    profiles: dict[str, dict] = {}

    live_seeds = []
    for handle in SEEDS:
        try:
            profile = api.profile(handle)
        except Exception as err:                       # squatted or renamed
            print(f"   seed {handle:<32} unavailable ({err})")
            continue
        if profile.get("postsCount", 0) < 20:
            # A parked handle with the right name and no posts contributes a
            # follow list that has nothing to do with AI discourse.
            print(f"   seed {handle:<32} skipped (only "
                  f"{profile.get('postsCount', 0)} posts)")
            continue
        live_seeds.append(profile)
        profiles[profile["did"]] = profile

    print(f"   {len(live_seeds)} live seeds")
    for profile in live_seeds:
        count = 0
        for follow in api.follows(profile["handle"],
                                  limit_total=follows_per_seed):
            votes[follow["did"]] = votes.get(follow["did"], 0) + 1
            profiles.setdefault(follow["did"], follow)
            count += 1
        print(f"   {profile['handle']:<34} follows {count}")

    for profile in live_seeds:                         # seeds are members too
        votes[profile["did"]] = max(votes.get(profile["did"], 0), min_votes)

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    members: dict[str, dict] = {}
    for did, vote_count in ranked:
        if vote_count < min_votes or len(members) >= max_members:
            continue
        members[did] = {"did": did, "votes": vote_count,
                        **{k: profiles[did].get(k)
                           for k in ("handle", "displayName", "description")}}
    return members


def hydrate(api: Bsky, members: dict[str, dict]) -> None:
    """Fill in follower/post counts — the numbers the baselines need."""
    for did, member in list(members.items()):
        try:
            profile = api.profile(did)
        except Exception:
            members.pop(did, None)
            continue
        member.update({
            "handle": profile.get("handle"),
            "displayName": profile.get("displayName"),
            "description": profile.get("description"),
            "followers": profile.get("followersCount", 0),
            "following": profile.get("followsCount", 0),
            "posts_total": profile.get("postsCount", 0),
            "created_at": profile.get("createdAt"),
        })


def intra_follows(api: Bsky, members: dict[str, dict],
                  follows_cap: int) -> list[dict]:
    """Follow edges *inside* the community — the graph RT-J gets to traverse."""
    edges = []
    for did, member in members.items():
        for follow in api.follows(did, limit_total=follows_cap):
            if follow["did"] in members:
                edges.append({"src": did, "dst": follow["did"]})
    return edges


# ---------------------------------------------------------------------------
# stage 2 — posts
# ---------------------------------------------------------------------------
def collect_posts(api: Bsky, members: dict[str, dict], *, anchor: datetime,
                  universe_hours: int, history_days: int,
                  history_per_author: int, feed_depth: int) -> list[dict]:
    universe_start = anchor - timedelta(hours=universe_hours)
    history_start = anchor - timedelta(days=history_days)
    posts: list[dict] = []

    for did, member in members.items():
        history_kept = 0
        for item in api.author_feed(did, limit_total=feed_depth):
            post = item.get("post") or {}
            record = post.get("record") or {}
            if post.get("author", {}).get("did") != did:
                continue                                # a repost, not a post
            created = record.get("createdAt")
            if not created:
                continue
            try:
                when = iso(created)
            except ValueError:
                continue
            if when < history_start:
                break                                   # feed is newest-first
            in_universe = universe_start <= when <= anchor
            if not in_universe:
                if when > anchor or history_kept >= history_per_author:
                    continue                            # future, or enough
                history_kept += 1
            posts.append({
                "post_id": post["uri"],
                "author_did": did,
                "created_at": created,
                "text": record.get("text", ""),
                "lang": (record.get("langs") or [None])[0],
                "reply_to": ((record.get("reply") or {}).get("parent") or {})
                            .get("uri"),
                "has_link": bool(record.get("facets") and any(
                    f.get("features", [{}])[0].get("$type", "").endswith("#link")
                    for f in record.get("facets", []))),
                "has_image": bool((record.get("embed") or {}).get("images")),
                "in_universe": in_universe,
            })
    return posts


# ---------------------------------------------------------------------------
# stage 3 — engagement edges: who, and when
# ---------------------------------------------------------------------------
def walk_replies(node: dict, root_id: str, out: list[dict]) -> None:
    for reply in node.get("replies") or []:
        post = reply.get("post")
        if not post:
            continue
        record = post.get("record") or {}
        out.append({"post_id": root_id, "kind": "reply",
                    "actor_did": post["author"]["did"],
                    "at": record.get("createdAt") or post.get("indexedAt")})
        walk_replies(reply, root_id, out)


def keep_profile(store: dict[str, dict], actor: dict | None) -> None:
    """Remember who an engager is.

    "Who engaged" is half the thesis, so the bio that came free in the
    like/quote/reply payload is worth keeping — it is how the model can tell a
    research lead from a drive-by account without being handed a score."""
    if not actor or not actor.get("did") or actor["did"] in store:
        return
    store[actor["did"]] = {k: actor.get(k) for k in
                           ("did", "handle", "displayName", "description")}


def walk_reply_profiles(node: dict, store: dict[str, dict]) -> None:
    for reply in node.get("replies") or []:
        if reply.get("post"):
            keep_profile(store, reply["post"].get("author"))
        walk_reply_profiles(reply, store)


def collect_engagement(api: Bsky, posts: list[dict], *, likes_cap: int,
                       progress_every: int) -> tuple[list[dict], dict]:
    edges: list[dict] = []
    profiles: dict[str, dict] = {}
    for index, post in enumerate(posts, 1):
        uri = post["post_id"]
        try:
            for like in api.likes(uri, limit_total=likes_cap):
                keep_profile(profiles, like["actor"])
                edges.append({"post_id": uri, "kind": "like",
                              "actor_did": like["actor"]["did"],
                              "at": like.get("createdAt")})
            for actor in api.reposted_by(uri):
                # getRepostedBy carries no per-repost timestamp; the repost
                # record's own time is not exposed here, so these are stamped
                # by db.py as undated and only used where that is sound.
                keep_profile(profiles, actor)
                edges.append({"post_id": uri, "kind": "repost",
                              "actor_did": actor["did"], "at": None})
            for quote in api.quotes(uri):
                record = quote.get("record") or {}
                keep_profile(profiles, quote.get("author"))
                edges.append({"post_id": uri, "kind": "quote",
                              "actor_did": quote["author"]["did"],
                              "at": record.get("createdAt")})
            thread = api.thread(uri, depth=3)
            if thread.get("thread"):
                walk_replies(thread["thread"], uri, edges)
                walk_reply_profiles(thread["thread"], profiles)
        except Exception as err:
            print(f"   ! {uri[-16:]} {type(err).__name__}: {str(err)[:60]}")
        if index % progress_every == 0:
            print(f"   engagement {index}/{len(posts)} posts, "
                  f"{len(edges)} edges, {api.calls} calls", flush=True)
    return edges, profiles


def follower_flags(api: Bsky, posts: list[dict],
                   edges: list[dict]) -> dict[str, bool]:
    """``"<author_did>|<actor_did>" -> actor follows author``.

    One ``getRelationships`` call covers 30 actors, so the whole snapshot costs
    a few hundred calls rather than one per pair."""
    author_of = {p["post_id"]: p["author_did"] for p in posts}
    by_author: dict[str, set[str]] = {}
    for edge in edges:
        author = author_of.get(edge["post_id"])
        if author and author != edge["actor_did"]:
            by_author.setdefault(author, set()).add(edge["actor_did"])

    flags: dict[str, bool] = {}
    for index, (author, actors) in enumerate(by_author.items(), 1):
        actor_list = sorted(actors)
        followers = api.follows_any(author, actor_list)
        for actor in actor_list:
            flags[f"{author}|{actor}"] = actor in followers
        if index % 25 == 0:
            print(f"   follower test {index}/{len(by_author)} authors")
    return flags


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-hours-ago", type=int, default=48,
                    help="anchor = now - this. Must exceed the horizon so the "
                         "future window has actually elapsed.")
    ap.add_argument("--universe-hours", type=int, default=12,
                    help="posts created in the N hours before the anchor")
    ap.add_argument("--history-days", type=int, default=21)
    ap.add_argument("--history-per-author", type=int, default=6)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--max-members", type=int, default=140)
    ap.add_argument("--follows-per-seed", type=int, default=1000)
    ap.add_argument("--follows-cap", type=int, default=600)
    ap.add_argument("--feed-depth", type=int, default=120)
    ap.add_argument("--likes-cap", type=int, default=300)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--out", default=str(SNAPSHOT))
    args = ap.parse_args()

    api = Bsky()
    now = datetime.now(timezone.utc)
    anchor = (now - timedelta(hours=args.anchor_hours_ago)).replace(
        minute=0, second=0, microsecond=0)
    print(f"now {now:%Y-%m-%d %H:%M} UTC   anchor {anchor:%Y-%m-%d %H:%M} UTC")

    print("\n>> community from the follow graph")
    members = build_community(api, min_votes=args.min_votes,
                              max_members=args.max_members,
                              follows_per_seed=args.follows_per_seed)
    print(f"   {len(members)} members at >={args.min_votes} seed follows")
    hydrate(api, members)
    print(f"   {len(members)} hydrated")

    print("\n>> follow edges inside the community")
    edges_follow = intra_follows(api, members, args.follows_cap)
    print(f"   {len(edges_follow)} edges")

    print("\n>> posts")
    posts = collect_posts(api, members, anchor=anchor,
                          universe_hours=args.universe_hours,
                          history_days=args.history_days,
                          history_per_author=args.history_per_author,
                          feed_depth=args.feed_depth)
    universe = [p for p in posts if p["in_universe"]]
    print(f"   {len(posts)} posts, {len(universe)} in the universe window")

    print("\n>> engagement (who, and when)")
    engagements, actor_profiles = collect_engagement(
        api, posts, likes_cap=args.likes_cap,
        progress_every=args.progress_every)
    print(f"   {len(engagements)} edges from "
          f"{len(actor_profiles)} distinct accounts")

    print("\n>> follower test for every (author, engager) pair")
    flags = follower_flags(api, posts, engagements)
    print(f"   {len(flags)} pairs, "
          f"{sum(flags.values())} of them already following")

    DATA.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.write_text(json.dumps({
        "fetched_at": now.isoformat(),
        "anchor": anchor.isoformat(),
        "universe_hours": args.universe_hours,
        "seeds": SEEDS,
        "members": list(members.values()),
        "actor_profiles": list(actor_profiles.values()),
        "follow_edges": edges_follow,
        "posts": posts,
        "engagements": engagements,
        "follower_flags": flags,
    }, indent=1))
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{api.calls} API calls)")


if __name__ == "__main__":
    main()
