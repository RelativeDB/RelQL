"""A very small AT Protocol client — public endpoints, no account, no key.

Everything this example needs is served unauthenticated by Bluesky's public
appview:

    app.bsky.actor.getProfile        who an account is
    app.bsky.graph.getFollows        the follow graph
    app.bsky.graph.getRelationships  "does A follow B?", 30 pairs per call
    app.bsky.feed.getAuthorFeed      an account's posts
    app.bsky.feed.getLikes           WHO liked, and WHEN
    app.bsky.feed.getRepostedBy      WHO reposted
    app.bsky.feed.getQuotes          WHO quoted
    app.bsky.feed.getPostThread      WHO replied

The last four are the reason this example is on Bluesky rather than Twitter.
Twitter's public dumps give you ``likes: 43`` — a scalar. Bluesky gives you the
43 identities and the timestamp on each one, which is what turns engagement
into a graph you can traverse.

``app.bsky.feed.searchPosts`` is *not* public (it 403s without a session), so
the universe is built from the follow graph instead — see ``fetch.py``.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://public.api.bsky.app/xrpc"
UA = "relativedb-ai-discourse-example"


class Bsky:
    """Public appview client with retry and a polite floor on request rate."""

    def __init__(self, *, min_interval: float = 0.02, retries: int = 4):
        self.min_interval = min_interval
        self.retries = retries
        self._last = 0.0
        self.calls = 0

    def get(self, method: str, **params) -> dict:
        gap = self.min_interval - (time.time() - self._last)
        if gap > 0:
            time.sleep(gap)
        url = f"{API}/{method}?" + urllib.parse.urlencode(params, doseq=True)
        request = urllib.request.Request(url, headers={"User-Agent": UA})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    self.calls += 1
                    self._last = time.time()
                    return json.load(response)
            except urllib.error.HTTPError as err:
                # 400 is a permanent "no such actor / blocked / deleted" for
                # these endpoints; retrying it just burns time.
                if err.code == 400:
                    raise
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def paged(self, method: str, key: str, *, limit_total: int, **params):
        """Follow ``cursor`` until ``limit_total`` items or the list ends."""
        cursor, seen = None, 0
        while seen < limit_total:
            page = dict(params)
            if cursor:
                page["cursor"] = cursor
            try:
                data = self.get(method, **page)
            except urllib.error.HTTPError:
                return
            items = data.get(key) or []
            if not items:
                return
            for item in items:
                yield item
                seen += 1
                if seen >= limit_total:
                    return
            cursor = data.get("cursor")
            if not cursor:
                return

    # -- convenience wrappers -------------------------------------------
    def profile(self, actor: str) -> dict:
        return self.get("app.bsky.actor.getProfile", actor=actor)

    def follows(self, actor: str, *, limit_total: int = 1000):
        return self.paged("app.bsky.graph.getFollows", "follows",
                          limit_total=limit_total, actor=actor, limit=100)

    def author_feed(self, actor: str, *, limit_total: int = 100,
                    filter: str = "posts_no_replies"):
        return self.paged("app.bsky.feed.getAuthorFeed", "feed",
                          limit_total=limit_total, actor=actor, limit=100,
                          filter=filter)

    def likes(self, uri: str, *, limit_total: int = 300):
        return self.paged("app.bsky.feed.getLikes", "likes",
                          limit_total=limit_total, uri=uri, limit=100)

    def reposted_by(self, uri: str, *, limit_total: int = 200):
        return self.paged("app.bsky.feed.getRepostedBy", "repostedBy",
                          limit_total=limit_total, uri=uri, limit=100)

    def quotes(self, uri: str, *, limit_total: int = 100):
        return self.paged("app.bsky.feed.getQuotes", "posts",
                          limit_total=limit_total, uri=uri, limit=100)

    def thread(self, uri: str, *, depth: int = 3) -> dict:
        return self.get("app.bsky.feed.getPostThread", uri=uri, depth=depth)

    def follows_any(self, actor_did: str, others: list[str]) -> set[str]:
        """Subset of ``others`` that follow ``actor_did``.

        ``getRelationships`` reports the edge as it stands *right now*, not as
        it stood at the anchor. Follows only accumulate, so someone who saw a
        post, liked it, and then followed reads as a follower here — which
        makes the "reached a non-follower" label an undercount, never an
        overcount. The bias is against the signal, so a positive result is not
        an artifact of it."""
        found: set[str] = set()
        for start in range(0, len(others), 30):        # endpoint caps at 30
            batch = others[start:start + 30]
            try:
                data = self.get("app.bsky.graph.getRelationships",
                                actor=actor_did, others=batch)
            except urllib.error.HTTPError:
                continue
            for rel in data.get("relationships", []):
                if rel.get("followedBy"):
                    found.add(rel["did"])
        return found
