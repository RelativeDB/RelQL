"""GH Archive events, loaded as a relational graph.

One gzipped JSON file per hour from data.gharchive.org. Each line is an event
with a nested actor, repo and payload; this flattens them into tables so
XGBoost can have a matrix and RelativeDB can have the graph.

Ground truth for the bot task lives in ``labels.py``: hand-verified accounts,
not the ``[bot]`` suffix, which only marks accounts that announce themselves.

Aggregates count every event including the types kept out of the graph;
``SKIP_TYPES`` controls what becomes a row.
"""
from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

BASE = "https://data.gharchive.org"
DIR = Path(__file__).resolve().parents[1] / "corpus" / "gharchive"
# GH Archive publishes hourly, a few hours behind real time. Keep this recent:
# a bot detector built on two-year-old traffic is measuring accounts GitHub has
# already banned and abuse patterns that have since moved on.
def hours_for(*dates: str) -> list[str]:
    """Every hour of each date, as GH Archive names them."""
    return [f"{d}-{h}" for d in dates for h in range(24)]


DEFAULT_HOURS = hours_for("2026-07-17")

# PushEvent is 90% of the current feed and its payload was reduced to
# before/head/push_id/ref/repository_id -- no commits, no messages, no sizes.
# Dropping it at parse time costs almost no information and lets many more
# hours fit, which is what the sparse event types (stars, issues, forks) need.
SKIP_TYPES = frozenset({"PushEvent"})


def ensure(hours=DEFAULT_HOURS) -> list[Path]:
    DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for h in hours:
        p = DIR / f"{h}.json.gz"
        if not p.exists():
            print(f"downloading {h}", flush=True)
            # data.gharchive.org rejects urllib's default User-Agent with a
            # 403, which reads like a missing file but is not one.
            req = urllib.request.Request(
                f"{BASE}/{h}.json.gz",
                headers={"User-Agent": "Mozilla/5.0 (relativedb-benchmark)"})
            tmp = p.with_suffix(".part")
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.rename(p)     # never leave a half file looking complete
        out.append(p)
    return out


def is_english(text: str) -> bool:
    """Whether ``text`` is English, per langdetect.

    The text encoder (all-MiniLM-L12-v2) is English-only, so a Russian issue
    title is embedded as noise rather than as meaning. Rows it cannot read are
    dropped from the task instead of being scored as if it could.
    """
    t = (text or "").strip()
    if not t:
        return True
    # Script check first: langdetect needs length to be reliable, but a short
    # string in a non-Latin script ("状态:待用户反馈") is decidable on sight, and
    # a length short-circuit alone would wave it through.
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return False
    if len(t) < 12:                    # too short for langdetect to judge
        return True
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(t) == "en"
    except Exception:
        return True


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((k / n) * math.log2(k / n) for k in counts.values())


def junk_name_score(name: str) -> float:
    """How much a repo name looks like keyboard mash rather than a project.

    Throwaway repos are named ``ffffffffff``, ``a1b2c3d4e5``, ``test-9931``.
    Three things a real project name rarely does: repeat one character many
    times, run long with no separator at high character entropy, and put digits
    in most of its tokens. Scored on the repo, so it reaches an account through
    the graph rather than being precomputed onto the account.
    """
    nm = name.split("/")[-1]
    if not nm:
        return 0.0
    longest_run = max((len(m.group(0)) for m in re.finditer(r"(.)\1*", nm)),
                      default=1)
    tokens = [t for t in re.split(r"[-_.\s]+", nm) if t]
    digitish = sum(1 for t in tokens if any(ch.isdigit() for ch in t))
    unbroken = (len(nm) >= 12 and not re.search(r"[-_.]", nm)
                and _entropy(nm.lower()) > 3.4)
    return round(min(1.0, longest_run / 6) * 0.4
                 + min(1.0, digitish / max(1, len(tokens))) * 0.3
                 + (0.3 if unbroken else 0.0), 3)


def login_stem(login: str) -> str:
    """The login with trailing digits and a common suffix removed.

    ``lihongjun1``, ``lihongjun9`` and ``lihongjun20`` share a stem, as do
    ``scalr-autotester0`` and ``scalr-autotester13``. A stem with several
    accounts behind it, all active in the same window, is the "similar name in
    the same timeframe" signal: farms mint accounts from a template.
    """
    s = re.sub(r"\d+$", "", login)
    # A short trailing token after a separator is usually an instance marker,
    # not part of the name: aws-aemilia-pdx / -iad / -dub are one fleet, and
    # cutting it is what makes them look like one. Four characters keeps
    # region codes and "-cmd"/"-sudo" style suffixes while leaving real name
    # parts ("terraform", "konflux") alone.
    s = re.sub(r"[-_][A-Za-z0-9]{1,4}$", "", s)
    return s or login


@dataclass
class GH:
    """Eight linked tables. Depth is the point: an actor reaches an org only
    through events and repos, and an issue reaches a label definition only
    through the bridge, so a model has to traverse to use any of it."""

    actors: pd.DataFrame
    orgs: pd.DataFrame        # owner side of owner/name
    repos: pd.DataFrame       # -> orgs
    events: pd.DataFrame      # -> actors, repos
    issues: pd.DataFrame      # -> repos, actors
    comments: pd.DataFrame    # -> issues, actors, repos
    pulls: pd.DataFrame       # -> repos, actors
    issue_labels: pd.DataFrame  # bridge: issues -> label_defs
    label_defs: pd.DataFrame
    # Added after the fact, so they carry defaults: a cache pickled before
    # they existed still unpickles, it just has nothing in them.
    pull_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    reviews: pd.DataFrame = field(default_factory=pd.DataFrame)

    TABLES = ("actors", "orgs", "repos", "events", "issues", "comments",
              "pulls", "pull_events", "reviews", "issue_labels", "label_defs")

    def describe(self) -> dict:
        return {k: int(len(getattr(self, k))) for k in self.TABLES}


def load(hours=DEFAULT_HOURS, max_events: int | None = None,
         skip_types: frozenset = SKIP_TYPES) -> GH:
    # No cap by default. A cap here does not sample the corpus, it fills from
    # the front of the file list and stops: at 400k it took hours 00 and 06
    # whole, 3% of hour 12, and none of hour 18. Human activity is
    # timezone-dependent and bot activity is not, so that skew lands directly
    # on the thing the bot task measures.
    ev_rows, issue_rows, comment_rows, pull_rows = [], [], [], []
    pe_rows, review_rows = [], []
    # Per-actor and per-repo tallies over EVERY event, including the types kept
    # out of the graph.
    actor_agg = defaultdict(lambda: {"n_events": 0, "repos": set(), "types": set(),
                                     "n_push": 0, "own": 0,
                                     "first": None, "last": None})
    repo_agg = defaultdict(lambda: {"n": 0, "actors": set()})
    n = 0
    for path in ensure(hours):
        with gzip.open(path, "rt") as f:
            for line in f:
                if max_events and n >= max_events:
                    break
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                a, r, pl = e.get("actor"), e.get("repo"), e.get("payload") or {}
                # A handful of records carry no repo name (2 in 654k on
                # 2026-07-17, both ForkEvent). They cannot be placed in the
                # graph, and one of them killed the whole load.
                if not a or not r or "login" not in a or "name" not in r:
                    continue
                n += 1
                # Skipped types stay out of the GRAPH -- PushEvent is 90% of
                # the feed and its payload is now empty -- but they are still
                # counted here, because "how much does this account push" is a
                # fact about the account whether or not the rows are worth
                # carrying. Aggregates see every event; the graph sees fewer.
                agg = actor_agg[a["login"]]
                agg["n_events"] += 1
                agg["repos"].add(r["name"])
                agg["types"].add(e["type"])
                if e["type"] == "PushEvent":
                    agg["n_push"] += 1
                if r["name"].split("/")[0] == a["login"]:
                    agg["own"] += 1
                ts = e["created_at"]
                agg["first"] = ts if agg["first"] is None else min(agg["first"], ts)
                agg["last"] = ts if agg["last"] is None else max(agg["last"], ts)
                repo_agg[r["name"]]["n"] += 1
                repo_agg[r["name"]]["actors"].add(a["login"])
                if e["type"] in skip_types:
                    continue
                ev_rows.append({
                    "event_id": e["id"], "type": e["type"],
                    "actor_login": a["login"], "repo_name": r["name"],
                    "ts": e["created_at"], "public": bool(e.get("public", True)),
                    "action": pl.get("action") or "",
                    # PushEvent payloads no longer carry a commits array (they
                    # are before/head/push_id/ref/repository_id since some
                    # point before 2026-07), so this is 0 on current data.
                    "n_commits": len(pl.get("commits") or []),
                    "ref_type": pl.get("ref_type") or "",
                })
                if e["type"] == "IssueCommentEvent" and pl.get("comment"):
                    c, it = pl["comment"], pl.get("issue") or {}
                    cb = c.get("body") or ""
                    comment_rows.append({
                        "comment_id": str(c["id"]),
                        "issue_key": f"{r['name']}#{it.get('number', 0)}",
                        "repo_name": r["name"], "actor_login": a["login"],
                        "comment_body": cb,
                        "comment_len": len(cb),
                        "comment_assoc": c.get("author_association") or "",
                        "ts": e["created_at"]})
                if e["type"] == "PullRequestEvent" and pl.get("pull_request"):
                    pr = pl["pull_request"]
                    # The lifecycle, one row per event rather than one per PR.
                    # ``pulls`` keeps only a PR's first appearance, so the
                    # merge that resolves it is not in there -- and the 2026
                    # payload has no ``merged`` flag to read instead, only the
                    # action. Whether a PR merged is a fact about the *stream*
                    # here, which is why it needs its own table.
                    num = pl.get("number") or pr.get("number")
                    pe_rows.append({
                        "pe_id": e["id"],
                        "pull_key": f"{r['name']}!{num}",
                        "repo_name": r["name"], "actor_login": a["login"],
                        "action": pl.get("action") or "",
                        "base_ref": (pr.get("base") or {}).get("ref") or "",
                        "head_ref": (pr.get("head") or {}).get("ref") or "",
                        "ts": e["created_at"]})
                    pull_rows.append({
                        "pull_key": f"{r['name']}!{pr['number']}",
                        "repo_name": r["name"], "actor_login": a["login"],
                        "changed_files": int(pr.get("changed_files") or 0),
                        "additions": int(pr.get("additions") or 0),
                        "deletions": int(pr.get("deletions") or 0),
                        "pr_title": pr.get("title") or "",
                        "pr_assoc": pr.get("author_association") or "",
                        "ts": e["created_at"]})
                if e["type"] in ("PullRequestReviewEvent",
                                 "PullRequestReviewCommentEvent"):
                    pr = pl.get("pull_request") or {}
                    body = pl.get("review") or pl.get("comment") or {}
                    num = pl.get("number") or pr.get("number")
                    if num is not None and body.get("id") is not None:
                        review_rows.append({
                            "review_id": str(body["id"]),
                            "pull_key": f"{r['name']}!{num}",
                            "repo_name": r["name"], "actor_login": a["login"],
                            "kind": ("review" if e["type"].endswith("ReviewEvent")
                                     else "review_comment"),
                            # approved / changes_requested / commented; empty
                            # on a line comment, which has no verdict
                            "state": body.get("state") or "",
                            "review_len": len(body.get("body") or ""),
                            "ts": e["created_at"]})
                if e["type"] == "IssuesEvent" and pl.get("issue"):
                    it = pl["issue"]
                    labels = [l["name"] for l in (it.get("labels") or [])]
                    body = it.get("body") or ""
                    issue_rows.append({
                        "issue_key": f"{r['name']}#{it['number']}",
                        "repo_name": r["name"], "actor_login": a["login"],
                        # opened / closed / reopened. An issue that entered the
                        # window on a close was opened before it, where its
                        # reply history is not observable.
                        "action": pl.get("action") or "",
                        "title": it.get("title") or "",
                        "title_len": len(it.get("title") or ""),
                        "body": body,
                        "body_len": len(body),
                        "n_labels": len(labels),
                        "labels": labels,
                        "label": labels[0] if labels else "",
                        "comments": int(it.get("comments") or 0),
                        "state": it.get("state") or "",
                        "author_association": it.get("author_association") or "",
                        "ts": e["created_at"],
                    })
        if max_events and n >= max_events:
            break

    events = pd.DataFrame(ev_rows)
    events["ts"] = pd.to_datetime(events.ts, format="ISO8601", utc=True)
    events = events.drop_duplicates("event_id").sort_values("ts")

    issues = pd.DataFrame(issue_rows)
    if len(issues):
        issues["ts"] = pd.to_datetime(issues.ts, format="ISO8601", utc=True)
        issues = issues.drop_duplicates("issue_key").sort_values("ts")

    comments = pd.DataFrame(comment_rows)
    if len(comments):
        comments["ts"] = pd.to_datetime(comments.ts, format="ISO8601", utc=True)
        comments = comments.drop_duplicates("comment_id")
    pulls = pd.DataFrame(pull_rows)
    if len(pulls):
        pulls["ts"] = pd.to_datetime(pulls.ts, format="ISO8601", utc=True)
        pulls = pulls.drop_duplicates("pull_key")

    pull_events = pd.DataFrame(
        pe_rows, columns=["pe_id", "pull_key", "repo_name", "actor_login",
                          "action", "base_ref", "head_ref", "ts"])
    if len(pull_events):
        pull_events["ts"] = pd.to_datetime(pull_events.ts, format="ISO8601",
                                           utc=True)
        pull_events = pull_events.drop_duplicates("pe_id").sort_values("ts")
    reviews = pd.DataFrame(
        review_rows, columns=["review_id", "pull_key", "repo_name",
                              "actor_login", "kind", "state", "review_len",
                              "ts"])
    if len(reviews):
        reviews["ts"] = pd.to_datetime(reviews.ts, format="ISO8601", utc=True)
        reviews = reviews.drop_duplicates("review_id").sort_values("ts")

    # issue_labels / label_defs used to bridge issues to a shared label node.
    # They were dropped from the schema because label_name restates the target
    # on another table; the frames are kept empty so GH.describe() still works.
    issue_labels = pd.DataFrame(columns=["il_key", "issue_key", "label_name", "ts"])
    label_defs = pd.DataFrame(columns=["label_name", "label_uses", "label_len"])

    logins = sorted(actor_agg)
    actors = pd.DataFrame({
        "actor_login": logins,
        "n_events": [actor_agg[k]["n_events"] for k in logins],
        "n_repos": [len(actor_agg[k]["repos"]) for k in logins],
        "n_types": [len(actor_agg[k]["types"]) for k in logins],
        "push_share": [actor_agg[k]["n_push"] / max(1, actor_agg[k]["n_events"])
                       for k in logins],
    })
    # Declared GitHub Apps. Weak supervision only: these announce themselves,
    # so the accounts worth catching are the ones this misses. labels.py holds
    # the hand-verified ground truth.
    actors["is_bot"] = actors.actor_login.str.endswith("[bot]").astype(int)
    # The username as a feature, with the marker removed. Naming style is real
    # signal, but leaving "[bot]" in means the model learns the suffix and
    # nothing that transfers to undeclared automation.
    actors["login"] = actors.actor_login.str.replace(r"\[bot\]$", "", regex=True)
    stars = events[events.type == "WatchEvent"].groupby("actor_login").size()
    actors["n_stars"] = actors.actor_login.map(stars).fillna(0).astype(int)

    # "Similar name in the same timeframe": how many accounts active in this
    # window share this login's stem. A farm mints logins from a template, so
    # a stem with several accounts behind it is worth more than the name alone.
    actors["login_stem"] = actors.actor_login.map(login_stem)
    fam = actors.groupby("login_stem").actor_login.nunique()
    actors["login_family"] = actors.login_stem.map(fam).astype(int)
    # Share of activity on repos the account itself owns. Throwaway-repo
    # accounts push only to themselves; a contributor spreads across others'.
    actors["own_repo_share"] = [
        round(actor_agg[k]["own"] / max(1, actor_agg[k]["n_events"]), 3)
        for k in logins]
    for name, kind in (("n_created", "CreateEvent"), ("n_deleted", "DeleteEvent")):
        c = events[events.type == kind].groupby("actor_login").size()
        actors[name] = actors.actor_login.map(c).fillna(0).astype(int)

    # Degree counts: how many rows of each related table this account owns.
    # These are structural facts about the graph -- the number of edges of each
    # kind -- rather than derived rates, and they let an account that only ever
    # stars be told apart from one that opens PRs and comments.
    for name, frame in (("n_pulls", pulls), ("n_issues", issues),
                        ("n_comments", comments)):
        counts = (frame.groupby("actor_login").size() if len(frame)
                  else pd.Series(dtype=int))
        actors[name] = actors.actor_login.map(counts).fillna(0).astype(int)
    owners = events.assign(owner=events.repo_name.str.split("/").str[0])
    n_orgs = owners.groupby("actor_login").owner.nunique()
    actors["n_orgs"] = actors.actor_login.map(n_orgs).fillna(0).astype(int)

    rnames = sorted(repo_agg)
    repos = pd.DataFrame({
        "repo_name": rnames,
        "repo_events": [repo_agg[k]["n"] for k in rnames],
        "repo_actors": [len(repo_agg[k]["actors"]) for k in rnames],
        "repo_name_len": [len(x) for x in rnames],
    })
    repos["junk_name"] = [junk_name_score(x) for x in repos.repo_name]
    repos["org_login"] = repos.repo_name.str.split("/").str[0]
    og = repos.groupby("org_login")
    orgs = pd.DataFrame({
        "org_login": og.size().index,
        "org_repos": og.size().values,
        "org_events": og.repo_events.sum().values,
        "org_name_len": [len(x) for x in og.size().index],
    })
    return GH(actors=actors, orgs=orgs, repos=repos, events=events,
              issues=issues, comments=comments, pulls=pulls,
              issue_labels=issue_labels, label_defs=label_defs,
              pull_events=pull_events, reviews=reviews)
