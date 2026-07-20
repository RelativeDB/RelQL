"""Does a new account come back? A depth ablation on the GH Archive graph.

The question this benchmark exists to answer is not "what is the score" but
"does relational context earn its cost". That needs a task where the entity
row is genuinely empty and the answer is genuinely far away, and this feed
forces one on us.

GH Archive's 2026 stream is subsampled per object: of 50,911 pull requests in
the week, 49,258 appear exactly once, and only 176 have both an ``opened`` and
a ``merged`` event. Issue threads are the same -- 311 of 22,514 commented
issues also have their issue row. So every task of the form "entity X, does
event Y happen to it later" is dead on arrival here. What survives is
structure that aggregates over many objects: which accounts touch which
repositories, and when.

So the task is:

    An account appears in the feed for the first time. Its first event is on a
    repository it does not own, which already had two or more people working
    on it. Will the account act again within 24 hours?

Its row is a login string. That is the whole entity. Everything that could
answer the question is somewhere else:

  * how busy the repository it arrived at actually is -- three hops out, in
    that repository's own events and days;
  * whether the repository is a real project or a throwaway -- its name, its
    org, its issues and pull requests;
  * who else is there and what kind of accounts they are -- four hops out,
    through the people table and back into ``actors``.

To keep that honest, every window-spanning aggregate is stripped from the
static tables. ``repos.repo_events`` counts the whole week, so at an anchor in
the middle of it that count is partly the future; the same is true of every
per-actor tally, and of the ``n_events``-style columns the bot benchmark
serves. They are dropped. What is left on a static row is what does not change
-- a name and its length -- and all activity has to be reached through
timestamped rows, which the temporal bound can cut.

The population is selected on pre-arrival state only. "Already had two or more
people" counts distinct actors strictly before the arrival, never over the
week, because selecting on what a repository would go on to do is selecting on
the future for both classes at once.

Two derived tables carry structure the raw feed does not express:

``repo_people``  one row per (repo, actor) at first contact, which is what
                 makes the third hop legible. The events table holds this
                 implicitly, but a fanout-capped walk sees at most
                 ``bfs_width`` rows, and 32 events can be 32 people or one
                 person 32 times. Deduplicating to first contact turns the same
                 budget into 32 distinct people -- and their arrival times say
                 whether this repository holds on to anyone.

``repo_days``    one row per repository per day, stamped at the day's end so it
                 is wholly in the past at any later anchor. Same-day counts
                 only; nothing in it looks forward, which is why it needs no
                 embargo beyond the day boundary.

The measurement is an ablation on ``max_hops`` with everything else held
fixed at the reference implementation's eval geometry. Depth 0 is the login
plus a cohort of other new accounts -- RT's in-context normalization with no
traversal at all. Depth 1 adds the account's own arrival event. Depth 2
reaches the repository. Depth 3 opens it up. Depth 4 reaches the people. If
the curve is flat, the graph is not paying for itself here, and that is the
answer.

    python -m gh.retention probe        # population and table sizes, no model
    python -m gh.retention preflight    # checkpoint, device, depth, leak checks
    python -m gh.retention run          # the ablation
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

CACHE = Path("/tmp/gh_pr.pkl")
OUT = Path(__file__).resolve().parent / "retention_results.json"
HORIZON = pd.Timedelta("24h")   # "comes back" means acts again within this
MIN_PRIOR_PEOPLE = 2            # the repo must already be somebody's project
# How long an account must have been silent to count as arriving. This is a
# fixed lookback rather than "first seen in the window" on purpose: an account
# first seen on the third day was only checked against two days of history,
# one first seen on the sixth against five, so the earlier population holds
# far more established accounts that merely missed the warmup. That showed as
# a return rate decaying 41.5% -> 35.0% -> 24.7% -> 19.8% across the four days
# -- a definition artifact, and one that a temporal split turns into training
# on a different population than it tests on. A fixed window is the same test
# for every entity wherever it sits.
SILENCE = pd.Timedelta("48h")
# How many days at the end of the feed entities may come from. The archive's
# own density rises fivefold across this week -- 183k, 186k, 287k, 468k, 407k,
# 325k, 337k graph events a day -- and "silent for 48h" is only as strict as
# the stretch it looks back over. Early arrivals are checked against 369k
# events, late ones against 875k, so the early population quietly fills with
# established accounts whose prior activity was simply never recorded, and
# those return far more often. It shows as a return rate of 42.1%, 38.0%,
# 28.6%, 25.8% across the four candidate days. The last two are measured
# against comparable density, and they are the ones used.
ENTITY_DAYS = 2

QUERY = ("PREDICT actors.came_back = 1 FROM actors "
         "WHERE actors.actor_login IN :ids")


# ---------------------------------------------------------------- derivation
def arrivals(g: D.GH) -> pd.DataFrame:
    """Each account's arrival: an event after ``SILENCE`` of nothing.

    Every qualifying wake is an entity, not just an account's first. Keeping
    only the first would empty the later days of exactly the accounts that
    wake more than once -- the ones most likely to come back -- which is the
    same population drift the fixed lookback exists to remove. One row per
    (account, moment) is also the shape a RelBench train table has.

    ``prior_people`` counts distinct actors on that repository over the two
    complete calendar days before the arrival's own day. Two properties matter
    and they pull in opposite directions:

      * it must not see the future, or the population is selected on what the
        repository went on to do -- for both classes, but the future all the
        same. Whole days strictly before the arrival's day cannot.
      * it must be the same measurement for every entity. Counting over all
        history before the arrival is not: a repository reaching two people by
        the third day of the feed had two days to do it, one reaching two
        people by the sixth had five, so the early population sits on far more
        active repositories and returns more often. That is most of the
        remaining slope in the daily return rate.
    """
    ev = g.events[["actor_login", "repo_name", "type", "ts"]].sort_values(
        ["actor_login", "ts"])
    prev = ev.groupby("actor_login").ts.shift(1)
    wake = ev[prev.isna() | (ev.ts - prev >= SILENCE)].rename(
        columns={"ts": "t0"})
    wake = wake[wake.repo_name.str.split("/").str[0] != wake.actor_login]
    wake = wake.reset_index(drop=True)
    wake["wake_id"] = wake.index

    ev2 = g.events[["repo_name", "actor_login", "ts"]].assign(
        day=g.events.ts.dt.floor("D"))
    wake["day"] = wake.t0.dt.floor("D")
    out = []
    for day, part in wake.groupby("day"):
        window = ev2[(ev2.day >= day - 2 * pd.Timedelta("1D"))
                     & (ev2.day < day)]
        n = window.groupby("repo_name").actor_login.nunique()
        out.append(part.assign(prior_people=part.repo_name.map(n)))
    return pd.concat(out).fillna({"prior_people": 0}).drop(columns=["day"])


def came_back(g: D.GH, arr: pd.DataFrame) -> pd.Series:
    """1 when the account produces another event inside the horizon."""
    ev = g.events[["actor_login", "ts"]]
    m = ev.merge(arr[["wake_id", "actor_login", "t0"]], on="actor_login")
    back = set(m.loc[(m.ts > m.t0) & (m.ts <= m.t0 + HORIZON), "wake_id"])
    return arr.wake_id.isin(back).astype(int)


def repo_days(g: D.GH) -> pd.DataFrame:
    """Per-repo daily activity, stamped at the day's end.

    Same-day counts only. Nothing here looks forward, so the day boundary is
    the whole embargo it needs.
    """
    ev = g.events
    d = ev.assign(day=ev.ts.dt.floor("D"))
    rd = (d.groupby(["repo_name", "day"])
          .agg(rd_events=("event_id", "size"),
               rd_actors=("actor_login", "nunique")).reset_index())
    con = (g.events.sort_values("ts")
           .drop_duplicates(["repo_name", "actor_login"]))
    newp = (con.assign(day=con.ts.dt.floor("D"))
            .groupby(["repo_name", "day"]).size().rename("rd_new_people")
            .reset_index())
    rd = rd.merge(newp, on=["repo_name", "day"], how="left")
    for name, frame in (("rd_prs", g.pull_events), ("rd_issues", g.issues)):
        if len(frame):
            c = (frame.assign(day=frame.ts.dt.floor("D"))
                 .groupby(["repo_name", "day"]).size().rename(name)
                 .reset_index())
            rd = rd.merge(c, on=["repo_name", "day"], how="left")
        else:
            rd[name] = 0
    for c in ("rd_new_people", "rd_prs", "rd_issues"):
        rd[c] = rd[c].fillna(0).astype(int)
    rd["ts"] = rd.day + pd.Timedelta("1D")
    rd["rd_key"] = rd.repo_name + "@" + rd.day.dt.strftime("%Y%m%d")
    return rd.drop(columns=["day"])


def repo_people(g: D.GH) -> pd.DataFrame:
    """One row per (repo, actor) at first contact."""
    ev = g.events.sort_values("ts")
    rp = (ev.drop_duplicates(["repo_name", "actor_login"])
          [["repo_name", "actor_login", "type", "ts"]]
          .rename(columns={"type": "rp_kind"}))
    rp["rp_key"] = rp.repo_name + "|" + rp.actor_login
    return rp.reset_index(drop=True)


# -------------------------------------------------------------------- schema
def schema():
    from relativedb import LinkDef, Schema, TableDef, ValueType
    return (Schema.new_schema()
        # The login and nothing else. Every per-actor tally counts the whole
        # week, so at an anchor inside it they are partly the future; what an
        # account is has to be read off its rows, which the bound can cut.
        .table(TableDef.new_table("actors")
               .column("login", ValueType.TEXT)
               # declared so the target can name it, never served: whether
               # another account came back is not knowable at this anchor
               .column("came_back", ValueType.NUMBER)
               .primary_key("actor_login").build())
        .table(TableDef.new_table("orgs")
               .column("org_name_len", ValueType.NUMBER)
               .primary_key("org_login").build())
        .table(TableDef.new_table("repos")
               .column("repo_name_len", ValueType.NUMBER)
               .column("junk_name", ValueType.NUMBER)
               .primary_key("repo_name").build())
        .table(TableDef.new_table("events")
               .column("type", ValueType.TEXT)
               .column("action", ValueType.TEXT)
               .column("ref_type", ValueType.TEXT)
               .column("ts", ValueType.DATETIME)
               .primary_key("event_id").time_column("ts").build())
        .table(TableDef.new_table("repo_people")
               .column("rp_kind", ValueType.TEXT)
               .column("ts", ValueType.DATETIME)
               .primary_key("rp_key").time_column("ts").build())
        .table(TableDef.new_table("repo_days")
               .column("rd_events", ValueType.NUMBER)
               .column("rd_actors", ValueType.NUMBER)
               .column("rd_new_people", ValueType.NUMBER)
               .column("rd_prs", ValueType.NUMBER)
               .column("rd_issues", ValueType.NUMBER)
               .column("ts", ValueType.DATETIME)
               .primary_key("rd_key").time_column("ts").build())
        .table(TableDef.new_table("pull_requests")
               .column("base_ref", ValueType.TEXT)
               .column("head_ref", ValueType.TEXT)
               .column("pr_number", ValueType.NUMBER)
               .column("ts", ValueType.DATETIME)
               .primary_key("pull_key").time_column("ts").build())
        .table(TableDef.new_table("pull_events")
               .column("action", ValueType.TEXT)
               .column("ts", ValueType.DATETIME)
               .primary_key("pe_id").time_column("ts").build())
        .table(TableDef.new_table("reviews")
               .column("kind", ValueType.TEXT)
               .column("state", ValueType.TEXT)
               .column("review_len", ValueType.NUMBER)
               .column("ts", ValueType.DATETIME)
               .primary_key("review_id").time_column("ts").build())
        .table(TableDef.new_table("issues")
               .column("title", ValueType.TEXT)
               .column("body_len", ValueType.NUMBER)
               .column("action", ValueType.TEXT)
               .column("author_association", ValueType.TEXT)
               .column("ts", ValueType.DATETIME)
               .primary_key("issue_key").time_column("ts").build())
        .table(TableDef.new_table("comments")
               .column("comment_len", ValueType.NUMBER)
               .column("comment_assoc", ValueType.TEXT)
               .column("ts", ValueType.DATETIME)
               .primary_key("comment_id").time_column("ts").build())
        .link(LinkDef("repos", "org_login", "orgs"))
        .link(LinkDef("events", "repo_name", "repos"))
        .link(LinkDef("events", "actor_login", "actors"))
        .link(LinkDef("repo_people", "repo_name", "repos"))
        .link(LinkDef("repo_people", "actor_login", "actors"))
        .link(LinkDef("repo_days", "repo_name", "repos"))
        .link(LinkDef("pull_requests", "repo_name", "repos"))
        .link(LinkDef("pull_requests", "actor_login", "actors"))
        .link(LinkDef("pull_events", "pull_key", "pull_requests"))
        .link(LinkDef("pull_events", "repo_name", "repos"))
        .link(LinkDef("pull_events", "actor_login", "actors"))
        .link(LinkDef("reviews", "pull_key", "pull_requests"))
        .link(LinkDef("reviews", "repo_name", "repos"))
        .link(LinkDef("reviews", "actor_login", "actors"))
        .link(LinkDef("issues", "repo_name", "repos"))
        .link(LinkDef("issues", "actor_login", "actors"))
        .link(LinkDef("comments", "issue_key", "issues"))
        .link(LinkDef("comments", "repo_name", "repos"))
        .link(LinkDef("comments", "actor_login", "actors"))
        .build())


def frames(g: D.GH, scope: set) -> dict:
    """Rows served to the engine, restricted to ``scope`` repositories.

    Wiring materializes a Python object per row, and the week holds three
    million repositories -- most of them one branch push, touched once, with
    nothing to traverse. The collaborative subgraph is where every entity's
    neighbourhood lives, and it fits in a short run.
    """
    def cut(f):
        return f[f.repo_name.isin(scope)]

    ev, rp = cut(g.events), cut(repo_people(g))
    rd, iss = cut(repo_days(g)), cut(g.issues)
    cm, pe, rv = cut(g.comments), cut(g.pull_events), cut(g.reviews)
    pr = cut(g.pulls)
    repos = g.repos[g.repos.repo_name.isin(scope)].copy()
    live = set(ev.actor_login) | set(rv.actor_login) | set(pe.actor_login)
    actors = g.actors[g.actors.actor_login.isin(live)]
    orgs = g.orgs[g.orgs.org_login.isin(set(repos.org_login))]

    # pull_requests carries the branch names; the reduced 2026 payload has
    # nothing else left on a PR
    pk = pe[pe.action == "opened"].sort_values("ts").drop_duplicates("pull_key")
    pk = pk[["pull_key", "repo_name", "actor_login", "base_ref", "head_ref",
             "ts"]].copy()
    pk["pr_number"] = pk.pull_key.str.rsplit("!", n=1).str[-1].astype(int)
    return {
        "actors": actors[["actor_login", "login"]],
        "orgs": orgs[["org_login", "org_name_len"]],
        "repos": repos[["repo_name", "org_login", "repo_name_len",
                        "junk_name"]],
        "events": ev[["event_id", "repo_name", "actor_login", "type", "action",
                      "ref_type", "ts"]],
        "repo_people": rp[["rp_key", "repo_name", "actor_login", "rp_kind",
                           "ts"]],
        "repo_days": rd[["rd_key", "repo_name", "rd_events", "rd_actors",
                         "rd_new_people", "rd_prs", "rd_issues", "ts"]],
        "pull_requests": pk,
        # Rows whose pull_key or issue_key has no row of its own are kept: the
        # feed is sampled per object, so most reviews and comments name a
        # parent that is not in the week at all. The dangling reference simply
        # finds nothing when the walk follows it, and the row still says the
        # true thing -- that this repository is being reviewed and discussed.
        "pull_events": pe[["pe_id", "pull_key", "repo_name", "actor_login",
                           "action", "ts"]],
        "reviews": rv[["review_id", "pull_key", "repo_name", "actor_login",
                       "kind", "state", "review_len", "ts"]],
        "issues": iss[["issue_key", "repo_name", "actor_login", "title",
                       "body_len", "action", "author_association", "ts"]],
        "comments": cm[["comment_id", "issue_key", "repo_name", "actor_login",
                        "comment_len", "comment_assoc", "ts"]],
    }


# ---------------------------------------------------------------------- load
def load() -> D.GH:
    t0 = time.time()
    if not CACHE.exists():
        g = D.load(hours=D.hours_for(*[f"2026-07-{d:02d}"
                                       for d in range(11, 18)]))
        with open(CACHE, "wb") as fh:
            pickle.dump(g, fh, protocol=5)
    else:
        with open(CACHE, "rb") as fh:
            g = pickle.load(fh)
    print(f"loaded in {time.time()-t0:.0f}s | " + " ".join(
        f"{k}={v:,}" for k, v in g.describe().items() if v), flush=True)
    return g


def entities(g: D.GH):
    arr = arrivals(g)
    y = came_back(g, arr)
    # the lookback must be fully observed, and so must the horizon
    hi = g.events.ts.max() - HORIZON
    lo = max(g.events.ts.min() + SILENCE,
             hi.floor("D") - pd.Timedelta(days=ENTITY_DAYS - 1))
    keep = ((arr.t0 >= lo) & (arr.t0 <= hi)
            & (arr.prior_people >= MIN_PRIOR_PEOPLE))
    ent = arr[keep].assign(y=y[keep].to_numpy()).sort_values("t0")
    return arr, y, ent.reset_index(drop=True), lo, hi


def scope_of(g: D.GH, min_actors: int = 2) -> set:
    """The collaborative subgraph: repositories more than one person touched."""
    n = g.events.groupby("repo_name").actor_login.nunique()
    return set(n[n >= min_actors].index)


# --------------------------------------------------------------------- probe
def probe(args, g=None):
    g = g or load()
    arr, y, ent, lo, hi = entities(g)
    print(f"\narrivals after {SILENCE} of silence, on a repo they do not own"
          f"  : {len(arr):,}")
    print(f"window                                          : "
          f"{lo:%m-%d %H:%M} .. {hi:%m-%d %H:%M}")
    print(f"of those, the repo already had >={MIN_PRIOR_PEOPLE} people     : "
          f"{len(ent):,}")
    print(f"came back within {HORIZON}                      : "
          f"{ent.y.mean():.1%}")
    by = (ent.assign(day=ent.t0.dt.floor("D")).groupby("day")
          .agg(n=("y", "size"), rate=("y", "mean")))
    print("  by day: " + ", ".join(f"{d:%m-%d} {r.n} at {r.rate:.1%}"
                                   for d, r in by.iterrows()))
    print(f"  arrival event: {ent.type.value_counts().head(5).to_dict()}")

    sc = scope_of(g, args.min_actors)
    fr = frames(g, sc)
    print(f"\nscope: {len(sc):,} repos touched by >={args.min_actors} people "
          f"(covers {ent.repo_name.isin(sc).mean():.1%} of entity repos)")
    print("  " + "  ".join(f"{k}={len(v):,}" for k, v in fr.items()))
    print(f"  total rows to wire: {sum(len(v) for v in fr.values()):,}")

    e = ent.iloc[len(ent) // 2]
    print(f"\nwhat is reachable from {e.actor_login} at arrival "
          f"({e.t0:%m-%d %H:%M}, repo {e.repo_name}):")
    print(f"  hop 1  its own rows      "
          f"{int((fr['events'].actor_login == e.actor_login).sum()):5} events "
          f"in the week, 1 at the anchor")
    for name, f in (("repo's events", fr["events"]),
                    ("repo's people", fr["repo_people"]),
                    ("repo's days", fr["repo_days"]),
                    ("repo's PRs", fr["pull_requests"]),
                    ("repo's issues", fr["issues"])):
        s = f[f.repo_name == e.repo_name]
        print(f"  hop 3  {name:18} {int((s.ts <= e.t0).sum()):5} before the "
              f"anchor ({len(s):5} in the week)")
    return g, arr, y, ent


# ------------------------------------------------------------------ policies
def policy(hops: int, cells: int, cohort: int):
    from relativedb import ContextPolicy
    # The reference implementation's eval geometry (scripts/eval.py defaults):
    # ctx_size 8192, bfs_width 32, prefer_latest. Uniform width, so max_hops is
    # the only thing the ablation varies.
    return ContextPolicy(max_context_cells=cells, bfs_width=32, max_hops=hops,
                         cohort_size=cohort, prefer_latest=True)


def build(g, scope, verbose=True):
    from harness import datasets as H
    sch = schema()
    fr = frames(g, scope)
    t0 = time.time()
    wir = H._wire(sch, fr)
    if verbose:
        print(f"wired {sum(len(f) for f in fr.values()):,} rows across "
              f"{len(fr)} tables in {time.time()-t0:.0f}s", flush=True)
    return sch, fr, wir


# ----------------------------------------------------------------- preflight
def preflight(args):
    """Everything checkable before paying for a run."""
    import warnings

    from relativedb import Engine, ExecutionInput, SamplerMode
    from relativedb.model import DEFAULT_CLASSIFICATION_MODEL_URI, ModelConfig
    from relativedb.relql.ast import TaskType
    from relativedb.rt_native import (RT_DEVICE_MPS, RtNativeBackend, load_lib,
                                      resolve_model_path)

    warnings.filterwarnings("ignore")
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ""), flush=True)
        if not ok:
            fails.append(name)

    print("== checkpoint and encoder ==")
    cfg = ModelConfig()
    uri = cfg.model_uri_for(TaskType.BINARY_CLASSIFICATION)
    path = resolve_model_path(uri)
    check("classification checkpoint is RT-J",
          uri == DEFAULT_CLASSIFICATION_MODEL_URI and "rt-j" in uri, uri)
    meta = Path(path).parent / "config.json"
    ck = json.loads(meta.read_text()) if meta.exists() else {}
    check("checkpoint encoder matches the pinned one",
          ck.get("embedding_model", cfg.embedding_model) == cfg.embedding_model,
          f"{ck.get('embedding_model', '?')} vs {cfg.embedding_model}")
    check("d_text is 384 (MiniLM)", cfg.d_text == 384, str(cfg.d_text))
    if ck:
        print(f"       blocks={ck.get('num_blocks')} d_model={ck.get('d_model')}"
              f" heads={ck.get('num_heads')} task_type={ck.get('task_type')}")
    check("Metal device is available",
          load_lib().device_available(RT_DEVICE_MPS))

    print("\n== context geometry vs the reference eval defaults ==")
    p = policy(max(args.depths), args.cells, args.cohort)
    check("cell budget matches reference ctx_size", p.max_context_cells == 8192,
          str(p.max_context_cells))
    check("width matches reference bfs_width", p.fanout_at(0) == 32,
          str(p.fanout_at(0)))
    check("width is uniform, so max_hops is the only thing varying",
          p.fanouts is None)
    check("prefer_latest matches reference", p.prefer_latest is True)
    check("cohort seeding is on -- the reference fills the rest of the window "
          "with a global sample", p.cohort_size > 0, str(p.cohort_size))

    print("\n== data ==")
    g, arr, y, ent = probe(args)
    check("both classes present", 0.15 < ent.y.mean() < 0.85,
          f"{ent.y.mean():.1%} came back")
    check("enough labelled entities", len(ent) >= 2000, f"{len(ent):,}")
    check("the population is selected on pre-arrival state only",
          "prior_people" in ent.columns and int(ent.prior_people.min())
          >= MIN_PRIOR_PEOPLE,
          f"min prior_people {int(ent.prior_people.min())}")
    rd = repo_days(g)
    day = pd.to_datetime(rd.rd_key.str.rsplit("@", n=1).str[-1],
                         format="%Y%m%d", utc=True)
    check("repo_days is stamped after the day it covers",
          (rd.ts - day).min() >= pd.Timedelta("24h"),
          f"earliest stamp is day + {(rd.ts - day).min()}")

    sch, fr, wir = build(g, scope_of(g, args.min_actors))
    check("the target column is never served",
          "came_back" not in set(fr["actors"].columns),
          ", ".join(sorted(fr["actors"].columns)))
    banned = {"n_events", "n_repos", "n_types", "n_stars", "n_pulls",
              "n_issues", "n_comments", "n_orgs", "login_family", "push_share",
              "repo_events", "repo_actors", "org_repos", "org_events"}
    check("no window-spanning tally survives on a static table",
          not (banned & (set(fr["actors"].columns) | set(fr["repos"].columns)
                         | set(fr["orgs"].columns))))

    print("\n== traversal ==")
    row = ent.iloc[len(ent) // 2]
    anchor = row.t0.to_pydatetime()
    seed_only = {}
    for h in args.depths:
        eng = Engine(sch, wir, sampler_mode=SamplerMode.CSC,
                     context_policy=policy(h, args.cells, 0))
        ctx = eng.assemble_context("actors", row.actor_login, anchor)
        by = pd.Series([r.table for r in ctx.rows]).value_counts()
        seed_only[h] = {r.key for r in ctx.rows}
        print(f"  hops={h}: {len(ctx.rows):4} rows {ctx.cell_count:5} cells | "
              + " ".join(f"{k}={v}" for k, v in by.items()), flush=True)
    deep = max(args.depths)
    check("depth adds reach", len(seed_only[deep]) > len(seed_only[1]),
          f"{len(seed_only[1])} rows at 1 hop -> {len(seed_only[deep])} "
          f"at {deep}")
    check("the walk reaches the repository's own history",
          any(k[0] in ("repo_people", "repo_days")
              for k in seed_only[deep]),
          ", ".join(sorted({k[0] for k in seed_only[deep]})))
    check("the walk reaches other accounts",
          sum(1 for k in seed_only[deep] if k[0] == "actors") > 1,
          f"{sum(1 for k in seed_only[deep] if k[0] == 'actors')} actor rows")

    # The cohort shares the cell budget with the seed's own neighbourhood and
    # is expanded in the same hop loop. If it crowds the seed out then the
    # depth is nominal: the policy says four hops, the context holds one.
    eng = Engine(sch, wir, sampler_mode=SamplerMode.CSC,
                 context_policy=policy(deep, args.cells, args.cohort))
    full = eng.assemble_context("actors", row.actor_login, anchor)
    keys = {r.key for r in full.rows}
    kept = len(seed_only[deep] & keys) / max(1, len(seed_only[deep]))
    check("the cohort does not crowd out the seed's neighbourhood", kept >= 0.6,
          f"{kept:.0%} of the seed-only context survives alongside "
          f"{args.cohort} cohort rows ({len(full.rows)} rows, "
          f"{full.cell_count} cells)")

    print("\n== leakage ==")
    late = [r.key for r in full.rows
            if r.timestamp is not None and r.timestamp > row.t0]
    check("no row in the context postdates the anchor", not late,
          f"{len(late)} rows after {row.t0}")
    own = [r for r in full.rows if r.key == ("actors", row.actor_login)]
    check("the entity's own row carries no answer",
          bool(own) and "came_back" not in own[0].cells,
          ", ".join(sorted(own[0].cells)) if own else "seed missing")
    mine = [r for r in full.rows if r.table == "events"
            and r.parents.get("actor_login") == row.actor_login]
    check("the account's own future events are not in its context",
          all(r.timestamp <= row.t0 for r in mine),
          f"{len(mine)} own events, latest {max((r.timestamp for r in mine), default=None)}")

    print("\n== cost ==")
    b = RtNativeBackend(schema=sch, wiring=wir)
    eng2 = Engine(sch, wir, sampler_mode=SamplerMode.CSC,
                  context_policy=policy(deep, args.cells, args.cohort),
                  model_backend=b)
    sample = ent.sample(n=4, random_state=0)
    t0 = time.perf_counter()
    probs = []
    for r in sample.itertuples():
        out = eng2.execute(ExecutionInput(query=QUERY,
                                          anchor_time=r.t0.to_pydatetime(),
                                          params={"ids": [r.actor_login]}))
        probs += [p.probability for p in out.predictions]
    per = (time.perf_counter() - t0) / len(sample)
    check("every entity scores", len(probs) == len(sample), str(probs))
    check("scores are not degenerate", len(set(np.round(probs, 6))) > 1,
          str(np.round(probs, 4)))
    n = (args.train + args.test) * len(args.depths)
    print(f"  {per:.2f}s per row at the deepest setting -> under "
          f"{per*n/60:.0f} min for {args.train}+{args.test} rows across "
          f"{len(args.depths)} depths (shallow hops cost far less)", flush=True)

    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 0 if not fails else 1


# ----------------------------------------------------------------------- run
def run(args):
    import warnings

    from relativedb import Engine, ExecutionInput, SamplerMode
    from relativedb.rt_native import RtNativeBackend
    from sklearn.metrics import roc_auc_score

    warnings.filterwarnings("ignore")
    g = load()
    arr, y, ent, lo, hi = entities(g)
    sch, fr, wir = build(g, scope_of(g, args.min_actors))
    ent = ent[ent.repo_name.isin(set(fr["repos"].repo_name))]

    # A temporal split: fit on the earlier arrivals, score the later ones. The
    # classes stay at their natural rate; balancing the test set would make the
    # AUC a different number than the one this task has.
    cut = ent.t0.quantile(args.split)
    early, late = ent[ent.t0 <= cut], ent[ent.t0 > cut]
    tr = early.sample(n=min(args.train, len(early)),
                      random_state=1).sort_values("t0")
    te = late.sample(n=min(args.test, len(late)),
                     random_state=2).sort_values("t0")
    print(f"\ntrain {len(tr)} ({tr.y.mean():.1%} came back, "
          f"{tr.t0.min():%m-%d %H:%M}..{tr.t0.max():%m-%d %H:%M}) | "
          f"test {len(te)} ({te.y.mean():.1%} came back, "
          f"{te.t0.min():%m-%d %H:%M}..{te.t0.max():%m-%d %H:%M})", flush=True)

    # Every row carries its own anchor -- the moment the account arrived. A
    # shared anchor would either hide the entity, whose arrival postdates the
    # cut, or show it what it did afterwards.
    anchors = sorted({r.t0.to_pydatetime() for r in tr.itertuples()})
    labels = {(r.actor_login, r.t0.to_pydatetime()): float(r.y)
              for r in tr.itertuples()}

    results = []
    for hops in args.depths:
        pol = policy(hops, args.cells, args.cohort)
        eng = Engine(sch, wir, sampler_mode=SamplerMode.CSC, context_policy=pol,
                     model_backend=RtNativeBackend(schema=sch, wiring=wir))
        t0 = time.perf_counter()
        head = eng.finetune(QUERY, anchors, entity_ids=list(tr.actor_login),
                            params={"ids": list(tr.actor_login)},
                            labels=labels, epochs=args.epochs,
                            learning_rate=1e-2)
        fit_s = time.perf_counter() - t0

        eng2 = Engine(sch, wir, sampler_mode=SamplerMode.CSC, context_policy=pol,
                      model_backend=RtNativeBackend(schema=sch, wiring=wir,
                                                    head=head))
        t0, preds, cells = time.perf_counter(), [], []
        for r in te.itertuples():
            out = eng2.execute(ExecutionInput(query=QUERY,
                                              anchor_time=r.t0.to_pydatetime(),
                                              params={"ids": [r.actor_login]}))
            p = next((x.probability for x in out.predictions), np.nan)
            preds.append(float(p) if p is not None else np.nan)
        score_s = time.perf_counter() - t0
        preds = np.asarray(preds, float)
        ok = np.isfinite(preds)
        auc = float(roc_auc_score(te.y.to_numpy()[ok], preds[ok]))
        rec = {"hops": hops, "auc": auc, "examples": int(head.n_examples),
               "loss_before": float(head.initial_loss),
               "loss_after": float(head.final_loss),
               "scored": int(ok.sum()),
               "distinct_scores": int(len(np.unique(np.round(preds[ok], 5)))),
               "fit_s": round(fit_s, 1), "score_s": round(score_s, 1)}
        results.append(rec)
        print(f"  hops={hops}  AUC={auc:.4f}  loss {head.initial_loss:.3f}->"
              f"{head.final_loss:.4f}  distinct={rec['distinct_scores']}  "
              f"({fit_s:.0f}s fit + {score_s:.0f}s score)", flush=True)

    base = float(te.y.mean())
    print(f"\ndepth ablation -- everything else fixed (cells={args.cells}, "
          f"bfs_width=32, cohort={args.cohort}, {len(tr)} train, {len(te)} "
          f"test, {base:.1%} came back)")
    print(f"  {'hops':>5} {'AUC':>8} {'loss after':>11}   what the walk reaches")
    reach = {0: "the login, and a cohort of other arrivals",
             1: "+ the account's own arrival event",
             2: "+ the repository it arrived at",
             3: "+ that repository's events, people, days, PRs, issues",
             4: "+ the other accounts on it"}
    for r in results:
        print(f"  {r['hops']:>5} {r['auc']:>8.4f} {r['loss_after']:>11.4f}   "
              f"{reach.get(r['hops'], '')}")
    OUT.write_text(json.dumps(
        {"task": f"new account acts again within {HORIZON}", "query": QUERY,
         "min_prior_people": MIN_PRIOR_PEOPLE, "min_actors": args.min_actors,
         "epochs": args.epochs, "cells": args.cells, "cohort": args.cohort,
         "train": len(tr), "test": len(te), "positive_rate": base,
         "results": results}, indent=2))
    print(f"\nwrote {OUT}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["probe", "preflight", "run"])
    ap.add_argument("--min-actors", type=int, default=2,
                    help="repositories below this many distinct actors are "
                         "left out of the wired subgraph")
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cells", type=int, default=8192)
    ap.add_argument("--cohort", type=int, default=32)
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--split", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    raise SystemExit({"probe": lambda a: (probe(a), 0)[1],
                      "preflight": preflight, "run": run}[args.mode](args))


if __name__ == "__main__":
    main()
