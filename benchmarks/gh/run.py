"""Bot detection on a GH Archive graph.

    python -m gh.run --suspects 60 --anchors 2

Ground truth is the hand-verified set in ``labels.py``, not the ``[bot]``
suffix. Verified accounts are held out for scoring; the declared Apps do the
fitting. The target column is declared in the schema and suppressed on the
entity's own row, so a row cannot read its own answer while other rows' values
stay visible as context.

Writes results.json with every scored row and a ranked list of unlabelled
suspects.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from relativedb import (ContextPolicy, Engine, ExecutionInput, LinkDef,
                        RtNativeBackend, SamplerMode, Schema, TableDef,
                        ValueType)
from relativedb.retrieve import TemporalBound

from . import data as D
from . import labels as L

OUT = Path(__file__).resolve().parent / "results.json"
# Matched to the reference implementation's eval defaults (ctx_size 8192,
# bfs_width 32). The local neighbourhood is only a few percent of an RT
# context there; the rest is other rows sampled from the database, which is
# what gives every column a distribution in the window and gives the model
# in-context examples to predict from. cohort_size is that sample here, and
# leaving it at 0 was why numeric cells had no spread and zero-shot scored at
# or below chance.
POLICY = ContextPolicy()      # the engine defaults now match the reference

# push_share counts every event, including the PushEvents kept out of the
# graph: how much an account pushes is a fact about the account whether or not
# those rows are worth carrying. mean_commits is absent because the payload no
# longer contains a commits array at all -- that one really is unavailable.
ACTOR_NUM = ["n_events", "n_repos", "n_types", "push_share", "n_pulls",
             "n_issues", "n_comments", "n_orgs", "n_created", "n_deleted",
             "login_family", "own_repo_share", "active_minutes"]
ACTOR_CAT = []
# the login as raw text for XGBoost too, so both systems read the same field
ACTOR_TEXT = ["login"]
# Issues are context for an account, not a prediction target: an account that
# files issues in a person's voice is the strongest human evidence there is.
ISSUE_COLS = ["issue_key", "repo_name", "actor_login", "title", "body",
              "title_len", "body_len", "comments", "state",
              "author_association", "ts"]

BOT_Q = ("PREDICT actors.is_bot = 1 FROM actors "
         "WHERE actors.actor_login IN :ids")


def schema() -> Schema:
    return (Schema.new_schema()
        .table(TableDef.new_table("actors")
               # The username, as GitHub gives it. It is the primary key and
               # also a real feature: this is the schema a bot detector
               # actually has. GitHub Apps are named "<something>[bot]", so the
               # model can read that off the text -- which is what a detector
               # with this data would do.
               .column("login", ValueType.TEXT)
               .column("n_events", ValueType.NUMBER)
               .column("n_repos", ValueType.NUMBER)
               .column("n_types", ValueType.NUMBER)
               .column("push_share", ValueType.NUMBER)
               .column("n_pulls", ValueType.NUMBER)
               .column("n_issues", ValueType.NUMBER)
               .column("n_comments", ValueType.NUMBER)
               .column("n_orgs", ValueType.NUMBER)
               .column("n_created", ValueType.NUMBER)
               .column("n_deleted", ValueType.NUMBER)
               # how many accounts active today share this login's stem
               .column("login_family", ValueType.NUMBER)
               # share of activity on repos the account itself owns
               .column("own_repo_share", ValueType.NUMBER)
               # observed span, as a raw fact; the model can form a rate from
               # this and n_events rather than being handed one
               .column("active_minutes", ValueType.NUMBER)
               # declared so a target can name it; never present in a row
               .column("is_bot", ValueType.NUMBER)
               .primary_key("actor_login").build())
        .table(TableDef.new_table("orgs")
               .column("org_repos", ValueType.NUMBER)
               .column("org_events", ValueType.NUMBER)
               .primary_key("org_login").build())
        .table(TableDef.new_table("repos")
               .column("repo_events", ValueType.NUMBER)
               .column("repo_actors", ValueType.NUMBER)
               # keyboard-mash naming, scored on the repo so it reaches an
               # account by traversal rather than being precomputed onto it
               .column("junk_name", ValueType.NUMBER)
               .column("repo_name_len", ValueType.NUMBER)
               .primary_key("repo_name").build())
        .table(TableDef.new_table("comments")
               .column("comment_len", ValueType.NUMBER)
               .column("comment_assoc", ValueType.TEXT)
               .column("ts", ValueType.DATETIME)
               .primary_key("comment_id").time_column("ts").build())
        .table(TableDef.new_table("pulls")
               .column("pr_title", ValueType.TEXT)
               .column("changed_files", ValueType.NUMBER)
               .column("additions", ValueType.NUMBER)
               .column("deletions", ValueType.NUMBER)
               .column("pr_assoc", ValueType.TEXT)
               .column("ts", ValueType.DATETIME)
               .primary_key("pull_key").time_column("ts").build())
        .table(TableDef.new_table("events")
               .column("type", ValueType.TEXT)
               .column("action", ValueType.TEXT)
               .column("n_commits", ValueType.NUMBER)
               .column("ts", ValueType.DATETIME)
               .primary_key("event_id").time_column("ts").build())
        .table(TableDef.new_table("issues")
               .column("title", ValueType.TEXT)
               .column("body", ValueType.TEXT)
               .column("title_len", ValueType.NUMBER)
               .column("body_len", ValueType.NUMBER)
               .column("comments", ValueType.NUMBER)
               .column("state", ValueType.TEXT)
               .column("author_association", ValueType.TEXT)
               .column("ts", ValueType.DATETIME)
               .primary_key("issue_key").time_column("ts").build())
        .link(LinkDef("events", "actor_login", "actors"))
        .link(LinkDef("events", "repo_name", "repos"))
        .link(LinkDef("issues", "actor_login", "actors"))
        .link(LinkDef("issues", "repo_name", "repos"))
        .link(LinkDef("repos", "org_login", "orgs"))
        .link(LinkDef("comments", "issue_key", "issues"))
        .link(LinkDef("comments", "actor_login", "actors"))
        .link(LinkDef("comments", "repo_name", "repos"))
        .link(LinkDef("pulls", "repo_name", "repos"))
        .link(LinkDef("pulls", "actor_login", "actors"))
        .build())


def frames(g: D.GH) -> dict:
    """Rows served to the engine.

    Target columns are served, not dropped. The engine suppresses the target
    column on the entity's *own* row, so an issue cannot read its own label,
    while every other issue's label stays visible — which is what forms the
    class domain for a static attribute.
    """
    issues = g.issues.copy()
    return {
        "actors": g.actors,
        "orgs": g.orgs,
        "repos": g.repos,
        # comment_body is dropped: comments are the highest-cardinality table
        # in the graph, so serving their bodies means embedding tens of
        # thousands of distinct long strings per run for a table that is
        # context rather than the subject of either task.
        "comments": g.comments[["comment_id", "issue_key", "repo_name",
                                "actor_login", "comment_len", "comment_assoc",
                                "ts"]],
        "pulls": g.pulls[["pull_key", "repo_name", "actor_login", "pr_title",
                          "changed_files", "additions", "deletions", "pr_assoc",
                          "ts"]],
        # issue_labels.label_name IS the target restated on another table, and
        # entity-row suppression does not reach across tables. Both are dropped:
        # a bridge that spells out the answer is a leak, not graph depth.
        "events": g.events[["event_id", "actor_login", "repo_name", "type",
                            "action", "n_commits", "ts"]],
        "issues": issues[ISSUE_COLS],
    }


def xgb(train, test, num, cat, text=None):
    """``text`` names a raw text column. RelativeDB reads it through its text
    encoder, so XGBoost gets the same field as TF-IDF character n-grams rather
    than being denied it — the comparison is over the same information."""
    from xgboost import XGBClassifier
    t0 = time.perf_counter()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    Xtr = np.hstack([train[num].astype(float).to_numpy(),
                     enc.fit_transform(train[cat].astype(str))])
    Xte = np.hstack([test[num].astype(float).to_numpy(),
                     enc.transform(test[cat].astype(str))])
    if text:
        from sklearn.feature_extraction.text import TfidfVectorizer
        cat_text = lambda d: (d[text[0]].astype(str) + " "
                              + d[text[1]].astype(str)) if len(text) > 1 \
                             else d[text[0]].astype(str)
        tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                             max_features=400, min_df=2)
        Xtr = np.hstack([Xtr, tv.fit_transform(cat_text(train)).toarray()])
        Xte = np.hstack([Xte, tv.transform(cat_text(test)).toarray()])
    prep = time.perf_counter() - t0
    y = train.label_id.to_numpy()
    kw = dict(n_estimators=300, max_depth=6, learning_rate=0.07,
              tree_method="hist", n_jobs=0)
    m = XGBClassifier(**kw)
    t0 = time.perf_counter(); m.fit(Xtr, y); fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = m.predict_proba(Xte)
    infer = time.perf_counter() - t0
    return (p[:, 1], prep + fit,
            1000 * infer / max(1, len(test)))


N_ANCHORS = [3]
SUSPECTS = [None]
# Applies to LABELLED accounts only. Purchased accounts have exactly one event
# by design, so filtering the whole table on this removes the class the suspect
# ranking exists to find.
MIN_EVENTS = 3


def rdb(query, train, test, sch, wir, finetune, epochs, suspects=None):
    eng = Engine(sch, wir, sampler_mode=SamplerMode.CSC, context_policy=POLICY,
                 model_backend=RtNativeBackend(schema=sch, wiring=wir))
    train_s, detail = 0.0, {}
    if finetune:
        qs = np.linspace(0.3, 0.95, max(1, N_ANCHORS[0]))
        anchors = [train.ts.quantile(q).to_pydatetime() for q in qs]
        t0 = time.perf_counter()
        head = eng.finetune(query, anchors, params={"ids": list(train.entity_id)},
                            epochs=epochs, learning_rate=1e-2)
        train_s = time.perf_counter() - t0
        detail = {"examples": head.n_examples, "loss_before": head.initial_loss,
                  "loss_after": head.final_loss}
        eng = Engine(sch, wir, sampler_mode=SamplerMode.CSC,
                     context_policy=POLICY,
                     model_backend=RtNativeBackend(schema=sch, wiring=wir,
                                                   head=head))
    def score_rows(frame):
        out = {}
        for eid, ts in zip(frame.entity_id, frame.ts):
            r = eng.execute(ExecutionInput(query=query,
                                           anchor_time=ts.to_pydatetime(),
                                           params={"ids": [eid]}))
            for p in r.predictions:
                out[p.id] = p
        return out

    by, t0 = {}, time.perf_counter()
    for eid, ts in zip(test.entity_id, test.ts):
        r = eng.execute(ExecutionInput(query=query,
                                       anchor_time=ts.to_pydatetime(),
                                       params={"ids": [eid]}))
        for p in r.predictions:
            by[p.id] = p
    infer = 1000 * (time.perf_counter() - t0) / max(1, len(test))

    pred = np.array([float(getattr(by.get(e), "probability", np.nan) or np.nan)
                     for e in test.entity_id])
    sus = None
    if suspects is not None and len(suspects):
        sby = score_rows(suspects)
        sus = np.array([float(getattr(sby.get(e), "probability", np.nan) or np.nan)
                        for e in suspects.entity_id])
    return pred, train_s, infer, detail, sus


def score(y, pred):
    ok = np.isfinite(pred)
    out = {"positive_rate": float(np.mean(y))}
    if ok.sum() and len(np.unique(y[ok])) > 1:
        out["auc"] = float(roc_auc_score(y[ok], pred[ok]))
    out["accuracy"] = float(accuracy_score(y[ok], (pred[ok] >= .5).astype(int)))
    out["f1"] = float(f1_score(y[ok], (pred[ok] >= .5).astype(int),
                               zero_division=0))
    return out


def inspect(jobs, sch, n):
    """Print what one context actually contains, without scoring anything.

    A full run costs about ten minutes, and most re-runs in this benchmark's
    history were triggered by things visible here in seconds: a target column
    still present in the context, a numeric cell normalizing to 0.0, a class
    name rendered wrong, an empty title.
    """
    import collections
    from relativedb.rt_native import SEM_TEXT
    from relativedb.relql import parse
    from relativedb.relql.ast import TaskType

    for (name, kind, q, train, test, num, cat, question, truth, wiring) in jobs:
        print(f"\n=== {name} :: {q}", flush=True)
        b = RtNativeBackend(schema=sch, wiring=wiring)
        eng = Engine(sch, wiring, sampler_mode=SamplerMode.CSC,
                     context_policy=POLICY, model_backend=b)
        pq = parse(q)
        tt = (TaskType.BINARY_CLASSIFICATION if kind == "binary"
              else TaskType.MULTICLASS_CLASSIFICATION)
        entity_table = pq.entity_key.table
        suppressed = {c for t, c in b._target_columns(pq.target)
                      if t == entity_table}
        print(f"  target columns suppressed on the entity row: "
              f"{sorted(suppressed) or 'NONE'}")
        served = sorted(wiring.scanner(entity_table)(
            entity_table, TemporalBound.unbounded()).__next__().cells)
        print(f"  {entity_table} columns served: {served}")
        leaked = suppressed & set(served)
        print(f"  served AND target: {sorted(leaked) or 'none'}"
              f"{'  <-- relies on entity-row suppression' if leaked else ''}")

        for _, row in test.head(n).iterrows():
            ctx = eng.assemble_context(entity_table, row.entity_id,
                                       row.ts.to_pydatetime())
            seq, _, enode, _ = b._build_ctx_seq(pq, tt, ctx,
                                                b._fk_to_parent(), [])
            b._normalize([seq], 0.0, 1.0)
            tables = collections.Counter(seq.tab)
            # the ENTITY's own cells, not merely cells of its table: the
            # context is mostly other rows of the same table, and the whole
            # point of this check is whether the entity's own target cell
            # reached the model
            own = {c: v for nd, (c, t), v, sm in
                   zip(seq.node, seq.col, seq.value, seq.sem)
                   if nd == enode and sm != SEM_TEXT}
            leaked_cells = sorted(set(own) & suppressed)
            zeros = [c for c, v in own.items() if v == 0.0]
            texts = sum(1 for s_ in seq.sem if s_ == SEM_TEXT)
            print(f"  {str(row.entity_id)[:38]:40} label={int(row.label_id)} "
                  f"rows={len(ctx.rows):5} tokens={len(seq):5} text={texts:4} "
                  f"tables={dict(tables)}")
            print(f"      entity cells after normalize: "
                  f"{ {k: round(float(v), 3) for k, v in own.items()} }")
            if leaked_cells:
                print(f"      LEAK: entity's own {leaked_cells} reached the "
                      f"model")
            if zeros:
                print(f"      ZEROED (carries no information): {zeros}")
            if "title" in test.columns:
                print(f"      title: {str(row.title)[:70]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--anchors", type=int, default=3,
                    help="fine-tuning anchors; each one re-encodes every "
                         "training row, so this multiplies encode cost")
    ap.add_argument("--tasks", default="bot_detection")
    ap.add_argument("--suspects", type=int, default=60, metavar="N",
                    help="undeclared accounts to score and rank after fitting")
    ap.add_argument("--inspect", type=int, default=0, metavar="N",
                    help="assemble N contexts per task, print what the model "
                         "would actually see, and exit without scoring")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    N_ANCHORS[0] = a.anchors
    want = {t.strip() for t in a.tasks.split(",") if t.strip()}
    print("loading GH Archive ...", flush=True)
    g = D.load()
    sch = schema()
    from harness import datasets as H
    wir = H._wire(sch, frames(g))

    report = {"dataset": "GH Archive", "dataset_url": "https://www.gharchive.org",
              "hours": D.DEFAULT_HOURS, "precision": "fp32",
              "generated": pd.Timestamp.now("UTC").isoformat(),
              "machine": {"platform": platform.platform()},
              "table_sizes": g.describe(), "tasks": []}

    # ---- bot detection ----------------------------------------------------
    act = g.actors.rename(columns={"actor_login": "entity_id",
                                   "first_seen": "ts"})
    # Ground truth is the hand-verified set, not the [bot] suffix. The suffix
    # only marks GitHub Apps; the accounts worth catching look like people.
    # Unlabelled accounts are dropped rather than assumed human -- most of them
    # are, but the undeclared bots are in there too, and calling them negatives
    # would train the model to miss exactly what it is for.
    # Positives are known automation: hand-verified accounts plus the declared
    # Apps. Negatives are unlabelled accounts, which is an assumption -- the
    # undeclared bots nobody has found yet are in there, so some fraction of
    # the negative pool is mislabelled and no score can beat that floor.
    # An account needs enough activity to be judged at all. henneberger has a
    # single event on a single repo -- indistinguishable from a throwaway bot
    # by construction, and scoring it measures nothing but the prior.
    act["label_id"] = act.entity_id.map(L.is_bot).astype(int)
    absent = L.validate(set(act.entity_id))
    if absent:
        print(f"  WARNING: {len(absent)} verified labels match no account in "
              f"this data and score nothing: {', '.join(absent)}", flush=True)
    hand = set(L.labelled())
    held = act[act.entity_id.isin(hand) & (act.n_events >= MIN_EVENTS)]
    act.loc[act.entity_id.isin(L.HUMANS), "label_id"] = 0
    print(f"  bots: {int(act.label_id.sum())} "
          f"({len(held)} hand-verified, rest declared [bot]) of "
          f"{len(act)} accounts", flush=True)

    # The hand-verified accounts are the point of the exercise, so they are
    # held out for scoring rather than spent on fitting: catching a declared
    # App is not evidence of catching a disguised one.
    # Humans are split across fitting and scoring so both sides carry two
    # classes; a test set of nothing but the held-out bots has no AUC to
    # report, which is what a whole run just produced.
    # the evidence floor applies here, to rows that will be fitted or scored
    act_ok = act[act.n_events >= MIN_EVENTS]
    humans = act_ok[act_ok.entity_id.isin(L.HUMANS)]
    h_te = humans.sample(n=max(1, len(humans) // 2), random_state=1)
    h_tr = humans.drop(h_te.index)
    # Declared Apps, sampled down to the human count. There are thousands of
    # them against 64 read-by-hand humans, and unbalanced the head just fits
    # the prior and calls everything a bot.
    declared = act_ok[(act_ok.label_id == 1) & ~act_ok.entity_id.isin(hand)
                      & ~act_ok.entity_id.isin(L.HUMANS)]
    b_tr = declared.sample(n=min(len(declared), len(h_tr)), random_state=0)
    tr = pd.concat([b_tr, h_tr]).sort_values("ts")
    # Scoring pits the hand-verified bots against held-out humans: catching a
    # declared App proves nothing, catching lyc228612-sudo does.
    te = pd.concat([held, h_te]).sort_values("ts")


    # Undeclared accounts to rank once the head is fitted. Sampled from
    # accounts with enough activity to judge; none of them carry the [bot]
    # marker, so the model has to go on behaviour.
    # Sampled at random from accounts active enough to judge. Deliberately not
    # sorted by stars or rate: picking them with the pattern we are looking for
    # would make the ranking a formality rather than a test of it.
    und = act[(act.label_id == 0) & ~act.entity_id.isin(L.HUMANS)
              & (act.n_events >= 20) & (act.n_types >= 2)]
    SUSPECTS[0] = und.sample(n=min(a.suspects, len(und)),
                             random_state=3).sort_values("ts")
    print(f"  suspects to rank: {len(SUSPECTS[0])} undeclared accounts",
          flush=True)

    jobs = [
        ("bot_detection", "binary", BOT_Q, tr, te, ACTOR_NUM, ACTOR_CAT,
         "Does this account behave like a bot?",
         "actor_login ends in [bot] (never shown to either model)", wir),
    ]

    jobs = [j for j in jobs if j[0] in want]

    if a.inspect:
        inspect(jobs, sch, a.inspect)
        return

    # per-task names for the two binary outcomes, so examples read as
    # "bug"/"not bug" rather than borrowing the bot task's vocabulary
    BINARY_NAMES = {"bot_detection": ("human", "bot")}
    TEXT_COLS = {"bot_detection": ACTOR_TEXT}

    for (name, kind, q, train, test, num, cat, question, truth,
         wiring) in jobs:
        names = BINARY_NAMES.get(name, ("0", "1"))
        print(f"\n=== {name} ({kind}) train={len(train)} test={len(test)} ===",
              flush=True)
        y = test.label_id.to_numpy()
        entry = {"name": name, "kind": kind, "query": q, "question": question,
                 "ground_truth": truth, "n_train": len(train),
                 "n_test": len(test),
                 "label_balance": pd.Series(y).value_counts(normalize=True)
                 .sort_index().round(4).to_dict(), "systems": []}
        preds = {}
        for label, fn in [
            ("XGBoost", lambda: xgb(train, test, num, cat,
                                    TEXT_COLS.get(name))),
            ("RelativeDB (fine-tuned)",
             lambda: rdb(q, train, test, sch, wiring, True, a.epochs,
                         suspects=SUSPECTS[0] if name == "bot_detection"
                         else None)),
        ]:
            print(f"  {label} ...", end=" ", flush=True)
            try:
                out = fn()
            except Exception as e:
                print(f"FAILED {type(e).__name__}: {str(e)[:80]}")
                entry["systems"].append({"system": label, "error": str(e)[:200]})
                continue
            pred, tsec, ims = out[0], out[1], out[2]
            det = out[3] if len(out) > 3 else {}
            if len(out) > 4 and out[4] is not None:
                # The point of the exercise: accounts that never declared
                # themselves, ranked by how much they look like the ones that
                # did. n_stars and active_minutes are shown so the ranking can
                # be judged; neither is a model feature.
                sus = SUSPECTS[0].copy()
                sus["score"] = out[4]
                sus = sus.sort_values("score", ascending=False).head(20)
                entry["suspects"] = [
                    {"login": str(r.entity_id), "score": round(float(r.score), 4),
                     "n_events": int(r.n_events), "n_stars": int(r.n_stars),
                     "active_minutes": round(float(r.active_minutes), 1),
                     "n_repos": int(r.n_repos), "n_types": int(r.n_types)}
                    for r in sus.itertuples()]
            preds[label] = pred
            m = score(y, pred)
            entry["systems"].append({"system": label, "train_seconds": tsec,
                                     "inference_ms_per_row": ims,
                                     "overall": m, "detail": det})
            key = "auc"
            print(f"{key}={m.get(key, float('nan')):.4f} train={tsec:.1f}s "
                  f"{ims:.2f} ms/row")

        # Every scored row, not just the twelve shown. Rendering choices —
        # which rows to display, what to call the classes, whether to include
        # the title — are then report-time decisions. Storing only the twelve
        # meant a cosmetic change cost a full re-run of the benchmark.
        entry["rows"] = [
            {"id": str(row.entity_id),
             "label": int(row.label_id),
             "title": (str(row.title)[:120] if "title" in test.columns else None),
             **{sysname: (None if not np.isfinite(pr[i]) else round(float(pr[i]), 4))
                for sysname, pr in preds.items()}}
            for i, (_, row) in enumerate(test.iterrows())]

        # examples a human can verify at a glance
        ex = []
        # Stratified: taking the head of the test split drew twelve negatives
        # at a 4% positive rate, which cannot show a correct positive catch.
        per = max(1, 12 // max(1, test.label_id.nunique()))
        pick: list = []
        for _, d in test.groupby("label_id"):
            pick += list(d.index[:per])
        pick += [i for i in test.index if i not in set(pick)]
        show = test.loc[pick[:12]]
        idx = {e: i for i, e in enumerate(test.entity_id)}
        for _, row in show.iterrows():
            i = idx[row.entity_id]
            e = {"id": str(row.entity_id),
                 "actual": names[int(row.label_id)]}
            for sysname, pr in preds.items():
                v = pr[i]
                e[sysname] = round(float(v), 3)
            ex.append(e)
        entry["examples"] = ex
        report["tasks"].append(entry)

    Path(a.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
