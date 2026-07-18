"""Extensive test run over the downloaded corpus.

    python benchmarks/run.py            # full run (movielens + online_retail)
    python benchmarks/run.py --quick    # fewer customers, faster

Prints a human report and writes benchmarks/results/report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.harness import audit_fixes, audit_grammar, audit_leakage  # noqa: E402
from benchmarks.harness import backtest as B                # noqa: E402
from benchmarks.harness import datasets as D                # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def _fmt(x, nd=3):
    return "  n/a" if x is None else f"{x:.{nd}f}"


def _print_binary(r: B.TaskResult):
    m = r.metrics
    print(f"\n[{r.dataset}] {r.task}  ({r.horizon})  n={r.n_entities}")
    print(f"    query: {r.query}")
    print(f"    pos_rate={_fmt(m['positive_rate'])}  AUROC={_fmt(m['auroc'])}  "
          f"PR_AUC={_fmt(m['pr_auc'])}  Brier={_fmt(m['brier'])}  "
          f"lift@10%={_fmt(m['lift_at_10pct'],2)}  distinct_scores={m['distinct_scores']}")
    for name, bm in r.baselines.items():
        # score-only baselines (recency, activity) are rank scores, not
        # probabilities — Brier is meaningless for them, so show it only for
        # the genuinely probabilistic constant baseline.
        brier = f"  Brier={_fmt(bm['brier'])}" if name == "constant_prevalence" else ""
        print(f"      vs {name:24s} AUROC={_fmt(bm['auroc'])}{brier}")
    _verdict_binary(r)
    for n in r.notes:
        print(f"    ! {n}")


def _verdict_binary(r: B.TaskResult):
    eng = r.metrics["auroc"]
    best_naive = max((b["auroc"] for b in r.baselines.values()
                      if b["auroc"] is not None), default=None)
    if eng is None:
        print("    => degenerate (single class at these anchors) — no ranking signal")
    elif best_naive is not None and eng <= best_naive + 0.005:
        print(f"    => NO LIFT over naive (engine {eng:.3f} <= naive {best_naive:.3f})")
    else:
        print(f"    => adds signal (engine {eng:.3f} > naive {best_naive})")


def _print_reg(r: B.TaskResult):
    m = r.metrics
    print(f"\n[{r.dataset}] {r.task}  ({r.horizon})  n={r.n_entities}")
    print(f"    query: {r.query}")
    print(f"    MAE={_fmt(m['mae'],2)}  RMSE={_fmt(m['rmse'],2)}  R2={_fmt(m['r2'])}  "
          f"Spearman={_fmt(m['spearman'])}  true_mean={_fmt(m['true_mean'],2)}")
    for name, bm in r.baselines.items():
        print(f"      vs {name:24s} MAE={_fmt(bm['mae'],2)}  Spearman={_fmt(bm['spearman'])}")
    eng = r.metrics["mae"]
    best = min((b["mae"] for b in r.baselines.values() if b["mae"] is not None),
               default=None)
    if eng is not None and best is not None:
        print(f"    => {'beats' if eng < best - 1e-9 else 'does NOT beat'} "
              f"best naive on MAE ({eng:.2f} vs {best:.2f})")
    for n in r.notes:
        print(f"    ! {n}")


def _print_rank(r: B.TaskResult):
    m = r.metrics
    print(f"\n[{r.dataset}] {r.task}  ({r.horizon})  n={r.n_entities}  k={m.get('k')}")
    print(f"    query: {r.query}")
    print(f"    Recall@K={_fmt(m['recall_at_k'])}  MAP@K={_fmt(m['map_at_k'])}  "
          f"NDCG@K={_fmt(m['ndcg_at_k'])}  HitRate={_fmt(m['hit_rate'])}  "
          f"listcov={_fmt(m['list_coverage'])}")
    for name, bm in r.baselines.items():
        print(f"      vs {name:24s} Recall@K={_fmt(bm['recall_at_k'])}  HitRate={_fmt(bm['hit_rate'])}")
    for n in r.notes:
        print(f"    ! {n}")


def _use_backend(ds) -> str:
    """Wire the dataset engine to the native RT-J model backend. Raises if the
    native engine or its checkpoints are unavailable — there is no model-free
    fallback scorer."""
    from relativedb.rt_native import RtNativeBackend
    ds.engine.model_backend = RtNativeBackend(schema=ds.schema)
    return "rtj-native"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--backend", choices=["rtj"], default="rtj",
                    help="scoring backend: 'rtj' (the native RT-J model — needs "
                         "librt_c + cached checkpoints). There is no model-free "
                         "fallback; a run with rtj unavailable errors out.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap entities scored per dataset (recommended, since "
                         "rtj does per-entity model inference).")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    report = {"tasks": [], "audits": {}}

    # ----- grammar audit (dataset-independent) -----
    print("=" * 78)
    print("PARSER / GRAMMAR AUDIT")
    print("=" * 78)
    g = audit_grammar.run()
    report["audits"]["grammar"] = g
    from relativedb.pql.native import native_available
    print(f"shared C++ parser (librt_c) available: {native_available()}  "
          f"(enable engine-wide with RELATIVEDB_USE_NATIVE_PARSER=1)")
    report["audits"]["native_parser_available"] = native_available()
    print(f"conformance corpus: {g['corpus_parsed']}/{g['corpus_total']} parsed")
    print(f"coverage: aggs={g['coverage']['agg_used']}")
    print(f"          ops missing from corpus={g['coverage']['op_missing']}")
    print(f"          task dist={g['coverage']['task_dist']}")
    if g["should_parse_fail"]:
        print("SHOULD-PARSE probes that were REJECTED:")
        for q, why, err in g["should_parse_fail"]:
            print(f"    ✗ ({why}) {q}\n        {err}")
    if g["should_reject_pass"]:
        print("SHOULD-REJECT probes that were ACCEPTED:")
        for q, why in g["should_reject_pass"]:
            print(f"    ✗ ({why}) {q}")
    if not g["findings"]:
        print("no grammar findings")

    # ----- datasets -----
    t0 = time.time()
    ml = D.movielens(max_users=args.limit)
    ret = D.online_retail(max_customers=args.limit or (400 if args.quick else 1500))
    print(f"\nloaded corpus in {time.time()-t0:.1f}s  "
          f"(movielens={len(ml.entity_ids)} users, retail={len(ret.entity_ids)} customers)")
    backend_label = _use_backend(ml)
    _use_backend(ret)
    print(f"scoring backend: {backend_label}")
    report["backend"] = backend_label

    tasks = []
    # Online Retail — the repeat-behavior domain: churn, count, value, ranking
    tasks += [B.churn_task(ret, "purchases", 90, 90),
              B.count_task(ret, "purchases", 90, 90),
              B.value_task(ret, "purchases", "amount", 90, 90),
              B.ranking_task(ret, "purchases", "stock_code", 90, 90, k=10)]
    # MovieLens — activity regression + the zero-repeat ranking dead-end
    tasks += [B.count_task(ml, "ratings", 60, 180),
              B.churn_task(ml, "ratings", 60, 180),
              B.ranking_task(ml, "ratings", "movie_id", 180, 365, k=10)]

    print("\n" + "=" * 78)
    print("PREDICTIVE BACKTEST  (engine vs. naive baselines, real held-out future)")
    print("=" * 78)
    for r in tasks:
        report["tasks"].append(asdict(r))
        if r.kind == "binary":
            _print_binary(r)
        elif r.kind == "regression":
            _print_reg(r)
        else:
            _print_rank(r)

    # ----- truncation-instrumentation guard -----
    print("\n" + "=" * 78)
    print("TRUNCATION INSTRUMENTATION GUARD (observability only, no prediction change)")
    print("=" * 78)
    fx = audit_fixes.run(ret)
    report["audits"]["instrumentation"] = fx
    for name, val in fx["checks"].items():
        print(f"    {name}: {val}")
    for f in fx["findings"]:
        print(f"    ! {f}")
    if not fx["findings"]:
        print("    truncation is surfaced under the default cap, silent when wide")

    # ----- leakage audits -----
    print("\n" + "=" * 78)
    print("TEMPORAL-CORRECTNESS (F24) AUDIT")
    print("=" * 78)
    for ds in (ml, ret):
        la = audit_leakage.run(ds)
        report["audits"].setdefault("leakage", {})[ds.name] = la
        print(f"[{ds.name}] contexts={la['contexts_checked']}  direct_leaks={la['direct_leaks']}  "
              f"injection_caught={la['injection_caught']}  mono_viol={la['monotonicity_violations']}")
        for f in la["findings"]:
            print(f"    ! {f}")

    # ----- collect all findings -----
    all_findings = list(g["findings"]) + list(fx["findings"])
    miss = g["coverage"]["op_missing"] + g["coverage"]["agg_missing"]
    if miss:
        all_findings.append(
            f"shared conformance corpus (examples.pql) never exercises {sorted(miss)} — "
            f"the 3 hand-maintained parsers (py/java/rust) can diverge on these undetected; "
            f"a single parser+CSC in the C++ layer would remove the divergence surface")
    for ds_name, la in report["audits"].get("leakage", {}).items():
        all_findings += [f"[{ds_name}] {x}" for x in la["findings"]]
    for r in tasks:
        if r.kind == "binary" and r.metrics["auroc"] is not None:
            bn = max((b["auroc"] for b in r.baselines.values()
                      if b["auroc"] is not None), default=None)
            if bn is not None and r.metrics["auroc"] <= bn + 0.005:
                all_findings.append(
                    f"[{r.dataset}] {r.task}: engine AUROC {r.metrics['auroc']:.3f} "
                    f"gives no lift over naive baseline {bn:.3f}")
        for n in r.notes:
            all_findings.append(f"[{r.dataset}] {r.task}: {n}")

    print("\n" + "=" * 78)
    print(f"CONSOLIDATED FINDINGS ({len(all_findings)})")
    print("=" * 78)
    for i, f in enumerate(all_findings, 1):
        print(f"  {i}. {f}")

    report["findings"] = all_findings
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {RESULTS / 'report.json'}")
    print("for the cross-dataset generalizability matrix (retail + movielens + "
          "brightkite), run: benchmarks/run_suite.py")


if __name__ == "__main__":
    main()
