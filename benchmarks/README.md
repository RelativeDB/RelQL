# benchmarks — real-corpus backtest harness

Extensive, measurement-driven testing of relativedb / RelQL against **real
relational data with a strict point-in-time backtest**. See
[`FINDINGS.md`](FINDINGS.md) for results.

## Run

```bash
python3 -m venv .venv-bench
.venv-bench/bin/pip install numpy pandas scikit-learn requests openpyxl
.venv-bench/bin/pip install -e python
.venv-bench/bin/python benchmarks/run.py          # audits + deep dive (retail + movielens)
.venv-bench/bin/python benchmarks/run_suite.py    # generalizability matrix across all 3 datasets
```

Datasets auto-download on first run into `benchmarks/corpus/` (git-ignored):
MovieLens `ml-latest-small` (grouplens), Online Retail II (UCI), and Brightkite
check-ins (SNAP). `run.py` writes `results/report.json`; `run_suite.py` writes
`results/suite.json` (both git-ignored).

`run.py` is the depth view (per-task baselines, parser/leakage/instrumentation
audits). `run_suite.py` is the breadth view: a datasets × tasks matrix with
per-split (per-anchor) mean ± std and an overall "beats naive" fraction — so a
backend change is judged across the whole grid, not one dataset's one number.

## Layout

| File | What |
|---|---|
| `harness/datasets.py` | loaders → schema + CSC engine + precomputed truth arrays |
| `harness/backtest.py` | the point-in-time protocol + task catalogue (churn, count, value, ranking) + naive baselines |
| `harness/metrics.py` | AUROC / PR-AUC / Brier / lift; MAE / RMSE / R² / Spearman; Recall@K / MAP@K / NDCG@K |
| `harness/audit_grammar.py` | parse the shared corpus, coverage, should-parse / should-reject fuzz probes |
| `harness/audit_leakage.py` | temporal-correctness (F24): direct, injection, monotonicity |
| `harness/audit_fixes.py` | regression guard for the context-truncation instrumentation |
| `harness/suite.py` | generalizability grid: datasets × tasks, engine-vs-naive + per-split stability |
| `run.py` | depth view: audits + per-task backtest on retail + movielens |
| `run_suite.py` | breadth view: the generalizability matrix across all 3 datasets |

RelQL parsing and the CSC index are single-sourced in the C++ layer (`librt_c`),
a hard dependency — the same native library the RT-J model requires. The Python
CSC binding is checked against a brute-force reference by
`python/tests/test_native_csc.py`; parser correctness rides on the C++
conformance test (`cpp/src/test_pql.cpp`) plus `python/tests/test_pql_parser.py`.

## Method

For each anchor `T`: context = rows `≤ T` (engine-enforced), target = RelQL
window `(T, T+h]`, truth computed **independently** from the raw frames (never
by re-running the engine), scored against naive baselines. A metric only counts
as "signal" when the engine beats the one-liner a user could write without it.
