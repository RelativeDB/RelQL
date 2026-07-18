# relativedb — corpus backtest findings

Measured evaluation of the Python engine against **real, held-out future data**
from two downloaded relational datasets, plus the changes made and the parser +
CSC consolidation into the C++ layer.

Reproduce: `.venv-bench/bin/python benchmarks/run.py`
(datasets auto-download into the git-ignored `benchmarks/corpus/`).

## Corpus

| Dataset | Source | Rows | Entities | Notes |
|---|---|---|---|---|
| MovieLens (ml-latest-small) | grouplens.org | 100,836 ratings | 610 users | each movie rated once (0 repeats); bursty, non-stationary |
| Online Retail II | UCI ML repo | 1,067,371 lines | ~5,900 customers (top 1,500 used) | customers re-buy SKUs |
| Brightkite check-ins | SNAP | ~4.5M check-ins | ~58k users (top 1,500 used) | mobility; locations recur; a 3rd, independent domain |

Protocol: for each anchor `T`, context = rows `≤ T` (engine-enforced), target =
PQL window `(T, T+h]`, truth = the actual outcome from real future rows, scored
against naive baselines.

## Changes made

1. **Context-truncation instrumentation.** The `ContextPolicy` fanout cap
   dropped children silently, biasing windowed COUNT/SUM/AVG low.
   `PredictionResult.stats` now reports `contexts_truncated`, and a
   `ContextTruncationWarning` fires when a count-like windowed aggregate runs
   over a capped context. Predictions are unchanged. Under the default
   `bfs_width=32`, retail forward COUNT saturates near 11 (true counts up to
   ~1476). Guarded by `harness/audit_fixes.py`.

2. **Parser + CSC moved to the shared C++ layer** (behavior-preserving; see
   below).

## Parser + CSC consolidation (C++)

Both the PQL grammar and the CSC adjacency now live **once** in `librt_c`. The
per-language hand-written parsers and CSC binary-search were deleted — the
Python/Java/Rust bindings call the C ABI directly with **no fallback**;
`librt_c` is a hard dependency (the same library the RT-J model requires).

| Component | C++ | C ABI (in `librt_c`) | Bindings | Correctness test |
|---|---|---|---|---|
| PQL parser | `cpp/src/pql.{hpp,cpp}` | `pql_parse` → JSON AST | `pql/native.py`, Java `NativePqlParser`, Rust `pql/native.rs` | C++ `test_pql` (44 parse + 10 reject) + each binding's JSON→AST tests |
| CSC index | `cpp/src/csc.{hpp,cpp}` | `csc_build`/`csc_children`/`csc_free` | `csc_native.{py,rs}`, Java `NativeCsc` | C++ `csc_test` (22,502 vs brute force) + per-binding brute-force equivalence |

- Each language keeps only its id↔dense mapping, row storage, and JSON→AST
  deserialization; the grammar and the time-bounded adjacency are single-sourced.
- Deleted: Python `parser.py` internals (439→123), Java ANTLR grammar +
  `AstBuilder` + the ANTLR plugin/deps, Rust `parser.rs` (825); plus each
  language's in-language CSC adjacency/binary-search. ~1,800 lines of duplicated
  algorithm removed. The native-vs-hand-written *equivalence* tests were dropped
  (nothing left to compare against); correctness now rides on the C++
  conformance tests + per-binding JSON/CSC checks.
- Missing `librt_c` is a hard error at parse / CSC-build, not a silent fallback.

## Findings (across the 3-dataset generalizability grid)

`run_suite.py` scores each engine task vs. its best naive baseline on every
dataset, with per-split (per-anchor) mean ± std. The engine beats naive in
**4 of 10 scorable cells**. What generalizes and what doesn't:

| Task | retail | movielens | brightkite | reading |
|---|---|---|---|---|
| **churn** (AUROC vs recency) | 0.59 / **0.67** | 0.78 / **0.86** | 0.60 / **0.96** | loses to a one-line recency baseline on **all three** — robust finding |
| **activity_count** (MAE vs naive) | **76 / 84** ✓ | 27 / **10** | 173 / **142** | the regression win is **retail-specific**, not general — persistence loses on the non-stationary domains |
| **buy_it_again** (Recall@10 vs popularity) | **0.08 / 0.03** ✓ | 0.00 / 0.01 | **0.39 / 0.08** ✓ | works where items recur (retail, brightkite ~5×); structurally 0 on once-per-item (movielens) |
| **forward_value** (MAE vs naive) | **1610 / 1772** ✓ | — | — | retail only |

Corrections to the earlier two-dataset read: the count-regression "win" did
**not** generalize — it holds on retail and fails on MovieLens/Brightkite. The
coarse-binary/recency finding and the repeat-only-ranking finding both
replicate across all three.

Other findings:
- Churn probability is coarse (≤`num_history_windows+1` levels, 3 distinct on
  the default), compounding the recency gap above.
- `examples.pql` never exercises `!=`, `>=`, `LIKE`, `ENDS WITH`, `IS NOT NULL`,
  `AVG`, `MIN`, `COUNT_DISTINCT`, `FIRST`; the Java/Rust hand-written parsers can
  diverge on these until they delegate to `librt_c`.

### Holding up
- Temporal correctness (F24): 0 leaks, injection caught, contexts monotonic.
- Buy-it-again on repeat domains (retail, brightkite): beats popularity ~3–5×.
- Per-split std is small (e.g. brightkite churn 0.585 ± 0.076), so these
  conclusions are stable across anchors, not artifacts of one split.

## Next steps
1. Point the Java (ANTLR) and Rust parsers at the same C ABI + JSON contract;
   flip native parser/CSC to default and delete the duplicated code.
2. Add more datasets/splits to the harness before changing the backend, so a
   change can be judged for generalizability across the suite.
