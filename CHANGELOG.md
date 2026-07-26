# Changelog

Notable changes to the `relativedb` Python package and the `librt_c` engine
behind it. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Only 0.1.0 and 0.1.1 are on PyPI. The engine is pre-1.0: the Python API and
the RelQL grammar may still change between minor versions.

## [Unreleased]

### Added
- **`ABLATE TABLE` actually ablates.** It parsed, validated, and then
  silently changed nothing — the plan even printed "declared, not applied".
  Every row of the ablated table is now dropped from each scored context.
  Unknown table names and the entity table are rejected with an
  `ExecutionError` instead of ablating nothing.
- **`EXPLAIN ABLATE`** (alias for the existing `EXPLAIN ABLATION`) scores the
  query as written, then re-scores once per candidate ablation — every
  non-entity schema table present in the contexts and every declared non-key
  column carrying values (time columns excluded; the target column included,
  since the entity's own cell is masked either way and its ablation measures
  reliance on sibling rows' past outcomes) — and ranks candidates by
  `mean_abs_delta`, how far dropping them moves the predictions. Near zero
  means the model is not using it — a good ablation; large means
  load-bearing. The report lands in `ExplainResult.ablation` and in
  `render()`; `predictions` carries the baseline. One cohort forward per
  candidate, so cost scales with schema width.

### Fixed
- **`ASSUMING` counterfactuals actually counterfact.** An assignment on the
  entity table (`ASSUMING c.plan = 'premium'`) now intervenes on the entity's
  own row only; sibling rows of the same table stay factual. Previously every
  context row of the table was overwritten, which — combined with zero-shot
  normalization deriving column statistics per context — flattened the column
  to a constant that normalized to zero regardless of the assumed value:
  `ASSUMING x = TRUE` and `ASSUMING x = FALSE` produced identical model
  inputs. Assignments on non-entity tables keep the documented broad
  semantics ("these orders are shipped").
- **EXPLAIN CONTEXT counts FK-feature cells.** Links that opt into feature
  emission (`LinkDef.feature_type`) always added one token per populated FK
  to the model input, but the EXPLAIN CONTEXT cell accounting ignored them —
  a table whose rows carry nothing but FKs reported zero cells, making the
  knob look inert to anyone measuring it. What EXPLAIN prints now matches
  what the model gets.

### Added
- **`AssumptionNotAppliedWarning` covers three inert-assumption shapes:**
  assigned table absent from context (existing), the entity's own row missing
  from the context, and a numeric/boolean assignment that leaves the column
  constant in-context — where zero-shot normalization would erase which
  constant it was.
- **`ContextCompositionWarning`** — one aggregate, end-of-run report per
  `execute()` when ≥50% of a cohort's contexts hit the cell budget or one
  schema table holds ≥60% of all context cells (cohorts above ~256 cells;
  virtual task rows excluded). Both failure shapes previously surfaced only
  as a per-entity warning stream or a hand-run EXPLAIN CONTEXT. It also
  reports a schema table that contributed context rows but zero feature
  cells — nothing of it reached the model. Table shares are now measured in
  *emitted* cells (declared non-key non-null columns plus feature-typed FK
  values) rather than raw cell counts.
- **`InvisibleTableWarning`** at `Engine` construction for a table that is
  statically unseeable: no non-key columns and no `feature_type` on any of
  its links. Serialization emits one token per feature cell and FK links
  ride on those tokens, so a pure edge table (follows/likes with no payload)
  enters the context at zero cost and never reaches the model — including
  the link structure its rows exist to represent. The warning points at the
  fix: when the key itself carries signal (a handle, a name, a category),
  set `feature_type` on that link so the value is emitted; otherwise add a
  payload column or summarize the edges. The legacy `BreadthFirstTraversal`
  cell-cost model now also counts feature-typed FK values, matching the
  reference traversal and what serialization emits.

## [0.1.2] — 2026-07-25

### Added
- **Shared-context scoring.** Cohort members that share an anchor time are
  scored in one forward pass over one assembled context instead of one pass
  each. `Engine.execute` picks the shared path automatically; `execute_many`
  exposes it directly. Seeding is coverage-aware and injects self-labels, so
  the shared context still carries each member's own history.
- **Columnar context population.** The context store and the traversal that
  fills it are array-backed rather than per-row objects, which is where the
  assembly time went once the forward pass stopped dominating.
- **Hurdle idiom for zero-inflated regression.** Targets that are mostly zero
  with a heavy tail (spend, quantity) are modelled as an occurrence
  probability times a conditional magnitude rather than as one regression,
  which is what the single head was quietly averaging away.
- **Cohort focal rows via `task_focal_keys`.** An optional resolver names the
  rows a task is actually about, instead of the engine inferring them from
  the cohort.
- Quantized inference (`f16` / `q8` / `q4`) on CUDA, and an `f16` micro-batch
  CPU kernel. Selected with `RELATIVEDB_RT_QUANTIZED`.
- macOS wheels are `universal2` (arm64 + x86_64, deployment target 13.0), so
  one wheel serves both Mac architectures. The platform tag is derived from
  the built dylib's Mach-O headers and cannot drift from what was compiled.
- Linux wheels: manylinux_2_28 x86_64 and aarch64, alongside a pure sdist that
  resolves the engine at runtime.
- Test tiers. `pytest -m "not integration"` runs offline with nothing built;
  `pytest -m integration` needs `librt_c` and the RT-J checkpoint.
  `RELATIVEDB_REQUIRE_NATIVE=1` turns a missing engine or an unresolvable
  checkpoint into a failure instead of a skip, so a broken cache cannot make
  CI green having tested nothing.
- Coverage for both languages (`--cov=relativedb`, gcov/gcovr over `cpp/src`),
  reported per flag.
- `RELEASING.md`, `CONTRIBUTING.md`, this changelog, and issue/PR templates.

### Changed
- Shared-context chunk size is derived from the measured cost of injecting a
  member into a context, not from a fixed constant.
- The default context budget and the model sequence cap are both 2048 cells.
- Column attention truncates key lists to the relevance-ranked head, and
  QK-RMSNorm plus output gating are fused into the Metal attention kernels.
- Self-label history windows are scoped to the entity that owns them; a
  sibling's labels no longer leak into a window.
- The shared-graph temporal overlay is vectorized and independent of the
  order anchors arrive in, so results no longer depend on cohort ordering.
- The README quickstart is copy-paste runnable end to end: it builds a toy
  two-table graph in memory and prints real churn probabilities.

### Removed
- `HistoryBaselineBackend`. RT-J is the only scoring backend; a model-free
  baseline in the same interface invited accidental use in production.
- The DuckDB extension workflow. It built `crates/` and `duckdb-extension/`,
  neither of which exists in this repository.
- The Rust and Java jobs in the release workflow, for the same reason.

## [0.1.1] — 2026-07-22

### Fixed
- The native RelQL parser failed to load from an installed wheel — it looked
  for `librt_c` relative to a source checkout. Wheel installs now find the
  bundled library inside the package.

## [0.1.0] — 2026-07-22

First release.

- RelQL v2: `PREDICT` over `OVER (...)` / `WINDOW` temporal frames, `HORIZONS`,
  `AS OF`, `ASSUMING`, `RETURN`, `EXPLAIN`, `EXISTS` / `NOT EXISTS`.
- `Engine` with GraphQL-style retriever wiring: the engine owns parsing,
  planning, context assembly and model routing; all data access goes through
  callables you supply. No bundled database connectors.
- `librt_c`, a dependency-light C++20 implementation of the RT-J relational
  transformer — no torch, no Python at inference — with CPU (Accelerate or a
  portable GEMM), Metal/MPS and CUDA backends, verified against the PyTorch
  reference on a golden batch.
- Native RelQL parser and CSC adjacency index in the same shared library, so
  the bindings delegate instead of reimplementing.
- Metal task-head fitting (`Engine.fit_head`) for multiclass and ranking, and
  full-checkpoint MPS fine-tuning (`Engine.finetune`) for binary and
  regression targets.
- Apache-2.0 license; macOS arm64 wheel with the engine bundled.

[Unreleased]: https://github.com/RelativeDB/RelQL/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/RelativeDB/RelQL/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/RelativeDB/RelQL/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/RelativeDB/RelQL/releases/tag/v0.1.0
