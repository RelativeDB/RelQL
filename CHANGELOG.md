# Changelog

Notable changes to the `relativedb` Python package and the `librt_c` engine
behind it. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The engine is pre-1.0: the Python API and the RelQL grammar may still change
between minor versions.

## [Unreleased]

### Added
- **XGBoost flat-feature backend.** The C++ layer gains a flat-feature
  planner/evaluator (`cpp/src/flat.*`, exposed as `relql_flat_analyze` /
  `relql_flat_features` in `librt_c`): it decides whether a RelQL query can
  run as flat features at all (scalar regression/binary, one horizon, no
  RANK, no ASSUMING, no ABLATE), derives the feature columns from the query
  and schema (entity scalars, the target mirrored into recent past windows,
  the WHERE clause's own aggregates, and the standard per-table
  COUNT/recency/SUM/AVG/MAX recipe), and evaluates them over assembled
  contexts with the engine's window semantics (`(anchor+start, anchor+end]`,
  NaN for missing). `relativedb.xgb.XgboostBackend` (optional extra
  `relativedb[xgboost]`, XGBoost >= 3.3) wires that matrix into an XGBoost
  model: `Engine.fit_xgboost(query, anchors)` is the adaptation path next
  to `fit_head`/`finetune` with the same supervision contract (past-bounded
  contexts, database-exact derived labels), `save()`/`load()` persist the
  model with its feature schema, and the fitted backend serves
  `Engine.execute` through the `ModelBackend` protocol. When the installed
  XGBoost build has CUDA and a device is visible, fitting and scoring run
  with `device="cuda"`. `analyze_flat()` answers "can this query run here"
  without raising, so callers can route ineligible queries to the sequence
  model.

## [0.1.3] — 2026-07-26

### Changed
- **The declared anchor is the traversal's fallback cutoff.** An undated
  focal row used to mean "no cutoff": the walk graph admitted rows dated
  after `AS OF`, f2p parent edges were followed unconditionally (a session
  row stamped at close — after the anchor — was serialized into the
  context), the factory overlay mapped a missing cutoff to `+inf`, and the
  columnar path did the same. All of these now fall back to `bound.as_of`,
  and the walk-graph cache key includes it. The same fallback ends the
  starvation twin: a query-aware walk from an undated entity used to drop
  every dated child, so a customer's own orders never reached its context —
  they do now, up to the anchor. Direct-target sampling fixtures were
  regenerated for the new (correct) contexts.

### Fixed (2026-07 review sweep)
- **Derived training labels are database-exact.** `fit_head`/`finetune`
  labels (when no `labels` dict is supplied) were evaluated over an
  assembled context — a sample that carries peer entities' rows (a peer's
  event counted into this entity's label; measured: derived 3.0 vs truth 2)
  and truncates heavy entities' own events. Labels and ranking relevance now
  fetch the entity's rows through the wiring, bounded at each window's far
  edge (`end_delta`, fixing the offset-window tail cut too). Aggregating a
  table with no direct link to the entity raises. Dropped NULL-label
  examples are now counted and warned about.
- **Inline aggregation filters see FK columns.** `COUNT(t.* WHERE t.fk = v)`
  compared `row.cells` only, so FK predicates matched nothing (and
  `IS NULL` matched everything).
- **Bare cross-table columns in scalar positions raise.**
  `WHERE orders.status = 'open'` on `FROM customers` validated and then
  silently read the customer's row, filtering every entity out.
- **Loud evaluation everywhere:** a windowed aggregation with no anchor
  raises instead of silently ignoring the window; `SUM`/`AVG`/`MIN`/`MAX`
  over non-numeric values raise instead of returning 0.0/NULL; cross-type
  comparisons raise instead of returning False; a WHERE window facing the
  future (FOLLOWING) is rejected as unsatisfiable.
- **Empty contexts warn.** A typo'd or temporally inadmissible entity id
  used to be scored on zero rows and returned as a normal prediction; it
  now warns (raises under `RELATIVEDB_STRICT=1`).
- **Anchor integrity.** `per_entity_anchor` no longer overrides a declared
  `AS OF` forward (clamped with `min`), warns when an entity has no dated
  row, and refuses to assemble an unbounded context; WHERE stays at the
  query's factual anchor when `context_anchor_time` decouples the context;
  multiclass/ranking batches with mixed per-entity anchors are scored per
  anchor group instead of scanning candidates at `max(anchors)`.
- **A fully dangling link warns at index build** (every FK value unmatched —
  the int-vs-str key mismatch that silently severed a whole link).
- **EXPLAIN honesty:** aggregate `ASSUMING` bounds render in the plan
  (previously `assuming: none` plus a stale "cannot be applied" warning);
  EXPLAIN ANALYZE warns that shared-context/hurdle strategies are not
  applied to its predictions; the multiclass path warns that aggregation
  function/window/filter do not affect scoring; ranking without
  `RANK TOP K` warns that it returns top-1; the 1000-class/candidate caps
  warn when they truncate.

### Added
- **Stepped multi-horizon forecasting.** `OVER (7 DAYS FOLLOWING HORIZONS 4
  [STEP 7 DAYS])` now runs one model forward per horizon, re-asking the
  masked question with the target token stamped at `anchor + k*step` while
  context and self-label history stay at the base anchor. Previously the
  N-horizon forecast was one prediction copied N times, silently. `STEP`
  defaults to the frame width; an unbounded window requires an explicit
  `STEP`.

### Fixed
- **WHERE aggregations are database-exact.** `WHERE COUNT(orders.*) OVER
  (14 DAYS PRECEDING) > 0` was evaluated over the assembled context — a
  *sample* built for the model that carries peer entities' rows (so the
  count passed for a customer who never ordered) and, per task shape, can
  omit the entity's own children (so the same count read zero for a customer
  with orders). Cohort filters are facts: the entity's rows are now fetched
  through the wiring, exact regardless of context budget. Aggregations on a
  table with no direct link to the entity raise instead of guessing, as does
  hitting the 1M-row fetch cap.
- **No best-effort expression evaluation.** An inline aggregation filter the
  evaluator could not run used to keep the row silently — turning
  `COUNT(t.* WHERE ...)` into `COUNT(t.*)` — and a filter naming another
  table's column silently compared NULL and dropped every row. Both now
  raise and fail the statement.

### Added
- **`ASSUMING COUNT(t.*) OVER (...) >= k`** (and `>`, `=`, `<=`, `<`,
  `EXISTS`, `NOT EXISTS`): aggregate counterfactuals, realized structurally.
  The entity's in-window rows are cloned (newest first, re-timestamped
  inside the window, re-parented to the entity, template fetched from the
  database when the context has none) or dropped (oldest first) until the
  bound holds — so the model sees the assumed history, not a constraint it
  cannot read. Unsatisfiable shapes (other aggregate functions, filtered
  counts, fractional bounds, bounds on the entity table) raise.
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

[Unreleased]: https://github.com/RelativeDB/RelQL/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/RelativeDB/RelQL/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/RelativeDB/RelQL/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/RelativeDB/RelQL/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/RelativeDB/RelQL/releases/tag/v0.1.0
