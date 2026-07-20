# RelativeDB Stability, Performance, and Recall Improvement Roadmap

**Prepared:** 2026-07-19  
**Based on:** RELATIONAL_TRANSFORMER_COMPARISON.md and the implementation/worktree reviewed there  
**Purpose:** Convert the comparison findings into an ordered engineering and evaluation program

## 1. Executive recommendation

The next investment should be in the semantic path into RT-J, not in lower-level matrix multiplication.

The native transformer is already the most mature part of the system: its CPU and MPS implementations agree with the current reference fixture, its attention avoids quadratic materialization, and its batching performance is good. The largest present risks and opportunities occur before and after the kernel:

1. A request currently helps determine its own normalization. This makes scores unstable across batch composition.
2. Cohort evidence can alter the focal entity's historical label calculation.
3. The sampler does not reproduce the reference model's task-row and peer-context geometry.
4. Ranking evaluates a broad, weakly generated candidate set and pays transformer cost for each candidate.
5. The benchmark and release paths do not yet reliably gate regressions.

The recommended order is therefore:

1. **Stabilize semantics and measurement.**
2. **Improve evidence and candidate recall.**
3. **Optimize the now-stable workload.**
4. **Add supervised calibration and hybrid fallbacks with automatic rollback.**

This order matters. Performance work against batch-dependent inputs can preserve the wrong behavior more efficiently. Recall tuning against contaminated self-labels or stale benchmarks can reward leakage rather than useful signal.

## 2. What “recall” means in this roadmap

Recall is overloaded in this system. Every improvement and benchmark should state which of these it targets:

| Recall type | Definition | Typical failure |
|---|---|---|
| Evidence recall | Relevant historical rows, relations, and demonstrations that reach the model | Fanout or cell budgets omit useful events |
| Candidate recall | True future item/class is present in the domain passed to a reranker | Repeat-only or capped candidate generation excludes the answer |
| Predictive recall | Fraction of positive labels recovered at an operating threshold | Poor calibration, class imbalance, or weak representation |
| Ranking Recall@K | Fraction of relevant future items returned in the top K | Weak candidate generator or reranker |

Increasing context size only addresses evidence recall, and often inefficiently. It cannot recover an item excluded from the candidate set, and it does not select an appropriate classification threshold.

## 3. Current evidence and implications

| Evidence | Implication for the roadmap |
|---|---|
| C7 regression changed from 0.72297 to 0.55974 when C1 joined the batch | Fixed artifacts and batch-invariance tests are release blockers |
| C7 binary probability changed by about 0.0258 under the same condition | Classification calibration cannot be trusted until normalization is fixed |
| A cohort changed C7's own historical count from 1 to 2 | Focal facts and demonstration facts need separate ownership |
| C++ CPU/MPS max drift is about 0.00391 from PyTorch | Native optimization is not the primary correctness gap |
| MPS 80×16 FP32 is about 0.8 ms/entity vs. 7.6 ms for 1×16 | A serving scheduler can unlock much more throughput than another kernel rewrite |
| Q8/Q4 reduce resident memory substantially but do not improve all measured latencies | Quantization should be selected for capacity, with quality gates, not assumed to be faster |
| Churn loses to a recency baseline on three benchmark domains | Hybridization and calibrated task heads are more promising than context growth alone |
| Buy-it-again works on repeat domains and fails structurally on MovieLens | Candidate generation must cover novel as well as repeated items |
| Fine-tuned heads sometimes regress badly, including future-spend MAE | Training requires validation, early stopping, and automatic fallback |
| Ranking may score up to 1,000 candidates with a full sequence per candidate | Two-stage retrieval is the largest ranking performance lever |

## 4. North-star properties

The system should eventually satisfy these properties simultaneously.

### Stability

- One entity receives the same output regardless of batch membership, order, pagination, and chunk size.
- A cohort can add demonstrations but cannot change the focal entity's factual history or label.
- Every prediction has a single explicit temporal cutoff, applied to rows, candidates, classes, statistics, and labels.
- The loaded checkpoint, embedding model, preprocessing artifact, and head are mutually compatible and cryptographically identifiable.
- Unsupported query behavior fails at validation instead of returning plausible but false output.

### Performance

- Common work is performed once per engine, task, entity, or batch at the narrowest valid scope.
- Retrieval stops at a token-aware budget rather than assembling data that will later be discarded.
- Short sequences are dynamically batched and length-bucketed.
- Ranking uses inexpensive candidate retrieval followed by a bounded RT rerank.
- End-to-end latency is measured by stage, including retrieval and tokenization, not only the C++ forward pass.

### Recall

- The sampler covers the focal neighborhood, useful same-task demonstrations, and underrepresented relation types.
- Candidate generation has measured oracle coverage before reranking quality is judged.
- Binary thresholds and ranking cutoffs are tuned on validation data, never hard-coded solely for convenience.
- Models are compared against recency, persistence, popularity, and simple tabular baselines under the same temporal split.
- Any adaptation is rejected automatically when it fails a held-out quality gate.

## 5. Roadmap overview

| ID | Improvement | Stability | Performance | Recall | Dependency |
|---|---|---:|---:|---:|---|
| F1 | Repair benchmark/release gates and add stage telemetry | High | Medium | High | None |
| S1 | Persist fixed column and task statistics | Critical | Medium | High | F1 |
| S2 | Separate focal evidence from demonstrations | Critical | Medium | High | F1 |
| S3 | Apply per-entity temporal bounds everywhere | Critical | Low | High | F1 |
| S4 | Introduce a stable task/target representation | High | Low | High | S1–S3 |
| S5 | Enforce artifact compatibility and data contracts | High | Medium | Medium | F1 |
| S6 | Make budgets token-aware and truncation explicit | High | High | High | S2 |
| R1 | Add a seeded, tiered, task-aware sampler | High | Medium | High | S1–S6 |
| R2 | Add similarity and label-aware cohort selection | Medium | Medium | High | R1 |
| R3 | Build two-stage ranking candidate generation | Medium | Critical | Critical | S3, F1 |
| R4 | Harden head training and threshold calibration | High | Low | High | S1–S4, F1 |
| R5 | Add validated hybrid baselines/fallbacks | High | Low | High | F1, R4 |
| R6 | Implement real horizon conditioning | High | Medium | High | S4 |
| R7 | Version multiclass vocabularies/class retrieval | High | High | High | S3–S5 |
| P1 | Add dynamic batching and length buckets | Medium | Critical | Neutral | S1 |
| P2 | Batch retrieval and push down query constraints | Medium | High | Medium | S6 |
| P3 | Cache immutable preprocessing and schema work | Medium | High | Neutral | S1, S5 |
| P4 | Add incremental/mmap serving indexes | Medium | High | Medium | S3, S5 |
| P5 | Select device/precision by measured policy | Medium | High | Medium | F1 |
| P8 | Remove hidden CPU-only output paths | Medium | High | Neutral | P5 |
| O1 | Package native wheels and restore CI/release integrity | High | Medium | Neutral | S5 |

“Critical” indicates a likely order-of-magnitude lever or a prerequisite for trustworthy operation. “Neutral” means the change should preserve recall and requires a no-regression gate.

## 6. Foundation: measurement before optimization

### F1.1 Repair the benchmark entry points

Immediate changes:

- Attach RtNativeBackend in benchmarks/run_suite.py before suite.run.
- Attach a backend in benchmarks/harness/audit_fixes.py, or rewrite that audit so it never enters scoring.
- Replace the 5,000,000-cell diagnostic policy with a bounded policy appropriate to the assertion being tested.
- Fail before context assembly when a scoring command has no model backend.
- Mark existing findings with the commit, checkpoint, and backend that produced them; archive results that cannot be reproduced.

Acceptance:

- Both documented benchmark commands run from a clean checkout.
- A missing backend fails in less than one second and before any corpus-wide context assembly.
- Result JSON is not written after a partial or failed run.

### F1.2 Record complete experiment provenance

Every result artifact should include:

- source commit and dirty-tree hash;
- canonical query AST and task identifier;
- dataset identity/checksum and temporal split;
- schema hash and index generation;
- checkpoint URI, resolved revision, file hash, precision, and device;
- embedding model identity and revision;
- statistics artifact hash and cutoff;
- sampler policy, context seed, and candidate policy;
- head artifact hash, training cutoff, class list, and validation result;
- batch size, length buckets, warm/cold state, and machine metadata.

This turns benchmark drift into an explainable artifact difference rather than an investigation.

### F1.3 Add stage-level telemetry

Measure these independently:

1. query parse/validation;
2. entity enumeration;
3. row retrieval/index traversal;
4. WHERE and self-label evaluation;
5. context selection;
6. token construction and normalization;
7. text/schema embedding lookup;
8. native forward;
9. candidate reranking;
10. result shaping.

For each stage record count, p50, p95, p99, maximum, bytes, rows, cells, emitted tokens, and cache hit rate where applicable.

Required context counters:

- focal rows and tokens;
- demonstration rows and tokens;
- rows omitted by time, fanout, relation quota, and global budget;
- relation/table coverage;
- number of historical labeled demonstrations;
- disconnected/tokenless nodes;
- context and sequence truncation;
- candidate count before/after each generation stage.

### F1.4 Establish quality scorecards

Do not use one “accuracy” number. Use:

| Task | Primary | Secondary | Calibration/coverage |
|---|---|---|---|
| Binary | AUROC and PR-AUC | recall at fixed precision; precision at fixed recall | Brier score, ECE, score cardinality |
| Regression | MAE or normalized MAE | RMSE, Spearman, tail MAE | prediction interval coverage |
| Multiclass | macro-F1 | balanced accuracy, top-3 recall | log loss, ECE |
| Ranking | Recall@K and NDCG@K | MAP@K, hit rate | candidate recall@M, catalog/list coverage |
| Forecast | horizon-weighted MAE | per-horizon MAE and rank correlation | coverage by horizon |

Always report naive baselines and the number of eligible/observable labels.

## 7. Stability workstream

### S1. Fixed statistics artifacts

#### Problem

RtNativeBackend._normalize currently derives physical and synthetic-label statistics from the sequences in the active score call unless ColumnStats happens to be attached. This makes output depend on batch composition. Fine-tuning fits physical statistics, but synthetic task-label scaling remains request-dependent.

#### Design

Create a versioned PreprocessingArtifact with:

- per-table/per-column numeric mean and standard deviation;
- global datetime mean and standard deviation using one documented convention;
- per-task target transformation;
- fitting cutoff and row counts;
- schema hash, task hash, checkpoint compatibility, and artifact version.

Task transformations should be explicit:

- binary targets: no z-score; retain 0/1;
- regression: train-label mean/std or a documented robust transform such as log1p plus fixed scaling;
- count/value tasks: optional nonnegative transform selected from training only;
- multiclass: fixed ordered class vocabulary;
- ranking: fixed candidate feature transforms; no request-derived label statistics.

The backend should have only two modes:

1. artifact-backed production mode; or
2. explicitly named experimental in-context mode.

It should never fall back per column or per request without surfacing the regime in PredictionResult.

#### Acceptance

- Same focal output under single-row, mixed batch, reversed order, duplicates, and multiple chunk sizes; target maximum delta at most 1e-6 on one device/precision.
- Artifact fit uses no row later than its cutoff.
- Serialization round-trip produces bit-identical transforms.
- Missing/incompatible artifacts fail before retrieval in production mode.

#### Expected impact

- Stability: eliminates the demonstrated batch dependence.
- Performance: avoids repeated mean/std scans of every sequence.
- Recall: makes calibration and head training reproducible; prevents thresholds from moving with cohort composition.

### S2. Separate focal evidence from demonstrations

#### Problem

EntityContext.rows currently merges focal and cohort subgraphs. _self_labels evaluates the target over all rows, so cohort events can change the focal label.

#### Design

Represent context ownership explicitly:

- focal_rows: facts reachable from the focal entity;
- demonstrations: one or more peer contexts, each with its own entity ID, rows, timestamp, and known task label;
- shared_rows: optional immutable dimension rows referenced by either group.

Build self-labels from focal_rows only. Materialize each peer's historical label from that peer's rows, not by mixing all peers into one row bag. Preserve node connections inside each demonstration while preventing cross-entity aggregate evaluation.

If changing EntityContext is too disruptive initially, attach owner_entity_id and role to every admitted row/node and require evaluation functions to select one owner.

#### Acceptance

- Focal self-labels are invariant to cohort size, cohort order, and peer row count.
- Adding a demonstration can change the model score, but cannot change factual counters or label values in explain output.
- Tests cover identical primary keys in different tables and shared dimension rows.
- Explain context visually separates focal, demonstration, and shared evidence.

### S3. Per-entity temporal bounds

#### Problem

Multiclass and ranking domain enumeration use the maximum anchor in a batch. Statistics and cohort selection can also become unsafe if their cutoff is not explicit.

#### Design

- Carry TemporalBound on every EntityContext and candidate-generation request.
- Group entities by identical bound only as a performance optimization.
- Enumerate/filter labels and candidates at each effective bound.
- Key caches by table, column, index generation, and bound bucket; never reuse later-state domains for earlier predictions.
- Fit statistics only from a declared training cutoff.
- Require time-aware indexes for candidate and class scans.

#### Acceptance

- Injecting a class, candidate, or row immediately after one entity's anchor never affects it, even when batched with a later entity.
- Per-entity and one-at-a-time execution agree.
- Temporal leakage audit covers focal rows, demonstrations, candidates, statistics, labels, and counterfactual assumptions.

### S4. Stable target representation

#### Problem

The reference model masks a real target cell with its actual task table and column semantics. RelativeDB emits a generic task.label row. The generic representation discards schema semantics that the model consumes.

#### Design

Use separate paths:

1. **Entity-column autocomplete:** mask the actual focal cell in place and retain its real table and column embedding.
2. **Materialized tasks:** expose an explicit task table in Schema, with entity link, timestamp, and target column; mask the real task cell.
3. **Derived RelQL targets:** compile the canonical query into a stable TaskSpec containing task name, target semantic type, entity relation, horizon, aggregation, filters, and version.

For derived targets, first evaluate two representations:

- a stable materialized task-row schema whose table/column names are user-declared; and
- a task-specific trained head over the frozen backbone.

Do not assume a hashed synthetic column phrase is understood by a checkpoint that never saw that vocabulary. Test representation alternatives on validation tasks.

#### Acceptance

- Autocomplete tokens match a reference fixture field by field.
- Task identity remains unchanged across process restarts and query formatting differences.
- Semantically different horizons/filters cannot collide.
- The selected representation improves official or held-out metrics without breaking batch invariance.

### S5. Artifact and data-contract enforcement

Add a resolved ModelManifest containing dimensions, architecture generation, embedding model/revision, checkpoint type, semantic routing, and file hash.

At load:

- read config.json;
- call the existing embedding mismatch guard;
- verify tensor names, dimensions, and decoder availability;
- verify the head's backbone and preprocessing hashes;
- reject unsupported legacy/variable geometry explicitly;
- select quantized derivatives only if their source hash matches.

At row ingestion:

- omit None/NaN cells rather than emitting zero-value tokens;
- validate value types against Schema;
- reject or explicitly support list-valued FKs;
- report dangling FKs and duplicate IDs;
- normalize all timestamps to one UTC contract;
- enforce stable scanner ordering or sort by a documented key;
- report tokenless connector rows before serving.

### S6. Token-aware budgets and fail-closed behavior

#### Problem

Context assembly counts row cells, not final tokens. Synthetic targets, task history, skipped cells, and tokenizer truncation create two budgets. Work can be retrieved and then silently lose the tail.

#### Design

- Use a single token budget owned by a ContextBudget object.
- Reserve target/task tokens first.
- Reserve minimum quotas for focal rows, each relation/table, and demonstrations.
- Estimate or materialize token cost before admitting a row.
- Stop retrieval when no quota can accept another row.
- Preserve provenance for every omitted row.
- Treat target loss, disconnected focal paths, or unexplained truncation as errors.
- Allow explicit best-effort mode only when PredictionResult carries a degraded flag.

#### Acceptance

- Emitted token count never exceeds the configured limit.
- No separately assembled rows are discarded by an unreported tail clip.
- Increasing a budget is monotonic in retained evidence for a fixed seed/policy.
- Truncation rate and quality-by-truncation cohort appear in benchmark output.

## 8. Recall workstream

### R1. Seeded, tiered, task-aware sampling

#### Goal

Increase useful evidence per token rather than simply increasing context size.

#### Proposed sampler

Build every context in explicit tiers:

**Tier 0 — focal neighborhood**

- Always retain the target/task row and focal entity features.
- Traverse direct parents needed to interpret focal/event rows.
- Select temporally valid children relevant to target aggregations and filters.
- Preserve a configurable mix of recent, frequent, high-value, and rare events.
- Reserve relation quotas so one high-fanout table cannot consume the context.

**Tier 1 — same-task demonstrations**

- Select peer entities with known labels before the focal anchor.
- Retrieve each peer's self-contained local neighborhood.
- Prefer peers similar in schema features, history shape, and graph neighborhood.
- Allow label balancing for training and evaluation contexts using historical labels only.
- Diversify peers so near-duplicates do not consume the demonstration budget.

**Tier 2 — broader graph evidence**

- Use seeded random walks or graph-frequency scores to locate informative same-table nodes.
- Fill unused relation quotas.
- Optionally use a FAISS/entity-embedding index for similarity retrieval.

**Tier 3 — deterministic fallback**

- Fill remaining capacity with temporally valid same-table rows using a seeded order.
- Do not fall back to raw scanner order, which can encode ingestion artifacts.

#### Determinism

Derive the default context seed from:

- sampler version;
- schema/index generation;
- canonical task ID;
- focal entity ID;
- anchor;
- user-provided global seed.

One seed must produce one context. Multi-seed ensembling should be an explicit quality mode that records all seeds.

#### Task-aware row value

Use the query only to prioritize legal historical evidence:

- COUNT/EXISTS: preserve event occurrence and recency coverage;
- SUM/AVG: preserve value distribution and high-value tails;
- FIRST/LAST: prioritize chronological boundary rows;
- LIST_DISTINCT/ranking: preserve item diversity, repeats, and transition/co-occurrence evidence;
- filtered aggregations: reserve rows capable of satisfying the filter.

Never use future truth to choose focal evidence.

#### Acceptance

- Evidence recall is measurable against an uncapped oracle context on small datasets.
- Each relation receives its declared minimum quota when eligible rows exist.
- Same seed is bit-reproducible; different seeds have measured quality variance.
- At a fixed 8,192-token budget, the new sampler improves validation metrics over scanner-order cohorts.
- Quality is plotted against context tokens and latency, not reported at one arbitrary budget.

### R2. Similarity, diversity, and label-aware cohorts

The current “first scanner rows” cohort is neither similar nor representative.

Candidate cohort signals can include:

- normalized static entity features;
- recency/frequency/value history summaries;
- graph degree and relation-presence fingerprints;
- pooled frozen RT representations;
- recent event/item embeddings;
- task-specific historical output.

Selection should combine relevance and diversity, for example:

1. retrieve top-N similar peers;
2. partition by historical label or target quantile where legal;
3. select with maximal marginal relevance or cluster coverage;
4. retrieve self-contained peer contexts.

Evaluate cohort policies independently:

- random seeded;
- recency/frequency;
- graph random walk;
- embedding similarity;
- similarity plus diversity;
- similarity plus legal label balance.

Report known-label count, label distribution, similarity distribution, and duplicate rate. Do not accept a cohort method solely because it produces a larger model-score spread.

### R3. Two-stage ranking and candidate generation

#### Problem

The current ranking path can enumerate up to 1,000 parent rows and perform a full transformer sequence for every candidate and entity. It is expensive, and enumeration by table order does not maximize candidate recall.

#### Stage 1 — candidate union

Generate a temporally eligible union from several inexpensive sources:

- the entity's previously interacted items;
- item-to-item co-occurrence/transition neighbors;
- items used by similar entities;
- graph-neighbor items;
- time-decayed global/category popularity;
- content/schema embedding neighbors;
- business-rule or eligibility candidates;
- optional exploration/new-item candidates.

Deduplicate and record the source of every candidate. Use source quotas so popularity does not eliminate personalized or novel candidates.

This directly addresses the MovieLens structural failure: a repeat-only set cannot recall a future item that has never been seen by the entity.

#### Stage 2 — lightweight pruning

Score the union using cached/inexpensive features:

- recency and frequency;
- co-occurrence strength;
- popularity;
- entity/item embeddings;
- simple linear or tree model;
- eligibility and freshness.

Keep a bounded M, initially evaluated across values such as 50, 100, and 200.

#### Stage 3 — RT reranking

Build candidate-conditioned RT sequences only for the retained M. Batch or chunk them by sequence length and device memory. Apply a trained pairwise/listwise head when it passes validation; retain zero-shot score as a feature or fallback.

#### Required measurements

- candidate recall@M before RT;
- oracle Recall@K if the reranker perfectly ordered the candidate set;
- final Recall@K/NDCG@K;
- fraction of relevant items that are novel to the entity;
- contribution and hit rate by candidate source;
- candidates and transformer forwards per entity;
- latency and memory by M.

#### Acceptance

- Candidate recall@M reaches a declared target, preferably at least 95% of the attainable relevant items on repeat-capable benchmark tasks.
- Full RT candidates are reduced by at least 80% from the current cap without reducing final Recall@K.
- Novel-item recall is nonzero on datasets where future novel items exist.
- Candidate generation is temporally valid and invariant to batch composition.

### R4. Harden task-head training and calibration

#### Problem

The current head fitter trains for a fixed epoch count and reports training loss. Local experiments show that lower training loss can coincide with much worse held-out quality.

#### Training protocol

- Use time-based train/validation/test splits.
- Fit column, task-label, and feature statistics on train only.
- Persist the exact class vocabulary and group construction.
- Add validation evaluation during training.
- Use early stopping and restore best validation weights.
- Select learning rate, weight decay, and epoch cap on validation only.
- Support class weights or focal loss for imbalanced binary/multiclass tasks.
- Tune regression transforms and robust losses for sparse/heavy-tailed values.
- Use pairwise or listwise ranking losses with entity group boundaries.
- Mine hard negatives from the stage-1 candidate generator.
- Calibrate classification probabilities using validation-only Platt/isotonic/temperature methods as appropriate.
- Tune thresholds for the product objective: recall at required precision, cost-weighted utility, or capacity.

#### Automatic rollback

Never activate a fitted head merely because training loss fell. Require:

- minimum validation sample size and class coverage;
- improvement over the released zero-shot head;
- improvement over the best declared simple baseline, or an explicit reason to ship otherwise;
- no stability/leakage failure;
- calibration within tolerance;
- confidence interval or repeated-split evidence where sample size permits.

If a head fails, serve the zero-shot or hybrid fallback and retain the failed artifact only for diagnosis.

#### Expected recall impact

- Class weighting and threshold tuning directly increase minority-class recall.
- Validation prevents catastrophic future-spend and repeat-purchase regressions.
- Hard negatives focus ranking capacity on confusing candidates.
- Fixed transforms let the head learn stable distinctions instead of request-specific scaling.

### R5. Validated hybrid models and fallbacks

The benchmarks show that one-line baselines can be stronger than RT-J on specific tasks:

- recency for churn;
- persistence for activity/value;
- popularity for some ranking domains.

Treat these as signals, not merely competitors.

Create a small validation-trained combiner using:

- zero-shot RT score/features;
- recency, frequency, value, and trend;
- context coverage/truncation indicators;
- candidate-source scores;
- entity cold-start depth;
- optional fitted-head output.

Possible forms:

- calibrated weighted blend;
- logistic/linear meta-model;
- compact gradient-boosted model;
- rule-based fallback for no-context/cold-start cases.

Report RT-only, baseline-only, and hybrid metrics separately so the system's source of value remains visible. The hybrid should be rejected if it improves aggregate performance only by degrading an important subgroup.

### R6. Real horizon-conditioned forecasting

The current multi-horizon path repeats one scalar.

Choose one supported design:

1. separate TaskSpec and target row per horizon;
2. a shared backbone plus one validated head per horizon;
3. a multi-output head conditioned on explicit horizon features;
4. repeated inference with horizon-specific task timestamp/semantics.

Train/evaluate every horizon separately and jointly. Enforce useful structural constraints where applicable, such as nonnegative counts, but do not force cumulative monotonicity on non-cumulative targets.

Reject multiple horizons at validation until one design is implemented.

### R7. Stable multiclass vocabularies and class retrieval

The zero-shot multiclass path scans distinct labels at request time, sorts them lexicographically, and caps the first 1,000. This can exclude the correct class for reasons unrelated to the entity and repeats expensive table scans/embedding work.

For bounded tasks:

- fit an ordered class vocabulary from training data;
- persist frequency, first/last valid time, embedding, and class ID;
- include an explicit unknown/other policy;
- make fitted heads and calibration reference the vocabulary hash;
- apply per-entity temporal eligibility without changing class IDs.

For very large/open class domains:

- retrieve a class candidate set using task/entity/text/graph similarity;
- measure class candidate recall before decoder quality;
- rerank the class candidates using the predicted text embedding or trained head;
- include frequency/novelty priors only through validation-tested combination;
- surface abstention when no class is sufficiently supported.

Acceptance:

- Class IDs and probabilities are stable across scanner order and request batching.
- The correct observed class is never lost to an unexplained lexicographic cap.
- Class candidate recall and open-set coverage are reported.
- Repeated table scanning and embedding of an unchanged vocabulary are eliminated.

## 9. Performance workstream

### P1. Dynamic batching and length bucketing

The measured MPS results show a large batching opportunity: approximately 7.6 ms/entity for 1×16 FP32 versus 0.8 ms/entity for 80×16.

Add an inference scheduler that:

- groups requests by checkpoint, precision, device, task/output head, and preprocessing artifact;
- buckets by sequence length to reduce padding;
- uses a short configurable batching window;
- caps tokens rather than only rows per batch;
- splits oversized contexts;
- preserves request order at result assembly;
- reports queue time separately from compute time;
- supports latency and throughput service classes.

Benchmark:

- batch sizes 1, 4, 16, 32, 80;
- sequence buckets such as ≤32, ≤128, ≤512, ≤2,048, ≤8,192;
- cold and warm model states;
- CPU and MPS;
- mixed short/long arrival distributions.

Acceptance:

- No batch-composition output change.
- Warm p95 latency stays within the service objective.
- Sustained throughput improves materially—target at least 3× on representative short-sequence MPS traffic if queueing volume permits.
- Padding ratio and queue wait are visible.

### P2. Batched retrieval and query pushdown

Current parent retrieval is batched by table, but child traversal invokes a callback per parent/link.

Extend RetrieverWiring with optional bulk APIs:

- entities(table, ids, bound);
- children_many(link, parent_ids, bound, per_parent_limit);
- scan/project(table, columns, bound, predicate);
- candidate domain retrieval with bound and eligibility;
- cohort_many(table, anchor_ids, bounds, limit).

Keep scalar callbacks for compatibility, but surface when they cause N+1 retrieval.

Push down:

- primary-key cohorts;
- temporal bounds;
- columns actually used by schema/tokenization/query filters;
- relation-specific limits;
- eligibility filters;
- class/candidate projection;
- sort/limit when callback storage can execute it.

Acceptance:

- Retrieval call count is included in explain/analyze.
- High-fanout workloads use O(relations × hops) bulk calls rather than O(rows × relations).
- Bulk and scalar implementations are context-equivalent.

### P3. Cache immutable work at the right scope

**Engine/index scope**

- schema hashes and column semantic maps;
- FK-to-parent maps;
- table/column schema phrase embeddings;
- task canonicalization and validation;
- resolved model manifests and model handles;
- fixed preprocessing artifacts.

**Index generation scope**

- time-sorted row arrays;
- cohort/entity embeddings;
- class vocabularies by cutoff bucket;
- candidate source indexes;
- pretokenized static dimension rows where valid.

**Request/batch scope**

- focal base sequence reused across candidates;
- text-value embeddings;
- repeated parent/dimension rows;
- normalized immutable values;
- candidate feature matrix.

All caches need bounded memory, versioned keys, hit/miss telemetry, and explicit invalidation. Avoid caches keyed only by a path or table name when data generation, bound, embedding revision, or quantization changes meaning.

### P4. Incremental and memory-mapped serving indexes

CSC is currently a full immutable scanner snapshot rebuilt with Engine.

For production:

- split immutable base index from append-only deltas;
- periodically compact deltas;
- version each snapshot and pin one version for a complete request;
- store time-sorted adjacency and row metadata in mmap-friendly arrays;
- keep arbitrary application IDs in a compact stable dense mapping;
- make candidate/cohort indexes use the same generation;
- support atomic generation swap without changing active requests.

This improves startup, memory sharing, and refresh latency while preserving point-in-time reproducibility.

### P5. Device and precision policy

Use observed workload policy, not a universal “quantized is faster” rule.

From current measurements:

- MPS FP32 is fastest on the shown 80×16 and 1×2,048 shapes.
- CPU F16/Q8 can modestly improve some shapes.
- Q8/Q4 primarily reduce resident memory.
- Q4 has materially larger golden drift than FP32/Q8.

Define deployment profiles:

| Profile | Preferred use | Required quality gate |
|---|---|---|
| FP32 MPS | Throughput/long-context on supported Apple systems | Golden and task metrics |
| FP32 CPU | Portable reference serving | Golden and task metrics |
| F16/Q8 CPU | Memory-constrained CPU with measured benefit | No task metric degradation beyond tolerance |
| Q8/Q4 MPS | Capacity-constrained multi-model hosting | Recall/calibration and golden drift gate |
| CUDA FP32 | Explicit supported GPU deployment | Independent golden, batching, and soak tests |

Autoselection should use device availability plus workload profile. It should not silently change precision under memory pressure without recording the change.

### P6. Ranking-specific execution optimization

After two-stage generation:

- construct the focal base sequence once;
- normalize fixed values once;
- append/rewire only candidate-specific nodes;
- batch candidate sequences in length buckets;
- chunk by token-memory budget;
- cache candidate item tokens/schema embeddings;
- reuse candidate source features in the trained head;
- consider a lightweight dual-encoder score to eliminate most full RT calls.

Prefix/state caching inside the relational transformer should be treated as research, not an immediate commitment. Candidate rewiring can change attention paths across layers, so an invalid cache could improve speed while corrupting rankings.

### P7. Concurrency, warmup, and resource controls

- Preload configured checkpoints and preprocessing artifacts at service start.
- Warm representative length buckets.
- Make model-handle and embedding caches thread-safe.
- Bound concurrent long-context and ranking jobs separately.
- Add admission control by total tokens/candidates.
- Prevent one 1,000-candidate query from starving short binary requests.
- Expose model load, memory pressure, queue depth, and native errors.
- Soak-test repeated engine/index swaps and Python object finalizers.

### P8. Remove hidden CPU-only output paths

Zero-shot multiclass currently uses the extended text decoder path, which is CPU-only even when normal scoring selected MPS. CUDA target-feature extraction is also explicitly unsupported.

Options, in order of implementation risk:

1. extract target features on MPS, then apply the relatively small 512×384 text decoder on CPU;
2. add device-aware text decoder execution after MPS backbone inference;
3. implement and test target-feature/text-head support on CUDA.

Report effective device separately for backbone, decoder, and fitted head. Never label a request “MPS” if its dominant stage silently ran on CPU.

Acceptance:

- Multiclass outputs match the current CPU path within a declared tolerance.
- Warm multiclass latency improves on MPS workloads.
- Device fallback is explicit in PredictionResult telemetry.
- CUDA is not advertised for fine-tuned/multiclass paths until end-to-end support is tested.

## 10. Recall–performance tradeoff experiments

Every sampler/ranking change should produce a Pareto curve, not one point.

### Context curve

Evaluate token budgets such as 256, 512, 1,024, 2,048, 4,096, and 8,192:

- quality metric;
- evidence recall;
- relation coverage;
- known-label demonstrations;
- end-to-end p50/p95;
- native forward time;
- truncation/degraded rate.

### Cohort curve

Evaluate 0, 4, 16, 64, and 256 demonstrations or the closest feasible peer budgets:

- quality and calibration;
- batch stability;
- label/diversity coverage;
- tokens and latency.

### Ranking curve

Evaluate candidate M and final K:

- candidate recall@M;
- final Recall@K/NDCG;
- forwards/entity;
- latency/memory;
- novel-item recall.

### Seed ensemble curve

Evaluate one, two, four, and eight seeded contexts:

- mean quality;
- variance across seeds;
- calibration;
- cost multiplier.

Ship the smallest configuration on the Pareto frontier that satisfies the product quality target.

## 11. Delivery sequence and release gates

The phases below are gated by evidence, not calendar dates.

### Phase 0 — trustworthy harness

Deliver:

- repaired benchmark entry points;
- provenance schema;
- stage telemetry;
- frozen baseline results for the current behavior;
- invariance probes promoted to automated tests.

Gate:

- clean-checkout benchmark commands complete;
- failures never write successful-looking result artifacts;
- baseline metrics and stage timings are reproducible.

### Phase 1 — semantic stability

Deliver:

- versioned preprocessing/task-stat artifact;
- focal/demonstration ownership separation;
- per-entity temporal domains;
- fail-fast artifact/data validation;
- one token-aware budget.

Gate:

- batch/order/chunk invariance passes;
- cohort cannot alter focal labels;
- temporal injection audit passes across every evidence/domain path;
- unsupported behavior fails before scoring;
- no benchmark metric is accepted if its stability gate fails.

### Phase 2 — representation and sampling

Deliver:

- real-cell autocomplete;
- stable TaskSpec/materialized task-row experiment;
- seeded tiered sampler;
- relation quotas and context explainability;
- sampler differential fixtures against the reference where applicable.

Gate:

- token fields match reference fixtures for supported autocomplete/materialized cases;
- validation quality beats the Phase 1 context at the same or lower token budget;
- same seed is reproducible and seed variance is reported;
- no relation is silently starved.

### Phase 3 — recall

Deliver:

- similarity/diversity cohorts;
- two-stage ranking candidate generation;
- validation-gated task heads;
- classification threshold calibration;
- hybrid baseline/fallback;
- horizon-conditioned forecasts or explicit rejection.

Gate:

- candidate recall is high enough that reranker quality is the limiting factor;
- final ranking beats popularity on target repeat domains without losing novel-item coverage;
- binary recall target is met at its required precision;
- every activated head/hybrid beats its fallback on held-out data;
- subgroup and cold-start results do not regress outside declared tolerance.

### Phase 4 — serving performance

Deliver:

- dynamic scheduler and length buckets;
- bulk retrieval APIs and pushdown;
- bounded caches;
- ranking chunking;
- index generation/versioning;
- measured device/precision selection.

Gate:

- same-device outputs remain invariant;
- task metrics remain within quality tolerances;
- p50/p95/p99, throughput, memory, and queueing targets pass;
- overload degrades by admission/rejection, not silent truncation or precision changes.

### Phase 5 — distribution and operations

Deliver:

- native platform wheels;
- one consistent distribution name;
- clean install smoke tests;
- corrected release workflows/docs;
- model/index migration and rollback tooling;
- production dashboards and soak tests.

Gate:

- clean environment can install and parse/score without an out-of-band repository build;
- supported platform matrix is explicit;
- release artifacts reproduce tested hashes;
- reference and RelativeDB repositories remain clean after all automated tests.

## 12. Proposed release scorecard

These are initial engineering gates, to be replaced by product-specific service objectives when available.

### Stability gates

| Metric | Initial gate |
|---|---|
| Same entity, different batch/order/chunk | max absolute output delta ≤ 1e-6 on same device/format |
| Repeated seeded context | identical token tensors |
| Cohort effect on focal labels | exactly zero |
| Future-row/class/candidate injection | exactly zero effect before anchor |
| Unreported target/context truncation | zero |
| Unsupported ABLATE or multi-horizon behavior | validation error until implemented |
| Artifact mismatch | deterministic fail before retrieval/forward |
| Clean wheel native smoke | pass on every supported wheel |

### Performance gates

| Metric | Initial direction/gate |
|---|---|
| Short-context MPS throughput | at least 3× current single-request throughput when batching volume exists |
| Ranking full RT candidates | at least 80% reduction at unchanged Recall@K |
| Retrieval calls | bulk path grows by relation/hop, not traversed row count |
| Wasted assembled rows | zero unreported post-assembly tail discard |
| Cache behavior | bounded size plus hit/miss/eviction telemetry |
| Long-context/ranking overload | bounded by token/candidate admission control |
| Quantization | use only when capacity benefit is demonstrated and quality gate passes |

### Recall/quality gates

| Metric | Initial gate |
|---|---|
| Candidate recall@M | ≥95% of attainable relevant items where feasible |
| Ranking | beat time-aware popularity on target domains; report novel-item recall |
| Binary | beat recency/activity baselines on PR-AUC or meet recall-at-precision objective |
| Regression | beat persistence/global mean on declared primary metric |
| Multiclass | improve macro-F1, not only majority-class accuracy |
| Fine-tuned head | held-out improvement over zero-shot and declared baseline |
| Hybrid | held-out improvement with no unacceptable subgroup regression |
| Quantized model | no quality/calibration degradation beyond declared tolerance |

The gates should not be gamed by silently excluding difficult entities. Coverage and abstention rates belong beside every quality metric.

## 13. Test strategy

### 13.1 Property and invariance tests

Generate randomized small schemas/graphs and assert:

- batch membership, order, duplicates, and chunking do not change a focal result;
- context selection is deterministic for a seed;
- adding cohort data does not change focal aggregates;
- raising the token budget retains a superset for a fixed policy;
- no timestamp later than the effective bound enters any path;
- candidate/class domains are per entity;
- serialization does not change transforms or task identity;
- scanner order does not affect results when the policy promises order independence.

### 13.2 Differential tests

- Reference Rust sampler vs. RelativeDB supported sampler modes on the same tiny preprocessed graph.
- Reference PyTorch vs. C++ on multiple generated mask geometries, lengths, and batch shapes.
- Scalar retrieval vs. bulk retrieval.
- Retriever mode vs. CSC mode.
- FP32 CPU vs. FP32 MPS within tolerance.
- FP32 vs. F16/Q8/Q4 task metrics and calibration.
- Single-candidate scoring vs. batched/chunked candidate scoring.

### 13.3 Leakage tests

Inject a uniquely identifiable future:

- event row;
- parent/dimension row update;
- cohort peer;
- class;
- ranking candidate;
- statistics outlier;
- task label;
- counterfactual assignment.

Assert zero effect at earlier anchors and expected effect after the relevant time.

### 13.4 Quality tests

Maintain:

- tiny deterministic correctness fixtures;
- fast sampled benchmark smoke set;
- full three-domain temporal suite;
- Olist/GH experiments with stable provenance;
- at least a subset of official RelBench tasks;
- cold-start, sparse-history, high-fanout, and long-tail slices.

Quality CI should have two layers:

- per-PR smoke gates with broad regression tolerances;
- scheduled/full gates with stable datasets, confidence intervals, and stricter promotion rules.

### 13.5 Performance tests

Measure complete request latency, not only forward:

- pinned single entity;
- 80 short entities;
- one medium/long context;
- mixed-length batch;
- ranking at each candidate stage;
- cold model/index;
- warm model/index;
- concurrent short and long traffic;
- index refresh during active reads.

Fail performance tests on statistically meaningful regression over repeated trials, not one noisy sample.

### 13.6 Distribution/soak tests

- build and install each platform wheel in a clean environment;
- parse, CSC build, CPU score, and supported device score;
- repeatedly load/unload model/head/index artifacts;
- run multi-threaded scoring and cache eviction;
- switch index generations atomically;
- simulate missing/corrupt/mismatched artifacts;
- verify memory returns to a steady range.

## 14. File-level implementation map

| Area | Primary files | Planned responsibility |
|---|---|---|
| Context ownership/budget | python/src/relativedb/engine.py | Focal vs. demonstration contexts, token budget, fail-fast backend check, stats |
| Row/retrieval contracts | python/src/relativedb/retrieve.py | Bulk callbacks, row provenance, data validation, stable ordering |
| Evaluation | python/src/relativedb/evaluate.py | Owner-scoped aggregations and temporal-safe label evaluation |
| Preprocessing/model manifest | python/src/relativedb/model.py plus a new artifact module | Fixed statistics, task transforms, hashes, checkpoint/embedding compatibility |
| Tensor/target construction | python/src/relativedb/rt_native.py | Stable normalization, task representation, fixed domains, candidate pipeline |
| Index serving | python/src/relativedb/csc.py and csc_native.py | Bulk lookup, versioned generations, time-aware domain/cohort APIs |
| Native index ABI | cpp/src/csc_c.h/.cpp and csc.hpp/.cpp | children-many and compact/mmap-friendly index operations |
| Native inference | cpp/src/rt_c.h/.cpp and rt.* | Scheduler-friendly token batches, device/format telemetry, broader differential fixtures |
| Head training | cpp/src/rt_train* and rt_native.py | Validation hooks, best-weight restore, weighted/robust/ranking losses |
| RelQL validation | python/src/relativedb/relql/* and cpp/src/relql* | Reject unsupported ablation/horizons; stable canonical TaskSpec |
| Benchmark harness | benchmarks/harness/*, run.py, run_suite.py | Provenance, stage metrics, candidate/evidence recall, promotion gates |
| Native tests | cpp/src/test_* and benchmarks/xlang_fixture | Expanded geometry/device/format conformance |
| Python tests | python/tests/* | Invariance, temporal domains, artifacts, sampler, packaging contracts |
| Packaging/release | python/pyproject.toml, cpp/CMakeLists.txt, .github/workflows/* | Native wheels, clean install matrix, removal/restoration of stale jobs |
| Documentation | README.md, cpp/README.md, website/* | Supported contracts, model size/format, performance and quality provenance |

## 15. Highest-value first implementation batch

This is the smallest sequence likely to improve all three objectives without a large architecture rewrite.

### Change 1 — fail fast and freeze the harness

- Fix run_suite.py and audit_fixes.py backend wiring.
- Move backend/config validation before context assembly.
- Add provenance and stage timers.
- Save a current baseline.

### Change 2 — introduce PreprocessingArtifact

- Persist physical column, datetime, and task-target transforms.
- Key it to schema/task/checkpoint/cutoff.
- Remove implicit request-batch normalization in production mode.

### Change 3 — add invariance gates

- Promote the demonstrated batch and cohort probes into tests.
- Add temporal class/candidate injection tests.
- Block further quality claims until they pass.

### Change 4 — partition focal and peer contexts

- Ensure self-labels use focal rows only.
- Encode each peer as a separate demonstration.
- Expose ownership in explain output.

### Change 5 — unify the token budget

- Reserve target/focal/relation/demo tokens.
- Stop retrieval at admission.
- Eliminate unreported tail truncation.

### Change 6 — add candidate-recall instrumentation

- Measure candidate recall@M and novel-item rate with the current candidate set.
- This reveals whether ranking needs better retrieval, reranking, or both.

### Change 7 — implement the first two-stage candidate union

- Personal repeats + co-occurrence + popularity + novel/content source.
- Cheap score to top 100.
- RT rerank the retained set in chunks.

### Change 8 — add validation/rollback to heads

- Time validation split, early stopping, best checkpoint, calibration, baseline comparison.
- Do not activate failed heads.

### Change 9 — add dynamic short-sequence batching

- Start with one checkpoint/device and two or three length buckets.
- Preserve request order and record queue time.

### Change 10 — package the tested native runtime

- Produce platform wheels containing the library.
- Add clean install smoke tests.

## 16. Key design decisions to make explicitly

### Production statistics

Choose whether RelativeDB:

- requires scanners to fit an offline/initial serving artifact;
- accepts a user-supplied artifact;
- or supports both.

Do not retain silent per-request fitting as the production default.

### Online vs. snapshot serving

Decide the freshness contract:

- immutable versioned snapshot;
- append-only delta plus compaction;
- or live retrievers with a caller-guaranteed snapshot token.

A single request must observe one coherent generation.

### Task representation

Choose among:

- materialized real task table;
- user-declared stable derived task schema;
- task-specific head;
- or a supported combination by query class.

This should be decided by validation evidence, not convenience.

### Recall objective

For each product task define:

- positive event;
- label observation window;
- required precision/recall or ranking K;
- abstention/capacity policy;
- cost of false positive vs. false negative;
- subgroup constraints.

AUROC alone cannot select a production threshold.

### Checkpoint scope

Either:

- formally support only RT-J 12×512 safetensors and validate it strictly; or
- make the C++ geometry/config path dynamic and test legacy/configurable models.

Ambiguous partial support is less stable than either choice.

## 17. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Reference-like stochastic sampling hurts repeatability | Deterministic derived seeds; explicit multi-seed mode |
| Similarity cohorts leak target information | Use only features/labels available before each anchor; leakage injections |
| Fixed statistics become stale | Version by cutoff/index generation; scheduled refit; drift telemetry |
| Larger contexts improve one metric while hurting latency | Token-quality Pareto curves and relation quotas |
| Q4 saves memory but harms recall/calibration | Task-level promotion gate; retain FP32/Q8 fallback |
| Hybrid hides weak RT behavior | Report component and hybrid metrics separately |
| Head overfits tiny/imbalanced data | Validation, early stop, confidence checks, automatic fallback |
| Candidate generator reinforces popularity | Source quotas, diversity, novel-item coverage, subgroup metrics |
| Cached domains cross temporal boundaries | Bound/index generation in every cache key |
| Task representation diverges from pretraining | Reference differential fixtures and held-out task evaluation |
| Incremental index changes mid-request | Pin generation for request lifetime |
| Performance scheduler increases tail latency | Separate latency/throughput classes and bounded queue window |

## 18. Deferred work

The following should not be the first response to the current findings:

- another low-level GEMM rewrite;
- full foundation-model pretraining;
- automatic q4 everywhere;
- increasing every context to the maximum;
- shipping stochastic sampling without recorded seeds;
- activating heads based only on training loss;
- adding more task syntax whose execution is a placeholder;
- claiming reference parity from the kernel fixture alone.

Full or continued pretraining may eventually be valuable, especially for a stable RelQL task representation. It becomes worth evaluating after the serving input contract, sampler, and evaluator are reliable.

## 19. Expected outcome

If the roadmap is followed in order:

- **Stability** improves first through fixed artifacts, focal/demo separation, temporal bounds, explicit budgets, and fail-fast compatibility.
- **Recall** improves through higher-value evidence, representative demonstrations, candidate-source coverage, validated heads, calibrated thresholds, and hybrids with strong simple signals.
- **Performance** improves through bounded work, two-stage ranking, bulk retrieval, caching, batching, and workload-aware device selection.

The central shift is from “run RT-J over whatever context the current request happens to assemble” to “run a versioned, measurable predictive plan.” That plan should identify its data generation, task, cutoff, statistics, sampler, candidates, model, and fallback. Once those are stable, both quality and speed become optimizable rather than anecdotal.
