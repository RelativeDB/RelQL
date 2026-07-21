# RelativeDB Stability, Performance, and Recall Improvement Roadmap

**Prepared:** 2026-07-19  
**Based on:** RELATIONAL_TRANSFORMER_COMPARISON.md and the implementation/worktree reviewed there  
**Purpose:** Convert the comparison findings into an ordered engineering and evaluation program

**Updated:** 2026-07-20 after the first semantic-alignment implementation.
Status annotations distinguish completed foundations from the remaining
production-hardening and quality-validation work. See
`REFERENCE_ALIGNMENT_IMPLEMENTATION.md` for the implemented API and test log.

## 1. Executive recommendation

The next investment should be in the semantic path into RT-J, not in lower-level matrix multiplication.

The native transformer remains the most mature part of the system. The first
semantic-alignment batch has now removed two demonstrated correctness defects
and introduced foundations for the third:

1. Request-batch normalization is removed. Zero-shot normalization is
   entity-local, while reference/statistics normalization uses persisted
   physical-column, datetime, and task-target statistics.
2. Focal ownership is explicit. Derived queries materialize stable timestamped
   task rows, and every peer label is evaluated against that peer's history at
   the legal historical cutoff.
3. Direct targets mask physical cells. `ReferenceTraversal` is pluggable and is
   now the default over one immutable graph snapshot. It implements the
   reference graph walk, target/visited/fallback tiers, prioritized BFS,
   temporal rules, stable global node IDs, and rand 0.9.1 RNG streams. The old
   cohort BFS remains only as an explicit compatibility plugin.
4. Ranking evaluates a broad, weakly generated candidate set and pays transformer cost for each candidate.
5. Release and packaging paths do not yet reliably gate regressions.

The recommended order is therefore:

1. **Stabilize semantics and measurement.**
2. **Improve evidence and candidate recall.**
3. **Optimize the now-stable workload.**
4. **Add supervised calibration and hybrid fallbacks with automatic rollback.**

The stability foundations are now in place, so the next work should harden
their artifacts and invariance gates, then proceed to candidate recall and
performance. A new independent `evaluation/` harness now provides reference-
context RT, SQL/XGBoost, native zero-shot, and native fine-tuned comparisons;
future evaluation must continue to exclude future information.

## 2. What “recall” means in this roadmap

Recall is overloaded in this system. Every future evaluation should state which of these it targets:

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
| Historical: C7 regression and binary outputs changed when C1 joined the batch | Resolved by per-entity zero-shot normalization; retained as motivation and a permanent invariance gate |
| Historical: a cohort changed C7's own historical count from 1 to 2 | Resolved for self-label evaluation through `focal_row_keys`; full peer demonstration separation remains |
| Retained verification passes after cleanup and reference alignment | Batch invariance, persisted stats, task identity, traversal injection, exact RNG/index-sampling vectors, snapshot immutability, FK/PK rules, and leakage defense have regression coverage |
| C++ CPU/MPS max drift is about 0.00391 from PyTorch | Native optimization is not the primary correctness gap |
| MPS 80×16 FP32 is about 0.8 ms/entity vs. 7.6 ms for 1×16 | A serving scheduler can unlock much more throughput than another kernel rewrite |
| Q8/Q4 reduce resident memory substantially but do not improve all measured latencies | Quantization should be selected for capacity, with quality gates, not assumed to be faster |
| Head fitting currently reports training loss without a held-out promotion gate | Training requires validation, early stopping, and automatic fallback |
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

| ID | Improvement | Status | Stability | Performance | Recall | Dependency |
|---|---|---|---:|---:|---:|---|
| S1 | Persist fixed column and task statistics | Foundation complete | Critical | Medium | High | None |
| S2 | Separate focal evidence from demonstrations | Complete for reference task rows | Critical | Medium | High | None |
| S3 | Apply per-entity temporal bounds everywhere | Partial | Critical | Low | High | None |
| S4 | Introduce a stable task/target representation | Complete | High | Low | High | S1–S3 |
| S5 | Enforce artifact compatibility and data contracts | Not started | High | Medium | Medium | None |
| S6 | Make budgets token-aware and truncation explicit | Reference cell geometry complete; provenance remains | High | High | High | S2 |
| R1 | Add a seeded, tiered, task-aware sampler | Reference parity complete; optional FAISS/balancing remain | High | Medium | High | S1–S6 |
| R2 | Add similarity and label-aware cohort selection | Not started | Medium | Medium | High | R1 |
| R3 | Build two-stage ranking candidate generation | Not started | Medium | Critical | Critical | S3 |
| R4 | Harden head training and threshold calibration | Gap confirmed by held-out regression | High | Low | High | S1–S4 |
| R5 | Add validated hybrid baselines/fallbacks | Not started | High | Low | High | R4 |
| R6 | Implement real horizon conditioning | Not started | High | Medium | High | S4 |
| R7 | Version multiclass vocabularies/class retrieval | Not started | High | High | High | S3–S5 |
| P1 | Add dynamic batching and length buckets | Not started | Medium | Critical | Neutral | S1 |
| P2 | Batch retrieval and push down query constraints | Not started | Medium | High | Medium | S6 |
| P3 | Cache immutable preprocessing and schema work | Not started | Medium | High | Neutral | S1, S5 |
| P4 | Add incremental/mmap serving indexes | Not started | Medium | High | Medium | S3, S5 |
| P5 | Select device/precision by measured policy | Partial | Medium | High | Medium | None |
| P8 | Remove hidden CPU-only output paths | Not started | Medium | High | Neutral | P5 |
| O1 | Package native wheels and restore CI/release integrity | Not started | High | Medium | Neutral | S5 |

“Critical” indicates a likely order-of-magnitude lever or a prerequisite for trustworthy operation. “Neutral” means the change should preserve recall and requires a no-regression gate.

## 6. Evaluation infrastructure

The previous `benchmarks/` tree remains intentionally deleted. Its entry
points, bespoke datasets, formats, scorecards, and historical conclusions were
not restored. The replacement is the independent `evaluation/` package: a
ported 21-task catalog, optional RT/SQL-XGBoost/native/native-fine-tuned
runners, exact reference sampled tensors for both native paths, official keyed
RelBench scoring, and Markdown/JSON reports. The executed rel-f1 pair proves
the end-to-end path; broad quality claims still require the complete catalog,
seeds, confidence intervals, and declared promotion thresholds.

## 7. Stability workstream

### S1. Fixed statistics artifacts

**Status: foundation complete; production artifact metadata remains.**

#### Problem

At the time of the original review, `RtNativeBackend._normalize` derived
physical and synthetic-label statistics across the active score batch. This
was the demonstrated source of batch-dependent output.

Current code has two explicit modes. `ZERO_SHOT` derives statistics separately
per entity context. `REFERENCE`/`STATISTICS` requires `ColumnStats` containing
physical-column, global datetime, and derived-task transforms. Missing entries
raise instead of silently falling back. Fitted adapter heads persist both the
mode and statistics, and adapter fitting collects labels before fitting task stats.
The current task-stat implementation applies numeric mean/std uniformly;
task-specific transforms below remain a design and validation step.

#### Design

Extend the implemented `ColumnStats`/head-sidecar foundation into a versioned
`PreprocessingArtifact` with:

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

The backend now has the intended two modes:

1. artifact-backed `REFERENCE`/`STATISTICS` mode; and
2. explicitly named artifact-free `ZERO_SHOT` mode.

Strict missing-stat behavior is implemented. Surfacing the active regime in
`PredictionResult` and adding schema/task/checkpoint hashes remain.

#### Acceptance

- Same focal output under single-row, mixed batch, reversed order, duplicates, and multiple chunk sizes; target maximum delta at most 1e-6 on one device/precision.
- Artifact fit uses no row later than its cutoff.
- Serialization round-trip produces bit-identical transforms.
- Missing/incompatible artifacts fail before retrieval in production mode.

Implemented regression coverage already proves per-entity zero-shot batch
invariance and persisted-stat transform use. Order/duplicate/chunk tests and
full artifact compatibility checks remain acceptance work.

#### Expected impact

- Stability: eliminates the demonstrated batch dependence.
- Performance: avoids repeated mean/std scans of every sequence.
- Recall: makes calibration and head training reproducible; prevents thresholds from moving with cohort composition.

### S2. Separate focal evidence from demonstrations

**Status: complete for the reference task-row representation.**

#### Problem

Historically, `EntityContext.rows` merged focal and cohort subgraphs and
`_self_labels` evaluated the target over all rows. `EntityContext` now records
`focal_row_keys`; both traversals propagate ownership, assumptions preserve it,
and self-label evaluation uses `focal_rows_by_table()`.

#### Design

The implemented representation contains:

- focal_rows: facts reachable from the focal entity;
- demonstrations: timestamped peer task rows, each connected to its own entity
  graph and carrying a known historical label;
- shared_rows: optional immutable dimension rows referenced by either group.

Peer task rows are now materialized for configurable prior windows. FOLLOWING
labels may see through the end of their own window but are capped at the focal
anchor. Graph edges preserve peer ownership and the reference task-table P2F
restriction. Richer role/provenance rendering in explain output remains useful
but is not a sampling-parity gap.

#### Acceptance

- Focal self-labels are invariant to cohort size, cohort order, and peer row count.
- Adding a demonstration can change the model score, but cannot change factual counters or label values in explain output.
- Tests cover identical primary keys in different tables and shared dimension rows.
- Explain context visually separates focal, demonstration, and shared evidence.

### S3. Per-entity temporal bounds

**Status: row traversal and statistics cutoff complete; batch-wide
class/candidate domains remain.**

#### Problem

Multiclass and ranking domain enumeration still use the maximum anchor in a
batch. In contrast, both traversal implementations defensively recheck every
row against the focal `TemporalBound`, and reference statistics fitted during
fine-tuning use the last training anchor as an explicit cutoff.

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

**Status: complete.**

#### Problem

The original implementation emitted one generic `task.label` row. Current code
uses `TaskSpec.from_query`: it canonicalizes the validated target AST and
derives a stable task ID/table/column. Direct entity-column autocomplete masks
the physical cell on the entity node.

#### Design

The implemented paths are:

1. **Entity-column autocomplete:** mask the actual focal cell in place and retain its real table and column embedding.
2. **Materialized tasks:** implemented as stable timestamped task rows with an
   entity edge and task-specific target column; the focal target is masked and
   emitted first.
3. **Derived RelQL targets:** implemented as a canonical `TaskSpec` whose
   identity includes the entity, target AST, task type, horizons,
   aggregations, and filters represented in that AST.

`TaskSpecFactory` remains injectable for deployments that want registry-owned
table/column names rather than the canonical hash-derived names.

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
- validate scalar/list FK value types; list FKs are supported as multiple graph
  edges and, when opted in as features, one stable text token;
- report dangling FKs and duplicate IDs;
- normalize all timestamps to one UTC contract;
- enforce stable scanner ordering or sort by a documented key;
- report tokenless connector rows before serving.

### S6. Token-aware budgets and fail-closed behavior

**Status: traversal admission is bounded and reported; unified model-token
budget remains.**

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
- Truncation rate and quality-by-truncation cohort are exposed to the future
  evaluation system.

## 8. Recall workstream

### R1. Seeded, tiered, task-aware sampling

**Status: reference parity complete for graph-walk sampling; optional reference
FAISS retrieval and label balancing remain.**

#### Goal

Increase useful evidence per token rather than simply increasing context size.

#### Implemented reference contract

`Engine(..., traversal=...)` accepts a `GraphTraversal`.
`ReferenceTraversal` is the default and consumes a one-time immutable
bidirectional snapshot. `BreadthFirstTraversal` explicitly preserves the old
pull-per-hop behavior.

The reference path now performs 10,000 length-20 walks by default over all F2P
and temporally valid P2F neighbors, then fills context in the same three tiers:

1. target BFS;
2. visited same-table seeds ordered by timestamp/count/random priority;
3. randomly sampled unvisited same-table fallback.

Within every seed BFS, F2P uses a LIFO priority stack and P2F draws from the
shallowest frontier. DB children are uniformly sampled to width 32. Task-table
P2F is legal only from a same-task seed. `visited_at_depth` is shared across
seeds, node cells emit once, and the target cell is always first.

Randomness ports rand 0.9.1 `StdRng` (ChaCha12), its `seed_from_u64` PCG
expansion, Canon integer sampling, and index sampling. Regression vectors were
generated by a Rust oracle and match exactly.

The remaining optional reference modes are:

- FAISS same-table similarity in place of walk ranking;
- historical-label balancing when assembling training/evaluation contexts.

#### Determinism

The configured context seed, step, and snapshot-global target node ID feed the
same wrapping seed arithmetic as the reference. One seed produces one context.

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

The compatibility BFS can still fall back to first scanner rows, which are
neither similar nor representative. `ReferenceTraversal` improves diversity
and determinism through walk-ranked peer seeds, but it does not yet use feature
similarity, task labels, or maximal-marginal-relevance selection.

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

This addresses the structural limitation of repeat-only retrieval: it cannot
recall a future item that has never been seen by the entity.

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

- Candidate recall@M reaches a declared target on future evaluation tasks.
- Full RT candidates are reduced by at least 80% from the current cap without reducing final Recall@K.
- Novel-item recall is nonzero on datasets where future novel items exist.
- Candidate generation is temporally valid and invariant to batch composition.

### R4. Make fine-tuning safe to promote

**Status:** implemented for binary classification and regression on native
C++/MPS. The old scalar frozen-head path is no longer called fine-tuning.
`Engine.fit_head()` remains available for multiclass and ranking adapters.

The full-model trainer now has the controls needed for a long run:

- it trains on the task train split and scores the validation split separately;
- it starts with zero-shot as the best checkpoint;
- it accumulates small physical MPS batches into the requested effective batch;
- it clips the global gradient norm before AdamW updates;
- it saves model weights, optimizer moments, sampler position, and validation
  history at atomic recovery points;
- it restores the best checkpoint and lowers the learning rate when validation
  gets worse; and
- it stops after the configured validation patience is exhausted.

A falling training loss is not enough to ship a model. Promotion requires a
held-out validation improvement over the current best checkpoint, valid class
coverage, and no leakage or stability failure. Test data is used only for the
final report after model selection. Generated checkpoints and scores remain
private under `evaluation/runs/`.

The default long-context evaluation keeps 8,192 cells. On memory-limited Apple
Silicon, only the physical microbatch is reduced; gradient accumulation keeps
the effective batch at 32. This is deliberately slower than the reference
script's usual 1,024-cell task fine-tuning. Performance work should focus on
reducing Metal synchronization and improving command submission without
changing the training contract.

Remaining work:

- complete substantive classification and regression runs through validation;
- run the selected checkpoints once on the untouched test split;
- add probability calibration and product-specific threshold selection;
- add class weighting or focal loss where validation shows a recall benefit; and
- extend full-model training beyond scalar tasks only after their loss and
  grouping rules are defined against the reference implementation.

### R5. Validated hybrid models and fallbacks

Simple domain signals—such as recency, persistence, and popularity—may be
stronger than RT-J on particular tasks.

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

Short-sequence batching is expected to provide a substantial throughput
opportunity. The new head-to-head harness records runner wall time, but its
current two-task execution is a functional comparison rather than a controlled
serving-performance study.

Add an inference scheduler that:

- groups requests by checkpoint, precision, device, task/output head, and preprocessing artifact;
- buckets by sequence length to reduce padding;
- uses a short configurable batching window;
- caps tokens rather than only rows per batch;
- splits oversized contexts;
- preserves request order at result assembly;
- reports queue time separately from compute time;
- supports latency and throughput service classes.

Future measurement should cover:

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

## 10. Evaluation expansion

The replacement harness deliberately retains nothing from the removed legacy
implementation. Next, expand its current reference-context RelBench slice to
all 21 tasks and multiple deterministic seeds, add confidence intervals and
calibration/latency artifacts, then define promotion thresholds. Ranking and
product-specific recall still need separate leakage-safe datasets because the
reference catalog does not exercise those product paths.

## 11. Delivery sequence and release gates

The phases below are gated by evidence, not calendar dates.

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
- no future quality result is accepted if its stability gate fails.

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

## 12. Current release correctness gates

These gates are independent of the predictive evaluation harness.

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

Predictive-quality gates now use the independent `evaluation/` harness.
Throughput gates still require a controlled serving-performance design rather
than the current end-to-end smoke timings.

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

### 13.4 Distribution/soak tests

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
| Context ownership/budget | python/src/relativedb/engine.py | Implemented focal ownership; remaining peer roles, unified token budget, and stats reporting |
| Traversal policy | python/src/relativedb/traversal.py | Pluggable BFS/reference traversal; remaining relation quotas, task-aware selection, and quality promotion |
| Row/retrieval contracts | python/src/relativedb/retrieve.py | Bulk callbacks, row provenance, data validation, stable ordering |
| Evaluation | python/src/relativedb/evaluate.py | Owner-scoped aggregations and temporal-safe label evaluation |
| Preprocessing/model manifest | python/src/relativedb/model.py and rt_native.py `ColumnStats` | Implemented modes/transforms; remaining artifact hashes, counts, checkpoint/embedding compatibility |
| Stable task identity | python/src/relativedb/task.py | Implemented canonical `TaskSpec`; remaining materialized task schemas and validation metrics |
| Tensor/target construction | python/src/relativedb/rt_native.py | Implemented stable normalization/direct targets; remaining fixed domains and candidate pipeline |
| Index serving | python/src/relativedb/csc.py and csc_native.py | Bulk lookup, versioned generations, time-aware domain/cohort APIs |
| Native index ABI | cpp/src/csc_c.h/.cpp and csc.hpp/.cpp | children-many and compact/mmap-friendly index operations |
| Native inference | cpp/src/rt_c.h/.cpp and rt.* | Scheduler-friendly token batches, device/format telemetry, broader differential fixtures |
| Head training | cpp/src/rt_train* and rt_native.py | Validation hooks, best-weight restore, weighted/robust/ranking losses |
| RelQL validation | python/src/relativedb/relql/* and cpp/src/relql* | Reject unsupported ablation/horizons; stable canonical TaskSpec |
| Native tests | cpp/src/test_* and cpp/testdata | Expanded geometry/device/format conformance |
| Python tests | python/tests/* | Invariance, temporal domains, artifacts, sampler, packaging contracts |
| Packaging/release | python/pyproject.toml, cpp/CMakeLists.txt, .github/workflows/* | Native wheels, clean install matrix, removal/restoration of stale jobs |
| Documentation | README.md, cpp/README.md, website/* | Supported contracts, model size/format, performance and quality provenance |

## 15. Highest-value first implementation batch

This is the smallest sequence likely to improve all three objectives without a large architecture rewrite.

### Change 1 — introduce PreprocessingArtifact

**Status: partially complete.**

- Completed: persist physical column, datetime, and task-target transforms in
  `ColumnStats` and head sidecars.
- Completed: remove implicit request-batch normalization; zero-shot is
  per-entity and reference mode is strict.
- Remaining: key a first-class artifact to schema/task/checkpoint/cutoff and
  surface its identity in results.

### Change 2 — add invariance gates

**Status: partially complete.**

- Completed: promote batch invariance, persisted-stat behavior, task identity,
  traversal determinism/injection, and leaky-retriever defense into tests.
- Remaining: explicit cohort-order/self-label, duplicate, pagination, chunk,
  and temporal class/candidate injection matrices.
- Block further quality claims until they pass.

### Change 3 — partition focal and peer contexts

**Status: partially complete.**

- Completed: track focal row keys and ensure self-labels use focal rows only.
- Remaining: encode each peer as a separate labeled demonstration and expose
  ownership in explain output.

### Change 4 — unify the token budget

- Reserve target/focal/relation/demo tokens.
- Stop retrieval at admission.
- Eliminate unreported tail truncation.

### Change 5 — add candidate-recall instrumentation

- Measure candidate recall@M and novel-item rate with the current candidate set.
- This reveals whether ranking needs better retrieval, reranking, or both.

### Change 6 — implement the first two-stage candidate union

- Personal repeats + co-occurrence + popularity + novel/content source.
- Cheap score to top 100.
- RT rerank the retained set in chunks.

### Change 7 — add validation/rollback to heads

- Time validation split, early stopping, best checkpoint, calibration, baseline comparison.
- Do not activate failed heads.

### Change 8 — add dynamic short-sequence batching

- Start with one checkpoint/device and two or three length buckets.
- Preserve request order and record queue time.

### Change 9 — package the tested native runtime

- Produce platform wheels containing the library.
- Add clean install smoke tests.

## 16. Key design decisions to make explicitly

### Production statistics

**Decision made:** support both user-supplied/fitted reference statistics and
artifact-free zero-shot operation through an explicit configuration enum.

Remaining production decision: whether scanners fit the reference artifact at
engine/index initialization or deployments must always supply one. In either
case, extend the current `ColumnStats` representation with version and
compatibility metadata.

RelativeDB now:

- accepts a user-supplied `ColumnStats` artifact;
- fits bounded statistics automatically during fine-tuning; and
- supports explicit per-entity `ZERO_SHOT` operation without an artifact.

Silent cross-entity request fitting is removed.

### Online vs. snapshot serving

Decide the freshness contract:

- immutable versioned snapshot;
- append-only delta plus compaction;
- or live retrievers with a caller-guaranteed snapshot token.

A single request must observe one coherent generation.

### Task representation

**Decision made for the first implementation:** physical-cell masking for
entity autocomplete and canonical stable synthetic `TaskSpec` identities for
derived RelQL targets. `TaskSpecFactory` is injectable for an external task
registry.

Still choose, using held-out validation, when to prefer:

- materialized real task table;
- user-declared stable derived task schema;
- task-specific head;
- or the implemented canonical synthetic schema.

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

The first stability foundations—explicit normalization modes, focal label
ownership, stable task identity, and pluggable deterministic traversal—are now
implemented. If the remaining roadmap is followed in order:

- **Stability** advances from those foundations through versioned artifact
  compatibility, complete peer/demo separation, per-entity domains, explicit
  token budgets, and fail-fast loading.
- **Recall** improves through higher-value evidence, representative demonstrations, candidate-source coverage, validated heads, calibrated thresholds, and hybrids with strong simple signals.
- **Performance** improves through bounded work, two-stage ranking, bulk retrieval, caching, batching, and workload-aware device selection.

The central shift is from “run RT-J over whatever context the current request happens to assemble” to “run a versioned, measurable predictive plan.” That plan should identify its data generation, task, cutoff, statistics, sampler, candidates, model, and fallback. Once those are stable, both quality and speed become optimizable rather than anecdotal.
