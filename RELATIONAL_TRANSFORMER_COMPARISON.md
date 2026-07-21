# Relational Transformer vs. RelativeDB

## Comprehensive implementation-difference and conformance report

**Reviewed:** 2026-07-19  
**Reference:** `~/relational-transformer` at `eece04847de7b52d6fe7a718c277abec7bb18c83`  
**Implementation under review:** `~/relativedb` at `c361db29d9584dcedf7bf89145d89256807333f8`, plus the existing uncommitted worktree changes listed below

**Implementation update:** 2026-07-20. Sections marked “Current status” were
reconciled after RelativeDB added configurable zero-shot/reference
normalization, stable `TaskSpec` target identities, focal-row ownership, and a
pluggable reference-style traversal. Historical probes remain in the report as
evidence of the defects that motivated those changes. Implementation details
and configuration examples are recorded in
`REFERENCE_ALIGNMENT_IMPLEMENTATION.md`.

## 1. Purpose, scope, and interpretation

This is a repository-wide comparison of Stanford's `relational-transformer` implementation with RelativeDB's implementation in this repository. It covers repository structure, data contracts, preprocessing, context sampling, tensor construction, model math, checkpoints, runtime backends, query/task semantics, training, evaluation, packaging, tests, CI, and documentation.

The two repositories are not forks. Only `.gitignore` and `README.md` occur at the same tracked path; every other tracked path differs. A literal line diff would therefore be mostly meaningless. “All differences” in this report means:

1. every material system, feature, API, and behavioral difference found by a full tracked-file review;
2. every observed conformance point;
3. every correctness or release risk found through targeted execution;
4. an exhaustive tracked-file/subsystem disposition in Appendix A.

Generated corpora, build products, caches, checkpoint payloads, and notebooks'
binary cell output are not compared byte-for-byte.

### Worktree state

The reference repository was clean. This repository already contained user changes before the review:

- modified: `README.md`, `python/src/relativedb/engine.py`, `python/src/relativedb/rt_native.py`, `python/tests/test_engine.py`, `python/tests/test_rt_native.py`, and `python/tests/test_xlang_parity.py`;

Those changes were reviewed as part of the current implementation and were not reverted or rewritten. This report is the only file added by the review.

## 2. Executive conclusion

RelativeDB is **not a reimplementation of the reference product**. It is a new online predictive-query system around a faithful native inference port of one particular reference checkpoint family:

- The **model core is strongly conformant**. The current reference PyTorch model exactly reproduces RelativeDB's committed golden fixture, and RelativeDB's C++ CPU and MPS paths reproduce it within about `3.91e-3` maximum absolute drift.
- The **end-to-end prediction system is not reference-equivalent**. RelativeDB changes the target-row representation, normalization, cohort construction, graph sampling, temporal handling, supported tasks, and output decoding. These differences are large enough to change predictions even when the native transformer kernel itself is correct.
- RelativeDB adds substantial product surface absent from the reference: RelQL, online callbacks, point-in-time query planning, assumptions, explanations, native CPU/MPS/CUDA inference, quantization, ranking/multiclass adapters, frozen-backbone head training, and full-checkpoint scalar-task MPS fine-tuning.
- RelativeDB omits substantial research surface present in the reference: RelBench preprocessing, rkyv/mmap datasets, full-model pretraining, DDP/SWA, task/config-aware checkpoint loading, legacy RT/PluRel compatibility, and most of the `rel2tab` framework. Reference traversal and official evaluation are now available through the product sampler and isolated `evaluation/` adapter respectively.

The original review's highest-priority defects were outside the transformer
math. Batch-dependent normalization is now removed, targets have stable task
identities, and context construction now implements the reference sampler over
an immutable bidirectional snapshot. Derived queries materialize timestamped
task rows with peer labels, and the target task cell is always emitted first.
Primary keys never become features. Foreign keys remain graph structure by
default; RelativeDB's only intentional sampling/input extension is an opt-in,
non-targetable FK feature token, including stable text serialization for list
FKs while retaining every graph edge.

## 3. Prioritized findings

| Priority | Finding | Evidence | Consequence |
|---|---|---|---|
| Resolved P0 | Default zero-shot normalization was fitted across the current scoring batch. | Historical probe in §10.2; current regression test `test_zero_shot_normalization_is_batch_invariant` | Zero-shot is now entity-local and batch invariant; reference mode uses strict persisted statistics. |
| Resolved P0 | RelativeDB used a generic synthetic `task.label` target. | Current `python/src/relativedb/task.py`, `traversal.py`, and sequence builder | Entity-column autocomplete masks the physical cell; derived targets use canonical identities and materialized timestamped task rows with peer labels. |
| Resolved P0 | Cohort rows entered the focal entity's self-label aggregation. | Historical probe in §10.3; current `EntityContext.focal_row_keys` | Self-label evaluation now uses the focal subgraph only. Fully separated peer demonstration objects/explain output remain future work. |
| Resolved P1 | Context construction differed from the reference sampler. | Current `python/src/relativedb/traversal.py`; `rustler/src/fly.rs:1116-1998` | `ReferenceTraversal` is now the default and implements the reference graph walks, three seed tiers, F2P-priority/P2F-width BFS, temporal rules, cell geometry, stable node IDs, and rand 0.9.1 ChaCha12 stream. Legacy BFS is an explicit plugin. |
| P1 | The built Python wheel does not contain or build `librt_c`. | clean wheel/install probe in §10.7 | The advertised parser, CSC, and native inference are unavailable after a normal wheel install. |
| P1 | Forecast horizons repeat one scalar rather than making horizon-specific predictions. | `python/src/relativedb/rt_native.py:960` | A multi-horizon query returns duplicated values, not a forecast curve. |
| P1 | `ABLATE` parses and appears in plans but is not executed. | `python/src/relativedb/engine.py:1012` | A scientifically meaningful reference ablation is currently a declared no-op. |
| P1 | Multiclass/ranking candidate enumeration uses the maximum bound across a batch. | `python/src/relativedb/rt_native.py:1334` | Later candidates/classes can become visible to entities with earlier anchors. |
| P1 | Checkpoint metadata is ignored and the documented embedding guard is not called. | `python/src/relativedb/model.py:64`; `rt_native.py:672-709` | Architecture/encoder mismatches can load without the intended fail-fast validation. |
| P1 | Release workflows reference removed Rust, Java, and DuckDB trees. | `.github/workflows/release-libraries.yml`; `.github/workflows/duckdb-extension.yml` | Current release jobs cannot succeed as written. |
| P2 | Package name and README install command disagree. | `python/pyproject.toml:6`; `README.md:91` | `pip install relativedb` does not match distribution name `relationdb`. |
| P2 | Model-size and checkpoint-format documentation is inconsistent. | `README.md:26,83`; `cpp/README.md:44,195-236` | The current 86M-parameter BF16 checkpoint is described as 22M and “fp32”/171 MB. |
| P2 | `python/pyproject.toml` names a missing package README. | `python/pyproject.toml:9` | Wheel/sdist builds warn and ship incomplete project metadata. |

## 4. System identity and architecture

| Dimension | Reference: `relational-transformer` | RelativeDB |
|---|---|---|
| Primary purpose | Research/training/evaluation implementation of Relational Transformer | Online, storage-agnostic predictive query engine using an RT-J runtime |
| Python distribution / import | `relational-transformer` / `rt` | `relationdb` / `relativedb` |
| Version | 1.1.0 | 0.1.0 |
| Python floor | 3.12 | 3.10 |
| Build backend | maturin/PyO3 | setuptools plus a separate CMake build |
| Native code | Rust preprocessing and sampling extension | C++ parser, CSC, inference, quantization, and training; Metal/CUDA source |
| Data mode | Offline RelBench/manifest/Parquet preprocessing | Live user callbacks or an in-memory CSC snapshot |
| User task interface | Python `Task` records and scripts | RelQL strings, AST, execution API, and Python schema/retriever objects |
| Model role | Configurable PyTorch training and inference | Hard-coded RT-J native inference; native scalar full-checkpoint fine-tuning plus optional frozen-head adaptation |
| Main devices | PyTorch CUDA; CPU/MPS eager possible for small examples | CPU, MPS, and optional CUDA inference; native MPS full-model/head training |
| Evaluation | Curated RelBench tasks and `rel2tab` baselines | Independent `evaluation/` four-runner RelBench harness; legacy `benchmarks/` remains deleted |
| Documentation | Markdown guides and examples | Root/product docs plus Docusaurus site |

### 4.1 Reference data flow

`RelBench/manifest + Parquet → Rust preprocessor → rkyv/mmap graph + embeddings → Rust sampler → PyTorch RT → RelBench evaluator`

Full pretraining uses the same sampling stack with PyTorch optimization, distributed execution, checkpointing, and stochastic weight averaging.

### 4.2 RelativeDB data flow

`Schema + Row callbacks/scanners → live retriever or CSC snapshot → RelQL validation/planning → Python context builder/tokenizer → C++ RT-J → typed query results`

`Engine.fit_head()` freezes the native backbone, extracts target features,
trains a small linear head on Metal, and serves that adapter over backbone
features. `Engine.finetune()` is a separate C++/MPS path: it differentiates
through the encoders, sparse relational attention, every transformer block,
normalizations, learned mask, and numeric decoder, then exports a complete
safetensors checkpoint. Binary/regression are supported; multiclass/ranking
remain on the explicitly named frozen adapter path.

These are different system boundaries. The reference owns dataset transformation and training. RelativeDB owns online query semantics and serving.

## 5. Repository scale and history

| Measure | Reference | RelativeDB |
|---|---:|---:|
| Tracked files | 86 | 120 |
| Commits in inspected history | 51 | 21 |
| First inspected commit date | 2025-07-10 | 2026-07-17 |
| Snapshot date | 2026-07-10 | 2026-07-19 |
| Core Python LOC | 3,350 in `src/rt` | 4,749 in `python/src/relativedb` |
| Native LOC | 3,413 Rust | 7,144 C++/Metal/CUDA |
| Core test LOC | 218 Python | 2,049 Python + 537 C++ |
| Extra research/evaluation LOC | 4,823 `rel2tab` + 4,127 scripts | About 923 Python lines in the new isolated `evaluation/` package |
| Site/docs LOC | 856 docs/examples | 1,855 website source/docs |

The LOC figures are physical line counts for the listed tracked source groups, not complexity measures.

## 6. Detailed difference catalog

### 6.1 Installation, packaging, and dependency model

**Reference**

- Uses maturin, and the wheel is designed to contain `rt._rustler`.
- Requires Python 3.12 and declares PyTorch, safetensors, embedding, serialization, and runtime dependencies.
- Keeps RelBench as a separately installed pinned Git dependency in the Pixi environment.
- Offers `eval` and `test` extras and Pixi tasks for preprocessing, pretraining, evaluation, baselines, and tests.
- The checked Pixi platform is Linux x86-64 only.
- Does not contain GitHub Actions in this snapshot, but has pre-commit configuration.

**RelativeDB**

- Builds a pure-Python wheel with setuptools. Native components are compiled independently with CMake and discovered dynamically.
- Requires Python 3.10 and fewer declared Python dependencies.
- Has no packaging hook for `librt_c`, no platform wheel build, and no fallback parser implementation used as the supported public path.
- The project name is `relationdb`, while the README says `pip install relativedb`.
- The project README path is resolved relative to `python/`, where no `README.md` exists.
- Declares Apache-2.0 metadata but has no tracked `LICENSE` file. The reference also has no tracked license file or license metadata in its project table.
- Has GitHub Actions, but two workflows target directories that no longer exist.

### 6.2 Schema and row contracts

**Reference**

- Consumes RelBench 3-style manifests and Parquet data.
- Infers/records semantic types and normalizes database identity into dense integer node indices.
- Removes PK/FK cells from feature emission and stores graph links separately.
- Supports scalar and list-valued foreign keys during preprocessing.
- Represents explicit forecast labels as real rows in real task tables. Autocomplete targets are real existing feature cells.
- Stores preprocessed nodes, adjacency, statistics, text vocabulary, and embeddings as disk artifacts suitable for mmap.

**RelativeDB**

- Requires an explicit Python `Schema` of `TableDef`, `LinkDef`, and `ValueType`.
- Accepts arbitrary application IDs and `Row` values from entity, child, scanner, and optional cohort callbacks.
- Expects FK identity in `Row.parents`, separate from `Row.cells`.
- Has no manifest/Parquet preprocessor or portable serialized dataset format comparable to the reference.
- The CSC mode snapshots all scanner output when the engine initializes; it is not a continually updated database index.
- The public contract says null cells should be omitted. If a caller includes `None` in `Row.cells`, the tokenizer still emits a typed zero value rather than omitting the cell.
- `Row.parents` accepts scalar and list/tuple FK identities. List FKs create
  one graph edge per value. When FK feature emission is enabled, the whole list
  is additionally serialized as one stable compact text token.

### 6.3 Statistics and normalization

**Reference**

- Computes numeric statistics per physical column during preprocessing.
- Uses fixed train-derived task statistics for validation and test.
- Reuses those statistics across batches, entities, and evaluation runs.
- Uses sample standard deviation for numeric columns. Datetime preprocessing uses its own Welford/population convention.
- Keeps semantic channels separate, unless evaluation intentionally maps booleans through the numeric path with `bool_as_num=True`.

**RelativeDB**

- Exposes `ModelConfig.normalization_mode` and a backend override with two
  explicit contracts: `ZERO_SHOT` and `REFERENCE` (`STATISTICS` is an alias).
- `ZERO_SHOT` derives numeric, boolean, datetime, and derived-label statistics
  independently inside each entity sequence. Other entities in the request do
  not contribute to those transforms.
- `REFERENCE` requires `ColumnStats`; missing physical or task statistics fail
  closed instead of falling back to request data.
- `ColumnStats` uses sample standard deviation for numeric columns and the
  reference Welford/population convention for the global datetime transform.
- Adapter fitting fits physical statistics at the training cutoff, adds persisted
  task-target statistics after labels are collected, and saves both the mode
  and statistics with the head sidecar.
- Zero-shot intentionally remains artifact-free and context-relative. It is
  stable across batching, but is not numerically identical to preprocessing-
  time reference statistics. Reference equivalence requires `REFERENCE` mode.

The batch-dependence demonstrated in §10.2 is therefore historical and now
covered by a regression test. Artifact versioning, schema/checkpoint hashes,
row counts, and drift management remain open.

### 6.4 Target representation

This is the most fundamental semantic divergence.

**Reference**

- Selects an actual target node and actual target column.
- Emits the target cell first.
- Masks that cell while retaining its real table and column-name embedding.
- Uses known same-task rows from the same physical task table as in-context examples.
- For autocomplete, masks an actual numeric or boolean feature cell in its original row.

**RelativeDB**

- Builds a stable `TaskSpec` from the validated target AST, entity table, and
  task type. Formatting-equivalent queries share an identity; semantic changes
  such as a different horizon or filter produce another identity.
- Bare entity-column autocomplete masks the actual physical table/column cell
  on the focal entity node, retaining the schema phrase the checkpoint expects.
- Derived RelQL targets use deterministic task table and target column names
  derived from the canonical identity, rather than one shared `task.label`.
- Produces historical self-labels by evaluating the query only over
  `focal_row_keys`, so global peer rows cannot rewrite the focal target history.
- Derived queries materialize a focal unknown task row and configurable prior
  task windows for every entity. Historical labels are evaluated at their own
  cutoff, may see their FOLLOWING window, and are capped at the focal anchor.
  Task rows connect bidirectionally to their entity and obey the reference rule
  that task-table P2F edges are traversed only from a same-task seed.

The RT-J weights can be numerically correct and still receive an out-of-distribution prompt geometry because schema/table/column semantics are part of every token.

### 6.5 Context and graph sampling

| Behavior | Reference sampler | RelativeDB |
|---|---|---|
| Global context default | 8,192 cells | 8,192 cells |
| Local context default | 256 | 256 |
| BFS width | 32 default evaluation width | 32 |
| Walk ranking | 10,000 graph random walks, length 20 by default | 10,000 bidirectional graph walks, length 20 |
| Same-table peers | Target BFS, walk-visited seeds, random unvisited fallback; optional FAISS | Same three tiers; FAISS remains future work |
| Peer ordering | Timestamp desc/count desc/random tie, or count desc/random tie | Same ordering and tie stream |
| Label balancing | Supported | Not supported in context construction |
| Seed reproducibility | rand 0.9.1 `StdRng` streams derived from context/step/node | Ported rand 0.9.1 ChaCha12, PCG seed expansion, integer ranges, and index sampling; oracle-tested against Rust |
| Budget fill | Target first, then tiered BFS/peer neighborhoods until cell capacity | Same target-first and local/global cell accounting |
| Time filtering | Target-time checks on same-table seeds and P2F expansion | Same target-time rules plus defensive snapshot-bound checks |
| Static rows | Admitted | Admitted |

Additional RelativeDB-specific differences:

- `ReferenceTraversal` is the engine default. It requires scanner material from
  which the engine builds one immutable CSC/bidirectional graph snapshot.
  `BreadthFirstTraversal` remains available explicitly for pull-per-hop legacy
  wiring.
- Snapshot row values and parent maps are copied and made immutable. Physical
  nodes keep snapshot-global IDs across focal contexts; virtual task node IDs
  are deterministic by entity and history slot.
- F2P is a LIFO priority frontier. P2F selects randomly from the shallowest
  level, applies target-time validity, samples DB fanout uniformly to width 32,
  and admits task P2F only for a same-task seed.
- The five-parent tensor limit now fails closed instead of silently dropping
  parent relationships.
- PKs are identity-only. FK feature tokens are disabled by default and are the
  sole deliberate divergence from reference preprocessing when enabled.

### 6.6 Transformer architecture: genuine parity

Both current RT-J implementations use:

- 12 relational blocks;
- model width 512;
- 8 attention heads of width 64;
- SwiGLU hidden width 2,048;
- text/schema embedding width 384;
- column, feature, and neighbor masked-attention passes in every block;
- RMSNorm, including Q/K normalization;
- learnable per-head scale multiplied by the logarithm of key count;
- attention score scaling of `1 / head_dim`, not `1 / sqrt(head_dim)`;
- output gating by `2 × sigmoid(gate)`;
- no positional encoding;
- number, text, datetime, boolean, and column-name encoders;
- maximum five feature-to-parent neighbors.

RelativeDB's `cpp/src/rt.hpp` is a close architectural port of `../relational-transformer/src/rt/model.py`. The golden tests provide unusually strong evidence that parameter naming, masks, residual order, norms, gating, and score reduction are aligned for RT-J.

### 6.7 Transformer implementation differences

**Reference PyTorch**

- Dimensions and block count come from checkpoint/config metadata.
- Uses `flex_attention`, with configurable sparse/materialized mask paths.
- Supports BF16 training/inference and normal PyTorch device semantics.
- Computes decoder outputs for number, text, datetime, and boolean channels.
- Detects older checkpoints without gate weights and enables legacy RT/PluRel attention math.
- Can load both safetensors and legacy `.pt` state dicts.

**RelativeDB C++**

- Hard-codes one RT-J geometry in C++ constants.
- Uses custom grouped sparse attention and never materializes an `S × S` matrix.
- Expands BF16 weights to FP32 for the normal path; activations are generally FP32.
- Uses only the output paths needed by its regression/binary/text adapter behavior.
- Routes boolean classification through the numeric head, matching the released RT-J evaluation setting but not every reference configuration.
- Does not implement the legacy attention mode and cannot load PluRel `.pt` checkpoints.
- Adds fused projections, persistent CPU workers, Accelerate kernels, Metal kernels, CUDA kernels, split-K attention, and quantization-aware GEMMs.

### 6.8 Checkpoints and model resolution

**Reference**

- Accepts a local weights file, a local checkpoint directory, or a normal Hub repository specification.
- Resolves revision/subfolder information and reads `config.json`.
- Instantiates dimensions and embedding/task metadata from checkpoint configuration.
- Prefers `model.safetensors` and falls back to `model.pt`.
- Supports repositories that contain multiple checkpoint subdirectories.

**RelativeDB**

- Accepts an explicit local file, a directory containing `model.safetensors`, or a required `hf://org/repo/subdir` URI.
- Does not expose revision selection.
- Does not read checkpoint `config.json`.
- Hard-codes RT-J dimensions and embedding width.
- Has `ModelConfig.check_checkpoint_embedding`, but production checkpoint resolution/loading never calls it. Only tests directly exercise the method.
- Routes classification/multiclass/ranking to one configured checkpoint and regression/forecasting to another.

RelativeDB adds q8, q4, and f16 converted formats. Q8 is per-row; Q4 uses groups of 32, with selected residual projections retained at q8. The reference contains no equivalent quantized runtime.

### 6.9 Devices and serving behavior

**Reference**

- Training is CUDA-oriented and can use distributed PyTorch.
- CPU/MPS can run eager examples when compiled CUDA-only attention is disabled.
- PyTorch owns device placement and model memory.

**RelativeDB**

- CPU is always the portability baseline.
- MPS is automatically preferred on supported macOS systems; otherwise CPU is used.
- CUDA exists behind an explicit CMake option and must be explicitly selected; automatic backend choice never selects CUDA.
- q8/q4/f16 are supported on CPU/MPS; current CUDA is FP32 only.
- The text-head extended forward used for zero-shot multiclass decoding is CPU-only.
- Frozen target-feature extraction works on CPU/MPS; the C ABI explicitly rejects CUDA for this operation.
- Frozen adapter-head training is Metal-only. At serving time, backbone feature extraction follows the selected CPU/MPS device, then the small saved head is evaluated on CPU.
- Full-checkpoint binary/regression fine-tuning is native MPS-only. It uses MPS matrix primitives for all GEMMs and custom SIMD-group Metal kernels for the exact sparse relational masks and their backward pass; it has no Torch dependency.
- Native training-forward, repeated-step, checkpoint, and 8,192-cell M3 evidence is recorded in `evaluation/runs/native-mps-finetune-verification.md`.

### 6.10 Task surface

**Reference**

- Public task types are node-level binary classification and regression.
- Explicit time-split task-table rows are called forecast tasks, but each item is still one labeled row at one timestamp.
- Autocomplete pretraining targets numeric and boolean cells.
- Text and datetime are modeled semantic channels but are not surfaced as official task target types.
- Link-prediction/recommendation tasks are intentionally skipped.

**RelativeDB**

- Infers regression, binary classification, multiclass, multilabel ranking, and forecasting from RelQL.
- Supports aggregations including count, distinct count, sum, average, min/max, list/array aggregation, first/last, and existence.
- Supports preceding/following windows, named windows, filters, scalar/arithmetic expressions, conditions, `WHERE`, `ASSUMING`, `AS OF`, `RETURN`, `EXPLAIN`, and `ABLATE`.
- Multiclass zero-shot prediction decodes a predicted text embedding against up to 1,000 distinct labels scanned from data, using nearest-neighbor similarity and a fixed softmax temperature.
- Ranking scans up to 1,000 distinct candidates, builds a candidate-conditioned context for each, performs model inference, and returns top-K.
- `RETURN` converts internal scores into a product-facing result shape; the reference exposes tensors/evaluator results.

These features are additions, not parity with reference behaviors. In particular:

- multi-horizon forecasting duplicates a scalar across all requested horizons;
- ranking has no reference link-prediction implementation to compare against;
- text-nearest-neighbor multiclass has no official reference evaluator path;
- `ASSUMING table.column = value` applies the assignment to every row of that table present in each context, rather than only the focal entity row;
- `ABLATE` is acknowledged but not applied.

### 6.11 Temporal semantics

**Reference**

- Every explicit task item has a task-row timestamp.
- Same-table peers and child expansion are constrained relative to the target timestamp.
- Train statistics are copied into validation/test artifacts.
- Leakage columns declared for task tables are removed from target contexts.

**RelativeDB**

- Supports a shared `ExecutionInput.anchor_time`, explicit `AS OF`, a bound parameter, or per-entity anchor derivation from the entity row.
- Can decouple the query anchor from the context anchor.
- Rechecks returned rows against a `TemporalBound`, so buggy retrievers cannot simply return later timestamped rows.
- Static rows are always admitted.
- In per-entity mode, class/candidate discovery uses the maximum anchor across the entire batch, not each entity's own bound.
- A static entity with per-entity anchor behavior can fall back to an unbounded context if no timestamp source is provided.

### 6.12 Training and adaptation

**Reference**

- Performs full backbone pretraining.
- Includes Muon plus AdamW parameter grouping, learning-rate scheduling, gradient clipping, distributed data parallelism, checkpoint resume, optimizer-state loading, and SWA.
- Trains from mixtures of explicit forecast tasks and schema-derived autocomplete tasks.
- Samples context geometry during training.
- The ordinary inference workflow is explicitly zero-shot without task fine-tuning, though the evaluator contains compatibility logic/comments for per-task checkpoint recipes.

**RelativeDB**

- Does not perform the reference repository's multi-database pretraining, DDP,
  Muon, or SWA workflow.
- Fine-tunes the complete RT-J checkpoint for binary and regression tasks with
  native C++/MPS forward, backward, gradient clipping, and AdamW updates.
- Uses the reference Rust sampler for task-specific train and validation
  contexts.
- Supports gradient accumulation, atomic model and optimizer checkpoints,
  exact sampler-position resume, validation selection, early stopping, and
  automatic restore-and-learning-rate-backoff when quality drops.
- Starts validation selection from the released zero-shot checkpoint. A
  trained model is not activated merely because its training loss decreased.
- Keeps frozen-head fitting as a separately named adapter path for multiclass
  and ranking tasks. Those adapters do not qualify as `ours_finetuned`.

RelativeDB now supports full-model task fine-tuning, but it still does not
replace the reference pretraining stack. These are different jobs: task
fine-tuning adapts a released checkpoint to one target, while pretraining builds
the shared checkpoint from many databases.

### 6.13 Evaluation and baselines

**Reference**

- Defines a curated 21-task RelBench evaluation set: 12 binary classification and 9 regression tasks.
- Maps predictions back to original task-table row order.
- Uses RelBench metrics, including AUROC and normalized MAE.
- Tunes context configurations on validation data and supports seed/config ensembles.
- Provides `rel2tab`, a general featurizer/predictor framework with global, entity, RT, precomputed, SQL, and RDBLearn-style features and mean, linear, ridge, LightGBM, XGBoost, TabPFN/TabICL-family predictors.
- Does not commit a comparable end-result JSON in the inspected snapshot.

**RelativeDB**

- Now has a clean `evaluation/` harness, implemented independently of the
  deleted legacy `benchmarks/` tree. It ports the reference 21-task catalog,
  official keyed RelBench scoring, and optional `rt`, SQL/XGBoost, `ours`, and
  `ours_finetuned` runners.
- `ours` consumes the reference evaluator's exact Rust-sampled tensor batches.
  `ours_finetuned` uses the same path but requires a complete task-specific
  model checkpoint; frozen head adapters are excluded from that runner.
- The executed two-task rel-f1 slice and its reproducible Markdown/JSON output
  are in `evaluation/runs/rel-f1-head-to-head/`. It is integration evidence,
  not a statistically complete 21-task quality claim.
- Unit, integration, native conformance, and sampler/normalization tests remain
  separate release gates.

#### Evaluation results

Generated evaluation outputs are private local artifacts and are not committed.
The harness records the effective context, split, checkpoint, validation history,
and official task metric for each runner. A full-model checkpoint can enter the
`ours_finetuned` column only after it beats the current best model on validation.
Frozen-head diagnostics are kept separate and are never presented as full-model
fine-tuning.

The paper is useful background, but the checked-out reference repository is the
implementation authority for sampler behavior and the current training recipe.
A comparison with paper numbers is not a controlled reproduction unless it uses
the same checkpoint, preprocessing split, context length, and selection rules.

### 6.14 Query planning and explainability

RelativeDB uniquely provides:

- parser and AST validation;
- schema-aware task inference;
- entity selection and bind parameters;
- point-in-time context assembly;
- plan/context/analyze explanations;
- context truncation instrumentation;
- JSON/text result serialization.

The reference has no query language or query planner. Its nearest analog is a context-visualization server that exposes sampler controls and renders sampled nodes/tokens. Reference schema-semantics ablation is implemented by a deterministic column-embedding permutation. RelativeDB's `ABLATE` syntax currently emits a warning and leaves execution unchanged.

### 6.15 Language and extension surface

RelativeDB currently exposes:

- Python API;
- C ABIs for the parser, CSC, inference, and head training;
- C++ CPU, Metal, and CUDA sources.

The repository history and stale documentation refer to Java, Rust, and DuckDB extension trees, but none is present in the current tracked snapshot. The reference exposes Python plus an internal PyO3/Rust extension and a standalone Rust preprocessor binary; it has no RelQL or database extension.

### 6.16 Documentation and release hygiene

Confirmed inconsistencies in RelativeDB:

- Root README says 22M parameters; the loaded RT-J configuration contains 85,565,091 parameters, conventionally 86M.
- Root checkpoint table labels a 171 MB file “fp32.” The inspected upstream safetensors has 400 BF16 tensors and is 171,169,942 bytes; expanding those weights to FP32 takes about 342 MB. `cpp/README.md` explains the distinction more accurately.
- Root install command says `pip install relativedb`; project metadata says `relationdb`.
- Several source/site comments refer to absent `CONTRACT.md`, `RelQL_EVOLUTION.md`, `kb/architecture.md`, and scratchpad specifications.
- `release-libraries.yml` still packages/tests absent `rust/` and `java/`.
- `duckdb-extension.yml` still targets absent root Cargo and `duckdb-extension/` files.
- The Python release job runs Python tests without building the C++ library first; clean-runner native-parser tests are not represented by the pure Python build.

The reference's docs are narrower but internally align more closely with its current paths. It has far less automated test/release coverage in the repository snapshot.

## 7. Feature disposition matrix

| Capability | Reference | RelativeDB | Disposition |
|---|---:|---:|---|
| RT-J transformer math | Yes | Yes | Conformant within measured native drift |
| Configurable RT dimensions | Yes | No | Missing |
| Legacy RT/PluRel attention | Yes | No | Missing |
| Safetensors | Yes | Yes | Shared |
| Legacy `.pt` checkpoints | Yes | No | Missing |
| Checkpoint config metadata | Yes | Ignored | Missing |
| Q8/Q4/F16 serving | No | Yes | Addition |
| CPU native runtime | PyTorch | C++ | Different implementation |
| MPS runtime | Eager PyTorch | Custom native backend | Addition |
| CUDA runtime | PyTorch | Optional custom backend | Different implementation |
| Full-model pretraining | Yes | Task fine-tuning only | Partial: no multi-dataset pretraining/DDP/SWA |
| Full-model task fine-tuning | Yes | Yes, scalar tasks on native MPS | Shared goal, different implementation |
| DDP/SWA/resume | Yes | No | Missing |
| Frozen-head training | Not primary workflow | Yes | Addition |
| RelBench manifest preprocessing | Yes | No | Missing |
| rkyv/mmap graph | Yes | No | Missing |
| Live callback retrieval | No | Yes | Addition |
| CSC snapshot | Native preprocessed adjacency | In-memory callback snapshot | Similar purpose, different contract |
| Reference random-walk sampler | Yes | No | Missing |
| Optional FAISS peer sampling | Yes | No | Missing |
| Label-balanced context | Yes | No | Missing |
| Fixed preprocessing stats | Yes | Yes in configurable reference mode | Shared; zero-shot mode is an intentional addition |
| Real task-row target | Yes | No | Replaced |
| RelQL | No | Yes | Addition |
| Point-in-time `AS OF` | Implicit target timestamp | Explicit query feature | Addition |
| Counterfactual assumptions | No | Yes | Addition |
| Explain plan/context/analyze | No | Yes | Addition |
| Working schema ablation | Yes | No-op | Incomplete |
| Binary classification | Yes | Yes | Shared, different orchestration |
| Regression | Yes | Yes | Shared, different orchestration |
| Multiclass text decoding | No official task path | Yes | Addition/experimental |
| Ranking/link prediction | Explicitly skipped | Yes | Addition/experimental |
| Multi-horizon forecast values | No | Duplicated scalar | Incomplete addition |
| Official RelBench evaluation | Yes | Yes, through `evaluation/` adapter | Ported |
| `rel2tab` baseline framework | Yes | Optional SQL/XGBoost runner, not the full framework | Partial port |
| Custom real-data evaluation | No comparable harness | Clean four-runner RelBench harness | Addition |
| Context visualization UI | Yes | Explain JSON/text | Different capability |
| Docusaurus product site | No | Yes | Addition |

## 8. Native conformance evidence

### 8.1 Parameter and fixture reproduction

I loaded the current reference `RelationalTransformer` using the same released
RT-J checkpoint and ran RelativeDB's committed native golden input from
`cpp/testdata` through it.

- Current reference model parameters: **85,565,091**.
- Reference PyTorch output vs. committed PyTorch golden fixture: **maximum absolute difference 0.0**.
- This confirms that the fixture is still representative of the inspected reference model, not merely of an older copied implementation.

### 8.2 C++ golden conformance

`cpp/build/rt_test` passed the same fixture on:

- CPU: maximum target-output drift approximately **3.911 × 10⁻³**;
- MPS: maximum target-output drift approximately **3.910 × 10⁻³**.

Per-sequence scalar scores agreed to roughly three or four decimal places. The observed residual is consistent with documented BF16-to-FP32 conversion and operation-order differences; there was no evidence of a mask, residual, or weight-mapping failure.

### 8.3 What this does and does not prove

It proves high confidence in:

- checkpoint tensor mapping;
- semantic encoders and decoder reduction used by the test;
- col/feat/nbr masks for the fixture;
- block order, residuals, norms, attention scale/gate, and FFN;
- CPU/MPS consistency.

It does **not** prove parity in:

- how a database becomes a sampled sequence;
- how target and peer labels are represented;
- statistics/normalization;
- task/candidate discovery;
- temporal bounds;
- ranking, multiclass, forecasting, or fine-tuned-head semantics;
- CUDA, quantized formats, or arbitrary unseen mask geometries.

## 9. Test and build matrix

| Check | Result | Notes |
|---|---|---|
| RelativeDB Python tests | **passed** | Post-alignment suite on 2026-07-20 |
| RelativeDB CMake configure/build | **passed** | Current C++ tree built successfully |
| RelQL C++ corpus | **passed** | Valid corpus accepted and invalid corpus rejected by `cpp/build/relql_test` |
| CSC C++ test | **passed** | `cpp/build/csc_test` |
| Native training test | **passed** | Multiclass loss 1.098612 → 0.010736; ranking 1.386294 → 0.093876 |
| Native RT-J golden, CPU | **passed** | max drift ≈ 0.003911 |
| Native RT-J golden, MPS | **passed** | max drift ≈ 0.003910 |
| Reference lightweight Python tests | **passed** | `test_api.py` + `test_import_safety.py`; PyTorch deprecation warnings remain |
| Reference Rust build/tests | **built** | `PYO3_PYTHON=/opt/homebrew/bin/python3.12 cargo test --locked --no-default-features` |
| Reference full Python suite | **not fully run** | Some cases require built extension/preprocess dependencies such as PyArrow/PluRel |
| RelativeDB wheel build | **built with warning** | pure `relationdb-0.1.0-py3-none-any.whl`; missing `python/README.md` |
| Isolated installed-wheel parse | **failed as expected from package contents** | `NativeParserUnavailable: librt_c not found` |

Reference Python tests were run with the available local Python 3.14 environment plus a temporary `einops` install, outside the project's declared Python 3.12 matrix. That is sufficient for the import/API smoke checks but not a substitute for the supported Pixi environment.

The first reference Cargo attempt auto-selected Python 3.8 and failed PyO3's ABI3/Python-3.12 requirement. Pinning `PYO3_PYTHON` to installed Python 3.12 resolved the build. This is an environment-selection issue, not a reference source failure.

## 10. Targeted behavioral research

### 10.1 Method

The unit/golden suites mostly isolate components. I therefore constructed small deterministic schemas and contexts to test properties that should hold across an inference API:

- batch composition should not change a focal entity's prediction;
- adding cohort examples should not alter the focal entity's own historical aggregation;
- supplying fixed physical-column statistics should remove physical-column batch normalization;
- planned features should affect execution;
- a built wheel should provide the functionality advertised by its Python API.

The probes used the current worktree and the current native checkpoint cache.

### 10.2 Historical batch-dependent prediction — resolved

For a toy churn schema, I scored entity C7 alone, entity C1 alone, and both in one call.

Observed fitted synthetic-label statistics:

| Scoring call | Label mean | Label standard deviation |
|---|---:|---:|
| C7 alone | 0.6667 | 0.4714 |
| C1 alone | 0.3333 | 0.4714 |
| C7 + C1 | 0.5000 | 0.50000001 |

C7's normalized `age` value was 0.0 when scored alone and approximately 1.0 when C1 was included. Its actual native predictions changed:

| Output for C7 | C7 alone | C7 in batch with C1 | Absolute change |
|---|---:|---:|---:|
| Regression value | 0.7229717667 | 0.5597380412 | 0.1632337255 |
| Binary probability | 0.4382060461 | 0.4639913257 | 0.0257852796 |

After supplying the current `ColumnStats` for physical columns, C7 regression still changed from 0.7654925430 to 0.6004700681 because the generic historical task-label statistics remained batch-fitted.

**Historical finding:** the reviewed API was not entity-wise invariant. Batch
size, filtering, pagination, and unrelated entities could alter predictions.

**Current status (2026-07-20):** resolved by per-entity zero-shot statistics
and strict persisted reference statistics. The new regression test compares a
focal sequence built alone and in a mixed batch and requires identical token
values. The old checkpoint ranking golden was regenerated because its lower
ranks encoded the removed batch-normalization behavior; raw native tensor/C
ABI goldens were not changed.

### 10.3 Historical cohort contamination of self-labels — resolved for focal labels

I assembled a context for C7 in the same toy schema:

- with `cohort_size=0`, the context contained C7 and its own orders/products, and the first 90-day preceding self-label count was **1**;
- with the default `cohort_size=256`, scanner-selected C1/C9 and their rows entered the context, and the same C7 self-label became **2**.

`_self_labels` evaluates the target aggregation over all `ctx.rows_by_table()` without first restricting rows to the focal entity. Cohort examples are useful as in-context demonstrations, but their event rows must not be folded into the focal entity's ground-truth expression.

**Historical finding:** cohort construction could change the task definition,
not just the evidence presented to the model.

**Current status (2026-07-20):** `EntityContext` records
`focal_row_keys`; traversal propagates focal ownership, assumptions preserve
it, and `_self_labels` evaluates `focal_rows_by_table()`. Cohorts can still
change model evidence and scores, as intended, but not the focal historical
label. The remaining modeling gap is representing each peer as a separately
labeled demonstration rather than a shared global row pool.

### 10.4 Batch-wide candidate temporal bound

`_batch_bound` returns the latest non-null anchor among all contexts. Multiclass label enumeration and ranking candidate enumeration scan with that one bound. If entities have different anchors, a candidate introduced between an early entity's anchor and the batch maximum can enter that early entity's domain.

This finding is source-proven rather than output-probed. The reference performs sampling relative to each target row's timestamp. RelativeDB should enumerate or filter candidates per effective entity bound, or explicitly require a common anchor for these task types.

### 10.5 Forecast horizons

The forecasting scorer obtains one regression value `v` and constructs:

`forecast = tuple([v] * n)`

where `n` is the number of requested horizons. No horizon token, horizon-specific label context, repeated forward call, or horizon-specific head is used.

**Finding:** syntax for multiple horizons exists, but the implementation reports one prediction repeatedly.

### 10.6 ABLATE behavior

The parser/AST/planner accept declared table ablations. Explain output warns:

> ablation not implemented — declared ABLATE tables are not removed

Execution uses the unmodified contexts. The reference's `ablate_schema_semantics` actually permutes column schema embeddings in the sampler.

**Finding:** do not use RelativeDB `ABLATE` results for causal or scientific conclusions in its current form.

### 10.7 Wheel isolation

I built `python/` into a clean temporary dist directory, installed the generated wheel into an isolated target, and called `relativedb.parse(...)`.

- Wheel tag: `py3-none-any`.
- No `librt_c` was present.
- Parse failed with `NativeParserUnavailable` and instructed the caller to build `cpp/` with CMake.

The same issue applies to the native CSC and model paths. The reference's maturin package, by contrast, is explicitly structured to compile/package `rt._rustler`.

## 11. Recommended remediation order

### P0 — make scoring semantically stable

1. **Completed: remove request-batch normalization.** Zero-shot is entity-local;
   reference/statistics mode consumes persisted column, datetime, and task
   transforms and fails on missing statistics.
2. **Partially completed: separate focal rows from demonstrations.** Self-label
   expressions are focal-only. Separate labeled demonstration objects and
   ownership-aware explain output remain.
3. **Partially completed: represent the target faithfully.** Entity
   autocomplete masks the physical cell, and derived targets use canonical
   stable `TaskSpec` identities. Materialized user-declared task tables and
   held-out representation comparisons remain.
4. Invariance tests now cover batch composition, stable task identity, custom
   traversal invocation, deterministic sampling, and temporal row filtering.
   Continue extending them to:
   - focal result unchanged under batch permutation;
   - focal result unchanged when unrelated entities are appended;
   - self-label unchanged under cohort size/order;
   - result unchanged across batch chunking.

### P1 — make declared product behavior true

5. Either implement horizon-conditioned forecasts or reject multiple `HORIZONS` values.
6. Apply `ABLATE` during context/token construction or reject it instead of returning unchanged predictions.
7. Use per-entity temporal bounds for class/candidate discovery.
8. Read checkpoint `config.json`, enforce dimensions and embedding identity, and invoke the existing mismatch guard.
9. Decide whether to support configurable/legacy checkpoints. If not, explicitly state “RT-J 12×512 only; safetensors only.”
10. Build platform wheels containing the C ABI library, or split the native runtime into an explicit installable package and make the dependency unavoidable.

### P1 — restore reproducibility

11. Expand the new independent `evaluation/` system from its executed rel-f1
    classification/regression slice to the full 21-task catalog before making
    broad predictive-quality claims.
12. Keep the reference-context path as the parity gate: RT, native zero-shot,
    and native inference from full task-fine-tuned checkpoints must consume
    identical sampled tensors and the same official keyed scorer.
13. Add a sampler-differential fixture: same preprocessed graph/target through
    Rust and RelativeDB tokenizers, with per-token tensor diffs.

### P2 — release and documentation cleanup

14. Choose one distribution name and align README, PyPI metadata, import examples, URLs, and artifact names.
15. Correct 22M → 86M and distinguish 171 MB BF16-on-disk from about 342 MB expanded FP32 resident memory.
16. Add `python/README.md` or point `readme` at an included file.
17. Remove or restore Java/Rust/DuckDB workflows and documentation.
18. Add a tracked license file consistent with declared metadata.
19. **Completed for the new alignment API:** `NormalizationMode`,
    `ColumnStats`, `TaskSpec`, traversal protocols/results, both built-in
    traversals, and `FineTunedHead` are exported consistently. Keep an API
    export test as new public types are added.

## 12. Suggested acceptance criteria for reference alignment

“Reference-compatible RT-J” should mean all of the following, not only a passing kernel golden:

1. A reference-preprocessed fixture produces the same token sequence fields: semantic type, node, table, column, target mask, F2P indices, values, schema embeddings, and timestamps.
2. Fixed dataset and task statistics match the reference artifacts.
3. PyTorch and C++ receive the same batch and agree within an explicitly versioned tolerance.
4. A focal prediction is invariant to request batching and iteration order.
5. Point-in-time sampling excludes post-target information per entity.
6. Context seed and sampler policy are recorded and reproducible.
7. The same official task rows produce evaluator-aligned AUROC/NMAE.
8. Unsupported reference features fail explicitly instead of silently changing semantics.

Today, criterion 3 is strongly satisfied for the committed fixture. Criteria 1, 2, 4, 5, and 7 are not satisfied end-to-end; 6 and 8 are partial.

## 13. Research log

### 13.1 Repository inspection

I performed:

- full `git ls-files` inventories for both trees;
- `git status`, branch, HEAD, commit count, and history/date inspection;
- path-intersection comparison;
- source line counts by language/subsystem;
- searches for architecture constants, task types, statistics, target construction, sampler controls, checkpoint loaders, native ABI functions, device routing, packaging metadata, workflow paths, stale binding claims, and docs.

### 13.2 Source review

Reference files reviewed in detail:

- all `src/rt/*.py`;
- all `rustler/src/*.rs` and Cargo/build metadata;
- all `scripts/*.py` plus recipes/Slurm wrappers;
- `rel2tab` configuration, featurizer, predictor, model, and backend implementations;
- all docs, example scripts, project metadata, and tests.

RelativeDB files reviewed in detail:

- all `python/src/relativedb/**/*.py`;
- all Python tests;
- all C/C++ headers and implementations, including Metal/CUDA/training/quantization;
- CMake and C++ tests/fixtures;
- root/product/style docs, Docusaurus configuration/content, package metadata, and workflows.

### 13.3 Execution

Commands/checks included:

- `python/.venv/bin/python -m pytest -q python/tests`;
- CMake configure/build and each C++ test executable;
- CPU and MPS golden RT-J inference;
- reference PyTorch fixture reproduction;
- reference lightweight pytest files;
- reference Cargo build/test with an explicit Python 3.12 interpreter;
- wheel build, isolated installation, and parser invocation;
- targeted in-memory batch/cohort normalization probes;
- `git diff --check` and final worktree status checks.

### 13.4 Effort boundaries

I did not during the original review:

- download and preprocess the full RelBench corpus;
- run reference full pretraining or the complete 21-task GPU evaluation;
- compare CUDA output on a CUDA host;
- assess model quality with a newly designed statistically powered dataset;
- rewrite any implementation issue found.

The subsequent evaluation port did run complete test rows for a representative
rel-f1 classification/regression pair across RT, SQL/XGBoost, native zero-shot,
and a historical frozen-head diagnostic. See
`evaluation/runs/rel-f1-head-to-head/results.md`. Full-catalog evaluation and
backbone pretraining remain separate larger experiments.

## 14. Final assessment

RelativeDB has a credible and well-tested native RT-J **kernel** plus a much
broader predictive-query product. Its strongest engineering remains the C++
inference stack. The semantic bridge is materially stronger after the
2026-07-20 changes: request-batch normalization and focal-label contamination
are resolved, direct targets retain physical schema identity, derived targets
are canonical, and traversal is pluggable with a reference-style option.

The system is still not end-to-end reference-equivalent. The largest remaining
semantic gaps are fully materialized task demonstrations in the product path,
per-entity candidate/class domains, horizon conditioning, and full-catalog
evaluation. Exact reference contexts and the official evaluator are now
available in the isolated evaluation path. These should drive quality work
rather than the two resolved invariance defects.

Accordingly:

- describe the current system as **“RT-J-backed”**, not end-to-end reference-equivalent;
- treat ranking, multiclass, multi-horizon forecasting, `ABLATE`, and head adaptation as experimental until their contracts and evaluations are tightened;
- keep batch and focal-label invariance as permanent release gates;
- validate reference/statistics artifacts and the new traversal on held-out
  tasks before making either the production default;
- use official task rows and the reference sampler/evaluator for any claim of RT-J quality parity.

The architecture can become a robust online counterpart to the research code, but the next increment of value is in data/task semantics and reproducibility, not additional GEMM optimization.

## Appendix A — exhaustive tracked-file/subsystem disposition

The inventory below accounts for every tracked file by exact path or containing file group. It records why raw path-by-path diffing is not useful and where each repository's unique functionality lives.

### A.1 Paths shared by name

| Path | Difference |
|---|---|
| `.gitignore` | Different ignored build/runtime artifacts for different language stacks |
| `README.md` | Reference research installation/inference overview vs. RelativeDB product/query-engine overview |

No other tracked relative path is shared.

### A.2 Reference-only root/config/docs/examples

- `.gitattributes`: reference repository text/attribute policy; no RelativeDB counterpart.
- `.pre-commit-config.yaml`: reference formatting/lint hooks; no root RelativeDB pre-commit configuration.
- `pyproject.toml`: maturin/Pixi project for `relational-transformer`; RelativeDB metadata instead lives under `python/pyproject.toml`.
- `docs/baselines.md`, `docs/context-visualization.md`, `docs/downloads.md`, `docs/inference.md`, `docs/preprocess.md`, `docs/pretrain.md`: research workflow documentation absent as equivalent workflows in RelativeDB.
- `examples/byod/colab.ipynb`, `examples/byod/mini-shop.duckdb`: bring-your-own-database demo that converts SQL data into RelBench form; RelativeDB instead connects callbacks directly.
- `examples/inference/0_make_demo_labels.py`, `1_data_prep.py`, `2_task_prep.py`, `3_predict.py`, `README.md`, `config.py`: staged DuckDB/Postgres/MySQL → RelBench → task → zero-shot example; no exact RelativeDB equivalent.

### A.3 Reference-only `src/rt`

- `src/rt/__init__.py`: lazy public imports.
- `src/rt/checkpoints.py`: config-aware local/Hub safetensors and legacy checkpoint resolution.
- `src/rt/config.py`: full model/train/eval/logger configuration dataclasses.
- `src/rt/data.py`: PyTorch dataset/dataloader wrapper over Rust sampling and tensor reshaping.
- `src/rt/embed.py`: sentence-transformer embedding support for preprocessed artifacts.
- `src/rt/eval_utils.py`: official task construction, prediction ordering, grid tuning, and ensembling.
- `src/rt/evaluator.py`: inference/evaluation orchestration and metric/submission integration.
- `src/rt/model.py`: configurable PyTorch RT/RT-J/legacy transformer and loss/prediction code.
- `src/rt/muon.py`: Muon optimizer implementation.
- `src/rt/pre.py`: preprocessed-dataset discovery/metadata/download helpers.
- `src/rt/recipes.py`: recipe/config assembly.
- `src/rt/swa.py`: stochastic weight averaging and train-state support.
- `src/rt/tasks.py`: explicit RelBench and schema-autocomplete task discovery.

RelativeDB distributes these responsibilities across its schema/retrieval/query engine/native runtime, but has no full equivalents for preprocessing, optimizer, recipes, or SWA.

### A.4 Reference-only Rust sampler/preprocessor

- `rustler/.gitignore`, `Cargo.lock`, `Cargo.toml`, `README.md`: Rust extension/binary project definition.
- `rustler/src/common.rs`: archived data structures and shared storage types.
- `rustler/src/pre.rs`: manifest/Parquet preprocessing, semantic values/statistics, adjacency, and artifact writing.
- `rustler/src/fly.rs`: production stochastic context sampler, target emission, temporal filtering, peer tiers, optional FAISS, and Python tensor export.
- `rustler/src/lib.rs`: PyO3 module registration.
- `rustler/src/main.rs`: standalone preprocessing entry.

RelativeDB has no matching offline preprocessor. Its nearest components are `engine.py`, `retrieve.py`, `csc.py`, and `csc_native.py`, which operate on user-supplied rows.

### A.5 Reference-only training/evaluation scripts

- `scripts/baseline.py`: `rel2tab` evaluation driver.
- `scripts/ctx_viz.py`: sampler context visualization HTTP application.
- `scripts/eval.py`: released checkpoint evaluator, tuning, and ensemble CLI.
- `scripts/mlock_recipe.py`: memory-locking recipe helper.
- `scripts/preprocess.py`: one/many/upload preprocessing CLI.
- `scripts/pretrain.py`: distributed full-model pretraining CLI.
- `scripts/recipe_rt_j.txt`: RT-J recipe.
- `scripts/slurm_preprocess.sh`, `scripts/slurm_pretrain.sh`: cluster launch wrappers.

RelativeDB now has native full-backbone task fine-tuning for scalar tasks, but
still has no matching multi-dataset pretraining/DDP/SWA workflow. Its new
evaluation adapter covers released RT, a selected `rel2tab` SQL/XGBoost
baseline, native zero-shot, and full-checkpoint native fine-tuned inference,
but not
the reference's full research training stack.

### A.6 Reference-only `rel2tab`

- Core: `rel2tab/README.md`, `__init__.py`, `config.py`, `featurize.py`, `featurizer.py`, `model.py`, `predictor.py`.
- Featurizers: `featurizers/__init__.py`, `entity_featurizer.py`, `global_featurizer.py`, `precomputed_featurizer.py`, `rdblearn_featurizer.py`, `rt_featurizer.py`, `sql_featurizer.py`.
- Dataset SQL features: `featurizers/sql_queries/__init__.py`, `rel_amazon.py`, `rel_avito.py`, `rel_event.py`, `rel_f1.py`, `rel_hm.py`, `rel_stack.py`, `rel_trial.py`.
- Predictors: `predictors/__init__.py`, `identity_predictor.py`, `lgbm_predictor.py`, `linear_predictor.py`, `mean_predictor.py`, `ridge_predictor.py`, `tab_predictor.py`, `tabicl_batched_predictor.py`, `xgboost_predictor.py`, `xgboost_tuned.py`.

RelativeDB has no maintained equivalent reusable featurizer/predictor framework.

### A.7 Reference-only tests

- `tests/conftest.py`: reference fixtures/import setup.
- `tests/test_api.py`: high-level model/task/checkpoint API tests.
- `tests/test_import_safety.py`: lazy/optional import behavior.
- `tests/test_rustler.py`: native preprocessing/sampler integration.

RelativeDB's test surface is much larger and is listed below.

### A.8 RelativeDB-only root/config/workflows

- `.mari/config.json`: local tooling configuration; no reference counterpart.
- `PRODUCT.md`: predictive-query product contract; reference documents research workflows instead.
- `STYLE.md`: repository naming/style guidance; no reference counterpart.
- `.github/workflows/deploy-docs.yml`: Docusaurus deployment.
- `.github/workflows/duckdb-extension.yml`: stale workflow for an absent DuckDB extension tree.
- `.github/workflows/release-libraries.yml`: Python plus stale Java/Rust release workflow.
- `python/pyproject.toml`: setuptools package metadata for `relationdb`; unlike reference root maturin project, it does not build native code.

### A.9 RelativeDB-only Python package

- `python/src/relativedb/__init__.py`: public/lazy Python exports.
- `python/src/relativedb/schema.py`: explicit application schema, value types, tables, links, and validation.
- `python/src/relativedb/retrieve.py`: `Row`, temporal-bound, callback, scanner, and wiring contracts.
- `python/src/relativedb/csc.py`: Python CSC-index interface/fallback orchestration.
- `python/src/relativedb/csc_native.py`: ctypes wrapper over native CSC.
- `python/src/relativedb/model.py`: task-to-checkpoint configuration and embedding-mismatch policy.
- `python/src/relativedb/engine.py`: context assembly, RelQL execution, filtering, assumptions, explain output, label extraction, and head-training orchestration.
- `python/src/relativedb/evaluate.py`: expression/window evaluation over assembled rows.
- `python/src/relativedb/rt_native.py`: text/schema embeddings, tokenization, normalization, native model calls, multiclass/ranking/forecast adapters, checkpoint resolution, and frozen-head code.
- `python/src/relativedb/relql/__init__.py`: RelQL exports.
- `python/src/relativedb/relql/ast.py`: query AST/task types.
- `python/src/relativedb/relql/native.py`: native parser discovery/ctypes bridge.
- `python/src/relativedb/relql/parser.py`: JSON-AST conversion and schema-aware validation.

The reference has no RelQL/query-engine counterparts. Its `model.py`, `data.py`, `tasks.py`, and Rust sampler collectively overlap only with portions of `rt_native.py` and `engine.py`.

### A.10 RelativeDB-only Python tests

- `python/tests/conftest.py`: library/native path fixtures.
- `python/tests/data/examples.relql`: accepted/rejected language corpus used by tests.
- `python/tests/test_engine.py`: context, execution, temporal, task, training, and output behavior.
- `python/tests/test_explain_asof.py`: explain and `AS OF` semantics.
- `python/tests/test_native_csc.py`: native CSC integration.
- `python/tests/test_relql_parser.py`: parser/validation corpus and edge cases.
- `python/tests/test_rt_native.py`: tokenizer/runtime/head/checkpoint behavior.

The reference's three test modules do not cover an equivalent product surface.

### A.11 RelativeDB-only C++ build and core

- `cpp/.gitignore`, `cpp/CMakeLists.txt`, `cpp/README.md`: native build, targets, and performance/conformance documentation.
- Parser: `cpp/src/relql.hpp`, `relql.cpp`, `relql_c.h`, `relql_c.cpp`.
- CSC: `cpp/src/csc.hpp`, `csc.cpp`, `csc_c.h`, `csc_c.cpp`.
- Transformer public/core: `cpp/src/rt.hpp`, `rt.cpp`, `rt_internal.hpp`, `rt_math.hpp`.
- C ABI: `cpp/src/rt_c.h`, `rt_c.cpp`.
- Quantization: `cpp/src/rt_quant.hpp`, `quantize.cpp`.
- Devices: `cpp/src/rt_metal.mm`, `rt_cuda.cu`.
- Training: `cpp/src/rt_train.hpp`, `rt_train.cpp`, `rt_train_metal.mm` for
  frozen adapters and `rt_full_train_metal.mm` for full-checkpoint scalar-task
  forward/backward/AdamW on MPS.

The reference's transformer is PyTorch; its Rust code preprocesses/samples and does not provide native transformer kernels.

### A.12 RelativeDB-only C++ tests/tools

- `cpp/src/test_relql.cpp`: parser conformance/rejection corpus.
- `cpp/src/test_csc.cpp`: high-volume native index correctness.
- `cpp/src/test_golden.cpp`: RT-J PyTorch/C++ golden conformance across devices/formats.
- `cpp/src/test_train.cpp`: multiclass/ranking head training convergence.
- `cpp/testdata/manifest.json`: native golden/checkpoint fixture metadata.
- `cpp/tools/dump_golden.py`: PyTorch/reference fixture generator.

The reference has no equivalent native-kernel golden or device test binaries.

### A.13 RelativeDB-only website

- Site configuration: `website/.gitignore`, `README.md`, `docusaurus.config.ts`, `package.json`, `package-lock.json`, `tsconfig.json`, `sidebars.ts`, `sidebars-relql.ts`.
- Content: `website/docs/intro.md`, `website/relql/index.md`.
- Application/styles: `website/src/pages/index.tsx`, `index.module.css`, `website/src/css/custom.css`.
- Hosting/static marker: `website/static/.nojekyll`.
- Brand/site assets: `website/static/img/favicon.ico`, `logo.svg`, `logo.png`, `logo-blue.svg`, `logo-crimson.svg`, `logo-plum.svg`, `logo-slate.svg`, `logo-teal.svg`, `logo-violet.svg`, `docusaurus-social-card.jpg`, `docusaurus.png`, `undraw_docusaurus_mountain.svg`, `undraw_docusaurus_react.svg`, `undraw_docusaurus_tree.svg`.

The reference uses repository Markdown and a context-visualizer script, not a product documentation site.

## Appendix B — key source anchors

These are the most useful starting points for independently checking the report.

| Topic | Reference anchor | RelativeDB anchor |
|---|---|---|
| Model geometry/math | `../relational-transformer/src/rt/model.py:31-135,247-576` | `cpp/src/rt.hpp:1-170`; `cpp/src/rt.cpp` |
| Task definitions | `../relational-transformer/src/rt/tasks.py:1-160` | `python/src/relativedb/relql/ast.py`; `parser.py` |
| Default evaluation sampler | `../relational-transformer/src/rt/eval_utils.py:164-185` | `python/src/relativedb/traversal.py`; `engine.py` `ContextPolicy` |
| Target/sampler construction | `../relational-transformer/rustler/src/fly.rs:1116-1739` | `python/src/relativedb/task.py`; `traversal.py`; `rt_native.py` `_build_ctx_seq` |
| Preprocessing/stats | `../relational-transformer/rustler/src/pre.rs` | `python/src/relativedb/rt_native.py` `ColumnStats`, `_label_stats`, and `_normalize_one` |
| Checkpoint resolution | `../relational-transformer/src/rt/checkpoints.py:41-140` | `python/src/relativedb/rt_native.py:662-709` |
| Embedding guard | checkpoint config consumed by model loader | `python/src/relativedb/model.py:64-72`, not called by loader |
| Full training | `../relational-transformer/scripts/pretrain.py`; `src/rt/swa.py`; `muon.py` | `python/src/relativedb/engine.py:817-979`; `cpp/src/rt_train*` |
| Forecast output | one target task row per timestamp | `python/src/relativedb/rt_native.py:957-960` |
| Candidate bound | target-row timestamp per sampled item | `python/src/relativedb/rt_native.py:1334-1343` |
| Ablation | `../relational-transformer/rustler/src/fly.rs:2359-2400` | `python/src/relativedb/engine.py:1007-1015` |
| Packaging | `../relational-transformer/pyproject.toml` | `python/pyproject.toml`; `cpp/CMakeLists.txt` |
| Evaluation | `../relational-transformer/src/rt/eval_utils.py`; `evaluator.py` | `evaluation/` four-runner adapter and official scorer |

## Appendix C — terminology

- **Reference**: the inspected `~/relational-transformer` snapshot.
- **RT-J**: the current gated/QK-normalized relational-transformer architecture/checkpoint family used by both implementations.
- **Kernel parity**: equivalent transformer computation given equivalent input tensors and weights.
- **End-to-end parity**: equivalent data preprocessing, sampling, tensor construction, model computation, and output decoding.
- **Focal entity**: the entity whose prediction is being returned.
- **Cohort/peer**: other examples added to provide in-context task demonstrations.
- **Self-label**: RelativeDB's historical evaluation of the RelQL target expression, emitted as a demonstration label.
