# Relational Transformer vs. RelativeDB

## Comprehensive implementation-difference and conformance report

**Reviewed:** 2026-07-19  
**Reference:** `~/relational-transformer` at `eece04847de7b52d6fe7a718c277abec7bb18c83`  
**Implementation under review:** `~/relativedb` at `c361db29d9584dcedf7bf89145d89256807333f8`, plus the existing uncommitted worktree changes listed below

## 1. Purpose, scope, and interpretation

This is a repository-wide comparison of Stanford's `relational-transformer` implementation with RelativeDB's implementation in this repository. It covers repository structure, data contracts, preprocessing, context sampling, tensor construction, model math, checkpoints, runtime backends, query/task semantics, training, evaluation, packaging, tests, CI, and documentation.

The two repositories are not forks. Only `.gitignore` and `README.md` occur at the same tracked path; every other tracked path differs. A literal line diff would therefore be mostly meaningless. “All differences” in this report means:

1. every material system, feature, API, and behavioral difference found by a full tracked-file review;
2. every observed conformance point;
3. every correctness or release risk found through targeted execution;
4. an exhaustive tracked-file/subsystem disposition in Appendix A.

Generated corpora, build products, caches, checkpoint payloads, notebooks' binary cell output, and untracked benchmark data are not compared byte-for-byte. They are discussed where they affect behavior or reproducibility.

### Worktree state

The reference repository was clean. This repository already contained user changes before the review:

- modified: `README.md`, `python/src/relativedb/engine.py`, `python/src/relativedb/rt_native.py`, `python/tests/test_engine.py`, `python/tests/test_rt_native.py`, and `python/tests/test_xlang_parity.py`;
- untracked: `benchmarks/gh/` and `benchmarks/olist/`.

Those changes were reviewed as part of the current implementation and were not reverted or rewritten. This report is the only file added by the review.

## 2. Executive conclusion

RelativeDB is **not a reimplementation of the reference product**. It is a new online predictive-query system around a faithful native inference port of one particular reference checkpoint family:

- The **model core is strongly conformant**. The current reference PyTorch model exactly reproduces RelativeDB's committed golden fixture, and RelativeDB's C++ CPU and MPS paths reproduce it within about `3.91e-3` maximum absolute drift.
- The **end-to-end prediction system is not reference-equivalent**. RelativeDB changes the target-row representation, normalization, cohort construction, graph sampling, temporal handling, supported tasks, and output decoding. These differences are large enough to change predictions even when the native transformer kernel itself is correct.
- RelativeDB adds substantial product surface absent from the reference: RelQL, online callbacks, point-in-time query planning, assumptions, explanations, native CPU/MPS/CUDA inference, quantization, ranking/multiclass adapters, and frozen-backbone head training.
- RelativeDB omits substantial research surface present in the reference: RelBench preprocessing, rkyv/mmap datasets, the production sampler's walk/seed tiers, full-model pretraining, DDP/SWA, the official evaluator, task/config-aware checkpoint loading, legacy RT/PluRel compatibility, and the `rel2tab` baseline framework.

The highest-priority issue is outside the transformer math: **default zero-shot predictions are batch-dependent** because RelativeDB fits numeric and synthetic-label normalization from the entities in the current scoring call. The same entity produced materially different regression and classification outputs when another entity was added to the batch. That violates ordinary scoring determinism and the reference pipeline's fixed preprocessing-stat contract.

## 3. Prioritized findings

| Priority | Finding | Evidence | Consequence |
|---|---|---|---|
| P0 | Default zero-shot normalization is fitted across the current scoring batch. | `python/src/relativedb/rt_native.py:1193`; live probe in §10.2 | A prediction for one entity changes when unrelated entities are added or removed. |
| P0 | RelativeDB replaces the real task-table target cell with a generic synthetic `task.label` target. | RelativeDB sequence builder vs. `rustler/src/fly.rs:1116-1266` | The model sees different table/column semantics from the representation on which the reference was trained and evaluated. |
| P0 | Cohort rows enter the current entity's self-label aggregation. | `python/src/relativedb/rt_native.py:993`; live probe in §10.3 | Other entities' histories can be counted as the target entity's history. |
| P1 | Context construction is geometrically different from the reference sampler. | `python/src/relativedb/engine.py:584-655`; `rustler/src/fly.rs:1116-1739` | Kernel parity does not imply end-to-end reference parity. |
| P1 | The built Python wheel does not contain or build `librt_c`. | clean wheel/install probe in §10.7 | The advertised parser, CSC, and native inference are unavailable after a normal wheel install. |
| P1 | Forecast horizons repeat one scalar rather than making horizon-specific predictions. | `python/src/relativedb/rt_native.py:960` | A multi-horizon query returns duplicated values, not a forecast curve. |
| P1 | `ABLATE` parses and appears in plans but is not executed. | `python/src/relativedb/engine.py:1012` | A scientifically meaningful reference ablation is currently a declared no-op. |
| P1 | Multiclass/ranking candidate enumeration uses the maximum bound across a batch. | `python/src/relativedb/rt_native.py:1334` | Later candidates/classes can become visible to entities with earlier anchors. |
| P1 | Checkpoint metadata is ignored and the documented embedding guard is not called. | `python/src/relativedb/model.py:64`; `rt_native.py:672-709` | Architecture/encoder mismatches can load without the intended fail-fast validation. |
| P1 | Release workflows reference removed Rust, Java, and DuckDB trees. | `.github/workflows/release-libraries.yml`; `.github/workflows/duckdb-extension.yml` | Current release jobs cannot succeed as written. |
| P1 | `run_suite.py` never attaches a native model backend. | `benchmarks/run_suite.py:37-44`; dataset constructors | The documented generalizability command does expensive assembly but cannot complete scoring. |
| P2 | Package name and README install command disagree. | `python/pyproject.toml:6`; `README.md:91` | `pip install relativedb` does not match distribution name `relationdb`. |
| P2 | Model-size and checkpoint-format documentation is inconsistent. | `README.md:26,83`; `cpp/README.md:44,195-236` | The current 86M-parameter BF16 checkpoint is described as 22M and “fp32”/171 MB. |
| P2 | `python/pyproject.toml` names a missing package README. | `python/pyproject.toml:9` | Wheel/sdist builds warn and ship incomplete project metadata. |
| P2 | Benchmark findings and binding claims contain stale counts/components. | `benchmarks/FINDINGS.md`; current tracked tree/tests | Published evidence is difficult to reproduce against the current implementation. |

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
| Model role | Configurable PyTorch training and inference | Hard-coded RT-J native inference; optional frozen-head adaptation |
| Main devices | PyTorch CUDA; CPU/MPS eager possible for small examples | CPU, MPS, and optional CUDA inference; Metal head training |
| Evaluation | Curated RelBench tasks and `rel2tab` baselines | Custom point-in-time benchmark harness plus Olist/GH/task-fit experiments |
| Documentation | Markdown guides and examples | Root/product docs plus Docusaurus site |

### 4.1 Reference data flow

`RelBench/manifest + Parquet → Rust preprocessor → rkyv/mmap graph + embeddings → Rust sampler → PyTorch RT → RelBench evaluator`

Full pretraining uses the same sampling stack with PyTorch optimization, distributed execution, checkpointing, and stochastic weight averaging.

### 4.2 RelativeDB data flow

`Schema + Row callbacks/scanners → live retriever or CSC snapshot → RelQL validation/planning → Python context builder/tokenizer → C++ RT-J → typed query results`

Fine-tuning freezes the native backbone, extracts target features, trains a small linear head on Metal, and serves that head over backbone features.

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
| Extra framework/benchmark LOC | 4,823 `rel2tab` + 4,127 scripts | 5,506 benchmark Python |
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
- `Row.parents` is scalar per FK column. It does not model a list-valued FK relationship directly.

### 6.3 Statistics and normalization

**Reference**

- Computes numeric statistics per physical column during preprocessing.
- Uses fixed train-derived task statistics for validation and test.
- Reuses those statistics across batches, entities, and evaluation runs.
- Uses sample standard deviation for numeric columns. Datetime preprocessing uses its own Welford/population convention.
- Keeps semantic channels separate, unless evaluation intentionally maps booleans through the numeric path with `bool_as_num=True`.

**RelativeDB**

- Historically/default zero-shot behavior computes normalization from the sequences in the current `score` call.
- Current worktree code adds `ColumnStats` and fits it automatically during `Engine.finetune`, which improves physical numeric/datetime column stability for that workflow.
- It does not automatically fit/persist physical-column statistics for normal zero-shot engine construction.
- Even when physical `ColumnStats` is supplied, synthetic historical task-label values are normalized from the current batch.
- Its datetime standard-deviation convention is not exactly the reference preprocessor's convention, despite “reference” language in nearby documentation.
- Replaces zero standard deviations defensively; the reference's boolean path can retain zero variance.

This normalization layer is the largest demonstrated source of end-to-end nondeterminism; see §10.2.

### 6.4 Target representation

This is the most fundamental semantic divergence.

**Reference**

- Selects an actual target node and actual target column.
- Emits the target cell first.
- Masks that cell while retaining its real table and column-name embedding.
- Uses known same-task rows from the same physical task table as in-context examples.
- For autocomplete, masks an actual numeric or boolean feature cell in its original row.

**RelativeDB**

- Builds an entity context, then creates a synthetic generic target row. Its masked `task.label` (and optional `task.timestamp`) tokens are emitted before the real context tokens.
- Produces historical “self-labels” by evaluating the RelQL target expression over assembled context rows.
- Therefore does not preserve the target task table's original name, target column name, task row topology, or task-table peer semantics.
- Bare entity-column autocomplete is also routed through the generic target representation instead of masking the real entity cell in place.

The RT-J weights can be numerically correct and still receive an out-of-distribution prompt geometry because schema/table/column semantics are part of every token.

### 6.5 Context and graph sampling

| Behavior | Reference sampler | RelativeDB |
|---|---|---|
| Global context default | 8,192 cells | 8,192 counted row cells |
| Local context default | 256 | No equivalent local neighborhood quota |
| BFS width | 32 default evaluation width | 32 default per hop |
| Walk ranking | 10,000 random walks, length 20 by default | None |
| Same-table peers | Multi-tier visited/unvisited selection, stochastic fallback; optional FAISS | Cohort callback, otherwise first scanner rows |
| Peer ordering | Recency/frequency controls, seeded randomness | Scanner order; child rows newest-first |
| Label balancing | Supported | Not supported in context construction |
| Seed reproducibility | Explicit shuffle/context seeds | No sampling seed because default selection is deterministic scanner/BFS order |
| Budget fill | Target first, then tiered BFS/peer neighborhoods until cell capacity | Entity/BFS/cohort rows, then independent token truncation |
| Time filtering | Target-time checks on same-table seeds and P2F expansion | Defensive `TemporalBound` applied to every callback result |
| Static rows | Admitted | Admitted |

Additional RelativeDB-specific differences:

- Parent rows are always followed; children are capped newest-first at each hop.
- Cohort defaults to 256, but a cohort is empty if neither a cohort callback nor a scanner is provided. The README quick-start configuration therefore does not actually obtain the default 256 peers.
- The cell budget counts `len(Row.cells)` plus a timestamp. It does not count the synthetic target/self-label tokens. The tokenizer later truncates to sequence length, so admitted graph context and final tensor context can differ.
- The reference limits feature-to-parent neighbors to five and asserts its tensor shape. RelativeDB also uses five, but silently truncates a longer parent-node list.
- RelativeDB's online retrieval provides stronger defensive rechecking of callback timestamps than the reference's parent traversal, but it also makes behavior dependent on application callback correctness and completeness.

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
- Fine-tuned head training is Metal-only. At serving time, backbone feature extraction follows the selected CPU/MPS device, then the small saved head is evaluated on CPU.

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

- Does not preprocess a multi-database pretraining corpus and cannot train the backbone.
- Extracts a frozen 512-dimensional target representation.
- Trains only a small task head: linear regression/binary or `512 × C + C` multiclass/ranking parameters.
- Builds labels either from supplied labels or by evaluating future RelQL windows.
- Persists a safetensors head and sidecar metadata/statistics.
- Does not implement DDP, optimizer resume, SWA, or reference training recipes.

Head training is a useful product adaptation, but it is not equivalent to the full-model reference training or to continued pretraining.

### 6.13 Evaluation and baselines

**Reference**

- Defines a curated 21-task RelBench evaluation set: 12 binary classification and 9 regression tasks.
- Maps predictions back to original task-table row order.
- Uses RelBench metrics, including AUROC and normalized MAE.
- Tunes context configurations on validation data and supports seed/config ensembles.
- Provides `rel2tab`, a general featurizer/predictor framework with global, entity, RT, precomputed, SQL, and RDBLearn-style features and mean, linear, ridge, LightGBM, XGBoost, TabPFN/TabICL-family predictors.
- Does not commit a comparable end-result JSON in the inspected snapshot.

**RelativeDB**

- Has a point-in-time harness over MovieLens, Online Retail II, and Brightkite, with naive recency/activity/popularity baselines.
- Has separate task-fit experiments using XGBoost, Digits, Olist, and untracked current GH/Olist work.
- Does not use the reference sampler or official task-table evaluator, so headline numbers are not apples-to-apples with released RT-J results.
- The committed findings state that the engine wins 4 of 10 scorable cells and that churn loses to recency on all three core datasets.
- The findings correctly warn about possible pretraining contamination in several public datasets and weak ranking/multiclass behavior.

Selected local experiment results are shown below. The Olist/GH rows come from the current untracked worktree reports, while Digits comes from a committed task-fit artifact; they are evidence from different protocols, not one benchmark leaderboard.

| Experiment | RelativeDB result | Comparator/result | Interpretation |
|---|---:|---:|---|
| Olist bad review, zero-shot | AUROC 0.547 | XGBoost 0.521 | Small positive comparison in this split |
| Olist bad review, head | AUROC 0.476 | XGBoost 0.521 | Adaptation regresses |
| Olist review stars, zero-shot | Accuracy 0.565 | XGBoost 0.548 | Accuracy higher, but macro-F1 0.144 vs. 0.176 |
| Olist future spend, zero-shot | MAE 38.25 | XGBoost 11.35 | Large regression gap |
| Olist future spend, head | MAE 709.64 | XGBoost 11.35 | Failed adaptation |
| Olist repeat purchase, zero-shot | AUROC 0.630 | XGBoost 0.584 | Positive comparison |
| Olist repeat purchase, head | AUROC 0.407 | XGBoost 0.584 | Adaptation regresses |
| GH bot, head | AUROC 0.523 | XGBoost 0.977 | Very large gap |
| Digits head | Accuracy 0.703 | released baseline in local report 0.10 | Head learns the synthetic task |

The committed `task_fit/olist_results.json` also contains alternative task-fit runs with different splits/recipes; those should not be merged with the table above without recording provenance.

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
- `benchmarks/task_fit/README.md` repeats the 22M claim.
- `benchmarks/FINDINGS.md` refers to current Java/Rust bindings that are absent and reports older parser corpus counts; current C++ tests pass 67 accepted and 22 rejected cases.
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
| Full-model pretraining | Yes | No | Missing |
| DDP/SWA/resume | Yes | No | Missing |
| Frozen-head training | Not primary workflow | Yes | Addition |
| RelBench manifest preprocessing | Yes | No | Missing |
| rkyv/mmap graph | Yes | No | Missing |
| Live callback retrieval | No | Yes | Addition |
| CSC snapshot | Native preprocessed adjacency | In-memory callback snapshot | Similar purpose, different contract |
| Reference random-walk sampler | Yes | No | Missing |
| Optional FAISS peer sampling | Yes | No | Missing |
| Label-balanced context | Yes | No | Missing |
| Fixed preprocessing stats | Yes | Partial/finetune only | Incomplete |
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
| Official RelBench evaluation | Yes | No | Missing |
| `rel2tab` baseline framework | Yes | No | Missing |
| Custom real-data backtests | No comparable harness | Yes | Addition |
| Context visualization UI | Yes | Explain JSON/text | Different capability |
| Docusaurus product site | No | Yes | Addition |

## 8. Native conformance evidence

### 8.1 Parameter and fixture reproduction

I loaded the current reference `RelationalTransformer` using the same released RT-J checkpoint and ran RelativeDB's committed x-language golden input through it.

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
| RelativeDB Python tests | **214 passed** | 47.05 s; one `utcfromtimestamp` deprecation warning |
| RelativeDB CMake configure/build | **passed** | Current C++ tree built successfully |
| RelQL C++ corpus | **67/67 accepted, 22/22 rejected** | `cpp/build/relql_test`, run from `cpp/` |
| CSC C++ test | **22,502/22,502 passed** | `cpp/build/csc_test` |
| Native training test | **passed** | Multiclass loss 1.098612 → 0.010736; ranking 1.386294 → 0.093876 |
| Native RT-J golden, CPU | **passed** | max drift ≈ 0.003911 |
| Native RT-J golden, MPS | **passed** | max drift ≈ 0.003910 |
| Reference lightweight Python tests | **7 passed** | `test_api.py` + `test_import_safety.py`; 14 PyTorch deprecation warnings |
| Reference Rust build/tests | **built; 0 Rust unit tests** | `PYO3_PYTHON=/opt/homebrew/bin/python3.12 cargo test --locked --no-default-features` |
| Reference full Python suite | **not fully run** | 9 tests collected; 2 require built extension/preprocess deps such as PyArrow/PluRel |
| RelativeDB wheel build | **built with warning** | pure `relationdb-0.1.0-py3-none-any.whl`; missing `python/README.md` |
| Isolated installed-wheel parse | **failed as expected from package contents** | `NativeParserUnavailable: librt_c not found` |
| Documented `run_suite.py --quick --no-stability` | **stopped after diagnosis** | loaded 3 datasets; no backend is attached; context assembly is very expensive before scoring reaches the backend check |

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

### 10.2 Batch-dependent prediction

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

**Finding:** the default API is not entity-wise invariant. Batch size, filtering, pagination, and the presence of unrelated entities can alter predictions. The reference preprocessing/statistics path is batch-invariant.

### 10.3 Cohort contamination of self-labels

I assembled a context for C7 in the same toy schema:

- with `cohort_size=0`, the context contained C7 and its own orders/products, and the first 90-day preceding self-label count was **1**;
- with the default `cohort_size=256`, scanner-selected C1/C9 and their rows entered the context, and the same C7 self-label became **2**.

`_self_labels` evaluates the target aggregation over all `ctx.rows_by_table()` without first restricting rows to the focal entity. Cohort examples are useful as in-context demonstrations, but their event rows must not be folded into the focal entity's ground-truth expression.

**Finding:** cohort construction can change the task definition, not just the evidence presented to the model.

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

### 10.8 Generalizability harness backend

`benchmarks/run.py` correctly attaches `RtNativeBackend` to its MovieLens and retail engines. `benchmarks/run_suite.py` constructs three datasets and calls `suite.run` without doing so. Each dataset constructor creates `Engine(..., model_backend=None)`.

A live `--quick --no-stability` run loaded:

- Online Retail: 400 entities;
- MovieLens: 610 entities;
- Brightkite: 400 entities.

It then spent substantial time assembling a 5,000,000-cell-budget context before model scoring. I interrupted it after capturing the stack. Static control flow shows the eventual `Engine._require_backend()` failure once the first context reaches scoring.

`benchmarks/harness/audit_fixes.py` has a similar default engine without a backend. It is called late in `run.py`, after normal task scoring, so the primary report can produce results before this audit fails.

## 11. Recommended remediation order

### P0 — make scoring semantically stable

1. **Persist all normalization statistics.** Require dataset/preprocessing statistics for zero-shot operation, or fit once at engine/index creation. Never fit statistics from the current request batch.
2. **Separate focal rows from demonstrations.** Evaluate RelQL self-label expressions over a focal-entity subgraph only. Keep cohort examples as separate labeled nodes.
3. **Represent the target faithfully.** For entity autocomplete, mask the real cell in place. For aggregate/query tasks, define and validate an explicit task-table schema representation whose table/column embeddings are stable and match training.
4. Add invariance tests:
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

11. Attach the backend in `run_suite.py` and `audit_fixes.py`; lower the diagnostic context policy so backend/config errors surface before multi-million-cell assembly.
12. Record benchmark provenance in every JSON: commit, dirty diff hash, checkpoint hash/URI, quantization, device, stats artifact, context policy, random seeds, dataset checksum, and contamination caveat.
13. Add a reference-style evaluator adapter on at least the curated RelBench tasks. This is the only defensible way to compare end-to-end results to RT-J.
14. Add a sampler-differential fixture: same preprocessed graph/target through Rust and RelativeDB tokenizers, with per-token tensor diffs.

### P2 — release and documentation cleanup

15. Choose one distribution name and align README, PyPI metadata, import examples, URLs, and artifact names.
16. Correct 22M → 86M and distinguish 171 MB BF16-on-disk from about 342 MB expanded FP32 resident memory.
17. Add `python/README.md` or point `readme` at an included file.
18. Remove or restore Java/Rust/DuckDB workflows and documentation.
19. Add a tracked license file consistent with declared metadata.
20. Export new public types consistently; `python/src/relativedb/__init__.py` currently has duplicate lazy handling for `FineTunedHead` and does not consistently list newer public types in `__all__`.

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
- benchmark harness, task-fit scripts, committed results/findings, and current untracked Olist/GH summaries;
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
- a live quick generalizability-suite diagnostic;
- `git diff --check` and final worktree status checks.

### 13.4 Effort boundaries

I did not:

- download and preprocess the full RelBench corpus;
- run reference full pretraining or the 21-task GPU evaluation;
- run every large RelativeDB benchmark to completion;
- compare CUDA output on a CUDA host;
- assess model quality with a newly designed statistically powered dataset;
- rewrite any implementation issue found.

Those would be separate multi-hour/day experiments and, for training, require appropriate GPU resources. The report distinguishes source-proven findings, live behavioral probes, committed benchmark evidence, and limitations.

## 14. Final assessment

RelativeDB has a credible and well-tested native RT-J **kernel** plus a much broader predictive-query product. Its strongest engineering is the C++ inference stack: faithful architecture, shared golden fixture, multiple devices, sparse attention, and quantization. Its most important weakness is the semantic bridge into that kernel. Target construction, batch-fitted normalization, cohort self-label contamination, and non-reference sampling can dominate the small native numerical drift by orders of magnitude.

Accordingly:

- describe the current system as **“RT-J-backed”**, not end-to-end reference-equivalent;
- treat ranking, multiclass, multi-horizon forecasting, `ABLATE`, and head adaptation as experimental until their contracts and evaluations are tightened;
- fix batch/cohort invariance before optimizing the native runtime further;
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
- `rustler/src/main.rs`: standalone preprocessing/benchmark entry.

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

RelativeDB benchmark scripts exercise serving/head adaptation, not equivalent full training/evaluation jobs.

### A.6 Reference-only `rel2tab`

- Core: `rel2tab/README.md`, `__init__.py`, `config.py`, `featurize.py`, `featurizer.py`, `model.py`, `predictor.py`.
- Featurizers: `featurizers/__init__.py`, `entity_featurizer.py`, `global_featurizer.py`, `precomputed_featurizer.py`, `rdblearn_featurizer.py`, `rt_featurizer.py`, `sql_featurizer.py`.
- Dataset SQL features: `featurizers/sql_queries/__init__.py`, `rel_amazon.py`, `rel_avito.py`, `rel_event.py`, `rel_f1.py`, `rel_hm.py`, `rel_stack.py`, `rel_trial.py`.
- Predictors: `predictors/__init__.py`, `identity_predictor.py`, `lgbm_predictor.py`, `linear_predictor.py`, `mean_predictor.py`, `ridge_predictor.py`, `tab_predictor.py`, `tabicl_batched_predictor.py`, `xgboost_predictor.py`, `xgboost_tuned.py`.

RelativeDB has isolated XGBoost/naive comparisons in `benchmarks/`, not a reusable featurizer/predictor framework.

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
- `python/tests/test_xlang_parity.py`: shared cross-language fixture parity.

The reference's three test modules do not cover an equivalent product surface.

### A.11 RelativeDB-only C++ build and core

- `cpp/.gitignore`, `cpp/CMakeLists.txt`, `cpp/README.md`: native build, targets, and performance/conformance documentation.
- Parser: `cpp/src/relql.hpp`, `relql.cpp`, `relql_c.h`, `relql_c.cpp`.
- CSC: `cpp/src/csc.hpp`, `csc.cpp`, `csc_c.h`, `csc_c.cpp`.
- Transformer public/core: `cpp/src/rt.hpp`, `rt.cpp`, `rt_internal.hpp`, `rt_math.hpp`.
- C ABI: `cpp/src/rt_c.h`, `rt_c.cpp`.
- Quantization: `cpp/src/rt_quant.hpp`, `quantize.cpp`.
- Devices: `cpp/src/rt_metal.mm`, `rt_cuda.cu`.
- Frozen-head training: `cpp/src/rt_train.hpp`, `rt_train.cpp`, `rt_train_metal.mm`.
- Benchmark executable: `cpp/src/bench.cpp`.

The reference's transformer is PyTorch; its Rust code preprocesses/samples and does not provide native transformer kernels.

### A.12 RelativeDB-only C++ tests/tools

- `cpp/src/test_relql.cpp`: parser conformance/rejection corpus.
- `cpp/src/test_csc.cpp`: high-volume native index correctness.
- `cpp/src/test_golden.cpp`: RT-J PyTorch/C++ golden conformance across devices/formats.
- `cpp/src/test_train.cpp`: multiclass/ranking head training convergence.
- `cpp/testdata/manifest.json`: native golden/checkpoint fixture metadata.
- `cpp/tools/dump_golden.py`: PyTorch/reference fixture generator.

The reference has no equivalent native-kernel golden or device test binaries.

### A.13 RelativeDB-only benchmark harness

- `benchmarks/README.md`, `FINDINGS.md`: harness instructions and committed conclusions.
- `benchmarks/run.py`: main MovieLens/retail native-backend backtest and audits.
- `benchmarks/run_suite.py`: three-dataset generalizability matrix; currently omits backend wiring.
- `benchmarks/harness/__init__.py`: harness package.
- `benchmarks/harness/datasets.py`: MovieLens, Online Retail II, and Brightkite loaders, schemas, callbacks, truth arrays.
- `benchmarks/harness/backtest.py`: churn/count/value/ranking task execution.
- `benchmarks/harness/metrics.py`: predictive metrics.
- `benchmarks/harness/suite.py`: dataset/task grid and stability summaries.
- `benchmarks/harness/audit_grammar.py`: language coverage audit.
- `benchmarks/harness/audit_leakage.py`: point-in-time leakage audit.
- `benchmarks/harness/audit_fixes.py`: context truncation instrumentation audit; currently creates an engine without a backend.

Reference evaluation is task-table/RelBench based and has no corresponding files.

### A.14 RelativeDB-only task-fit experiments

- `benchmarks/task_fit/README.md`: experimental protocol/results summary.
- `brightkite_clf_reg.py`: Brightkite classification/regression fitting.
- `churn_rtj_with_rfm_cells.py`: churn with RFM cells.
- `churn_spend_rtj.py`, `churn_spend_xgboost.py`: RT-J/XGBoost churn/spend comparisons.
- `data_efficiency.py`: labeled-data scaling.
- `digits_metal_finetune.py`: synthetic digits head experiment.
- `olist_metal_vs_xgboost.py`: Olist head/baseline comparison.
- `ranking_buy_it_again.py`: ranking experiment.
- `digits_head.safetensors`, `olist_review_head.safetensors`: committed trained-head artifacts.
- `digits_results.json`, `olist_results.json`: committed experiment outputs.

The reference's comparable baseline machinery is generalized under `rel2tab`; it does not commit these RelativeDB-specific artifacts.

### A.15 RelativeDB-only cross-language fixture

- `benchmarks/xlang_fixture/README.md`: fixture contract.
- `embeddings.tsv`, `movies.tsv`, `ratings.tsv`: deterministic fixture inputs.
- `golden.json`: reference PyTorch outputs/tensors consumed by C++ and Python tests.

This is the central bridge used to establish kernel conformance. The reference repository does not contain the fixture, but its current model reproduced it exactly in this review.

### A.16 RelativeDB-only website

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
| Default evaluation sampler | `../relational-transformer/src/rt/eval_utils.py:164-185` | `python/src/relativedb/engine.py:252-274` |
| Target/sampler construction | `../relational-transformer/rustler/src/fly.rs:1116-1739` | `python/src/relativedb/engine.py:584-655`; `rt_native.py:993-1290` |
| Preprocessing/stats | `../relational-transformer/rustler/src/pre.rs` | `python/src/relativedb/rt_native.py` `ColumnStats` and `_normalize` |
| Checkpoint resolution | `../relational-transformer/src/rt/checkpoints.py:41-140` | `python/src/relativedb/rt_native.py:662-709` |
| Embedding guard | checkpoint config consumed by model loader | `python/src/relativedb/model.py:64-72`, not called by loader |
| Full training | `../relational-transformer/scripts/pretrain.py`; `src/rt/swa.py`; `muon.py` | `python/src/relativedb/engine.py:817-979`; `cpp/src/rt_train*` |
| Forecast output | one target task row per timestamp | `python/src/relativedb/rt_native.py:957-960` |
| Candidate bound | target-row timestamp per sampled item | `python/src/relativedb/rt_native.py:1334-1343` |
| Ablation | `../relational-transformer/rustler/src/fly.rs:2359-2400` | `python/src/relativedb/engine.py:1007-1015` |
| Packaging | `../relational-transformer/pyproject.toml` | `python/pyproject.toml`; `cpp/CMakeLists.txt` |
| Evaluation | `../relational-transformer/src/rt/eval_utils.py`; `evaluator.py` | `benchmarks/harness/`; `benchmarks/task_fit/` |

## Appendix C — terminology

- **Reference**: the inspected `~/relational-transformer` snapshot.
- **RT-J**: the current gated/QK-normalized relational-transformer architecture/checkpoint family used by both implementations.
- **Kernel parity**: equivalent transformer computation given equivalent input tensors and weights.
- **End-to-end parity**: equivalent data preprocessing, sampling, tensor construction, model computation, and output decoding.
- **Focal entity**: the entity whose prediction is being returned.
- **Cohort/peer**: other examples added to provide in-context task demonstrations.
- **Self-label**: RelativeDB's historical evaluation of the RelQL target expression, emitted as a demonstration label.
