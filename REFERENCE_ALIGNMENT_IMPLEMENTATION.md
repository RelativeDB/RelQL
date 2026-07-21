# Reference-alignment implementation note

Date: 2026-07-20

This change implements the first three priorities from
`RELATIVEDB_IMPROVEMENT_ROADMAP.md`: stable normalization, stable target/task
representation, and pluggable graph traversal.

## Normalization

`ModelConfig.normalization_mode` is the public switch:

```python
from relativedb import ModelConfig, NormalizationMode

zero_shot = ModelConfig(normalization_mode=NormalizationMode.ZERO_SHOT)
reference = ModelConfig(normalization_mode=NormalizationMode.REFERENCE)
```

For direct inference in reference mode, fit and attach the artifact explicitly
(the explicit adapter-fitting path performs this step automatically):

```python
from relativedb import ColumnStats, RtNativeBackend, TemporalBound

stats = ColumnStats.fit(schema, wiring, TemporalBound.at_or_before(train_end))
backend = RtNativeBackend(
    schema=schema,
    wiring=wiring,
    column_stats=stats,
    normalization_mode=NormalizationMode.REFERENCE,
)
```

Derived numeric targets additionally require
`stats.with_task_values(task_spec, training_labels)` before scoring.

- `ZERO_SHOT` computes numeric and datetime statistics independently for each
  entity context. It requires no preprocessing artifact and is invariant to
  the other entities in the scoring batch.
- `REFERENCE` uses a `ColumnStats` artifact fitted before inference. Numeric
  columns use sample standard deviation, while the reference's shared datetime
  accumulator uses population standard deviation. Derived numeric task targets
  have separately persisted task statistics.
- Missing physical or task statistics fail closed in reference mode. The
  backend does not silently fall back to request statistics.
- Fitted adapter heads persist both the mode and preprocessing artifact. Loading a
  head restores the training-time contract, preventing training/serving skew.

`NormalizationMode.STATISTICS` is an alias for `REFERENCE`. A backend-level
mode can override the model configuration when constructing
`RtNativeBackend`, which is useful for serving a fixed artifact.

## Stable task and target representation

`TaskSpec.from_query` canonicalizes the validated target AST and derives a
stable SHA-256 task identity. Formatting-only query changes therefore produce
the same model-facing schema, while changes to a window, predicate, expression,
entity table, or task type produce a different identity.

Direct prediction of a column on the entity table now masks that physical
table/column cell. Derived targets use stable synthetic task table and target
column names based on the canonical identity. `TaskSpecFactory` is injectable
on `RtNativeBackend` for applications with an external task registry.

Derived targets now become real timestamped `Row` objects in the stable task
table identified by `TaskSpec`. The focal target row has an unknown/masked
label; configurable historical windows are materialized for every entity with
labels evaluated at their legal cutoff. FOLLOWING windows may observe through
their own end, but never past the focal prediction anchor. These task rows are
bidirectionally linked to their entity and receive deterministic global node
IDs.

Global peer rows are explicitly distinguished from the focal entity graph in
`EntityContext.focal_row_keys`. Peer labels are attached to their own task rows
and entity graphs, so adding peer context cannot rewrite focal history.

## Graph traversal

`Engine(..., traversal=...)` accepts any `GraphTraversal`.

- `ReferenceTraversal` is the default. The engine copies scanner rows into one
  immutable CSC/bidirectional graph snapshot at construction. Snapshot-global
  node IDs remain stable across focal contexts.
- `BreadthFirstTraversal` preserves the old cohort-seeded pull traversal and is
  available only when explicitly selected.
- The reference path uses the evaluator defaults: 8,192 global cells, 256 local
  cells, BFS width 32, 10,000 walks, and walk length 20.
- It implements target BFS, walk-visited same-table seeds, and random unvisited
  fallback. F2P is LIFO-priority; P2F is selected from the shallowest level,
  temporally filtered, and DB fanout is uniformly sampled to the width.
- Task-table P2F edges are traversed only for a seed from that same task table.
- The RNG is a direct port of rand 0.9.1 `StdRng`: PCG `seed_from_u64`,
  ChaCha12, integer-range sampling, and index sampling. Oracle vectors from a
  Rust program match exactly.

Reference traversal knobs live on `ContextPolicy`: `local_context_cells`,
`num_walks`, `walk_length`, and `seed`, in addition to the existing global
budget, hop, fanout, cohort, and recency controls.

```python
from relativedb import ContextPolicy, Engine

engine = Engine(
    schema,
    wiring,
    context_policy=ContextPolicy(
        max_context_cells=8192,
        local_context_cells=256,
        bfs_width=32,
        num_walks=10_000,
        walk_length=20,
        seed=7,
    ),
)
```

## Key and foreign-key feature contract

- Primary keys are identity only. A schema that declares its PK as a feature
  column is rejected, and a retriever-provided PK cell is defensively skipped.
- Foreign keys are graph edges and are not feature tokens by default, matching
  the reference.
- `LinkDef.feature_type` is the sole intentional extension. When set, the FK is
  emitted as a non-targetable feature without removing its graph edge.
- A list/tuple FK produces multiple graph edges. If also emitted as a feature,
  it must use `ValueType.TEXT` and is serialized as compact stable JSON.
- FK feature tokens count against the same context budget. More than five F2P
  parents fails closed instead of silently dropping edges.

## Verification performed

- Added regression coverage for zero-shot batch invariance.
- Added persisted column/task-stat normalization coverage.
- Added canonical task identity and target-sensitivity coverage.
- Added a custom traversal injection test that verifies query propagation.
- Added deterministic and temporal-safety tests for reference traversal,
  including a deliberately leaky retriever.
- Added immutable-snapshot/global-node-ID tests.
- Added Rust-oracle RNG vectors for rand 0.9.1 parity.
- Added materialized-task-row and target-first tensor tests.
- Added PK suppression and opt-in FK feature tests.
- Focused engine/traversal and native-backend verification passes. The raw C
  ABI goldens remain unchanged.
- Added an independent `evaluation/` package with optional `rt`, SQL/XGBoost,
  `ours`, and `ours_finetuned` runners. `ours_finetuned` now accepts only a
  complete task-specific model checkpoint; a frozen output-head adapter can no
  longer be mislabeled as end-to-end fine-tuning.
- Historical 128-cell run executed all four runners over every test row of `rel-f1/driver-top3` (726)
  and `rel-f1/driver-position` (760). The complete protocol, scores, deltas,
  wall times, and limitations are recorded in
  `evaluation/runs/rel-f1-head-to-head/results.md`.
- Direct native verification also passed for CSC, valid/invalid RelQL parsing,
  head-training convergence, and the CPU RT-J golden differential.

## Evaluation status

Evaluation outputs stay under `evaluation/runs/` and are not committed. The
publishable contract is simple: use the official task split and metric, keep the
requested context for each model, and record any model-specific fallback. A
fine-tuned checkpoint is accepted only when full validation improves over the
current best checkpoint. Historical frozen-head runs are diagnostics and are not
part of the full-model comparison.

### Correct fine-tuning and 8,192-cell contract

`Engine.finetune()` trains the complete RT-J checkpoint for binary
classification and regression tasks. The forward pass, backward pass, global
gradient clipping, and AdamW update run in native C++ on Metal/MPS. Torch is
not part of this path. The transformer blocks, input encoders, mask embeddings,
normalization parameters, and number decoder all update.

The trainer follows the reference task sampler. It uses the train split for
updates and the validation split for checkpoint selection. Zero-shot is the
initial best checkpoint. A trained checkpoint is promoted only when its full
validation score improves. When validation gets worse, the trainer restores
the best model and optimizer state, lowers the learning rate, and continues.
Early stopping remains a final guard after the configured recovery attempts.

Gradient accumulation separates the physical MPS batch from the effective
training batch. This lets an 18 GiB M3 use a small physical batch while keeping
the reference effective batch of 32. Each example can still contain all 8,192
cells. An out-of-memory fallback changes only the failing model's physical
batch or explicitly configured context; it does not reduce every runner.

Model weights and optimizer moments are written atomically at validation
boundaries. A recovery file records the sampler position, optimizer step,
current learning rate, validation history, and best score. A resumed run uses
that state instead of silently starting a new optimizer trajectory. Native
finite-difference checks cover representative encoder, attention, feed-forward,
normalization, and decoder parameters before a long run is promoted.

`Engine.fit_head()` remains a separate frozen-adapter API for multiclass and
multilabel-ranking tasks. It is not called full-model fine-tuning. The
`ours_finetuned` evaluator accepts only a complete task checkpoint and rejects
old scalar head files.

The head-to-head configuration now requests 8,192 global cells and 256 local
cells independently for RT, XGBoost, native zero-shot, and each full fine-tuned
checkpoint. Evaluator batch size is independently memory-tuned (four for the
neural runners, one for isolated XGBoost) without changing context length.
OOM fallbacks are per command/model and are
written to that runner's `context_usage.json`; they never lower another
runner's context length.

## Compatibility notes

The default traversal changed to `ReferenceTraversal`. Pull-only applications
must either provide scanners for the immutable snapshot or explicitly select
`BreadthFirstTraversal`. The default normalization is stable zero-shot, so
checkpoint outputs captured after request-batch normalization need to be
regenerated and versioned by mode. Existing saved fine-tuned heads without an
explicit mode are interpreted as reference/statistics-normalized.
