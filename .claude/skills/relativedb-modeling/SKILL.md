---
name: relativedb-modeling
description: >
  How to actually get good results out of RelativeDB/RelQL on a new dataset.
  Use when designing a schema for RT-J, writing RelQL queries, tuning the
  context policy, running ablations or counterfactuals, fine-tuning, or
  debugging "the model performs badly" on a task. Encodes the failure modes
  that cost real time: raw event tables starving the context, the
  local_context_cells trap, undated rows leaking the future, ASSUMING
  semantics, and how to tell a bad schema from a bad model.
license: Apache-2.0
compatibility: Requires the local relativedb checkout; native lib from cpp/build.
metadata:
  version: 1.0.0
  project: RelativeDB / RelQL
---

# Getting good results out of RelativeDB

## The one-paragraph version

RT-J's advantage is finding signal **across tables**. It is fed a fixed cell
budget of rows retrieved by BFS + random walks from the target. If any one table
dominates that budget, everything else is crowded out and the model looks
broken. The single most common cause of "the model is bad" is a **raw event
table** — likes, clicks, views, ticks — with tens of thousands of rows each
carrying one timestamp. Summarise those into period rows before they reach the
context.

---

## 0. Before debugging a model, validate the instrument

There is a known-good benchmark in this repo. If a new task performs badly, run
this first — it takes minutes and tells you whether the problem is your schema
or the environment:

```bash
RELATIVEDB_RT_LIB=$PWD/cpp/build/librt_c.dylib \
  .venv-eval/bin/python -m evaluation.head_to_head \
  --config evaluation/config.rel-f1.json --runners ours --tasks rel-f1/driver-top3
```

Expected `roc_auc` for `ours`:

| context cells | expected |
|---|---|
| 8,192 | ~0.91 |
| 2,048 | ~0.71 |

Reference points on the same task: RT-J reference 0.7112, XGBoost 0.6820. If you
reproduce these, the engine, checkpoint and native library are fine and the
problem is in your schema or query.

---

## 1. Schema design — the thing that actually matters

**Look at `rel-f1` for the target shape**: many tables (`drivers`, `results`,
`races`, `standings`, `constructors`), each row carrying several meaningful
columns, modest fan-out per entity.

### Do not hand the model a raw event log

Real measured example. Predicting future likes on a post, with `likes` modelled
as one row per like (157k rows, one `at` column each). Context for a single
target at a 2,048-cell budget:

| table | rows | cells | share |
|---|---:|---:|---:|
| likes | 1,136 | 2,272 | **69%** |
| posts | 114 | 684 | 21% |
| accounts | **6** | **12** | **0.4%** |

The model read 1,136 like timestamps and saw six accounts. Follower counts,
author identity, text and media were all crowded out. Spearman 0.22 against a
trivial baseline's 0.33.

### Summarise into period tables instead

Model the aggregate as its own timestamped table — this is exactly what F1's
`standings` is:

```
post_window(post_id, window_start, window_end,
            likes, distinct_actors, non_follower_likes,
            replies, reposts, rate_per_hour)
```

One row per post per hour. Each row is timestamped, so the temporal bound still
cuts it correctly and nothing needs rebuilding per anchor. 157k single-column
rows become ~10k rows with six meaningful columns.

Measured effect of compressing (dense per-post columns, small like fan-out):

| schema | Spearman | posts cells | likes cells |
|---|---:|---:|---:|
| raw event rows | 0.221 | 708 | 2,138 |
| dense summary columns | **0.355** | 1,298 | 1,270 |

**Capping fan-out is not a substitute.** Truncating the like list to 20, then 5
rows per post made it *worse* (0.190, 0.187) — that discards information without
freeing enough budget for the tables that matter. Compress, do not discard.

### Keep the raw table only if the target needs it

`PREDICT COUNT(likes.*) OVER (7 DAYS FOLLOWING)` requires `likes` to exist in
the schema. Keep it, but keep its fan-out into context small and put the signal
in summary columns.

---

## 2. Context composition is checked for you — but look anyway

`execute()` now emits `ContextCompositionWarning` when ≥50% of a cohort's
contexts hit the cell budget, or when one schema table holds ≥60% of all
context cells (cohorts above ~256 cells; virtual `task_*` rows excluded).
Those are the two failure shapes that previously burned hours silently.

For anything finer, `EXPLAIN CONTEXT` remains the tool:

```python
ex = engine.execute(ExecutionInput(query="EXPLAIN CONTEXT " + q,
                                   anchor_time=anchor, params={"ids": [one_id]}))
c = ex.context                      # a plain dict, NOT an object with .rows
print(c["total_cells"], c["contexts_hit_cell_budget"])
for t, v in c["tables"].items():
    print(f"  {t}: {v['rows']} rows, {v['cells']} cells")
```

Its cell counts now include FK-feature tokens (see §3a), so what EXPLAIN
prints is what the model gets.

**Rule of thumb:** if one table holds more than ~40% of cells, or an important
table has single-digit rows, fix the schema before tuning anything else.

### 3a. FK features (`LinkDef.feature_type`) work — and are now counted

Setting `feature_type` on a link emits the raw FK value as one extra token per
populated FK. It always worked at the model level; EXPLAIN CONTEXT simply did
not count those cells (a table whose rows carry nothing but FKs reported zero
cells), which made the knob look inert. Fixed: EXPLAIN now counts them.
Prefer real columns on edge tables over high-cardinality identifier text when
you want an edge to carry meaning.

### 3b. Pure-FK tables are invisible — and now loud

Serialization emits one token per feature cell, and a row's FK links ride on
the tokens it emits. A pure edge table — `follows(follower_id, followee_id)`
with no payload columns and no `feature_type` — therefore enters the context
at zero cell cost and **never reaches the model**: the connection each row
exists to represent is silently lost. This is why graph effects can come back
dead on an edge-table schema.

Two warnings now catch it: `InvisibleTableWarning` at `Engine` construction
(statically provable: no non-key columns, no feature-typed link) and a
`ContextCompositionWarning` at run time when a table contributed rows but
zero emitted cells. The fix, in order of preference:

1. **If the key itself carries signal** — a handle, a name, a category id —
   set `feature_type` on that link: `LinkDef("follows", "followee_id",
   "accounts", feature_type=ValueType.TEXT)`. The value is emitted as a
   feature token and the edge becomes visible.
2. Add a payload column the edge actually has (weight, kind, created_at).
3. Summarize the edges into a table with measured columns (§2).

## 3. Context policy

`ContextPolicy` defaults already match the reference evaluation
(`rt.eval_utils.build_evaluator`): `bfs_width=32`, `local_context_cells=256`,
`num_walks=10_000`, `walk_length=20`, `max_hops=2`.

```python
ContextPolicy(max_context_cells=8192, local_context_cells=256,
              bfs_width=32, num_walks=10_000, walk_length=20, seed=0)
```

### Traps

- **Do not set `local_context_cells = max_context_cells // 2`.** It is a natural
  looking thing to write and it is 16× the reference. Every cell spent on rows
  adjacent to the target is a cell the walk-ranked rows never get. Measured cost
  on one task: 0.676 vs 0.700 AUROC.
- **`max_hops` does not limit graph reach.** `ReferenceTraversal` builds its walk
  graph by *unbounded* BFS over the reachable component; the 10,000 walks of
  length 20 do the exploring. Raising `max_hops` is not the knob you want.
- **More cells rarely rescues a bad schema.** Measured: 2,048 → 8,192 → 16,384
  moved AUROC 0.648 → 0.710 → 0.712 and plateaued, with contexts truncating at
  every setting.
- **Dense many-to-many links add little and cost fan-out.** Dropping a
  `likes.actor_id -> accounts` link changed Spearman by +0.026.

---

## 4. Temporal correctness

**An undated row is admitted at every anchor.** From `TemporalBound.admits`:
*"A row with no timestamp is static and always admitted."*

That is right for genuine dimension tables (tags, categories) and a silent
future-leak for events. If an event source gives no timestamp — e.g.
`getRepostedBy` on Bluesky returns no per-repost time — **drop those rows**
rather than letting them in undated.

Checklist for any new time-series schema:

- every event table has `.time_column(...)` set
- rows with unparseable or missing timestamps are dropped, and the count is
  reported, not silently swallowed
- any precomputed/derived column is computed only from data at or before the
  row's own timestamp
- state clearly which relations are read "as of now" rather than as of the
  anchor (e.g. follow-graph edges), and which direction that biases

---

## 5. Queries

Verify a query parses before building an experiment around it:

```python
from relativedb.relql import parse
parse("PREDICT COUNT(engagements.* WHERE engagements.from_follower = FALSE) "
      "OVER (48 HOURS FOLLOWING) >= 1 FROM posts "
      "WHERE posts.post_id IN :ids RETURN PROBABILITY")
```

Confirmed working forms: `COUNT(t.* WHERE t.col = v)`, `COUNT_DISTINCT(t.col)`,
comparison targets (`>= k`), `RETURN PROBABILITY` / `RETURN EXPECTED VALUE`,
column targets (`PREDICT posts.topic ... WHERE posts.topic IS NULL`),
`EXPLAIN CONTEXT`, `ASSUMING`.

### Per-entity anchors are one execution each

`ExecutionInput` takes a single `anchor_time`. If each entity needs its own
anchor (e.g. "10 minutes after each post was created"), that is one `execute()`
per entity — budget accordingly. On a 41k-row like graph this was ~55s per call;
after trimming the pool to ~1,100 posts it dropped to ~4s.

Predictions are **batching-invariant**: scoring 123 entities in one call and one
at a time gives identical results (measured correlation 1.000).
`NormalizationMode.ZERO_SHOT` derives statistics per entity context by design, so
batching cannot change an existing prediction. Do not go looking for a batching
bug.

---

## 6. ASSUMING — semantics

### Who gets the assumed value (fixed 2026-07)

* **Entity-table assignment** — `ASSUMING c.plan = 'premium'` intervenes on
  the entity's **own row only**; sibling rows of the same table stay factual.
  This is what preserves the in-context scale the assumed value is normalized
  against: zero-shot normalization derives column stats inside each context,
  and if every row of the table were overwritten the column would become a
  constant, which normalizes to zero regardless of the constant — the value
  would be erased. (That was the historical bug: `ASSUMING x = TRUE` and
  `= FALSE` produced identical predictions.)
* **Other-table assignment** — `ASSUMING orders.status = 'shipped'` keeps the
  broad meaning: every context row of that table gets the value ("these
  orders are shipped").

### The engine now warns when an assumption cannot reach the model

`AssumptionNotAppliedWarning` fires when: the assigned table has no rows in
context; the entity's own row is missing from the context; or a numeric/
boolean assignment leaves the column **constant** in-context (zero-shot
normalization erases which constant it was — keep factual sibling rows, or
use reference normalization).

### Still run a positive control

Sweep `ASSUMING` over a column you already know moves the outcome before
trusting a flat curve on your variable of interest. Verified working: the
`text_length` sweep shows a monotonic dose-response (20 chars −14%, 300 chars
+2.6% on one sample).

## 7. Training

- **`fit_head` cannot do binary/regression.** It raises: *"frozen task-head
  fitting is limited to multiclass and multilabel-ranking adapters; scalar
  binary/regression tasks require full-backbone fine-tuning."* Use
  `Engine.finetune` for those — much slower.
- **`fit_head` / `finetune` need `params` bound** if the query references `:ids`,
  even though entities are named explicitly.
- **Small training sets memorise instantly.** 216 examples over a full checkpoint
  drove training loss to *exactly 0.0000* and dropped held-out AUROC 0.638 →
  0.570. Treat a loss of 0 as a red flag, not a success.
- Train each entity at its own cut-off by passing the diagonal:
  `labels={(entity_id, anchor): y}` — a pair not named is skipped rather than
  derived.
- To serve a fine-tuned checkpoint, pass its `model_uri` **and** its
  `normalization_mode` and `column_stats`:

```python
Engine(schema, wiring,
       model_config=ModelConfig(classification_model_uri=ckpt.model_uri,
                                normalization_mode=ckpt.normalization_mode),
       model_backend=RtNativeBackend(..., column_stats=ckpt.column_stats,
                                     normalization_mode=ckpt.normalization_mode))
```

---

## 8. Native library troubleshooting

Symptom:

```
RtNativeUnavailableError: found .../python/src/relativedb/librt_c.dylib but could
not bind the rt_c ABI: dlsym(...): symbol not found
```

The dylib bundled in `python/src/relativedb/` is stale relative to the Python
layer. Fixes, in order of preference:

1. Point at a fresh build without touching the tree:
   `RELATIVEDB_RT_LIB=$PWD/cpp/build/librt_c.dylib`
2. Re-sync the bundled copy from `cpp/build/`.

Check which candidates carry a symbol:

```bash
for f in $(find . -name "librt_c*.dylib" -not -path "*/.venv*"); do
  printf "%-55s " "$f"; nm -gU "$f" 2>/dev/null | grep -c rt_full_finetune_available
done
```

**Also:** installing the PyPI `relativedb` over an editable checkout replaces the
native library and causes this. In example `requirements.txt` files, prefer
`-e ../../python`.

---

## 9. Evaluating results honestly

RT-J's outputs are often compressed into a narrow band (measured: probabilities
spanning 0.469–0.508; regression predictions averaging 0.9 where truth averaged
17.4, including negative counts). Before concluding anything:

- print the **prediction spread** — a near-constant predictor can still post a
  respectable AUROC off tie-breaking noise
- check `contexts_hit_cell_budget` — every result above came from contexts that
  truncated
- compare against a **trivial baseline** computed on the same rows, not against
  chance. On engagement tasks "count what already happened" is very strong.
- for ablations, keep the query and labels identical and change only the
  database. If dropping a column forces the query to change, that is a different
  question, not an ablation — report it separately.
- small n hides everything: one task moved from 0.69 to 0.27 going from n=40 to
  n=150. Spearman's standard error at n=40 is roughly ±0.15.

---

## Quick reference

| symptom | first thing to check |
|---|---|
| model much worse than a trivial baseline | context composition — is one table eating the budget? |
| every context truncates | schema shape, not `max_context_cells` |
| predictions in a narrow band | prediction spread + context composition |
| `ASSUMING` has no effect | positive control on a known-good column; the value is currently inert |
| results change when batching | they should not — expect correlation 1.000 |
| suspiciously good result | n, and whether the baseline was computed on the same rows |
| native lib fails to bind | `RELATIVEDB_RT_LIB=$PWD/cpp/build/librt_c.dylib` |
| is the engine itself OK? | `evaluation/head_to_head.py` on `rel-f1/driver-top3` |
