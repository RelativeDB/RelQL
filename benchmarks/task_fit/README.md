# RT-J task-fit evaluation

Measured, honest evaluation of **what the RT-J checkpoint can and cannot do** on
real relational data with real future-outcome ground truth. Companion to
[`../FINDINGS.md`](../FINDINGS.md).

> ⚠️ **The legacy tables below are IN-DISTRIBUTION results, not zero-shot.**
> All three evaluation
> datasets are in the RT-J pretraining mixture
> (`~/relational-transformer/scripts/recipe_rt_j.txt`): `join-online-retail`,
> `join-movielens` (+ `-1m`/`-imdb`/`-bird` variants), and `join-brightkite`.
> The model's weights saw this data — including rows in what we treat as the
> "future" window, since pretraining sees the whole DB. So every number below is
> **optimistic** (potential contamination). The new Digits experiment below is
> the clean exception: it is absent from the released 485-database recipe.

## Headline

On simple star-schema tabular tasks, RT-J produces genuine signal but:

- **underperforms a supervised GBDT (XGBoost) everywhere tested**;
- beats naive baselines only in *some* domains (retail yes, Brightkite no) — it
  depends on how autocorrelated the domain is;
- its **one genuine edge is the extreme cold-start regime**: with zero labels and
  zero features it is worth roughly the **first ~15–18 labeled examples** before
  XGBoost overtakes it;
- **ranking and multiclass fail** — the model was never trained for them.

The gap to XGBoost is a **supervision gap, not a feature gap** (adding the GBDT's
own features to RT-J barely helped). Hence the next step: **fine-tune.**

## Setup

- **Datasets:** Online Retail II (UCI, ~1M lines) and Brightkite check-ins (SNAP,
  4.7M) for churn/regression; MovieLens for ranking/multiclass.
- **Anchors:** retail 2011-06-01, Brightkite 2009-06-01; 30-day forward window,
  90-day active lookback; point-in-time correct (retrievers return only rows
  `≤ anchor`); ground truth = actual future rows.
- **Population:** 300 active customers/users (fixed seed).
- **XGBoost:** sklearn-parity `xgboost 3.3.0`, 11 hand-built RFM features, trained
  on 4–5 earlier monthly cohorts, tested on the same 300.

## Binary classification (churn) & regression

| domain | task | metric | RT-J | naive baseline | XGBoost (supervised) | beats baseline? |
|---|---|---|---|---|---|---|
| Retail | churn | AUC | 0.607 | 0.581 (recency) | **0.744** | ✅ |
| Retail | spend, 30d | Spearman | 0.291 | 0.201 (persistence) | **0.400** | ✅ |
| Retail | spend, 30d | MAE | £287 | £318 | **£213** | ✅ |
| Brightkite | churn | AUC | 0.772 | **0.827** (recency) | **0.858** | ❌ |
| Brightkite | activity count, 30d | Spearman | 0.609 | **0.680** (persistence) | **0.710** | ❌ |

**Domain-dependent.** On retail (bursty purchases) RT-J beats the trivial
baselines; on Brightkite (habitual check-ins — recency/persistence are
near-ceiling) it loses to them. XGBoost wins in every case.

## Data efficiency — the cold-start niche (retail churn)

XGBoost AUC as a function of #labeled examples, vs RT-J's fixed 0.607:

| # labels | 3 | 5 | 8 | 12 | **18** | 25 | 80 | 11,169 |
|---|---|---|---|---|---|---|---|---|
| XGBoost AUC | 0.500 | 0.500 | 0.506 | 0.573 | **0.643** | 0.657 | 0.664 | 0.745 |

**With 0 labels + 0 features, RT-J (0.607) beats a tuned XGBoost until XGBoost has
~15–18 labeled outcomes.** Below ~12 labels the GBDT is near-random. That is
RT-J's honest value here: *useful signal from label #0, before a GBDT can
function* — the "context-efficient" thesis, quantified. The window is **narrow**
(~18 labels) because churn is easy for RFM features; on a harder, hard-to-feature
task the cold-start window should be wider (untested).

## Adding features barely helps (supervision gap, not feature gap)

The same 11 RFM features, attached as **cells** on the `customers` table:

| | Retail churn AUC | Retail spend Spearman |
|---|---|---|
| RT-J (raw history) | 0.607 | 0.291 |
| RT-J + 11 RFM cells | 0.612 (+0.005) | 0.290 |
| XGBoost (supervised) | 0.744 | 0.400 |

Giving RT-J the exact features that power XGBoost moves AUC by +0.005 — nothing.
Zero-shot, it has the features in context but no learned weights connecting them
to the label. **The lever is supervision (fine-tuning), not feature
engineering.** (The RFM cells did cut spend MAE £287→£235 via better scale
calibration, but rank-correlation was unchanged.)

## Tasks the model was NOT trained for — they fail

| task | metric | RT-J | baseline |
|---|---|---|---|
| ranking (`LIST_DISTINCT` buy-it-again) | recall@20 | 0.05 | 0.20 (popularity) |
| multiclass (movie genre) | — | near-constant ("War" everywhere) | — |

Grounded in the training recipe (`~/relational-transformer/src/rt/tasks.py:30-34`):
`link_prediction` (recommendation) is **explicitly excluded** from the task
mixture; there is **no multiclass task type** (only binary `clf` + `reg`); text
columns are "not targets." Ranking/multiclass *execute correctly and are
cross-language consistent* (see `../xlang_fixture/`) — they are just low quality
because the signal was never trained in.

## Next step: fine-tuning

**Goal: fine-tune RT-J on task-specific labels and see if it can beat XGBoost.**

Rationale, straight from the evidence above:
- The gap to XGBoost is a **supervision gap** — RT-J has the relational context and
  even the hand features, but no learned task mapping; adding features zero-shot
  didn't help (+0.005), so more supervision is the only remaining lever.
- The model is **tiny (22M params)** — fine-tuning is cheap (a short run on one
  GPU, not the 32×A100 pretraining).
- The pretrained backbone should give a **better starting point than XGBoost in
  the low-data regime**, so the interesting question is the *data-efficiency
  curve of fine-tuned RT-J vs XGBoost*: does fine-tuned RT-J beat XGBoost at
  every label count, or only close the gap at large N?

**Targets to beat:** retail churn AUC **0.744**, retail spend Spearman **0.400**;
Brightkite churn AUC **0.858**, count Spearman **0.710**. And critically, run it
on a **held-out** dataset (outside `recipe_rt_j.txt`) so the result is a clean,
uncontaminated comparison — the thing this in-distribution round could not
provide.

## Metal fine-tuning on a held-out dataset

Implemented in the dependency-light C++ backend as frozen-backbone head
fine-tuning. `sklearn.datasets.load_digits` has 1,797 handwritten 8×8 images
and 10 classes, and `digits` does not occur in RT-J's released 485-database
recipe. The benchmark uses a fixed stratified 80/20 split (seed 1729), computes
normalization statistics on the 1,437-example train split only, and evaluates
once on 360 held-out examples.

| held-out metric | released head (before) | Metal fine-tuned head (after) |
|---|---:|---:|
| accuracy | 0.100 | **0.703** |
| macro-F1 | 0.018 | **0.700** |
| cross-entropy | 2.303 | **0.987** |
| class-ranking MRR | 0.289 | **0.813** |
| class-ranking recall@3 | 0.267 | **0.914** |

The multiclass head has 5,130 trainable parameters (`10×512 + 10`). Its 2,000
full-batch AdamW steps ran on Metal in **1.29 s**, reducing train loss from
2.303 to 0.928; the saved safetensors adapter is 20 KB. Ranking is also a
native training objective: `rt_train_test` verifies grouped listwise softmax
on variable candidate groups. The class-ranking figures above rank all ten
digit labels by the multiclass logits.

Reproduce and overwrite the committed result/adapter:

```bash
python/.venv/bin/python benchmarks/task_fit/digits_metal_finetune.py
```

Exact machine-readable output: `digits_results.json`; adapter:
`digits_head.safetensors`.

## Olist versus XGBoost

Olist is a useful relational stress test, but **not a clean held-out dataset**:
the released RT-J recipe contains `join-spider2-brazilian-e-commerce`. Results
must therefore be read as a supervised adaptation comparison, not evidence of
zero-shot generalization.

Task: at order delivery, predict the later 1–5 star review. Orders whose review
predates delivery are excluded. Customer, order, item/product/seller, payment,
and actual-versus-estimated delivery features are all observable at the anchor.
The chronological split is 2018-05-01: 61,795 train and 25,748 test orders;
normalization and categorical encoders are fitted on train only. RT-J and
XGBoost receive the same source feature columns.

| held-out metric | train prior | released RT-J | Metal fine-tuned RT-J | XGBoost |
|---|---:|---:|---:|---:|
| accuracy | **0.647** | 0.642 | **0.647** | 0.646 |
| macro-F1 | 0.157 | 0.159 | 0.157 | **0.182** |
| cross-entropy | 1.055 | 1.601 | 1.043 | **1.033** |
| bad-review (1–2 star) AUC | 0.500 | 0.561 | 0.607 | **0.660** |
| class-ranking MRR | 0.789 | 0.764 | 0.789 | **0.790** |
| class-ranking recall@3 | 0.914 | 0.914 | 0.919 | **0.923** |

The test set is 64.7% five-star, so accuracy alone is misleading: the
fine-tuned RT-J head and train-prior baseline both choose five stars almost
everywhere. Fine-tuning still materially improves RT-J cross-entropy
(1.601→1.043), bad-review AUC (0.561→0.607), and recall@3 (0.914→0.919), but
XGBoost wins the discriminative metrics.

Cost on the same machine: XGBoost preprocessing plus 500-tree training took
5.58 s. RT-J frozen feature extraction took 154.24 s and its 2,565-parameter,
800-step Metal head optimization took another 22.82 s. RT-J features can be
cached for later heads, but end-to-end it is substantially slower here.

```bash
python/.venv/bin/python benchmarks/task_fit/olist_metal_vs_xgboost.py
```

The script downloads the public Olist archive when needed. Exact output:
`olist_results.json`; adapter: `olist_review_head.safetensors`.

## Reproduce

Run inside `python/.venv` with the native lib built and the checkpoint cached:

```bash
export SSL_CERT_FILE=/opt/homebrew/etc/ca-certificates/cert.pem   # py3.14 CA fix
export RELATIVEDB_RT_LIB=$PWD/cpp/build/librt_c.dylib
V=python/.venv/bin/python
$V benchmarks/task_fit/churn_spend_rtj.py           # RT-J churn AUC + spend (retail)
$V benchmarks/task_fit/churn_spend_xgboost.py       # supervised XGBoost baseline (retail)
$V benchmarks/task_fit/brightkite_clf_reg.py        # churn + count, 2nd domain, RT-J vs XGBoost
$V benchmarks/task_fit/ranking_buy_it_again.py      # ranking vs popularity (fails)
$V benchmarks/task_fit/churn_rtj_with_rfm_cells.py  # RT-J + RFM feature cells (barely helps)
$V benchmarks/task_fit/data_efficiency.py           # XGBoost AUC vs #labels — the cold-start crossover
$V benchmarks/task_fit/digits_metal_finetune.py     # held-out Digits, native Metal fine-tuning
$V benchmarks/task_fit/olist_metal_vs_xgboost.py    # Olist review score vs XGBoost (contaminated)
```

> **xgboost note:** `pip` is broken in the py3.14 venv (its vendored SSL bundle
> trips the stricter PEM parser). `xgboost 3.3.0` was installed by extracting the
> wheel directly into site-packages; `numpy`/`scipy` come via `scikit-learn`.
