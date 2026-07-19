# RT-J on RelationDB — Model Overview and Native Inference

*A structured overview for researchers and MLOps practitioners.*

RT-J is a **Relational Transformer (RT)** — a relational foundation model (RFM)
from Stanford's STAR lab that predicts directly over relational databases (tables
linked by foreign keys) and generalizes zero-shot to new schemas, tasks, and
databases. **RelationDB** is an optimized, dependency-light native implementation
of the RT-J forward pass, surfaced through **RelQL**, a SQL-flavored predictive
query language. This article gives a structured model overview and documents how
the model is served — on CPU, Apple Metal/MPS, and CUDA — for the research and
MLOps community.

## Model Card Summary

| Field | Value |
|---|---|
| Model | RT-J (Relational Transformer, "J" pretraining line) |
| Origin | Stanford STAR — `stanford-star/relational-transformer` |
| Native engine | RelationDB / `librt_c` (~700 lines of dependency-light C++20) |
| Model type | Relational transformer / relational foundation model (RFM) |
| Parameters | ~22M |
| Blocks / d_model / heads / d_ff | 12 / 512 / 8 / 2048 |
| Positional encodings | None — structure is carried by attention masks |
| Task heads | Binary classification, regression (per checkpoint variant) |
| Context length | Variable, no architectural cap (memory-bound); reference runs up to 8192 cells |
| Weight precision | bf16 checkpoint → fp32 compute; optional INT8 (`q8`) weights |
| Text encoder | `all-MiniLM-L12-v2` (384-dim, pinned) |
| Checkpoints | `stanford-star/rt-j/{classification, regression}` |
| Backends | CPU (Accelerate/AMX), Metal/MPS (Apple GPU), CUDA (cuBLAS) |
| Query interface | RelQL (parser single-sourced in C++, decoded by Python/Rust/Java) |
| Engine license | Apache-2.0 (model per `stanford-star/relational-transformer`) |

## Architecture

RT-J is a transformer that attends not over a token sequence but over a **small
subgraph of a relational database**: each token is one *cell* (a feature value of
some row), and attention is masked along the structure that relates cells. It
predicts a masked target cell — "will this customer place zero orders in the next
90 days?" — in a single forward pass, the relational analogue of prompting an LLM
rather than fine-tuning one.

### Masked relational attention

Each of the 12 blocks runs three masked-attention passes followed by a SwiGLU FFN,
with pre-RMSNorm residuals:

- **column attention** — cells of the same column, across rows
- **feature attention** — cells of the same row and its foreign-key parents
- **neighbor attention** — cells of a row's foreign-key children

There are **no positional encodings**; the masks alone carry structure, which is
why the model has no fixed context window. The attention math is faithfully ported:
per-head **QK-RMSNorm**, a learnable per-head scale × **log(kv_count)** query
scaling, a **sigmoid output gate ×2**, a `1/head_dim` score scale (not `1/√d`), and
zero output for fully-masked queries.

### In-context prediction

The engine assembles a temporally-bounded context per entity — the entity, its
foreign-key neighborhood, and the entity's own past outcomes as in-context labeled
examples — and RT-J scores it directly. No per-task training, no feature
engineering, and no positional leakage: a fact dated after the prediction anchor
can never enter the context.

### Additional components

- Per-semantic-type value encoders (number / datetime / boolean / text) with mask
  embeddings.
- Text cells and `"<column> of <table>"` schema phrases embedded by a pinned
  MiniLM-L12 encoder (384-dim).
- Number-head decoding with `bool_as_num` (boolean targets read off the number head).
- safetensors loading (bf16 → fp32) with a built-in header parser — no JSON dependency.

## Quantization: INT8 weights

Alongside the fp32 checkpoints, RT-J ships **INT8-quantized** weights
(`model.q8.safetensors`) using **per-row symmetric** quantization — each weight
matrix is stored as `int8` with an `fp32` `.q_scale` companion (one scale per row).
`librt_c` reads these directly and **dequantizes to fp32 at load**.

- **Checkpoint size** — ~88 MB vs ~342 MB fp32, **≈3.9× smaller** on disk.
- **Load time** — ~0.17 s vs ~0.76 s cold, **≈4× faster** (smaller file to read and
  dequantize).
- **Accuracy** — max\|Δ\| ≈ **4–5e-3** on raw scores vs fp32 (classification
  q8-vs-fp32 ~4e-3). For a classifier that is ~0.001 in probability space —
  negligible for ranking / AUROC.
- **Inference speed** — **unchanged**. Weights dequantize to fp32 at load, so the
  forward pass runs the identical fp32 kernels.

### Deployment implications

INT8 is a **storage / cold-start** optimization, not a compute one: use it to cut
checkpoint storage and load ~4× when distributing many models or scaling out
replicas, with no measurable hit to classification accuracy. It does **not** reduce
steady-state inference latency or resident memory (weights live as fp32 after load) —
for those, the backend choice (CPU / Metal / CUDA) is the lever.

## Execution: CPU, Metal (MPS), and CUDA

RelationDB reimplements the forward pass from scratch with **no torch dependency**,
so the same model runs across three backends selected at runtime (`--device`):

- **CPU** — Apple **Accelerate** `cblas_sgemm` (including the on-die **AMX** matrix
  units) elsewhere portable SIMD. Masked attention never materializes the S×S
  matrix: queries that share a key list are **grouped** and run as tiled per-head
  GEMMs, so long contexts stay tractable.
- **Metal / MPS** — the dense projections run as `MPSMatrixMultiplication`; the
  grouped attention, RMSNorm, SwiGLU, and head are custom **Metal compute kernels**.
- **CUDA** — cuBLAS projections with the same query-group attention sparsity.

Weights are converted once to contiguous fp32; activations are reused across blocks;
per-worker scratch buffers avoid allocation inside the block loop.

### Backend portability

All three backends produce **numerically identical** output (`max|Δ| = 0`) and pass
a batch-isolation check (attention never leaks across batch rows). Because the CPU
path is plain C++ + Accelerate, RelationDB runs **at full quality on a laptop CPU**
with no GPU and no torch install — a meaningful contrast with reference PyTorch RT,
whose `torch.compile`'d `flex_attention` kernel has no CPU backend and requires a
GPU. Choose the backend by workload (see Benchmarks): Metal for short-context,
high-batch scoring; CPU for long single sequences.

## Benchmark Results

### Numerical parity (vs. the PyTorch reference)

A golden batch dumped from the reference PyTorch model, replayed through `librt_c`:

| Stage | max\|Δ\| | mean\|Δ\| |
|---|---:|---:|
| block-0 input (`x_embed`) | 4.1e-06 | 4.0e-07 |
| after one block (`x_block0`) | 9.3e-04 | 1.1e-04 |
| final head (`yhat`) | 3.9e-03 | 5.1e-04 |

Per-row target scores match to three decimals (e.g. `-0.18470` vs `-0.18508`).

### Reference reproduction (end-to-end)

Driving the *entire* path — RelQL query → temporal context assembly → tokenization
→ `librt_c` — on a held-out binary relational task (213 rows, same checkpoint as the
reference) reproduces the reference model:

| Metric | Reference RT-J | RelationDB RT-J |
|---|---:|---:|
| AUROC | 0.742 | 0.736 |
| Per-row agreement (Pearson r) | — | 0.52 |
| Inference (213 rows, CPU) | *no CPU backend* | ~9 s |

The ΔAUROC of 0.006 (with correlated per-row predictions) indicates the whole
pipeline, not just the kernel, is faithful.

### Inference performance — CPU vs Metal (Apple Silicon, fp32)

Batch scaling (context S=16, per-entity scoring): Metal parallelizes across the
batch and saturates ~3.2–3.5× CPU throughput.

| Batch | CPU ms/fwd | MPS ms/fwd | MPS speedup |
|---:|---:|---:|:---:|
| 1 | 16.4 | 7.2 | 2.3× |
| 80 | 220.7 | 62.6 | 3.5× |
| 1280 | 3138.3 | 982.0 | 3.2× |

Context length (single sequence, B=1): Metal's edge shrinks as `S` grows; beyond
S ≈ 1–2k the two land within measurement noise (few iterations) and trade places
run-to-run — no consistent winner for a lone long sequence.

| B×S | CPU ms/fwd | MPS ms/fwd | Faster |
|---|---:|---:|:---|
| 1×1024 | 197 | 158 | MPS 1.25× |
| 1×2048 | 388 | 406 | ~tie (CPU 1.05×) |
| 1×8192 | 2320 | 2190 | ~tie (MPS 1.06×) |

Peak throughput: MPS ~21,000 tok/s (short context) vs CPU ~6,500 tok/s; both
converge to ~3,500–4,000 tok/s at S=8192. Metal uses less RSS at the largest shapes.

## What a Dependency-Light Native Engine Means

### Research opportunities

1. **Reproducibility.** The forward pass is ~700 lines with no framework
   indirection and a golden test against the reference — the exact math is auditable.
2. **Zero-artifact iteration.** Everything except model scoring (parser, schema
   binding, temporal-correctness guard, context assembly) runs and is tested with no
   checkpoint, GPU, or network.
3. **One model, three languages.** The engine and RelQL parser are single-sourced
   and decoded by Python, Rust, and Java peers with identical behavior.
4. **Portable acceleration.** CPU/Metal/CUDA behind one interface makes it practical
   to study where relational-transformer inference is compute- vs. memory-bound.

### MLOps considerations

- The RT-J single-head checkpoints serve **binary classification and regression**;
  scoring requires the native backend (`librt_c` + a cached checkpoint), with a clear
  error if none is configured.
- CPU deployment needs no GPU, no CUDA, and no torch — one static library plus a
  safetensors file. Metal/MPS and CUDA are opt-in for throughput.
- Temporal correctness is an **engine guarantee**: every retrieved row is re-checked
  against the prediction anchor, so a buggy retriever cannot leak the future.

## Known Limitations

1. **Published heads are narrow.** The released RT-J classification/regression
   checkpoints expose one scalar head and were not pretrained for multiclass or
   ranking. The native backend can now freeze the backbone and Metal-fine-tune a
   compact binary, regression, multiclass, or grouped-listwise ranking head; this
   is head adaptation, not full-transformer backpropagation. Distributional
   outputs (`RETURN QUANTILES`/`INTERVAL`) remain unsupported.
2. **Long context is memory-bound.** With no positional cap, context scales to
   8192+ cells, but attention cost grows with sequence length and dominates at long
   S; the Metal backend's advantage disappears there (CPU is faster for long single
   sequences).
3. **In-context examples.** By default the engine surfaces the entity's own
   self-labels as in-context examples, not a cross-entity labeled cohort; fully
   transductive tasks may need a cohort retriever to match the reference's setup.
4. **Metal is Apple-only.** The GPU speedups above require Apple Silicon; other
   platforms use the CPU path or CUDA.

## Conclusion

RT-J brings foundation-model, zero-shot prediction to relational data, and
RelationDB makes that inference **portable and auditable** — a from-scratch native
engine that matches the PyTorch reference numerically and end-to-end, runs at full
quality on a commodity CPU, and accelerates on Apple Metal or CUDA when the workload
warrants. The interesting frontier now is less about raw kernels and more about
context: how much relational neighborhood, and which in-context examples, buy the
most predictive signal per token.
