# rt.cpp — shared native layer (inference + parser + CSC)

A dependency-light C++20 implementation of **RT-J** (the Stanford Relational
Transformer successor, `stanford-star/rt-j`), verified bit-for-practical
against the PyTorch reference. ~700 lines, no torch, no Python at inference.

`librt_c` is the single shared backend for every language binding. Beyond
inference it now also hosts two components that were previously reimplemented
per language, so the bindings can delegate instead of diverging:

- **RelQL parser** (`src/relql.{hpp,cpp}`, C ABI `relql_parse` in `src/relql_c.h`) —
  hand-written lexer + recursive-descent parser producing a JSON AST. Implements
  the v2 grammar (`OVER (...)`/`WINDOW` frames, `HORIZONS`, `AS OF`, `RETURN`,
  `EXPLAIN`, `EXISTS`; see `RelQL_EVOLUTION.md`). Test: `./build/relql_test`. Python
  binding: `relativedb.relql.native`; cross-language equivalence:
  `python/tests/test_native_parser.py`.
- **CSC index** (`src/csc.{hpp,cpp}`, C ABI `csc_build`/`csc_children`/`csc_free`
  in `src/csc_c.h`) — lex-sorted adjacency + binary-searched "latest ≤ anchor"
  children. Test: `./build/csc_test`. Python binding: `relativedb.csc_native`;
  equivalence: `python/tests/test_native_csc.py`.

Build all: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4`.

## What it implements

The exact `rt/model.py` (main branch) forward pass:

- 12 blocks × [**col | feat | nbr** masked attention → SwiGLU FFN], pre-RMSNorm
  residuals; no positional encodings — structure is carried by the masks alone
- attention extras faithfully ported: per-head **QK-RMSNorm**, learnable
  per-head scale × **log(kv_count)** query scaling (including the upstream
  `.bfloat16()` rounding of counts), **sigmoid output gate ×2**, score scale
  `1/head_dim` (not `1/√d`), zero output for fully-masked queries
- per-sem-type value encoders + mask embeddings, `"col of table"` text
  embeddings, stable in-forward sort by column id, number-head decoding
  (`bool_as_num`)
- safetensors loading (bf16 → fp32) with a built-in header parser — no JSON dep

## Full-checkpoint MPS fine-tuning

`rt_full_train_metal.mm` implements scalar-task end-to-end fine-tuning without
Torch. Forward, activation-gradient, and weight-gradient GEMMs use
`MPSMatrixMultiplication`; custom MSL kernels implement the exact grouped
column/feature/neighbor attention and backward pass, RMSNorm/QK-RMSNorm,
gating, SwiGLU, Huber loss, global gradient clipping, and AdamW. The forward
tape checkpoints block boundaries and recomputes one block at a time during
backward, keeping 8,192-cell memory bounded on Apple Silicon.

`fit_model_metal_step` updates encoders, mask embeddings, every transformer
block, learned scales/norms, and the numeric decoder. Optimizer moments persist
on `Model`; `Model::save` exports the complete FP32 safetensors checkpoint.
The C ABI exposes `rt_model_finetune_step_metal`, `rt_model_save`, and optimizer
reset. The Python orchestration is `Engine.finetune()` for binary/regression.
Quantized inference checkpoints are rejected because their training precision
has already been discarded.

The committed native check verifies inference/training loss agreement,
representative early/late parameter updates, and full checkpoint round-trip.
An 8,192-cell batch-one step has also completed on M3 MPS without fallback.

## Metal task-head fitting (`rt_train`)

The native backend can now adapt RT-J without PyTorch. It freezes the
golden-verified transformer, extracts the final normalized 512-dimensional
target-cell state on CPU or Metal, and trains a compact task head on Metal.
This is deliberately **head fitting**, not a full 86M-parameter backward
pass: only `512*C + C` parameters change, so an adapter is cheap to train,
audit, save, and deploy.

| task | head / loss |
|---|---|
| binary classification | scalar sigmoid cross-entropy |
| regression | scalar squared error |
| multiclass | `C`-way softmax cross-entropy |
| ranking | scalar score + grouped listwise softmax cross-entropy |

`rt_train.hpp` exposes `FineTuneHead`, `fit_head_metal`, adapter safetensors
save/load, and portable CPU prediction. The C ABI mirrors it with
`rt_encode_targets_device` and `rt_finetune_head_*`. Scalar heads initialize
from the released number decoder. A multiclass head can initialize from the
released text decoder plus class-label MiniLM embeddings, preserving its
zero-shot class ordering before training. Feature standardization
reparameterizes every initialized head as `w' = w*σ`, `b' = b+w·μ`, so the
epoch-zero logits are exactly preserved. The public Python `Engine.fit_head`
surface is restricted to multiclass/ranking; scalar heads remain a lower-level
diagnostic and are not presented as task fine-tuning.

```bash
./build/rt_train_test
```

The native test covers multiclass and variable-group ranking losses.

## Backends (CPU / MPS / CUDA)

`rt::forward(model, batch, ForwardOpts{.device = rt::Device::CPU|MPS|CUDA})`
selects the compute device; the C ABI mirrors it as
`rt_forward_device(..., RT_DEVICE_*)` with `rt_device_available()` for probing.
Batch preparation (stable sort, query-group construction, work tiling, value
embeddings) is identical for all devices and always runs on the CPU; the 12
transformer blocks + head run on the selected backend. All backends share the
same query-group sparsity — no backend ever materializes an S×S mask — and all
pass the same golden-parity and batching-invariance tests.

- **CPU** (always built): Accelerate GEMMs/vDSP on Apple; elsewhere (or with
  `-DRT_PORTABLE`) a register-blocked 4×8 portable GEMM parallelized over row
  chunks on the same persistent thread pool.
- **MPS** (`-DRT_METAL=ON`, default on Apple): `MPSMatrixMultiplication` for the
  dense projections — `wo`/`w2` accumulate into the residual stream with β=1 —
  plus custom Metal kernels: simdgroup-per-row RMSNorm, in-place QK-RMSNorm,
  a grouped-attention kernel (one threadgroup per (group, query-tile) work
  item, one simdgroup per (query, head) pair streaming the shared key list
  with a single-pass online softmax), sigmoid gating, SwiGLU, and a fused
  output-norm+head. Weights upload once per model (fp32, unified memory);
  activation buffers grow on demand and are reused; one command buffer per
  forward. Forwards on one model serialize on the GPU; CPU forwards stay
  reentrant.
- **CUDA** (`-DRT_CUDA=ON`, needs the CUDA toolkit): the same design with
  cuBLAS SGEMMs (β=1 residual accumulation) and warp-level mirrors of the
  Metal kernels (warp per (query, head) pair, `__shfl_xor_sync` reductions).
  f16/q8/q4 projections run a CUDA port of the Metal `qgemm` (32×32
  shared-memory tiles, in-register dequant on the DRAM load, half2 loads for
  f16); weights stay quantized-resident.

## Optimization design (idioms from llama.cpp / vllm)

| Concern | Approach |
|---|---|
| Dense projections | one GEMM per projection over the whole (B·S, d) panel (Accelerate / MPS / cuBLAS); `wq/wk/wv/wg` are stacked into a single `[4d, d]` weight so QKV+gate is one GEMM |
| Masked attention | never materializes S×S: queries sharing a key list are **grouped** (column groups, (node, FK-set) groups, reverse-FK lists) in O(S) per batch row, and each group runs as per-head GEMMs over ≤64-query tiles — `scores = Q_g K_gᵀ`, max-subtracted softmax (`vvexpf`), `out = P V_g` — on the AMX units (CPU) or as online-softmax streaming kernels (GPU); GPU key lists >512 are flash split-K'd into 256-key chunks reduced with the online-softmax identity |
| Memory | weights converted once to contiguous fp32; activations reused across blocks; per-worker scratch buffers, no allocation inside the block loop; GPU weight upload once per model, activation buffers reused across forwards |
| Parallelism | attention/FFN elementwise work parallelized across (batch × group × query-tile) work items on a persistent thread pool (workers park between jobs); GEMMs use the BLAS library's internal threading; on GPU each work item is a threadgroup/block |

The three mask types come from the same structures the samplers produce:
`col` = same (column, table); `feat` = own row ∪ FK-parent rows (deduped);
`nbr` = reverse-FK children. See `kb/architecture.md` in the rt knowledge base.

## Quantization (`rt_quantize`)

```bash
./build/rt_quantize <in>/model.safetensors <out>/model.q8.safetensors --type q8|q4|f16
./build/rt_test testdata <out>/model.q8.safetensors --quantized [--device mps]
./build/rt_test testdata <out>/model.q4.safetensors --tol 100  [--device mps]
```

Three formats, applied to every transformer-block projection (qkvg / wo /
ffn — ~99% of the parameters); the value/col-name encoders, decoder head,
norms and biases stay fp32, like llama.cpp's embedding/output layers —
input-side error would otherwise propagate through all 12 blocks:

| Format | Layout | File | Golden drift (`yhat max\|Δ\|`) |
|---|---|---|---|
| fp32 | — | 342 MB | 3.9e-3 |
| **f16** | IEEE half payload | 172 MB | 3.9e-3 (≡ fp32) |
| **q8** | int8, per-output-row symmetric f32 scale (`I8` + `<name>.q_scale`) | 88 MB | 1.1e-2 |
| **q4** | uint4, groups of 32, f16 (scale, min)/group with min-MSE clip search (`U8` + `<name>.q4_scale`); `wo`/`ffn.w2` stay q8 (Q4_K_M-style — their error lands on the residual stream) | 64 MB | 1.5e-1 (scores keep sign + ranking) |

**Weights stay quantized-resident** — this is not load-time dequantization.
On CPU, `matmul_w` dequantizes 64-row weight tiles into per-thread scratch
(hot in cache) right before the Accelerate/portable GEMM, so DRAM weight
traffic is the quantized payload. On Metal, quantized projections skip MPS
entirely and run a custom `qgemm`: 32-row tiles below 128 tokens favor
latency/tails, while 64-row tiles reuse each staged weight panel across twice
as many rows. K-chunks are staged in threadgroup memory with **in-register
dequant on the DRAM load**, accumulated via `simdgroup_float8x8` MMA
(K-chunks of 32 align with
Q4's group size, so each staged row-chunk touches one scale pair). CUDA runs
the same qgemm design as a `k_qgemm` kernel (verified against the reference
dequant numerically; pending a run on CUDA hardware). fp32 checkpoints take
the exact same code paths as before (Accelerate / MPS / cuBLAS). On CPU, f16
micro-batches (≤4 rows) skip the dequant pass and stream half weights
through a NEON widening kernel (2–3x at M=1; above that the AMX GEMM wins
and tile-dequant is used — `RT_NO_F16_NEON=1` forces tile-dequant). CPU and
MPS produce identical drift per format.

Both RT-J checkpoints have all three variants next to their HF-cache
originals (`.../rt-j/snapshots/<hash>/{classification,regression}/
model.{q8,q4,f16}.safetensors`). The Java/Python/Rust resolvers select a
variant via env `RELATIVEDB_RT_QUANTIZED` (or system property
`relativedb.rt.quantized` in Java): `1`/`true`/`q8` → q8, `q4` → q4,
`f16` → f16 — off by default so fp32 golden parity stays untouched; explicit
file paths are always used as given.

## Verification

`rt_test` replays a golden batch dumped from the working PyTorch demo
(`rt/demo/run_rt.py` — 5 customers × 16 tokens of the churn example) through
three checkpoints:

```
sort mismatches : 0
x_embed    max|Δ| 4.05e-06     (block-0 input)
x_block0   max|Δ| 9.31e-04     (after one full block)
yhat       max|Δ| 3.91e-03     (final head outputs)
target scores (cpp vs torch):  -0.18470/-0.18508 · -0.33108/-0.33127
  +0.43363/+0.43323 · -0.14448/-0.14469 · +0.46848/+0.46808
GOLDEN TEST PASS
```

Differences are fp32 op-ordering drift (Accelerate vs torch GEMM reduction
order) accumulating over 12 layers; ranking and scores agree to ~3–4 decimals.

## Build & run

```bash
cmake -B build -S . && cmake --build build -j        # add -DRT_CUDA=ON for CUDA

# regenerate golden data (needs the rt repo's venv + HF cache):
/Users/henneberger/rt/.venv/bin/python tools/dump_golden.py

./build/rt_test testdata <path-to>/classification/model.safetensors \
    [--device cpu|mps|cuda]
```


## Scope / next steps

- Classification head only (`dec_dict.number`, matching `bool_as_num=True`
  releases); the regression checkpoint loads with the same code.
- Weights are fp32/f16/q8/q4 with in-kernel dequant (see Quantization).
  Activations are fp32 except the CPU q8 path: it quantizes them to int8 and
  runs SMMLA (i8mm) or SDOT (int8×int8) — a quarter of the weight bandwidth.
  Metal attention uses flash split-K for long key lists. Remaining wins:
  int8/fp16 tensor-core compute on the GPU, flash split-K attention on CUDA,
  and fusing the CPU activation-quantization into the preceding RMSNorm.
- The CUDA backend mirrors the golden-verified Metal design (all four weight
  formats) but has not been compiled or run yet (no CUDA toolchain on the
  dev machine) — build with `-DRT_CUDA=ON` and run `rt_test --device cuda`
  (fp32/f16/q8 `--quantized`, q4 `--tol 100`) on the first CUDA box to
  confirm golden parity. The `k_qgemm` dequant/tiling logic is already
  verified numerically against the CPU reference via a line-by-line CPU
  emulation over the real quantized checkpoints.
- Feed from the relativedb engines: the `Batch` struct is exactly the token
  batch the Java/Python/Rust samplers assemble, so this library is a natural
  native `ModelBackend` behind `relativedb-ffi`.
