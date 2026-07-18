# rt.cpp — shared native layer (inference + parser + CSC)

A dependency-light C++20 implementation of **RT-J** (the Stanford Relational
Transformer successor, `stanford-star/rt-j`), verified bit-for-practical
against the PyTorch reference. ~700 lines, no torch, no Python at inference.

`librt_c` is the single shared backend for every language binding. Beyond
inference it now also hosts two components that were previously reimplemented
per language, so the bindings can delegate instead of diverging:

- **RelQL parser** (`src/pql.{hpp,cpp}`, C ABI `pql_parse` in `src/pql_c.h`) —
  hand-written lexer + recursive-descent parser producing a JSON AST. Implements
  the v2 grammar (`OVER (...)`/`WINDOW` frames, `HORIZONS`, `AS OF`, `RETURN`,
  `EXPLAIN`, `EXISTS`; see `RelQL_EVOLUTION.md`). Test: `./build/pql_test`. Python
  binding: `relativedb.pql.native`; cross-language equivalence:
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

## Optimization design (idioms from llama.cpp / vllm)

| Concern | Approach |
|---|---|
| Dense projections | one GEMM per projection over the whole (B·S, d) panel (Accelerate / MPS / cuBLAS); `wq/wk/wv/wg` are stacked into a single `[4d, d]` weight so QKV+gate is one GEMM |
| Masked attention | never materializes S×S: queries sharing a key list are **grouped** (column groups, (node, FK-set) groups, reverse-FK lists) in O(S) per batch row, and each group runs as per-head GEMMs over ≤64-query tiles — `scores = Q_g K_gᵀ`, max-subtracted softmax (`vvexpf`), `out = P V_g` — on the AMX units (CPU) or as online-softmax streaming kernels (GPU) |
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
entirely and run a custom `qgemm`: one threadgroup per 32×32 output tile,
K-chunks staged in threadgroup memory with **in-register dequant on the DRAM
load**, accumulated via `simdgroup_float8x8` MMA (K-chunks of 32 align with
Q4's group size, so each staged row-chunk touches one scale pair). fp32
checkpoints take the exact same code paths as before (Accelerate / MPS).
CPU and MPS produce identical drift per format; CUDA is fp32-only for now.

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
    [--bench 20] [--device cpu|mps|cuda]
./build/rt_bench testdata <path-to>/classification/model.safetensors \
    [--device cpu|mps|cuda]
```

## Benchmarks (`rt_bench`, M-series CPU, fp32)

**Batching correctness** — batched vs single-row, batch-order-permuted, and
duplicated-row runs are all **bit-identical** (`max|Δ| = 0.0`): attention
provably never leaks across batch rows.

**Memory** — RSS 336 MB after load (≈ the 342 MB fp32 weight conversion of the
85.6M-param bf16 checkpoint); peaks at ~1.0 GB across B=8 × S=1024 /
B=1 × S=2048 forwards (qkvg/FFN activation panels + per-worker attention
scratch, freed per call). Warm checkpoint load: ~85 ms (page-cached; 0.7 s cold).

**Batch-size sweep** (S=16) — weights are streamed once per forward, so
batching amortizes the memory-bound floor ~5×:

| B | ms/fwd | ms/entity |
|---|---|---|
| 1 | 15 | 14.7 |
| 5 | 25 | 4.9 |
| 20 | 63 | 3.2 |
| 80 | 221 | **2.8** |

**Context-length sweep** (synthetic relational batches — entity/facts/items/
label-history shape):

| B | S | ms/fwd | tok/s |
|---|---|---|---|
| 1 | 256 | 55 | 4.7k |
| 1 | 1024 | 193 | 5.3k |
| 1 | 2048 | 405 | 5.1k |
| 8 | 1024 | 1323 | 6.2k |

(Interleaved A/B against the pre-GEMM-attention baseline: 1.2–1.9× depending
on shape — largest at S=2048, where group-batched attention replaces per-query
key streaming, and at B=1/S=16, where the persistent pool removes per-pass
thread spawns.) Growth beyond linear at large S comes from column-group
attention (fact columns form large groups → O(S·group) score work), the same
asymptotic FlexAttention pays upstream — it just runs as AMX GEMMs now.

**MPS vs CPU** (same M3 Pro, `--device mps`): the GPU wins wherever there is
batch or width to fill it — B=80/S=16: 63 ms vs 228 ms (**3.6×**),
B=8/S=1024: 628 ms vs 1342 ms (**2.1×**), B=20/S=16: 20 ms vs 71 ms — and is
roughly at parity on single-row long-context shapes (B=1/S=2048: 416 vs
388 ms), where a handful of huge-key-list groups leave the grouped-attention
kernel latency-bound rather than throughput-bound. Single-row S=16 inference
costs ~7 ms of fixed command-buffer overhead vs 15 ms on CPU.

## Scope / next steps

- Classification head only (`dec_dict.number`, matching `bool_as_num=True`
  releases); the regression checkpoint loads with the same code.
- Activations are fp32; weights are fp32/f16/q8/q4 with in-kernel dequant
  (see Quantization). Remaining wins: int8 activations on CPU (NEON sdot)
  for true int8 throughput, key-tiled (flash-style) GPU attention for the
  single-row long-context case, quantized formats on the CUDA backend.
- The CUDA backend mirrors the golden-verified Metal design but has not been
  compiled or run yet (no CUDA toolchain on the dev machine) — build with
  `-DRT_CUDA=ON` and run `rt_test --device cuda` on the first CUDA box to
  confirm golden parity.
- Feed from the relativedb engines: the `Batch` struct is exactly the token
  batch the Java/Python/Rust samplers assemble, so this library is a natural
  native `ModelBackend` behind `relativedb-ffi`.
