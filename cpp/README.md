# rt.cpp — shared native layer (inference + parser + CSC)

A dependency-light C++20 implementation of **RT-J** (the Stanford Relational
Transformer successor, `stanford-star/rt-j`), verified bit-for-practical
against the PyTorch reference. ~700 lines, no torch, no Python at inference.

`librt_c` is the single shared backend for every language binding. Beyond
inference it now also hosts two components that were previously reimplemented
per language, so the bindings can delegate instead of diverging:

- **PQL parser** (`src/pql.{hpp,cpp}`, C ABI `pql_parse` in `src/pql_c.h`) —
  hand-written lexer + recursive-descent parser producing a JSON AST. Test:
  `./build/pql_test`. Python binding: `relativedb.pql.native`; cross-language
  equivalence: `python/tests/test_native_parser.py`.
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

## Optimization design (idioms from llama.cpp / vllm)

| Concern | Approach |
|---|---|
| Dense projections | one Accelerate `cblas_sgemm` per projection over the whole (B·S, d) panel; `wq/wk/wv/wg` are stacked into a single `[4d, d]` weight so QKV+gate is one GEMM |
| Masked attention | never materializes S×S: queries sharing a key list are **grouped** (column groups, (node, FK-set) groups, reverse-FK lists) in O(S) per batch row, and each group runs as per-head GEMMs over ≤64-query tiles — `scores = Q_g K_gᵀ`, max-subtracted softmax (`vvexpf`), `out = P V_g` — on the AMX units |
| Memory | weights converted once to contiguous fp32; activations reused across blocks; per-worker scratch buffers, no allocation inside the block loop |
| Parallelism | attention/FFN elementwise work parallelized across (batch × group × query-tile) work items on a persistent thread pool (workers park between jobs); GEMMs use Accelerate's internal threading |

The three mask types come from the same structures the samplers produce:
`col` = same (column, table); `feat` = own row ∪ FK-parent rows (deduped);
`nbr` = reverse-FK children. See `kb/architecture.md` in the rt knowledge base.

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
cmake -B build -S . && cmake --build build -j

# regenerate golden data (needs the rt repo's venv + HF cache):
/Users/henneberger/rt/.venv/bin/python tools/dump_golden.py

./build/rt_test testdata <path-to>/classification/model.safetensors [--bench 20]
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

## Scope / next steps

- Classification head only (`dec_dict.number`, matching `bool_as_num=True`
  releases); the regression checkpoint loads with the same code.
- fp32 only. Obvious wins, in llama.cpp order of value: keep weights bf16 and
  convert in-register (halves bandwidth → ~2× at these shapes), Metal GEMMs,
  int8 quantization of the big projections.
- Feed from the relativedb engines: the `Batch` struct is exactly the token
  batch the Java/Python/Rust samplers assemble, so this library is a natural
  native `ModelBackend` behind `relativedb-ffi`.
