---
title: C++ inference engine
description: rt.cpp — native RT-J inference behind all three libraries.
---

# C++ inference engine (rt.cpp)

A dependency-light C++20 implementation of the RT-J forward pass — ~700 lines,
no torch, no Python at inference. It backs the `RtNativeBackend` in all three
libraries through one C ABI (`librt_c`).

## What it implements

- 12 blocks of column / feature / neighbor **masked attention** + SwiGLU FFN,
  pre-RMSNorm residuals; no positional encodings — structure is carried by the
  masks
- Faithful attention details: per-head QK-RMSNorm, log(kv-count) query
  scaling, sigmoid output gating, score scale `1/head_dim`
- Per-sem-type value encoders, number-head decoding, built-in safetensors
  loader (bf16 → fp32)

## Performance design

Idioms from llama.cpp / vLLM on Apple Accelerate: stacked-QKV GEMM panels over
the whole batch, grouped masked attention that never materializes S×S,
persistent thread pool, zero allocation inside the block loop.

## Build and verify

```bash
cd cpp
cmake -B build -S . && cmake --build build -j

./build/rt_test testdata <path>/classification/model.safetensors  # golden gate
./build/rt_bench <testdata> <model.safetensors>                   # batching + speed + memory
```

Targets: `rt` (static lib), **`librt_c`** (shared, the C ABI in `src/rt_c.h`),
`rt_test`, `rt_bench`.

The golden test replays a batch dumped from the PyTorch reference and matches
final scores to ~3–4 decimals — remaining differences are fp32 op-ordering
drift. The Java, Python, and Rust bindings each re-run this gate through
their own FFI layer.
