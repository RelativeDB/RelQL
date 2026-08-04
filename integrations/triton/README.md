# Pure Triton RT-J proof of concept

This directory retains the initial relational-attention experiment. The
end-to-end implementation has graduated into `relativedb-engine`. All model
computation is performed by Triton kernels. PyTorch is used only as Triton's
CUDA tensor allocator, for checkpoint/host/device transfers, and for
CUDA-event timing; it does not perform inference math.

The benchmark models an 8,192-cell column-attention pass with 16 relational
groups, 512 keys per group, 16-query work tiles, eight heads, and a 64-wide
head. It validates one work tile against a NumPy CPU implementation, then
measures a complete attention pass. RT-J executes three relational attention
types in each of twelve blocks, so the benchmark also reports the naive
36-pass projection. That projection is deliberately conservative: the real
feature and neighbor groups are usually much smaller than the capped
512-member column groups.

Run on an NVIDIA GPU:

```bash
python benchmark_attention.py
```

Run the complete 12-block model on an exported F1 corpus:

```bash
python -m relativedb_engine.triton_model \
  /checkpoints/model.f16.safetensors /corpus/f1-8192
```

`relativedb_engine.triton_model` includes exact stable input sorting, the column/feature/neighbor
work-list semantics, semantic encoders and masking, all attention and FFN
blocks, and the number decoder. Variable key lists are bucketed at
32/64/128/256/512 cells so small feature groups do not execute a masked
512-key attention loop. The optimized path additionally:

- pre-normalizes and packs reused column K/V rows;
- uses merged QKVG and W1/W3 weights without retaining duplicate source
  tensors;
- fuses output/down projections with their FP32 residual updates;
- uses Blackwell-selected `64x128x64` GEMM tiles;
- uses 64-key attention tiles for long groups and 32-key tiles for fragmented
  groups; and
- leaves 384-wide text/column embeddings in source order, gathering them on
  the GPU instead of copying roughly 100 MB during CPU sorting.
