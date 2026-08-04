"""Pure-Triton proof of concept for RT-J relational attention."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
import triton
import triton.language as tl


HEADS = 8
HEAD_DIM = 64
QUERY_TILE = 16
KEY_TILE = 64
KEYS_PER_GROUP = 512
GROUPS = 16
QUERY_TILES_PER_GROUP = 32
WORK_ITEMS = GROUPS * QUERY_TILES_PER_GROUP
N_CELLS = GROUPS * KEYS_PER_GROUP
NORM_EPS = 1e-5
LOG2E = math.log2(math.e)


@triton.jit
def relational_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    gate_ptr,
    q_norm_ptr,
    k_norm_ptr,
    head_scale_ptr,
    out_ptr,
    n_keys: tl.constexpr,
    chunks_per_group: tl.constexpr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    work = tl.program_id(0)
    head = tl.program_id(1)
    group = work // chunks_per_group

    m = tl.arange(0, block_m)
    d = tl.arange(0, head_dim)
    q_offsets = (
        ((work * block_m + m[:, None]) * n_heads + head) * head_dim
        + d[None, :]
    )
    q = tl.load(q_ptr + q_offsets).to(tl.float32)
    q_scale = tl.load(q_norm_ptr + d)[None, :]
    q_ss = tl.sum(q * q, axis=1)
    q = q * tl.rsqrt(q_ss[:, None] / head_dim + 1e-5) * q_scale
    log_keys = tl.log(tl.full((), n_keys, tl.float32))
    scale = tl.load(head_scale_ptr + head) * log_keys / head_dim * 1.4426950408889634
    q = (q * scale).to(tl.float16)

    running_max = tl.full((block_m,), -float("inf"), tl.float32)
    running_sum = tl.zeros((block_m,), tl.float32)
    accumulator = tl.zeros((block_m, head_dim), tl.float32)

    for key_start in range(0, n_keys, block_n):
        n = key_start + tl.arange(0, block_n)
        k_offsets = (
            ((group * n_keys + n[:, None]) * n_heads + head) * head_dim
            + d[None, :]
        )
        k = tl.load(k_ptr + k_offsets).to(tl.float32)
        k_scale = tl.load(k_norm_ptr + d)[None, :]
        k_ss = tl.sum(k * k, axis=1)
        k = k * tl.rsqrt(k_ss[:, None] / head_dim + 1e-5) * k_scale
        scores = tl.dot(q, tl.trans(k.to(tl.float16)))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        correction = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        running_sum = (
            running_sum * correction + tl.sum(probabilities, axis=1)
        )

        v = tl.load(v_ptr + k_offsets)
        accumulator = (
            accumulator * correction[:, None]
            + tl.dot(probabilities.to(tl.float16), v)
        )
        running_max = new_max

    gate = tl.load(gate_ptr + q_offsets).to(tl.float32)
    gate = 2.0 * tl.sigmoid(gate)
    output = accumulator / running_sum[:, None] * gate
    tl.store(out_ptr + q_offsets, output.to(tl.float16))


def cpu_reference(q, k, v, gate, q_norm, k_norm, head_scale):
    q = q.astype(np.float32)
    k = k.astype(np.float32)
    v = v.astype(np.float32)
    q = q / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + NORM_EPS)
    k = k / np.sqrt(np.mean(k * k, axis=-1, keepdims=True) + NORM_EPS)
    q *= q_norm
    k *= k_norm
    scores = np.einsum("mhd,nhd->hmn", q, k)
    scores *= (
        head_scale[:, None, None]
        * math.log(KEYS_PER_GROUP)
        / HEAD_DIM
    )
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    output = np.einsum("hmn,nhd->mhd", probabilities, v)
    return output * (2.0 / (1.0 + np.exp(-gate)))


def percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rng = np.random.default_rng(7)
    q_host = rng.normal(
        0, 0.15, (WORK_ITEMS * QUERY_TILE, HEADS, HEAD_DIM)
    ).astype(np.float16)
    k_host = rng.normal(
        0, 0.15, (GROUPS * KEYS_PER_GROUP, HEADS, HEAD_DIM)
    ).astype(np.float16)
    v_host = rng.normal(
        0, 0.15, (GROUPS * KEYS_PER_GROUP, HEADS, HEAD_DIM)
    ).astype(np.float16)
    gate_host = rng.normal(
        0, 0.15, (WORK_ITEMS * QUERY_TILE, HEADS, HEAD_DIM)
    ).astype(np.float16)
    q_norm_host = rng.normal(1, 0.02, (HEAD_DIM,)).astype(np.float32)
    k_norm_host = rng.normal(1, 0.02, (HEAD_DIM,)).astype(np.float32)
    head_scale_host = rng.normal(1, 0.02, (HEADS,)).astype(np.float32)

    started = time.perf_counter()
    q = torch.from_numpy(q_host).cuda()
    k = torch.from_numpy(k_host).cuda()
    v = torch.from_numpy(v_host).cuda()
    gate = torch.from_numpy(gate_host).cuda()
    q_norm = torch.from_numpy(q_norm_host).cuda()
    k_norm = torch.from_numpy(k_norm_host).cuda()
    head_scale = torch.from_numpy(head_scale_host).cuda()
    out = torch.empty_like(q)
    allocation_seconds = time.perf_counter() - started

    grid = (WORK_ITEMS, HEADS)
    launch = lambda: relational_attention[grid](
        q,
        k,
        v,
        gate,
        q_norm,
        k_norm,
        head_scale,
        out,
        n_keys=KEYS_PER_GROUP,
        chunks_per_group=QUERY_TILES_PER_GROUP,
        n_heads=HEADS,
        head_dim=HEAD_DIM,
        block_m=QUERY_TILE,
        block_n=KEY_TILE,
        num_warps=8,
        num_stages=3,
    )

    started = time.perf_counter()
    launch()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - started

    actual = out[:QUERY_TILE].float().cpu().numpy()
    expected = cpu_reference(
        q_host[:QUERY_TILE],
        k_host[:KEYS_PER_GROUP],
        v_host[:KEYS_PER_GROUP],
        gate_host[:QUERY_TILE],
        q_norm_host,
        k_norm_host,
        head_scale_host,
    )
    error = np.abs(actual - expected)

    for _ in range(10):
        launch()
    torch.cuda.synchronize()

    timings = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))

    p50 = percentile(timings, 50)
    result = {
        "device": torch.cuda.get_device_name(),
        "implementation": "triton-only-compute",
        "shape": {
            "cells": N_CELLS,
            "groups": GROUPS,
            "keys_per_group": KEYS_PER_GROUP,
            "query_work_items": WORK_ITEMS,
            "query_tile": QUERY_TILE,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
        },
        "correctness": {
            "max_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()),
        },
        "startup": {
            "allocation_seconds": allocation_seconds,
            "first_compile_and_launch_seconds": compile_seconds,
        },
        "attention_pass_ms": {
            "p50": p50,
            "p95": percentile(timings, 95),
            "mean": float(np.mean(timings)),
        },
        "naive_36_attention_pass_projection_ms": p50 * 36,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
