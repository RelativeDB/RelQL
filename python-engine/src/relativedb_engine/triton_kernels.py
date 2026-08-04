"""Triton kernels for the engine's primary FP16 CUDA inference path."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr,
    rows: tl.constexpr, in_dim: tl.constexpr, out_dim: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    has_bias: tl.constexpr, ieee: tl.constexpr, add_residual: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pid = tl.cdiv(out_dim, block_n)
    pid_m = pid // n_pid
    pid_n = pid % n_pid
    m = pid_m * block_m + tl.arange(0, block_m)
    n = pid_n * block_n + tl.arange(0, block_n)
    k = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_n), tl.float32)
    for k0 in range(0, in_dim, block_k):
        kk = k0 + k
        x = tl.load(
            x_ptr + m[:, None] * in_dim + kk[None, :],
            mask=(m[:, None] < rows) & (kk[None, :] < in_dim),
            other=0.0,
        )
        w = tl.load(
            weight_ptr + n[:, None] * in_dim + kk[None, :],
            mask=(n[:, None] < out_dim) & (kk[None, :] < in_dim),
            other=0.0,
        )
        # The weights are fp16 and the activation buffers are fp32. tl.dot
        # requires matching operands, so the activation meets the weight here
        # — the accumulator stays fp32, which is where the precision that
        # matters actually lives.
        x = x.to(w.dtype)
        if ieee:
            acc += tl.dot(x, tl.trans(w), input_precision="ieee")
        else:
            acc += tl.dot(x, tl.trans(w))
    if has_bias:
        acc += tl.load(bias_ptr + n, mask=n < out_dim, other=0.0)[None, :]
    if add_residual:
        acc += tl.load(
            residual_ptr + m[:, None] * out_dim + n[None, :],
            mask=(m[:, None] < rows) & (n[None, :] < out_dim),
            other=0.0,
        )
    tl.store(
        out_ptr + m[:, None] * out_dim + n[None, :],
        acc,
        mask=(m[:, None] < rows) & (n[None, :] < out_dim),
    )


def linear(x, weight, out, bias=None, ieee=False, residual=None):
    rows, in_dim = x.shape
    out_dim = weight.shape[0]
    block_m, block_n, block_k = 64, 128, 64
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(out_dim, block_n),)
    linear_kernel[grid](
        x, weight, bias if bias is not None else weight,
        residual if residual is not None else out, out,
        rows=rows, in_dim=in_dim, out_dim=out_dim,
        block_m=block_m, block_n=block_n, block_k=block_k,
        has_bias=bias is not None, ieee=ieee, add_residual=residual is not None,
        num_warps=8, num_stages=3,
    )


@triton.jit
def linear_gather_kernel(
    x_ptr, row_index_ptr, weight_ptr, bias_ptr, out_ptr,
    rows: tl.constexpr, in_dim: tl.constexpr, out_dim: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
):
    pid = tl.program_id(0)
    n_pid = tl.cdiv(out_dim, block_n)
    pid_m = pid // n_pid
    pid_n = pid % n_pid
    m = pid_m * block_m + tl.arange(0, block_m)
    source_m = tl.load(row_index_ptr + m, mask=m < rows, other=0)
    n = pid_n * block_n + tl.arange(0, block_n)
    k = tl.arange(0, block_k)
    acc = tl.zeros((block_m, block_n), tl.float32)
    for k0 in range(0, in_dim, block_k):
        kk = k0 + k
        x = tl.load(
            x_ptr + source_m[:, None] * in_dim + kk[None, :],
            mask=(m[:, None] < rows) & (kk[None, :] < in_dim),
            other=0.0,
        )
        w = tl.load(
            weight_ptr + n[:, None] * in_dim + kk[None, :],
            mask=(n[:, None] < out_dim) & (kk[None, :] < in_dim),
            other=0.0,
        )
        x = x.to(w.dtype)
        acc += tl.dot(x, tl.trans(w), input_precision="ieee")
    acc += tl.load(bias_ptr + n, mask=n < out_dim, other=0.0)[None, :]
    tl.store(
        out_ptr + m[:, None] * out_dim + n[None, :],
        acc,
        mask=(m[:, None] < rows) & (n[None, :] < out_dim),
    )


def linear_gather(x, row_index, weight, bias, out):
    rows = row_index.shape[0]
    in_dim = x.shape[1]
    out_dim = weight.shape[0]
    block_m, block_n, block_k = 64, 128, 64
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(out_dim, block_n),)
    linear_gather_kernel[grid](
        x, row_index, weight, bias, out,
        rows=rows, in_dim=in_dim, out_dim=out_dim,
        block_m=block_m, block_n=block_n, block_k=block_k,
        num_warps=8, num_stages=3,
    )


@triton.jit
def rmsnorm_kernel(
    x_ptr, scale_ptr, out_ptr,
    rows: tl.constexpr, width: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, block)
    mask = d < width
    x = tl.load(x_ptr + row * width + d, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    inv = tl.rsqrt(ss / width + 1e-5)
    scale = tl.load(scale_ptr + d, mask=mask, other=0.0)
    tl.store(out_ptr + row * width + d, x * inv * scale, mask=mask)


def rmsnorm(x, scale, out):
    block = triton.next_power_of_2(x.shape[1])
    rmsnorm_kernel[(x.shape[0],)](
        x, scale, out, rows=x.shape[0], width=x.shape[1], block=block,
        num_warps=8,
    )


@triton.jit
def scalar_embeddings_kernel(
    number_ptr, datetime_ptr, boolean_ptr,
    number_w_ptr, datetime_w_ptr, boolean_w_ptr,
    number_b_ptr, datetime_b_ptr, boolean_b_ptr,
    number_norm_ptr, datetime_norm_ptr, boolean_norm_ptr,
    sem_ptr, out_ptr,
    rows: tl.constexpr, width: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, block)
    mask = d < width
    sem = tl.load(sem_ptr + row)
    number = tl.load(number_ptr + row)
    datetime = tl.load(datetime_ptr + row)
    boolean = tl.load(boolean_ptr + row)
    number = tl.where(number == number, number, 0.0)
    datetime = tl.where(datetime == datetime, datetime, 0.0)
    boolean = tl.where(boolean == boolean, boolean, 0.0)
    nw = tl.load(number_w_ptr + d, mask=mask, other=0.0)
    dw = tl.load(datetime_w_ptr + d, mask=mask, other=0.0)
    bw = tl.load(boolean_w_ptr + d, mask=mask, other=0.0)
    nb = tl.load(number_b_ptr + d, mask=mask, other=0.0)
    db = tl.load(datetime_b_ptr + d, mask=mask, other=0.0)
    bb = tl.load(boolean_b_ptr + d, mask=mask, other=0.0)
    raw = tl.where(sem == 0, number * nw + nb,
          tl.where(sem == 2, datetime * dw + db, boolean * bw + bb))
    norm = tl.where(
        sem == 0,
        tl.load(number_norm_ptr + d, mask=mask, other=0.0),
        tl.where(
            sem == 2,
            tl.load(datetime_norm_ptr + d, mask=mask, other=0.0),
            tl.load(boolean_norm_ptr + d, mask=mask, other=0.0),
        ),
    )
    ss = tl.sum(raw * raw, axis=0)
    value = raw * tl.rsqrt(ss / width + 1e-5) * norm
    tl.store(out_ptr + row * width + d, value, mask=mask)


@triton.jit
def combine_embeddings_kernel(
    col_ptr, text_ptr, scalar_ptr, sem_ptr, target_ptr, padding_ptr,
    mask_ptr, out_ptr,
    rows: tl.constexpr, width: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, block)
    valid = d < width
    sem = tl.load(sem_ptr + row)
    target = tl.load(target_ptr + row) != 0
    padding = tl.load(padding_ptr + row) != 0
    col = tl.load(col_ptr + row * width + d, mask=valid, other=0.0)
    value = tl.where(
        sem == 1,
        tl.load(text_ptr + row * width + d, mask=valid, other=0.0),
        tl.load(scalar_ptr + row * width + d, mask=valid, other=0.0),
    )
    masked = tl.load(mask_ptr + sem * width + d, mask=valid, other=0.0)
    result = col + tl.where(target, masked, value)
    result = tl.where(padding, 0.0, result)
    tl.store(out_ptr + row * width + d, result, mask=valid)


@triton.jit
def swiglu_kernel(a_ptr, b_ptr, out_ptr, total: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    mask = i < total
    a = tl.load(a_ptr + i, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + i, mask=mask, other=0.0).to(tl.float32)
    silu = a / (1.0 + tl.exp(-a))
    tl.store(out_ptr + i, silu * b, mask=mask)


@triton.jit
def zero_kernel(out_ptr, total: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    tl.store(out_ptr + i, 0.0, mask=i < total)


@triton.jit
def relational_attention_kernel(
    qkvg_ptr, packed_k_ptr, packed_v_ptr,
    qidx_ptr, kidx_ptr, qstart_ptr, kstart_ptr, nq_ptr, nk_ptr, logkv_ptr,
    q_norm_ptr, k_norm_ptr, head_scale_ptr, out_ptr,
    n_heads: tl.constexpr, head_dim: tl.constexpr, packed_kv: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, max_keys: tl.constexpr,
):
    work = tl.program_id(0)
    head = tl.program_id(1)
    qstart = tl.load(qstart_ptr + work)
    kstart = tl.load(kstart_ptr + work)
    nq = tl.load(nq_ptr + work)
    nk = tl.load(nk_ptr + work)
    logkv = tl.load(logkv_ptr + work)
    m = tl.arange(0, block_m)
    d = tl.arange(0, head_dim)
    qrows = tl.load(qidx_ptr + qstart + m, mask=m < nq, other=0)
    qoff = qrows[:, None] * 2048 + head * head_dim + d[None, :]
    q = tl.load(qkvg_ptr + qoff, mask=m[:, None] < nq, other=0.0).to(tl.float32)
    qscale = tl.load(q_norm_ptr + d)[None, :]
    qss = tl.sum(q * q, axis=1)
    q = q * tl.rsqrt(qss[:, None] / head_dim + 1e-5) * qscale
    scale = tl.load(head_scale_ptr + head) * logkv / head_dim * 1.4426950408889634
    q = (q * scale).to(tl.float16)
    running_max = tl.full((block_m,), -float("inf"), tl.float32)
    running_sum = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, head_dim), tl.float32)
    for n0 in range(0, max_keys, block_n):
        n = n0 + tl.arange(0, block_n)
        kmask = n < nk
        if packed_kv:
            koff = ((kstart + n)[:, None] * n_heads + head) * head_dim + d[None, :]
            kval = tl.load(packed_k_ptr + koff, mask=kmask[:, None], other=0.0).to(tl.float32)
        else:
            krows = tl.load(kidx_ptr + kstart + n, mask=kmask, other=0)
            koff = krows[:, None] * 2048 + head * head_dim + d[None, :]
            kval = tl.load(qkvg_ptr + 512 + koff, mask=kmask[:, None], other=0.0).to(tl.float32)
            knorm = tl.load(k_norm_ptr + d)[None, :]
            kss = tl.sum(kval * kval, axis=1)
            kval = kval * tl.rsqrt(kss[:, None] / head_dim + 1e-5) * knorm
        scores = tl.dot(q, tl.trans(kval.to(tl.float16)))
        scores = tl.where(kmask[None, :] & (m[:, None] < nq), scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        correction = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        if packed_kv:
            vv = tl.load(packed_v_ptr + koff, mask=kmask[:, None], other=0.0)
        else:
            vv = tl.load(qkvg_ptr + 1024 + koff, mask=kmask[:, None], other=0.0)
        acc = acc * correction[:, None] + tl.dot(probabilities.to(tl.float16), vv)
        running_max = new_max
    gate = tl.load(qkvg_ptr + 1536 + qoff, mask=m[:, None] < nq, other=0.0).to(tl.float32)
    output = acc / running_sum[:, None] * (2.0 * tl.sigmoid(gate))
    outoff = (qrows[:, None] * n_heads + head) * head_dim + d[None, :]
    tl.store(out_ptr + outoff, output, mask=m[:, None] < nq)


@triton.jit
def prepare_kv_kernel(
    qkvg_ptr, kidx_ptr, k_norm_ptr, packed_k_ptr, packed_v_ptr,
    n_keys: tl.constexpr, n_heads: tl.constexpr, head_dim: tl.constexpr,
):
    key = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, head_dim)
    row = tl.load(kidx_ptr + key)
    source = row * 2048 + head * head_dim + d
    kval = tl.load(qkvg_ptr + 512 + source).to(tl.float32)
    norm = tl.load(k_norm_ptr + d)
    ss = tl.sum(kval * kval, axis=0)
    kval = kval * tl.rsqrt(ss / head_dim + 1e-5) * norm
    destination = (key * n_heads + head) * head_dim + d
    tl.store(packed_k_ptr + destination, kval)
    tl.store(packed_v_ptr + destination, tl.load(qkvg_ptr + 1024 + source))


@triton.jit
def swiglu_packed_kernel(
    packed_ptr, out_ptr,
    rows: tl.constexpr, width: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * block + tl.arange(0, block)
    mask = d < width
    a = tl.load(packed_ptr + row * (2 * width) + d, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(packed_ptr + row * (2 * width) + width + d, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * width + d, a / (1.0 + tl.exp(-a)) * b, mask=mask)
