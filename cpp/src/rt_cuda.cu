// rt_cuda.cu — CUDA backend for RT-J inference (mirror of the Metal design).
//
// fp32 dense projections run as cuBLAS SGEMMs (row-major y = x W^T via the
// col-major transpose trick); wo and w2 accumulate into the residual stream
// with beta=1. q8 projections run true int8: activations quantize per row on
// device (absmax/127, like the CPU SDOT path) and the GEMM runs int8 x int8
// on the dp4a units with scales folded in at the epilogue. f16/q4 run a
// custom qgemm (32x32 output tiles, shared-memory-staged in-register
// dequant — the fp32 weight tile never exists in DRAM). All quantized
// weights stay quantized-resident. Custom kernels handle the rest:
//  - rmsnorm_rows(_clear): one warp per row (pre-norms, d=512); the attn
//    pre-norm also clears the attention output row in the same pass
//  - attn: tiled like the Metal kernel — one 256-thread block per work item
//    of at most kMQ queries; K/V rows are staged into shared memory kTK at a
//    time so each staged row serves every (query, head) pair in the item,
//    with QK-RMSNorm fused at load and the sigmoid output gate fused at the
//    write. Same single-pass online softmax and bf16-rounded log(kv) query
//    scaling as the CPU/MPS paths.
//  - attn_part / attn_reduce: flash split-K for groups whose key list
//    exceeds kFlashSplit — chunks stream in parallel blocks, partial
//    {m, l, o} states merge with the online-softmax identity
//  - swiglu_packed / head: SwiGLU on the stacked [w1;w3] GEMM output, fused
//    output-norm + number head. w1/w3 upload stacked so the FFN up-projection
//    is one GEMM.
// Weights are uploaded once per model; activation/index buffers grow on
// demand and are reused. Forwards on one model are serialized by the ctx.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "rt_internal.hpp"

namespace rt {
namespace detail {
namespace {

constexpr int kD = kDModel;          // 512
constexpr int kC4 = 4 * kDModel;     // fused qkvg row stride
constexpr float kNormEps = 1e-6f;

#define RT_CU(call)                                                     \
  do {                                                                  \
    cudaError_t e_ = (call);                                            \
    if (e_ != cudaSuccess)                                              \
      throw std::runtime_error(std::string("rt/cuda: ") +               \
                               cudaGetErrorString(e_));                 \
  } while (0)

#define RT_CUBLAS(call)                                                 \
  do {                                                                  \
    cublasStatus_t s_ = (call);                                         \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                    \
      throw std::runtime_error("rt/cuda: cublas error " +               \
                               std::to_string((int)s_));                \
  } while (0)

struct AttnWorkGpu {
  int qstart, tq, kstart, nk, rowbase;
  float logkv;
};

// Host mirrors of the flash split-K work items (see attn_part/attn_reduce).
struct AttnPWorkGpu {
  int qstart, tq, kstart, nk, rowbase;
  float logkv;
  int part;      // float offset of this chunk's partials
};
struct AttnRWorkGpu {
  int qstart, tq, rowbase, part, nchunks;
};

// A group whose shared key list exceeds kFlashSplit is split into
// kFlashChunk-key chunks, each streamed by an independent block.
constexpr int kFlashChunk = 256;
constexpr int kFlashSplit = 512;
constexpr int kMQ = 8;    // queries per tiled-attention work item
constexpr int kTK = 6;    // keys staged per shared-memory tile
constexpr float kLog2e = 1.4426950408889634f;  // scores carry a log2(e) factor
                                               // so softmax exps are raw EX2

__device__ inline float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// y[M,N] = x[M,K] @ W[N,K]^T (+ y when ACC) with W quantized-resident — the
// CUDA port of the Metal qgemm. WT: 1 = F16, 2 = Q8, 3 = Q4 (rt::WType).
// One 256-thread block computes a 32x32 output tile. The K loop stages a
// 32x32 x-tile and a 32x32 *dequantized* W-tile in shared memory (dequant
// happens on the load from DRAM), then each thread accumulates 4 outputs of
// its column. K and N are multiples of 32 for every RT-J projection; M
// (tokens) is edge-guarded. Q4 note: K-chunks of 32 align exactly with Q4's
// group size, so each staged W-row chunk touches one (scale, min) pair.
template <int WT, bool ACC>
__global__ void k_qgemm(const float* __restrict__ x,
                        const uint8_t* __restrict__ w,
                        const uint8_t* __restrict__ ws, float* __restrict__ y,
                        int M, int N, int K) {
  __shared__ float Xt[32][33];
  __shared__ float Wt[32][33];
  const int n0 = blockIdx.x * 32, m0 = blockIdx.y * 32;
  const int tx = threadIdx.x, ty = threadIdx.y;  // block is (32, 8)
  const int tid = ty * 32 + tx;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};

  for (int k0 = 0; k0 < K; k0 += 32) {
    for (int i = tid; i < 32 * 32; i += 256) {         // stage x tile
      int r = i / 32, c = i % 32;
      Xt[r][c] = (m0 + r < M) ? x[(size_t)(m0 + r) * K + k0 + c] : 0.f;
    }
    if (WT == 3) {
      // One byte contains two adjacent Q4 weights. Have one thread unpack and
      // dequantize both values, halving payload/scale reads and loop work.
      for (int i = tid; i < 32 * 16; i += 256) {
        int n = i / 16, p = i % 16;
        size_t gn = (size_t)(n0 + n);
        const __half* sh = reinterpret_cast<const __half*>(ws) +
                           (gn * (K >> 5) + (k0 >> 5)) * 2;
        uint8_t b = w[gn * (size_t)(K >> 1) + (k0 >> 1) + p];
        float scale = __half2float(sh[0]), bias = __half2float(sh[1]);
        Wt[n][2 * p] = scale * (float)(b & 0xf) + bias;
        Wt[n][2 * p + 1] = scale * (float)(b >> 4) + bias;
      }
    } else if (WT == 1) {
      // Two adjacent halves per thread as one vectorized half2 load, widened
      // with a single __half22float2 (K is even; rows are half2-aligned).
      for (int i = tid; i < 32 * 16; i += 256) {
        int n = i / 16, p = i % 16;
        size_t gn = (size_t)(n0 + n);
        __half2 h = reinterpret_cast<const __half2*>(
            w + gn * (size_t)K * 2)[(k0 >> 1) + p];
        float2 f = __half22float2(h);
        Wt[n][2 * p] = f.x;
        Wt[n][2 * p + 1] = f.y;
      }
    } else {
      for (int i = tid; i < 32 * 32; i += 256) {       // stage + convert W
        int n = i / 32, kk = i % 32;
        size_t gn = (size_t)(n0 + n);
        Wt[n][kk] =
            reinterpret_cast<const float*>(ws)[gn] *
            (float)reinterpret_cast<const int8_t*>(w)[gn * K + k0 + kk];
      }
    }
    __syncthreads();
    for (int kk = 0; kk < 32; kk++) {
      float wv = Wt[tx][kk];
#pragma unroll
      for (int i = 0; i < 4; i++) acc[i] += Xt[ty + 8 * i][kk] * wv;
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    int r = ty + 8 * i;
    if (m0 + r < M) {
      float* d = y + (size_t)(m0 + r) * N + n0 + tx;
      if (ACC)
        *d += acc[i];
      else
        *d = acc[i];
    }
  }
}

// Per-row int8 activation quantization (absmax/127, mirrors the CPU SDOT
// path). One warp per row; scale written per row.
__global__ void k_quant_rows(const float* __restrict__ x,
                             int8_t* __restrict__ q, float* __restrict__ qs,
                             int K) {
  int row = blockIdx.x;
  int lane = threadIdx.x;
  const float* xr = x + (size_t)row * K;
  float amax = 0.f;
  for (int i = lane; i < K; i += 32) amax = fmaxf(amax, fabsf(xr[i]));
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const float s = amax > 0.f ? amax / 127.f : 1.f;
  const float inv = 1.f / s;
  int8_t* qr = q + (size_t)row * K;
  for (int i = lane; i < K; i += 32) qr[i] = (int8_t)__float2int_rn(xr[i] * inv);
  if (lane == 0) qs[row] = s;
}

// True-int8 GEMM for Q8 weights: y[M,N] (+)= (sx_m * sw_n) * (xq_m . wq_n),
// int8 x int8 dot products on the dp4a units. Same 32x32 tiling as k_qgemm
// but tiles are staged as packed int32 (4 bytes/lane) and the K loop runs
// dp4a; the int32 accumulator is exact (|dot| <= 127*127*K < 2^31 for
// K <= 2048) and scales fold in once at the epilogue. K, N multiples of 32;
// M edge-guarded.
template <bool ACC>
__global__ void k_qgemm_i8(const int8_t* __restrict__ xq,
                           const float* __restrict__ xs,
                           const uint8_t* __restrict__ w,
                           const uint8_t* __restrict__ ws,
                           float* __restrict__ y, int M, int N, int K) {
  __shared__ int Xt[32][9];            // 32 int8 = 8 ints per row, +1 pad
  __shared__ int Wt[32][9];
  const int n0 = blockIdx.x * 32, m0 = blockIdx.y * 32;
  const int tx = threadIdx.x, ty = threadIdx.y;  // block is (32, 8)
  const int tid = ty * 32 + tx;
  int acc[4] = {0, 0, 0, 0};

  for (int k0 = 0; k0 < K; k0 += 32) {
    {                                   // one int (4 bytes) per thread
      int r = tid / 8, c = tid % 8;
      Xt[r][c] = (m0 + r < M)
                     ? reinterpret_cast<const int*>(
                           xq + (size_t)(m0 + r) * K + k0)[c]
                     : 0;
      Wt[r][c] = reinterpret_cast<const int*>(
          w + (size_t)(n0 + r) * K + k0)[c];
    }
    __syncthreads();
    for (int kk = 0; kk < 8; kk++) {
      int wv = Wt[tx][kk];
#pragma unroll
      for (int i = 0; i < 4; i++)
        acc[i] = __dp4a(Xt[ty + 8 * i][kk], wv, acc[i]);
    }
    __syncthreads();
  }

  const float sw = reinterpret_cast<const float*>(ws)[n0 + tx];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int r = ty + 8 * i;
    if (m0 + r < M) {
      float* d = y + (size_t)(m0 + r) * N + n0 + tx;
      float v = xs[m0 + r] * sw * (float)acc[i];
      if (ACC)
        *d += v;
      else
        *d = v;
    }
  }
}

// out[row] = rmsnorm(in[row]) * scale, rows of length n. One warp per row.
// CLEAR also zeroes clear[row] in the same pass (the attention output row —
// positions no work item writes must contribute zero to the wo projection).
template <bool CLEAR>
__global__ void k_rmsnorm_rows(const float* __restrict__ in,
                               float* __restrict__ out,
                               const float* __restrict__ scale, int n,
                               float* __restrict__ clear,
                               __half* __restrict__ outh = nullptr,
                               __half* __restrict__ clearh = nullptr) {
  int row = blockIdx.x;
  int lane = threadIdx.x;
  const float* x = in + (size_t)row * n;
  float* y = out + (size_t)row * n;
  __half* yh = outh ? outh + (size_t)row * n : nullptr;
  float ss = 0.f;
  for (int i = lane; i < n; i += 32) ss += x[i] * x[i];
  ss = warp_sum(ss);
  float inv = rsqrtf(ss / n + kNormEps);
  for (int i = lane; i < n; i += 32) {
    const float v = x[i] * inv * scale[i];
    // fp16 consumers (tensor-core GEMM) get the value pre-converted so the
    // projection skips its f32->f16 pass; rounding point is unchanged.
    if (yh) yh[i] = __float2half(v);
    else y[i] = v;
    if (CLEAR) {
      if (clearh)
        clearh[(size_t)row * n + i] = __float2half(0.f);
      else
        clear[(size_t)row * n + i] = 0.f;
    }
  }
}

// Tiled attention (CUDA port of the Metal kernel). One 256-thread block
// (8 warps) per work item of at most kMQ queries; warp `sg` owns the
// (query, head) pairs {sg, sg+8, ...} with their online-softmax state in
// registers. K/V rows are staged into shared memory kTK at a time by all
// 256 threads, so each staged row serves every query and head in the item
// instead of being re-streamed from DRAM per (query, head). QK-RMSNorm is
// fused: queries normalize at load, keys normalize in the staged tile.
// The sigmoid output gate is fused at the write — each (query, head)
// segment is written by exactly one pair of one group.
template <typename Q>
__global__ void k_attn(const Q* __restrict__ qkvg, float* __restrict__ att,
                       const int* __restrict__ qidx,
                       const int* __restrict__ kidx,
                       const AttnWorkGpu* __restrict__ work,
                       const float* __restrict__ head_scale,
                       const float* __restrict__ q_norm,
                       const float* __restrict__ k_norm,
                       __half* __restrict__ atth = nullptr) {
  __shared__ float Kt[kTK * kD];
  __shared__ float Vt[kTK * kD];
  const AttnWorkGpu w = work[blockIdx.x];
  const int lane = threadIdx.x % 32;
  const int sg = threadIdx.x / 32;
  const int tid = threadIdx.x;
  const int npair = w.tq * kHeads;
  float q0v[kMQ], q1v[kMQ], mx[kMQ], den[kMQ], a0[kMQ], a1[kMQ];
  size_t orow[kMQ];
  for (int s = 0; s < kMQ; s++) {
    mx[s] = -INFINITY; den[s] = 0.f; a0[s] = 0.f; a1[s] = 0.f;
    int p = sg + s * 8;
    if (p < npair) {
      int r = p / kHeads, h = p % kHeads;
      size_t qrowi = (size_t)(w.rowbase + qidx[w.qstart + r]);
      const Q* q = qkvg + qrowi * kC4 + h * kHeadDim;
      float r0 = q[2 * lane], r1 = q[2 * lane + 1];
      float inv = rsqrtf(warp_sum(r0 * r0 + r1 * r1) / kHeadDim + kNormEps);
      float qscale = head_scale[h] * w.logkv / kHeadDim * kLog2e;
      q0v[s] = r0 * inv * q_norm[2 * lane] * qscale;
      q1v[s] = r1 * inv * q_norm[2 * lane + 1] * qscale;
      orow[s] = qrowi * kD + h * kHeadDim;
    }
  }
  for (int k0 = 0; k0 < w.nk; k0 += kTK) {
    const int kn = min(kTK, w.nk - k0);
    __syncthreads();                     // previous tile fully consumed
    for (int i = tid; i < kn * kD; i += 256) {
      int j = i / kD, d = i % kD;
      size_t krow = (size_t)(w.rowbase + kidx[w.kstart + k0 + j]) * kC4;
      Kt[j * kD + d] = qkvg[krow + kD + d];
      Vt[j * kD + d] = qkvg[krow + 2 * kD + d];
    }
    __syncthreads();
    // Fused QK-RMSNorm (key half): normalize the staged tile in place, one
    // warp per (key, head) segment.
    for (int seg = sg; seg < kn * kHeads; seg += 8) {
      int j = seg / kHeads, h = seg % kHeads;
      float* kk = Kt + j * kD + h * kHeadDim;
      float a = kk[2 * lane], b = kk[2 * lane + 1];
      float invk = rsqrtf(warp_sum(a * a + b * b) / kHeadDim + kNormEps);
      kk[2 * lane] = a * invk * k_norm[2 * lane];
      kk[2 * lane + 1] = b * invk * k_norm[2 * lane + 1];
    }
    __syncthreads();
    // Score the whole staged tile per pair first, then do ONE running-max
    // update, ONE rescale, and kn EX2s — instead of a max/corr/rescale per
    // key. Scores already carry log2(e) (folded into qscale), so the
    // softmax exps are raw exp2f, the SFU's native op.
    if (kn == kTK) {                     // full tile: unrolled fast path
      for (int s = 0; s < kMQ; s++) {
        int p = sg + s * 8;
        if (p >= npair) continue;
        int h = p % kHeads;
        const float* kb = Kt + h * kHeadDim;
        float sc[kTK];
#pragma unroll
        for (int j = 0; j < kTK; j++) {
          const float* k = kb + j * kD;
          sc[j] = warp_sum(q0v[s] * k[2 * lane] + q1v[s] * k[2 * lane + 1]);
        }
        float m6 = sc[0];
#pragma unroll
        for (int j = 1; j < kTK; j++) m6 = fmaxf(m6, sc[j]);
        float nm = fmaxf(mx[s], m6);
        float corr = exp2f(mx[s] - nm);
        float wt[kTK], dsum = 0.f;
#pragma unroll
        for (int j = 0; j < kTK; j++) { wt[j] = exp2f(sc[j] - nm); dsum += wt[j]; }
        den[s] = den[s] * corr + dsum;
        float acc0 = a0[s] * corr, acc1 = a1[s] * corr;
        const float* vb = Vt + h * kHeadDim;
#pragma unroll
        for (int j = 0; j < kTK; j++) {
          acc0 += wt[j] * vb[j * kD + 2 * lane];
          acc1 += wt[j] * vb[j * kD + 2 * lane + 1];
        }
        a0[s] = acc0; a1[s] = acc1; mx[s] = nm;
      }
    } else {
      for (int j = 0; j < kn; j++) {
        for (int s = 0; s < kMQ; s++) {
          int p = sg + s * 8;
          if (p >= npair) continue;
          int h = p % kHeads;
          const float* k = Kt + j * kD + h * kHeadDim;
          float score = warp_sum(q0v[s] * k[2 * lane] + q1v[s] * k[2 * lane + 1]);
          const float* v = Vt + j * kD + h * kHeadDim;
          float nm = fmaxf(mx[s], score);
          float corr = exp2f(mx[s] - nm);
          float wt = exp2f(score - nm);
          den[s] = den[s] * corr + wt;
          a0[s] = a0[s] * corr + wt * v[2 * lane];
          a1[s] = a1[s] * corr + wt * v[2 * lane + 1];
          mx[s] = nm;
        }
      }
    }
  }
  for (int s = 0; s < kMQ; s++) {
    int p = sg + s * 8;
    if (p >= npair) continue;
    int r = p / kHeads, h = p % kHeads;
    size_t grow = (size_t)(w.rowbase + qidx[w.qstart + r]) * kC4 + 3 * kD +
                  h * kHeadDim;
    float g0 = 2.f / (1.f + __expf(-(float)qkvg[grow + 2 * lane]));
    float g1 =
        2.f / (1.f + __expf(-(float)qkvg[grow + 2 * lane + 1]));
    if (atth) {
      __half* oh = atth + orow[s];
      oh[2 * lane] = __float2half(a0[s] / den[s] * g0);
      oh[2 * lane + 1] = __float2half(a1[s] / den[s] * g1);
    } else {
      float* o = att + orow[s];
      o[2 * lane] = a0[s] / den[s] * g0;
      o[2 * lane + 1] = a1[s] / den[s] * g1;
    }
  }
}

// Flash split-K attention for long key lists: same tiled streaming as
// Tensor-core attention for big-key groups (column groups dominate: ~500-key
// all-pairs blobs that are 94% of attention time). One 256-thread block per
// work item of up to 16 queries; warp w owns head w for all 16 queries.
// Q/K/V tiles are staged fp16 in dynamic smem (QK-RMSNorm fused into the
// stage, log2(e) folded into the query scale), scores come from wmma
// m16n16k16 HMMA with fp32 accumulate, the flash-2 online softmax runs on a
// per-warp fp32 scratch (lane pair per query row), and P.V is a second HMMA
// accumulated into an fp32 O tile in smem that each warp rescales by its
// rows' running correction. Small groups keep the scalar tiled kernel — the
// 16x16 staging isn't worth it under ~3 tiles of keys.
constexpr int kMmaQ = 16;      // queries per mma work item
constexpr int kMmaK = 32;      // keys per tile (two 16-wide HMMA n-frags)
constexpr int kMmaMinNk = 48;  // below this the scalar kernel wins
constexpr size_t kMmaSmem =
    (size_t)kMmaQ * kD * sizeof(__half)      // Qs
    + (size_t)kHeads * kMmaQ * kMmaK * sizeof(float)   // per-warp S scratch
    + (size_t)kHeads * kMmaQ * kMmaK * sizeof(__half)  // per-warp P
    + (size_t)kHeads * kMmaQ * sizeof(float);          // per-warp row corr

// K/V prepack for the tensor-core path: every mma work item of the same
// group used to re-read, re-normalize, and fp16-convert the same key rows
// (a 500-key column group split into 16-query items repeats that ~32x).
// One 256-thread block per key row (8 warps = 8 heads) writes RMS-normalized
// k_norm-folded fp16 K and raw fp16 V, indexed by flattened key-list
// position, so k_attn_mma loads tiles straight from global (L2-hot) with no
// per-tile conversion. The +kMmaK padding tail rows are zeroed; reads past a
// group's kn land there or in the next group's rows, and both are masked out
// of the softmax and P.
template <typename Q>
__global__ void k_prep_kv(const Q* __restrict__ qkvg,
                          const int32_t* __restrict__ kabs, int n,
                          const float* __restrict__ k_norm,
                          __half* __restrict__ kp, __half* __restrict__ vp) {
  const int i = blockIdx.x;
  const int lane = threadIdx.x % 32, h = threadIdx.x / 32;
  __half* kd = kp + (size_t)i * kD + h * kHeadDim;
  __half* vd = vp + (size_t)i * kD + h * kHeadDim;
  if (i >= n) {
    for (int d = lane; d < kHeadDim; d += 32) {
      kd[d] = __float2half(0.f);
      vd[d] = __float2half(0.f);
    }
    return;
  }
  const Q* kg = qkvg + (size_t)kabs[i] * kC4 + kD + h * kHeadDim;
  const Q* vg = qkvg + (size_t)kabs[i] * kC4 + 2 * kD + h * kHeadDim;
  float ss = 0.f;
  for (int d = lane; d < kHeadDim; d += 32) {
    const float v = (float)kg[d];
    ss += v * v;
  }
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss / kHeadDim + kNormEps);
  for (int d = lane; d < kHeadDim; d += 32) {
    kd[d] = __float2half((float)kg[d] * inv * k_norm[d]);
    vd[d] = __float2half((float)vg[d]);
  }
}

template <typename Q>
__global__ void k_attn_mma(const Q* __restrict__ qkvg,
                           float* __restrict__ att,
                           const int* __restrict__ qidx,
                           const __half* __restrict__ kp,
                           const __half* __restrict__ vp,
                           const AttnWorkGpu* __restrict__ work,
                           const float* __restrict__ head_scale,
                           const float* __restrict__ q_norm,
                           __half* __restrict__ atth = nullptr) {
  extern __shared__ unsigned char smraw[];
  __half* Qs = (__half*)smraw;                         // [kMmaQ][kD]
  float* Sw = (float*)(Qs + (size_t)kMmaQ * kD);       // [kHeads][16][16]
  __half* Pw = (__half*)(Sw + (size_t)kHeads * kMmaQ * kMmaK);  // [kHeads][16][16]
  float* Cw = (float*)(Pw + (size_t)kHeads * kMmaQ * kMmaK);    // [kHeads][16]
  const AttnWorkGpu w = work[blockIdx.x];
  const int lane = threadIdx.x % 32;
  const int wid = threadIdx.x / 32;    // warp == head

  // ---- stage Q (normalized, scaled, fp16) + zero O; once per item --------
  // Warp-cooperative: warp w stages rows {2w, 2w+1}; per head segment the
  // lanes split the 64 dims (2 each), warp-reduce the sumsq, and store
  // coalesced — no 64-value serial loops, no bank-aliased column stores.
  for (int r = wid * 2; r < wid * 2 + 2; r++) {
    __half* dst = Qs + (size_t)r * kD;
    if (r < w.tq) {
      const Q* qg = qkvg +
          (size_t)(w.rowbase + qidx[w.qstart + r]) * kC4;
      for (int h = 0; h < kHeads; h++) {
        const float v0 = qg[h * kHeadDim + lane];
        const float v1 = qg[h * kHeadDim + 32 + lane];
        float ss = v0 * v0 + v1 * v1;
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float inv = rsqrtf(ss / kHeadDim + kNormEps);
        const float qscale = head_scale[h] * w.logkv / kHeadDim * kLog2e * inv;
        dst[h * kHeadDim + lane] = __float2half(v0 * qscale * q_norm[lane]);
        dst[h * kHeadDim + 32 + lane] =
            __float2half(v1 * qscale * q_norm[32 + lane]);
      }
    } else {
      for (int d = lane; d < kD; d += 32) dst[d] = __float2half(0.f);
    }
  }
  // per-row softmax state, mirrored in the row's lane pair (r = lane>>1)
  const int myrow = lane >> 1;
  float mx = -INFINITY, den = 0.f;

  // O lives in mma.m16n8k16 accumulator fragments: 8 column slabs of the
  // head's 64 dims, 4 f32 each. The documented C layout puts this thread's
  // values at rows {lane/4, lane/4+8}, cols (lane%4)*2+{0,1} of each slab,
  // so the per-tile softmax correction is a plain register multiply.
  float oreg[8][4] = {};
  const int fr0 = lane >> 2, fr1 = (lane >> 2) + 8;
  const int fkb = (lane & 3) * 2;

  namespace wm = nvcuda::wmma;
  wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> af;
  wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> bf;
  wm::fragment<wm::accumulator, 16, 16, 16, float> cf;
  float* mySw = Sw + (size_t)wid * kMmaQ * kMmaK;
  __half* myPw = Pw + (size_t)wid * kMmaQ * kMmaK;

  __syncthreads();                     // Qs staged once
  for (int k0 = 0; k0 < w.nk; k0 += kMmaK) {
    const int kn = min(kMmaK, w.nk - k0);
    // K/V tiles come prepacked (normalized fp16) from global; rows past kn
    // belong to the padding tail or the next group and are masked below.
    const __half* Kt = kp + (size_t)(w.kstart + k0) * kD + wid * kHeadDim;

    // ---- scores: S = Q_h K_h^T via HMMA, fp32 accumulate ------------------
    for (int j = 0; j < kMmaK; j += 16) {
      wm::fill_fragment(cf, 0.f);
      for (int kk = 0; kk < kHeadDim; kk += 16) {
        wm::load_matrix_sync(af, Qs + wid * kHeadDim + kk, kD);
        wm::load_matrix_sync(bf, Kt + (size_t)j * kD + kk, kD);
        wm::mma_sync(cf, af, bf, cf);
      }
      wm::store_matrix_sync(mySw + j, cf, kMmaK, wm::mem_row_major);
    }
    __syncwarp();

    // ---- flash-2 online softmax on the scratch; lane pair per query row ---
    {
      const int cb = (lane & 1) * (kMmaK / 2);   // this lane's column half
      float lmax = -INFINITY;
      for (int c = 0; c < kMmaK / 2; c++)
        if (cb + c < kn) lmax = fmaxf(lmax, mySw[myrow * kMmaK + cb + c]);
      lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, 1));
      float nm = fmaxf(mx, lmax);
      float corr = exp2f(mx - nm);
      float dloc = 0.f;
      for (int c = 0; c < kMmaK / 2; c++) {
        float wt = (cb + c < kn && myrow < w.tq)
                       ? exp2f(mySw[myrow * kMmaK + cb + c] - nm)
                       : 0.f;
        myPw[myrow * kMmaK + cb + c] = __float2half(wt);
        dloc += wt;
      }
      den = den * corr + dloc + __shfl_xor_sync(0xffffffffu, dloc, 1);
      if ((lane & 1) == 0) Cw[wid * kMmaQ + myrow] = corr;
      mx = nm;
    }
    __syncwarp();

    // ---- O = corr*O + P V via raw mma.sync, accumulators in registers -----
    {
      const float cr0 = Cw[wid * kMmaQ + fr0];
      const float cr1 = Cw[wid * kMmaQ + fr1];
      for (int sl = 0; sl < 8; sl++) {
        oreg[sl][0] *= cr0; oreg[sl][1] *= cr0;
        oreg[sl][2] *= cr1; oreg[sl][3] *= cr1;
      }
      for (int kk2 = 0; kk2 < kMmaK; kk2 += 16) {
        const unsigned a0 =
            *(const unsigned*)(myPw + fr0 * kMmaK + kk2 + fkb);
        const unsigned a1 =
            *(const unsigned*)(myPw + fr1 * kMmaK + kk2 + fkb);
        const unsigned a2 =
            *(const unsigned*)(myPw + fr0 * kMmaK + kk2 + 8 + fkb);
        const unsigned a3 =
            *(const unsigned*)(myPw + fr1 * kMmaK + kk2 + 8 + fkb);
        const __half* vb =
            vp + (size_t)(w.kstart + k0 + kk2) * kD + wid * kHeadDim;
        for (int sl = 0; sl < 8; sl++) {
          const int col = sl * 8 + (lane >> 2);
          __half2 b0h = __halves2half2(vb[(size_t)fkb * kD + col],
                                       vb[(size_t)(fkb + 1) * kD + col]);
          __half2 b1h = __halves2half2(vb[(size_t)(fkb + 8) * kD + col],
                                       vb[(size_t)(fkb + 9) * kD + col]);
          const unsigned b0 = *(const unsigned*)&b0h;
          const unsigned b1 = *(const unsigned*)&b1h;
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(oreg[sl][0]), "+f"(oreg[sl][1]), "+f"(oreg[sl][2]),
                "+f"(oreg[sl][3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
    }
    __syncwarp();
  }

  // ---- finalize from registers: divide, gate, write this thread's cells ---
  for (int rr = 0; rr < 2; rr++) {
    const int q = rr ? fr1 : fr0;
    // q differs per lane, so the shuffle must run in all lanes (full mask)
    // before the tq guard — a divergent continue here is UB.
    const float dq = __shfl_sync(0xffffffffu, den, 2 * q);
    if (q >= w.tq) continue;
    const float invden = dq > 0.f ? 1.f / dq : 0.f;
    const size_t qrowi = (size_t)(w.rowbase + qidx[w.qstart + q]);
    const Q* g = qkvg + qrowi * kC4 + 3 * kD + wid * kHeadDim;
    float* o = att + qrowi * kD + wid * kHeadDim;
    __half* oh = atth ? atth + qrowi * kD + wid * kHeadDim : nullptr;
    for (int sl = 0; sl < 8; sl++) {
      const int d0 = sl * 8 + fkb, d1 = d0 + 1;
      const float v0 = oreg[sl][rr * 2 + 0] * invden;
      const float v1 = oreg[sl][rr * 2 + 1] * invden;
      const float g0 = 2.f / (1.f + __expf(-(float)g[d0]));
      const float g1 = 2.f / (1.f + __expf(-(float)g[d1]));
      if (oh) {
        oh[d0] = __float2half(v0 * g0);
        oh[d1] = __float2half(v1 * g1);
      } else {
        o[d0] = v0 * g0;
        o[d1] = v1 * g1;
      }
    }
  }
}

// v2 of the tiled kernel: warp-per-(query,head) pair, lane-per-key scoring.
// A 32-key tile of normalized K is staged TRANSPOSED (Kt[d][j], padded
// stride) in dynamic shared memory, so each lane owns one key and computes
// its full 64-dim dot with Q broadcast lane-by-lane via shuffles — one
// 5-shuffle warp reduction per 32 keys instead of one per key, and the two
// softmax exps run once per lane instead of serially per key. V rows are
// read straight from global memory: every warp touches the same 32 rows,
// so after the first pass they are L1-resident, and staging them would
// blow the 99KB smem budget that the padded K tile already dominates.
constexpr int kTK2 = 32;                    // keys per v2 tile (= warp lanes)
constexpr int kKtStride = kTK2 + 2;         // padded fp16 rows: bank-conflict-free
constexpr size_t kAttn2Smem =
    (size_t)kD * kKtStride * sizeof(__half) + kTK2 * sizeof(int);

__global__ void k_attn2(const float* __restrict__ qkvg, float* __restrict__ att,
                        const int* __restrict__ qidx,
                        const int* __restrict__ kidx,
                        const AttnWorkGpu* __restrict__ work,
                        const float* __restrict__ head_scale,
                        const float* __restrict__ q_norm,
                        const float* __restrict__ k_norm) {
  extern __shared__ __half smem2[];
  __half* Kt = smem2;                        // [kD][kKtStride], fp16
  int* krows = (int*)(smem2 + (size_t)kD * kKtStride);  // key row ids of tile
  const AttnWorkGpu w = work[blockIdx.x];
  const int lane = threadIdx.x % 32;
  const int sg = threadIdx.x / 32;
  const int tid = threadIdx.x;
  const int npair = w.tq * kHeads;
  const unsigned full = 0xffffffffu;
  float q0v[kMQ], q1v[kMQ], mx[kMQ], den[kMQ], a0[kMQ], a1[kMQ];
  int hh[kMQ];
  size_t orow[kMQ];
#pragma unroll
  for (int s = 0; s < kMQ; s++) {
    mx[s] = -INFINITY; den[s] = 0.f; a0[s] = 0.f; a1[s] = 0.f; hh[s] = 0;
    int p = sg + s * 8;
    if (p < npair) {
      int r = p / kHeads, h = p % kHeads;
      hh[s] = h;
      size_t qrowi = (size_t)(w.rowbase + qidx[w.qstart + r]);
      const float* q = qkvg + qrowi * kC4 + h * kHeadDim;
      float r0 = q[2 * lane], r1 = q[2 * lane + 1];
      float inv = rsqrtf(warp_sum(r0 * r0 + r1 * r1) / kHeadDim + kNormEps);
      float qscale = head_scale[h] * w.logkv / kHeadDim;
      q0v[s] = r0 * inv * q_norm[2 * lane] * qscale;
      q1v[s] = r1 * inv * q_norm[2 * lane + 1] * qscale;
      orow[s] = qrowi * kD + h * kHeadDim;
    }
  }
  for (int k0 = 0; k0 < w.nk; k0 += kTK2) {
    const int kn = min(kTK2, w.nk - k0);
    __syncthreads();                     // previous tile fully consumed
    // Stage + QK-RMSNorm(key half) + transpose: thread (j = t/8, h = t%8)
    // owns one (key, head) segment — 64 serial global reads that the GEMM
    // just wrote (L2-hot), one rsqrt, 64 padded-stride smem writes.
    {
      int j = tid >> 3, h = tid & 7;
      if (j < kn) {
        if (h == 0) krows[j] = w.rowbase + kidx[w.kstart + k0 + j];
        const float* kg = qkvg +
            (size_t)(w.rowbase + kidx[w.kstart + k0 + j]) * kC4 + kD +
            h * kHeadDim;
        // Two passes over the L2-hot row instead of a 64-float register
        // cache (which would spill to local memory).
        float ss = 0.f;
        for (int d = 0; d < kHeadDim; d++) { float kv = kg[d]; ss += kv * kv; }
        float invk = rsqrtf(ss / kHeadDim + kNormEps);
        for (int d = 0; d < kHeadDim; d++)
          Kt[(size_t)(h * kHeadDim + d) * kKtStride + j] =
              __float2half(kg[d] * invk * k_norm[d]);
      }
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < kMQ; s++) {
      int p = sg + s * 8;
      if (p >= npair) continue;
      const int h = hh[s];
      float acc = lane < kn ? 0.f : -INFINITY;
      const __half* kt = Kt + (size_t)h * kHeadDim * kKtStride + lane;
#pragma unroll
      for (int d = 0; d < kHeadDim; d++) {
        float qd = __shfl_sync(full, (d & 1) ? q1v[s] : q0v[s], d >> 1);
        acc += qd * __half2float(kt[(size_t)d * kKtStride]);
      }
      float m_tile = acc;
      for (int off = 16; off > 0; off >>= 1)
        m_tile = fmaxf(m_tile, __shfl_xor_sync(full, m_tile, off));
      float nm = fmaxf(mx[s], m_tile);
      float wt = __expf(acc - nm);         // 0 for masked lanes (-INF)
      float corr = __expf(mx[s] - nm);
      den[s] = den[s] * corr + warp_sum(wt);
      a0[s] *= corr; a1[s] *= corr;
      mx[s] = nm;
      for (int j = 0; j < kn; j++) {
        float wtj = __shfl_sync(full, wt, j);
        const float* v = qkvg + (size_t)krows[j] * kC4 + 2 * kD +
                         h * kHeadDim;
        a0[s] += wtj * v[2 * lane];
        a1[s] += wtj * v[2 * lane + 1];
      }
    }
  }
#pragma unroll
  for (int s = 0; s < kMQ; s++) {
    int p = sg + s * 8;
    if (p >= npair) continue;
    int r = p / kHeads, h = p % kHeads;
    size_t grow = (size_t)(w.rowbase + qidx[w.qstart + r]) * kC4 + 3 * kD +
                  h * kHeadDim;
    float g0 = 2.f / (1.f + __expf(-qkvg[grow + 2 * lane]));
    float g1 = 2.f / (1.f + __expf(-qkvg[grow + 2 * lane + 1]));
    float* o = att + orow[s];
    o[2 * lane] = a0[s] / den[s] * g0;
    o[2 * lane + 1] = a1[s] / den[s] * g1;
  }
}

// k_attn, but only over this work item's key chunk, emitting per-
// (query, head, chunk) partials {running max m, denom l, unnormalized
// weighted-V sum o[64]} at float offset w.part + (r*8+h)*66.
template <typename Q>
__global__ void k_attn_part(const Q* __restrict__ qkvg,
                            float* __restrict__ partials,
                            const int* __restrict__ qidx,
                            const int* __restrict__ kidx,
                            const AttnPWorkGpu* __restrict__ work,
                            const float* __restrict__ head_scale,
                            const float* __restrict__ q_norm,
                            const float* __restrict__ k_norm) {
  __shared__ float Kt[kTK * kD];
  __shared__ float Vt[kTK * kD];
  const AttnPWorkGpu w = work[blockIdx.x];
  const int lane = threadIdx.x % 32;
  const int sg = threadIdx.x / 32;
  const int tid = threadIdx.x;
  const int npair = w.tq * kHeads;
  float q0v[kMQ], q1v[kMQ], mx[kMQ], den[kMQ], a0[kMQ], a1[kMQ];
  for (int s = 0; s < kMQ; s++) {
    mx[s] = -INFINITY; den[s] = 0.f; a0[s] = 0.f; a1[s] = 0.f;
    int p = sg + s * 8;
    if (p < npair) {
      int r = p / kHeads, h = p % kHeads;
      size_t qrowi = (size_t)(w.rowbase + qidx[w.qstart + r]);
      const Q* q = qkvg + qrowi * kC4 + h * kHeadDim;
      float r0 = q[2 * lane], r1 = q[2 * lane + 1];
      float inv = rsqrtf(warp_sum(r0 * r0 + r1 * r1) / kHeadDim + kNormEps);
      float qscale = head_scale[h] * w.logkv / kHeadDim;
      q0v[s] = r0 * inv * q_norm[2 * lane] * qscale;
      q1v[s] = r1 * inv * q_norm[2 * lane + 1] * qscale;
    }
  }
  for (int k0 = 0; k0 < w.nk; k0 += kTK) {
    const int kn = min(kTK, w.nk - k0);
    __syncthreads();
    for (int i = tid; i < kn * kD; i += 256) {
      int j = i / kD, d = i % kD;
      size_t krow = (size_t)(w.rowbase + kidx[w.kstart + k0 + j]) * kC4;
      Kt[j * kD + d] = qkvg[krow + kD + d];
      Vt[j * kD + d] = qkvg[krow + 2 * kD + d];
    }
    __syncthreads();
    for (int seg = sg; seg < kn * kHeads; seg += 8) {
      int j = seg / kHeads, h = seg % kHeads;
      float* kk = Kt + j * kD + h * kHeadDim;
      float a = kk[2 * lane], b = kk[2 * lane + 1];
      float invk = rsqrtf(warp_sum(a * a + b * b) / kHeadDim + kNormEps);
      kk[2 * lane] = a * invk * k_norm[2 * lane];
      kk[2 * lane + 1] = b * invk * k_norm[2 * lane + 1];
    }
    __syncthreads();
    for (int j = 0; j < kn; j++) {
      for (int s = 0; s < kMQ; s++) {
        int p = sg + s * 8;
        if (p >= npair) continue;
        int h = p % kHeads;
        const float* k = Kt + j * kD + h * kHeadDim;
        float score = warp_sum(q0v[s] * k[2 * lane] + q1v[s] * k[2 * lane + 1]);
        const float* v = Vt + j * kD + h * kHeadDim;
        float nm = fmaxf(mx[s], score);
        float corr = __expf(mx[s] - nm);
        float wt = __expf(score - nm);
        den[s] = den[s] * corr + wt;
        a0[s] = a0[s] * corr + wt * v[2 * lane];
        a1[s] = a1[s] * corr + wt * v[2 * lane + 1];
        mx[s] = nm;
      }
    }
  }
  for (int s = 0; s < kMQ; s++) {
    int p = sg + s * 8;
    if (p >= npair) continue;
    int r = p / kHeads, h = p % kHeads;
    float* o = partials + w.part + (size_t)(r * 8 + h) * 66;
    if (lane == 0) { o[0] = mx[s]; o[1] = den[s]; }
    o[2 + 2 * lane] = a0[s];
    o[2 + 2 * lane + 1] = a1[s];
  }
}

// Merge the chunk partials with the online-softmax identity
// l = Σ l_c·exp(m_c-M), o = Σ o_c·exp(m_c-M); gate fused at the write.
template <typename Q>
__global__ void k_attn_reduce(const float* __restrict__ partials,
                              float* __restrict__ att,
                              const int* __restrict__ qidx,
                              const AttnRWorkGpu* __restrict__ work,
                              const Q* __restrict__ qkvg,
                              __half* __restrict__ atth = nullptr) {
  const AttnRWorkGpu w = work[blockIdx.x];
  const int lane = threadIdx.x % 32;
  const int sg = threadIdx.x / 32;
  const int nsg = blockDim.x / 32;
  const size_t cstride = (size_t)w.tq * 8 * 66;
  for (int p = sg; p < w.tq * kHeads; p += nsg) {
    int r = p / kHeads, h = p % kHeads;
    const float* base = partials + w.part + (size_t)(r * 8 + h) * 66;
    float M = -INFINITY;
    for (int c = 0; c < w.nchunks; c++) M = fmaxf(M, base[c * cstride]);
    float l = 0.f, o0 = 0.f, o1 = 0.f;
    for (int c = 0; c < w.nchunks; c++) {
      const float* pc = base + c * cstride;
      float f = __expf(pc[0] - M);
      l += pc[1] * f;
      o0 += pc[2 + 2 * lane] * f;
      o1 += pc[2 + 2 * lane + 1] * f;
    }
    size_t qrowi = (size_t)(w.rowbase + qidx[w.qstart + r]);
    size_t grow = qrowi * kC4 + 3 * kD + h * kHeadDim;
    float g0 = 2.f / (1.f + __expf(-(float)qkvg[grow + 2 * lane]));
    float g1 =
        2.f / (1.f + __expf(-(float)qkvg[grow + 2 * lane + 1]));
    if (atth) {
      __half* oh = atth + qrowi * kD + h * kHeadDim;
      oh[2 * lane] = __float2half(o0 / l * g0);
      oh[2 * lane + 1] = __float2half(o1 / l * g1);
    } else {
      float* o = att + qrowi * kD + h * kHeadDim;
      o[2 * lane] = o0 / l * g0;
      o[2 * lane + 1] = o1 / l * g1;
    }
  }
}

// ffa = silu(ffa) * ffb (separate w1/w3 fallback when dtypes differ).
__global__ void k_swiglu(float* __restrict__ ffa,
                         const float* __restrict__ ffb, size_t total) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  float a = ffa[gid];
  ffa[gid] = (a / (1.f + __expf(-a))) * ffb[gid];
}

// ffa[row, d] = silu(ff13[row, d]) * ff13[row, kDFF + d] — SwiGLU on the
// stacked [w1; w3] GEMM output.
__global__ void k_swiglu_packed(const float* __restrict__ ff13,
                                float* __restrict__ ffa, size_t total,
                                __half* __restrict__ ffah = nullptr) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  size_t row = gid / kDFF, d = gid % kDFF;
  float a = ff13[row * (2 * kDFF) + d];
  float b = ff13[row * (2 * kDFF) + kDFF + d];
  const float v = (a / (1.f + __expf(-a))) * b;
  if (ffah) ffah[gid] = __float2half(v);
  else ffa[gid] = v;
}

// Same operation when the stacked projection writes its intermediate in
// fp16. Accumulation inside cuBLAS remains fp32; only the materialized
// w1/w3 activation is narrowed, halving both its write and this kernel's read.
__global__ void k_swiglu_packed_h16(const __half* __restrict__ ff13,
                                    float* __restrict__ ffa, size_t total,
                                    __half* __restrict__ ffah = nullptr) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  size_t row = gid / kDFF, d = gid % kDFF;
  float a = __half2float(ff13[row * (2 * kDFF) + d]);
  float b = __half2float(ff13[row * (2 * kDFF) + kDFF + d]);
  const float v = (a / (1.f + __expf(-a))) * b;
  if (ffah) ffah[gid] = __float2half(v);
  else ffa[gid] = v;
}

// Frozen-backbone features: tfeat[b] = Σ_target-cells rmsnorm(x[row]) *
// norm_scale, the output-normalized final hidden state summed over the batch
// row's target cells (matches the CPU/Metal reduction). One warp per batch
// row; rows are streamed so multiple targets accumulate in order.
__global__ void k_target_feats(const float* __restrict__ x,
                               const uint8_t* __restrict__ is_target,
                               const float* __restrict__ norm_scale,
                               float* __restrict__ tfeat, int S) {
  int b = blockIdx.x;
  int lane = threadIdx.x;
  float acc[kD / 32];
#pragma unroll
  for (int i = 0; i < kD / 32; i++) acc[i] = 0.f;
  for (int s = 0; s < S; s++) {
    size_t row = (size_t)b * S + s;
    if (!is_target[row]) continue;
    const float* xr = x + row * kD;
    float ss = 0.f;
    for (int i = lane; i < kD; i += 32) ss += xr[i] * xr[i];
    ss = warp_sum(ss);
    float inv = rsqrtf(ss / kD + kNormEps);
#pragma unroll
    for (int i = 0; i < kD / 32; i++) {
      int d = lane + 32 * i;
      acc[i] += xr[d] * inv * norm_scale[d];
    }
  }
#pragma unroll
  for (int i = 0; i < kD / 32; i++) tfeat[(size_t)b * kD + lane + 32 * i] = acc[i];
}

// yhat[row] = dec_b + dot(rmsnorm(x[row]) * norm_scale, dec_w). Warp per row.
__global__ void k_head(const float* __restrict__ x,
                       const float* __restrict__ norm_scale,
                       const float* __restrict__ dec_w, float dec_b,
                       float* __restrict__ yhat) {
  int row = blockIdx.x;
  int lane = threadIdx.x;
  const float* xr = x + (size_t)row * kD;
  float ss = 0.f;
  for (int i = lane; i < kD; i += 32) ss += xr[i] * xr[i];
  ss = warp_sum(ss);
  float inv = rsqrtf(ss / kD + kNormEps);
  float d = 0.f;
  for (int i = lane; i < kD; i += 32)
    d += xr[i] * inv * norm_scale[i] * dec_w[i];
  d = warp_sum(d);
  if (lane == 0) yhat[row] = dec_b + d;
}

// Weight on the GPU: fp32 buffer (cuBLAS path) or quantized payload + scales
// (custom qgemm path). Uploaded once per model.
struct GpuWeight {
  WType type{};
  float* f32 = nullptr;                // F32 payload
  uint8_t* q = nullptr;                // F16/Q8/Q4 payload
  uint8_t* s = nullptr;                // Q8/Q4 scales
  int out = 0, in = 0;
};

struct BlockWeights {
  GpuWeight wqkvg[3], wo[3];           // per attention type (col, feat, nbr)
  GpuWeight w1, w2, w3, w13;           // w13 = stacked [w1; w3] (one GEMM)
  float* norm[4];
  float *q_norm[3], *k_norm[3], *head_scale[3];
};

// One independent execution lane: its own stream, cuBLAS handle, and every
// grow-on-demand activation/index buffer. Weights live on CudaCtx and are
// shared read-only across slots, so K slots run K forwards concurrently on
// the device with no shared mutable state.
struct CudaSlot {
  cublasHandle_t blas = nullptr;
  cudaStream_t stream = nullptr;
  // grow-on-demand activation / index buffers
  float *x = nullptr, *xn = nullptr, *qkvg = nullptr, *att = nullptr;
  float *ffa = nullptr, *ffb = nullptr, *ff13 = nullptr;
  float *yhat = nullptr, *tap = nullptr;
  __half* xh = nullptr;                // f16 activations (tensor-core GEMMs)
  int8_t* xq = nullptr;                // int8 activations for Q8 projections
  float* xqs = nullptr;                // per-row activation scales
  int *qidx[3] = {}, *kidx[3] = {};
  int32_t* kabsd[3] = {};              // absolute token row per kflat entry
  __half *kp = nullptr, *vp = nullptr; // prepacked normalized-K / V (fp16)
  __half *xnh = nullptr, *ffah = nullptr;  // fp16 producer outputs (fused)
  __half* qkvgh = nullptr;                 // fp16 Q/K/V/gate intermediate
  __half* ff13h = nullptr;                 // fp16 stacked FFN intermediate
  __half* atth = nullptr;                  // fp16 attention output (fused)
  AttnWorkGpu* work[3] = {};
  AttnWorkGpu* mwork[3] = {};          // tensor-core (mma) work items
  AttnPWorkGpu* pwork[3] = {};         // flash split-K chunk items
  AttnRWorkGpu* rwork[3] = {};         // flash split-K reduce items
  float* partials = nullptr;           // split-K partial {m, l, o[64]} states
  uint8_t* tgt = nullptr;              // sorted_is_target for feature gather
  float* tfeat = nullptr;              // [B, kD] target features
  // device-embed channel buffers ([BS,384] x2, [BS] x3 floats, [BS] x3 bytes,
  // plus two [BS,kD] projection scratch halves)
  float *ch_col = nullptr, *ch_text = nullptr;
  float *ch_num = nullptr, *ch_dat = nullptr, *ch_bool = nullptr;
  uint8_t *ch_sem = nullptr, *ch_pad = nullptr, *ch_tgt = nullptr;
  float *etmp_col = nullptr, *etmp_text = nullptr;
  size_t cap_bs = 0, cap_q[3] = {}, cap_k[3] = {}, cap_w[3] = {};
  size_t cap_mw[3] = {}, cap_pw[3] = {}, cap_rw[3] = {}, cap_part = 0;
  size_t cap_tgt = 0, cap_tf = 0, cap_xh = 0, cap_ch = 0;
  size_t cap_ka[3] = {}, cap_kv = 0;   // kabs entries / prepack rows

  ~CudaSlot() {
    for (float* p : {x, xn, qkvg, att, ffa, ffb, ff13, yhat, tap, xqs,
                     partials, tfeat, ch_col, ch_text, ch_num, ch_dat,
                     ch_bool, etmp_col, etmp_text})
      cudaFree(p);
    cudaFree(xh);
    cudaFree(xq);
    cudaFree(tgt);
    for (uint8_t* p : {ch_sem, ch_pad, ch_tgt}) cudaFree(p);
    for (int a = 0; a < 3; a++) {
      cudaFree(qidx[a]);
      cudaFree(kidx[a]);
      cudaFree(kabsd[a]);
      cudaFree(work[a]);
      cudaFree(mwork[a]);
      cudaFree(pwork[a]);
      cudaFree(rwork[a]);
    }
    cudaFree(kp);
    cudaFree(vp);
    cudaFree(xnh);
    cudaFree(ffah);
    cudaFree(qkvgh);
    cudaFree(ff13h);
    cudaFree(atth);
    if (blas) cublasDestroy(blas);
    if (stream) cudaStreamDestroy(stream);
  }
};

// Embedding-stage weights (block-0 input construction on device): the
// col-name/text projections' packed matrices, per-type biases and scalar
// encoder rows (zeros where the checkpoint has no bias), the input rmsnorm
// scales, and the mask embeddings, all [4,kD]-packed by SemType.
struct EmbedWeights {
  float* col_w = nullptr;              // [kD, kDText]
  float* text_w = nullptr;             // [kD, kDText]
  float* col_b = nullptr;              // [kD] (zeros when null)
  float* text_b = nullptr;             // [kD]
  float* scal_w = nullptr;             // [4, kD]: enc[t].w column (in==1)
  float* scal_b = nullptr;             // [4, kD]
  float* norm_col = nullptr;           // [kD]
  float* norm_enc = nullptr;           // [4, kD]
  float* mask = nullptr;               // [4, kD]
};

struct CudaCtx {
  BlockWeights blk[kBlocks] = {};      // shared, read-only after upload
  float *norm_out = nullptr, *dec_w = nullptr;
  float dec_b = 0.f;
  EmbedWeights emb;
  std::vector<void*> owned;            // every weight cudaMalloc for cleanup

  // Slot pool: lanes are created on demand up to RT_CUDA_SLOTS (default 2)
  // and callers block when all are busy. One slot degrades to the previous
  // one-mutex-per-model behavior.
  std::mutex pool_mu;
  std::condition_variable pool_cv;
  std::vector<std::unique_ptr<CudaSlot>> slots;
  std::vector<CudaSlot*> free_list;
  const int max_slots = [] {
    const char* e = std::getenv("RT_CUDA_SLOTS");
    const int v = e ? std::atoi(e) : 2;
    return v > 0 ? v : 1;
  }();

  CudaSlot* acquire() {
    std::unique_lock<std::mutex> lk(pool_mu);
    for (;;) {
      if (!free_list.empty()) {
        CudaSlot* s = free_list.back();
        free_list.pop_back();
        return s;
      }
      if ((int)slots.size() < max_slots) {
        auto s = std::make_unique<CudaSlot>();
        RT_CU(cudaStreamCreate(&s->stream));
        RT_CUBLAS(cublasCreate(&s->blas));
        RT_CUBLAS(cublasSetStream(s->blas, s->stream));
        slots.push_back(std::move(s));
        return slots.back().get();
      }
      pool_cv.wait(lk);
    }
  }

  void release(CudaSlot* s) {
    {
      std::lock_guard<std::mutex> lk(pool_mu);
      free_list.push_back(s);
    }
    pool_cv.notify_one();
  }

  ~CudaCtx() {
    for (void* p : owned) cudaFree(p);
  }
};

struct SlotLease {                     // RAII: release on every exit path
  CudaCtx& ctx;
  CudaSlot* s;
  explicit SlotLease(CudaCtx& c) : ctx(c), s(c.acquire()) {}
  ~SlotLease() { ctx.release(s); }
};

float* dev_upload(CudaCtx* ctx, const float* p, size_t n) {
  float* d = nullptr;
  RT_CU(cudaMalloc(&d, n * sizeof(float)));
  RT_CU(cudaMemcpy(d, p, n * sizeof(float), cudaMemcpyHostToDevice));
  ctx->owned.push_back(d);
  return d;
}

uint8_t* dev_upload_bytes(CudaCtx* ctx, const uint8_t* p, size_t bytes) {
  uint8_t* d = nullptr;
  RT_CU(cudaMalloc(&d, bytes));
  RT_CU(cudaMemcpy(d, p, bytes, cudaMemcpyHostToDevice));
  ctx->owned.push_back(d);
  return d;
}

GpuWeight upload_weight(CudaCtx* ctx, const Weight& w) {
  GpuWeight g;
  g.type = w.type;
  g.out = w.out;
  g.in = w.in;
  if (w.type == WType::F32) {
    g.f32 = dev_upload(ctx, w.f32, (size_t)w.out * w.in);
    return g;
  }
  // qgemm requires 32-aligned projection shapes (true for every RT-J weight).
  if (w.in % 32 != 0 || w.out % 32 != 0)
    throw std::runtime_error("rt/cuda: quantized weight dims must be "
                             "multiples of 32");
  g.q = dev_upload_bytes(ctx, w.q, row_bytes(w.type, w.in) * (size_t)w.out);
  size_t sb = scale_bytes(w.type, w.in) * (size_t)w.out;
  if (sb) g.s = dev_upload_bytes(ctx, w.qs, sb);
  return g;
}

// Stacked [w1; w3] upload so the FFN up-projection runs as one GEMM. Rows
// quantize independently in every format, so stacking is plain payload (and
// scale) concatenation. Returns type-less (out=0) when the dtypes differ.
GpuWeight upload_stacked(CudaCtx* ctx, const Weight& w1, const Weight& w3) {
  GpuWeight g;
  if (w1.type != w3.type || w1.in != w3.in) return g;
  g.type = w1.type;
  g.out = w1.out + w3.out;
  g.in = w1.in;
  if (w1.type == WType::F32) {
    float* d = nullptr;
    size_t n1 = (size_t)w1.out * w1.in, n3 = (size_t)w3.out * w3.in;
    RT_CU(cudaMalloc(&d, (n1 + n3) * sizeof(float)));
    RT_CU(cudaMemcpy(d, w1.f32, n1 * 4, cudaMemcpyHostToDevice));
    RT_CU(cudaMemcpy(d + n1, w3.f32, n3 * 4, cudaMemcpyHostToDevice));
    ctx->owned.push_back(d);
    g.f32 = d;
    return g;
  }
  size_t rb1 = row_bytes(w1.type, w1.in) * (size_t)w1.out;
  size_t rb3 = row_bytes(w3.type, w3.in) * (size_t)w3.out;
  uint8_t* q = nullptr;
  RT_CU(cudaMalloc(&q, rb1 + rb3));
  RT_CU(cudaMemcpy(q, w1.q, rb1, cudaMemcpyHostToDevice));
  RT_CU(cudaMemcpy(q + rb1, w3.q, rb3, cudaMemcpyHostToDevice));
  ctx->owned.push_back(q);
  g.q = q;
  size_t sb1 = scale_bytes(w1.type, w1.in) * (size_t)w1.out;
  size_t sb3 = scale_bytes(w3.type, w3.in) * (size_t)w3.out;
  if (sb1 + sb3) {
    uint8_t* s = nullptr;
    RT_CU(cudaMalloc(&s, sb1 + sb3));
    RT_CU(cudaMemcpy(s, w1.qs, sb1, cudaMemcpyHostToDevice));
    RT_CU(cudaMemcpy(s + sb1, w3.qs, sb3, cudaMemcpyHostToDevice));
    ctx->owned.push_back(s);
    g.s = s;
  }
  return g;
}

CudaCtx* make_ctx(const Model& m) {
  auto* ctx = new CudaCtx();
  try {
    for (int b = 0; b < kBlocks; b++) {
      const Block& blk = m.blocks[b];
      BlockWeights& g = ctx->blk[b];
      for (int a = 0; a < 3; a++) {
        g.wqkvg[a] = upload_weight(ctx, blk.attn[a].wqkvg);
        g.wo[a] = upload_weight(ctx, blk.attn[a].wo);
        g.q_norm[a] = dev_upload(ctx, blk.attn[a].q_norm, kHeadDim);
        g.k_norm[a] = dev_upload(ctx, blk.attn[a].k_norm, kHeadDim);
        g.head_scale[a] = dev_upload(ctx, blk.attn[a].head_scale, kHeads);
        g.norm[a] = dev_upload(ctx, blk.norm[a], kD);
      }
      g.norm[3] = dev_upload(ctx, blk.norm[3], kD);
      g.w13 = upload_stacked(ctx, blk.w1, blk.w3);
      if (!g.w13.out) {                // dtype mismatch: separate projections
        g.w1 = upload_weight(ctx, blk.w1);
        g.w3 = upload_weight(ctx, blk.w3);
      }
      g.w2 = upload_weight(ctx, blk.w2);
    }
    ctx->norm_out = dev_upload(ctx, m.norm_out, kD);
    ctx->dec_w = dev_upload(ctx, m.dec_number.w, kD);
    ctx->dec_b = m.dec_number.b[0];
    {
      // embedding-stage weights for the device-embed path
      std::vector<float> zeros(kD, 0.f);
      auto up_b = [&](const float* b) {
        return dev_upload(ctx, b ? b : zeros.data(), kD);
      };
      ctx->emb.col_w = dev_upload(ctx, m.enc_col_name.w, (size_t)kD * kDText);
      ctx->emb.text_w = dev_upload(ctx, m.enc[kText].w, (size_t)kD * kDText);
      ctx->emb.col_b = up_b(m.enc_col_name.b);
      ctx->emb.text_b = up_b(m.enc[kText].b);
      std::vector<float> sw(4 * kD, 0.f), sb(4 * kD, 0.f), ne(4 * kD, 0.f),
          me(4 * kD, 0.f);
      for (int t = 0; t < 4; t++) {
        if (t != kText && m.enc[t].w)          // [out, in==1]: w[j]
          for (int j = 0; j < kD; j++) sw[(size_t)t * kD + j] = m.enc[t].w[j];
        if (t != kText && m.enc[t].b)
          std::memcpy(&sb[(size_t)t * kD], m.enc[t].b, kD * sizeof(float));
        if (m.norm_enc[t])
          std::memcpy(&ne[(size_t)t * kD], m.norm_enc[t], kD * sizeof(float));
        if (m.mask_emb[t])
          std::memcpy(&me[(size_t)t * kD], m.mask_emb[t], kD * sizeof(float));
      }
      ctx->emb.scal_w = dev_upload(ctx, sw.data(), sw.size());
      ctx->emb.scal_b = dev_upload(ctx, sb.data(), sb.size());
      ctx->emb.norm_col = dev_upload(ctx, m.norm_col_name, kD);
      ctx->emb.norm_enc = dev_upload(ctx, ne.data(), ne.size());
      ctx->emb.mask = dev_upload(ctx, me.data(), me.size());
    }
    return ctx;
  } catch (...) {
    delete ctx;
    throw;
  }
}

__global__ void k_f32_to_f16(const float* __restrict__ in,
                             __half* __restrict__ out, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

// Block-0 input from sorted channels (device-embed path). One block per
// token, 128 threads over kD=512: x = rmsnorm(colv @ col_w + col_b) *
// norm_col, then either the sem-type mask embedding (masked target) or the
// rmsnorm'd value encoding, added on top. Mirrors rt.cpp's embed stage
// (rmsnorm: x / sqrt(mean(x^2) + 1e-6) * scale; scalar NaN -> 0).
__global__ void k_embed_gather(
    const float* __restrict__ tcol, const float* __restrict__ ttext,
    const float* __restrict__ numv, const float* __restrict__ datv,
    const float* __restrict__ boolv, const uint8_t* __restrict__ sem,
    const uint8_t* __restrict__ pad, const uint8_t* __restrict__ tgt,
    const float* __restrict__ col_b, const float* __restrict__ text_b,
    const float* __restrict__ scal_w, const float* __restrict__ scal_b,
    const float* __restrict__ norm_col, const float* __restrict__ norm_enc,
    const float* __restrict__ mask, float* __restrict__ x) {
  const size_t i = blockIdx.x;
  const int tid = threadIdx.x;
  float* xr = x + i * kD;
  __shared__ float red[128];
  if (pad[i]) {
    for (int j = tid; j < kD; j += 128) xr[j] = 0.f;
    return;
  }
  float v[kD / 128];
  float ss = 0.f;
  for (int c = 0, j = tid; j < kD; j += 128, c++) {
    v[c] = tcol[i * kD + j] + col_b[j];
    ss += v[c] * v[c];
  }
  red[tid] = ss;
  __syncthreads();
  for (int w = 64; w > 0; w >>= 1) {
    if (tid < w) red[tid] += red[tid + w];
    __syncthreads();
  }
  float inv = 1.f / sqrtf(red[0] / kD + 1e-6f);
  __syncthreads();
  for (int c = 0, j = tid; j < kD; j += 128, c++)
    xr[j] = v[c] * inv * norm_col[j];
  const int t = sem[i];
  if (tgt[i]) {
    for (int j = tid; j < kD; j += 128) xr[j] += mask[(size_t)t * kD + j];
    return;
  }
  ss = 0.f;
  if (t == kText) {
    for (int c = 0, j = tid; j < kD; j += 128, c++) {
      v[c] = ttext[i * kD + j] + text_b[j];
      ss += v[c] * v[c];
    }
  } else {
    float val = t == kNumber ? numv[i] : t == kDatetime ? datv[i] : boolv[i];
    if (isnan(val)) val = 0.f;
    for (int c = 0, j = tid; j < kD; j += 128, c++) {
      v[c] = val * scal_w[(size_t)t * kD + j] + scal_b[(size_t)t * kD + j];
      ss += v[c] * v[c];
    }
  }
  red[tid] = ss;
  __syncthreads();
  for (int w = 64; w > 0; w >>= 1) {
    if (tid < w) red[tid] += red[tid + w];
    __syncthreads();
  }
  inv = 1.f / sqrtf(red[0] / kD + 1e-6f);
  for (int c = 0, j = tid; j < kD; j += 128, c++)
    xr[j] += v[c] * inv * norm_enc[(size_t)t * kD + j];
}

// y[M,N] = x[M,K] @ W[N,K]^T (+ beta * y), all row-major, via the col-major
// transpose identity: y_cm[N,M] = W_cm^T[N,K] @ x_cm[K,M].
void gemm(CudaSlot& s, const float* x, const float* w, float* y, int M, int N,
          int K, float beta) {
  const float alpha = 1.f;
  RT_CUBLAS(cublasSgemm(s.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w,
                        K, x, K, &beta, y, N));
}

// Projection dispatch: fp32 uses cuBLAS SGEMM; f16 casts activations to half
// and runs a tensor-core GemmEx (fp32 accumulate) — the raw f16 rows ARE the
// device matrix, no dequant; q8 quantizes activations on device and runs the
// true-int8 dp4a GEMM (mirrors the CPU SDOT path); q4 runs the
// dequant-in-register qgemm. Weights stay quantized-resident. beta is only
// ever 0 or 1. RT_CUDA_F16_QGEMM=1 forces f16 back onto the qgemm kernel
// (parity bisection knob).
void proj(CudaSlot& s, const float* x, const GpuWeight& w, float* y, int M,
          float beta, const __half* xh_pre = nullptr) {
  if (w.type == WType::F32) {
    gemm(s, x, w.f32, y, M, w.out, w.in, beta);
    return;
  }
  const dim3 grid((unsigned)(w.out / 32), (unsigned)((M + 31) / 32));
  const dim3 block(32, 8);
  const bool acc = beta != 0.f;
  cudaStream_t st = s.stream;
  switch (w.type) {
    case WType::F16: {
      static const bool force_qgemm =
          std::getenv("RT_CUDA_F16_QGEMM") != nullptr;
      if (!force_qgemm) {
        const __half* a16 = xh_pre;
        if (!a16) {
          const size_t n = (size_t)M * w.in;
          if (s.cap_xh < n) {
            if (s.xh) RT_CU(cudaFree(s.xh));
            s.xh = nullptr;
            RT_CU(cudaMalloc(&s.xh, n * sizeof(__half)));
            s.cap_xh = n;
          }
          k_f32_to_f16<<<(int)((n + 255) / 256), 256, 0, st>>>(x, s.xh, n);
          a16 = s.xh;
        }
        const float alpha = 1.f;
        RT_CUBLAS(cublasGemmEx(
            s.blas, CUBLAS_OP_T, CUBLAS_OP_N, w.out, M, w.in, &alpha, w.q,
            CUDA_R_16F, w.in, a16, CUDA_R_16F, w.in, &beta, y, CUDA_R_32F,
            w.out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        break;
      }
      if (acc)
        k_qgemm<1, true><<<grid, block, 0, st>>>(x, w.q, w.s, y, M, w.out,
                                                 w.in);
      else
        k_qgemm<1, false><<<grid, block, 0, st>>>(x, w.q, w.s, y, M, w.out,
                                                  w.in);
      break;
    }
    case WType::Q8:
      k_quant_rows<<<M, 32, 0, st>>>(x, s.xq, s.xqs, w.in);
      if (acc)
        k_qgemm_i8<true><<<grid, block, 0, st>>>(s.xq, s.xqs, w.q, w.s, y,
                                                 M, w.out, w.in);
      else
        k_qgemm_i8<false><<<grid, block, 0, st>>>(s.xq, s.xqs, w.q, w.s,
                                                  y, M, w.out, w.in);
      break;
    default:  // Q4
      if (acc)
        k_qgemm<3, true><<<grid, block, 0, st>>>(x, w.q, w.s, y, M, w.out,
                                                 w.in);
      else
        k_qgemm<3, false><<<grid, block, 0, st>>>(x, w.q, w.s, y, M, w.out,
                                                  w.in);
  }
}

// F16-weight projection with an F16 materialized result. Tensor-core
// multiply and accumulation stay fp32; cuBLAS performs the final conversion
// while storing C, avoiding a separate full-width fp32 intermediate.
void proj_h16(CudaSlot& s, const GpuWeight& w, __half* y, int M,
              const __half* x) {
  const float alpha = 1.f, beta = 0.f;
  RT_CUBLAS(cublasGemmEx(
      s.blas, CUBLAS_OP_T, CUBLAS_OP_N, w.out, M, w.in, &alpha, w.q,
      CUDA_R_16F, w.in, x, CUDA_R_16F, w.in, &beta, y, CUDA_R_16F, w.out,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template <typename T>
void grow(T** p, size_t* cap, size_t need) {
  if (*cap >= need) return;
  if (*p) RT_CU(cudaFree(*p));
  *p = nullptr;
  RT_CU(cudaMalloc(p, need * sizeof(T)));
  *cap = need;
}

}  // namespace

bool cuda_available() {
  static bool ok = [] {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }();
  return ok;
}

void run_blocks_cuda(const Model& m, Prepared& prep, Output& out,
                     bool debug_taps, bool want_target_features) {
  // ---- lazy per-model context --------------------------------------------
  static std::mutex init_mu;
  std::shared_ptr<void>& slot = m.device_ctx[(int)Device::CUDA];
  {
    std::lock_guard<std::mutex> lk(init_mu);
    if (!slot) slot.reset(make_ctx(m), [](void* p) { delete (CudaCtx*)p; });
  }
  CudaCtx& ctx = *(CudaCtx*)slot.get();
  bool use_h16_ff13 = std::getenv("RT_CUDA_F16_QGEMM") == nullptr;
  for (int b = 0; b < kBlocks && use_h16_ff13; b++)
    use_h16_ff13 &= ctx.blk[b].w13.out != 0 &&
                    ctx.blk[b].w13.type == WType::F16;
  bool use_h16_qkvg = std::getenv("RT_CUDA_F16_QGEMM") == nullptr &&
                      std::getenv("RT_CUDA_ATTN_V2") == nullptr;
  for (int b = 0; b < kBlocks && use_h16_qkvg; b++)
    for (int a = 0; a < 3; a++)
      use_h16_qkvg &= ctx.blk[b].wqkvg[a].type == WType::F16;

  const int B = prep.B, S = prep.S;
  const size_t BS = (size_t)B * S;

  // ---- flatten group indices / work items for the GPU --------------------
  // Pure host work on prep — runs before taking the ctx lock so concurrent
  // forwards overlap their CPU flattening with another forward's GPU time.
  // Small groups (nk <= kFlashSplit) run the single-pass tiled kernel; large
  // groups split into kFlashChunk-key chunks for attn_part -> attn_reduce.
  // Both tiled kernels take at most kMQ queries per item (per-pair softmax
  // state lives in registers), so prep's kQTile items are sub-tiled here.
  std::vector<int32_t> qflat[3], kflat[3], kabs[3];
  std::vector<AttnWorkGpu> wflat[3], mflat[3];
  std::vector<AttnPWorkGpu> pflat[3];
  std::vector<AttnRWorkGpu> rflat[3];
  static const bool use_mma = std::getenv("RT_CUDA_ATTN_MMA") == nullptr ||
                              std::string(std::getenv("RT_CUDA_ATTN_MMA")) != "0";
  size_t part_floats[3] = {0, 0, 0};
  const std::vector<Groups>* gsets[3] = {&prep.g_col, &prep.g_feat,
                                         &prep.g_nbr};
  for (int a = 0; a < 3; a++) {
    std::vector<int> qbase(B), kbase(B);
    int q = 0, k = 0;
    for (int b = 0; b < B; b++) {
      qbase[b] = q; kbase[b] = k;
      const Groups& G = (*gsets[a])[b];
      q += (int)G.q.size();
      k += (int)G.k.size();
    }
    qflat[a].reserve(q); kflat[a].reserve(k); kabs[a].reserve(k);
    for (int b = 0; b < B; b++) {
      const Groups& G = (*gsets[a])[b];
      qflat[a].insert(qflat[a].end(), G.q.begin(), G.q.end());
      kflat[a].insert(kflat[a].end(), G.k.begin(), G.k.end());
      for (int v : G.k) kabs[a].push_back(b * S + v);
    }
    size_t poff = 0;
    for (const Work& W : prep.work[a]) {
      const Groups& G = (*gsets[a])[W.b];
      const int qs = qbase[W.b] + G.qoff[W.g] + W.q0;
      const int tq = W.q1 - W.q0;
      const int ks = kbase[W.b] + G.koff[W.g];
      const int nk = G.koff[W.g + 1] - G.koff[W.g];
      if (nk <= kFlashSplit) {
        if (use_mma && nk >= kMmaMinNk) {
          for (int sub = 0; sub < tq; sub += kMmaQ)
            mflat[a].push_back({qs + sub, std::min(kMmaQ, tq - sub), ks, nk,
                                W.b * S, W.logkv});
        } else {
          for (int sub = 0; sub < tq; sub += kMQ)
            wflat[a].push_back({qs + sub, std::min(kMQ, tq - sub), ks, nk,
                                W.b * S, W.logkv});
        }
        continue;
      }
      const int nchunks = (nk + kFlashChunk - 1) / kFlashChunk;
      const int item_base = (int)poff;
      for (int c = 0; c < nchunks; c++) {
        const int c0 = c * kFlashChunk;
        const int cnk = std::min(kFlashChunk, nk - c0);
        for (int sub = 0; sub < tq; sub += kMQ)
          pflat[a].push_back({qs + sub, std::min(kMQ, tq - sub), ks + c0, cnk,
                              W.b * S, W.logkv,
                              item_base + c * (tq * 8 * 66) + sub * 8 * 66});
      }
      rflat[a].push_back({qs, tq, W.b * S, item_base, nchunks});
      poff += (size_t)nchunks * tq * 8 * 66;
    }
    part_floats[a] = poff;
  }
  const size_t part_max =
      std::max({part_floats[0], part_floats[1], part_floats[2], (size_t)1});

  SlotLease lease(ctx);
  CudaSlot& sl = *lease.s;
  cudaStream_t st = sl.stream;

  // ---- buffers -----------------------------------------------------------
  const bool packed_ffn = ctx.blk[0].w13.out != 0;
  if (sl.cap_bs < BS) {
    for (float** p : {&sl.x, &sl.xn, &sl.att, &sl.tap}) {
      if (*p) RT_CU(cudaFree(*p));
      *p = nullptr;
      RT_CU(cudaMalloc(p, BS * kD * sizeof(float)));
    }
    if (sl.qkvg) RT_CU(cudaFree(sl.qkvg));
    if (sl.qkvgh) RT_CU(cudaFree(sl.qkvgh));
    sl.qkvg = nullptr;
    sl.qkvgh = nullptr;
    if (use_h16_qkvg)
      RT_CU(cudaMalloc(&sl.qkvgh, BS * (size_t)kC4 * sizeof(__half)));
    else
      RT_CU(cudaMalloc(&sl.qkvg, BS * (size_t)kC4 * sizeof(float)));
    if (packed_ffn) {
      for (float** p : {&sl.ffa}) {
        if (*p) RT_CU(cudaFree(*p));
        *p = nullptr;
        RT_CU(cudaMalloc(p, BS * (size_t)kDFF * sizeof(float)));
      }
      if (sl.ff13) RT_CU(cudaFree(sl.ff13));
      if (sl.ff13h) RT_CU(cudaFree(sl.ff13h));
      sl.ff13 = nullptr;
      sl.ff13h = nullptr;
      if (use_h16_ff13)
        RT_CU(cudaMalloc(&sl.ff13h,
                         BS * (size_t)(2 * kDFF) * sizeof(__half)));
      else
        RT_CU(cudaMalloc(&sl.ff13,
                         BS * (size_t)(2 * kDFF) * sizeof(float)));
    } else {
      for (float** p : {&sl.ffa, &sl.ffb}) {
        if (*p) RT_CU(cudaFree(*p));
        *p = nullptr;
        RT_CU(cudaMalloc(p, BS * (size_t)kDFF * sizeof(float)));
      }
    }
    if (sl.yhat) RT_CU(cudaFree(sl.yhat));
    sl.yhat = nullptr;
    RT_CU(cudaMalloc(&sl.yhat, BS * sizeof(float)));
    for (__half** p : {&sl.xnh, &sl.ffah, &sl.atth}) {
      if (*p) RT_CU(cudaFree(*p));
      *p = nullptr;
    }
    RT_CU(cudaMalloc(&sl.xnh, BS * (size_t)kD * sizeof(__half)));
    RT_CU(cudaMalloc(&sl.ffah, BS * (size_t)kDFF * sizeof(__half)));
    RT_CU(cudaMalloc(&sl.atth, BS * (size_t)kD * sizeof(__half)));
    if (sl.xq) RT_CU(cudaFree(sl.xq));
    sl.xq = nullptr;
    RT_CU(cudaMalloc(&sl.xq, BS * (size_t)kDFF));
    if (sl.xqs) RT_CU(cudaFree(sl.xqs));
    sl.xqs = nullptr;
    RT_CU(cudaMalloc(&sl.xqs, BS * sizeof(float)));
    sl.cap_bs = BS;
  }
  grow(&sl.partials, &sl.cap_part, part_max);
  {
    size_t kvrows = 0;
    for (int a = 0; a < 3; a++)
      if (!mflat[a].empty())
        kvrows = std::max(kvrows, kflat[a].size() + kMmaK);
    if (kvrows && sl.cap_kv < kvrows) {
      if (sl.kp) RT_CU(cudaFree(sl.kp));
      if (sl.vp) RT_CU(cudaFree(sl.vp));
      sl.kp = sl.vp = nullptr;
      RT_CU(cudaMalloc(&sl.kp, kvrows * kD * sizeof(__half)));
      RT_CU(cudaMalloc(&sl.vp, kvrows * kD * sizeof(__half)));
      sl.cap_kv = kvrows;
    }
  }
  for (int a = 0; a < 3; a++) {
    grow(&sl.qidx[a], &sl.cap_q[a], std::max<size_t>(1, qflat[a].size()));
    grow(&sl.kidx[a], &sl.cap_k[a], std::max<size_t>(1, kflat[a].size()));
    grow(&sl.kabsd[a], &sl.cap_ka[a], std::max<size_t>(1, kabs[a].size()));
    grow(&sl.work[a], &sl.cap_w[a], std::max<size_t>(1, wflat[a].size()));
    grow(&sl.mwork[a], &sl.cap_mw[a], std::max<size_t>(1, mflat[a].size()));
    grow(&sl.pwork[a], &sl.cap_pw[a], std::max<size_t>(1, pflat[a].size()));
    grow(&sl.rwork[a], &sl.cap_rw[a], std::max<size_t>(1, rflat[a].size()));
    RT_CU(cudaMemcpyAsync(sl.qidx[a], qflat[a].data(), qflat[a].size() * 4,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(sl.kidx[a], kflat[a].data(), kflat[a].size() * 4,
                          cudaMemcpyHostToDevice, st));
    if (!mflat[a].empty())
      RT_CU(cudaMemcpyAsync(sl.kabsd[a], kabs[a].data(), kabs[a].size() * 4,
                            cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(sl.work[a], wflat[a].data(),
                          wflat[a].size() * sizeof(AttnWorkGpu),
                          cudaMemcpyHostToDevice, st));
    if (!mflat[a].empty())
      RT_CU(cudaMemcpyAsync(sl.mwork[a], mflat[a].data(),
                            mflat[a].size() * sizeof(AttnWorkGpu),
                            cudaMemcpyHostToDevice, st));
    if (!pflat[a].empty())
      RT_CU(cudaMemcpyAsync(sl.pwork[a], pflat[a].data(),
                            pflat[a].size() * sizeof(AttnPWorkGpu),
                            cudaMemcpyHostToDevice, st));
    if (!rflat[a].empty())
      RT_CU(cudaMemcpyAsync(sl.rwork[a], rflat[a].data(),
                            rflat[a].size() * sizeof(AttnRWorkGpu),
                            cudaMemcpyHostToDevice, st));
  }
  if (prep.device_embed) {
    if (sl.cap_ch < BS) {
      for (float** p : {&sl.ch_col, &sl.ch_text}) {
        if (*p) RT_CU(cudaFree(*p));
        *p = nullptr;
        RT_CU(cudaMalloc(p, BS * (size_t)kDText * sizeof(float)));
      }
      for (float** p : {&sl.ch_num, &sl.ch_dat, &sl.ch_bool}) {
        if (*p) RT_CU(cudaFree(*p));
        *p = nullptr;
        RT_CU(cudaMalloc(p, BS * sizeof(float)));
      }
      for (uint8_t** p : {&sl.ch_sem, &sl.ch_pad, &sl.ch_tgt}) {
        if (*p) RT_CU(cudaFree(*p));
        *p = nullptr;
        RT_CU(cudaMalloc(p, BS));
      }
      for (float** p : {&sl.etmp_col, &sl.etmp_text}) {
        if (*p) RT_CU(cudaFree(*p));
        *p = nullptr;
        RT_CU(cudaMalloc(p, BS * (size_t)kD * sizeof(float)));
      }
      sl.cap_ch = BS;
    }
    RT_CU(cudaMemcpyAsync(sl.ch_col, prep.colv.data(),
                          BS * (size_t)kDText * 4, cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(sl.ch_text, prep.textv.data(),
                          BS * (size_t)kDText * 4, cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(sl.ch_num, prep.numv.data(), BS * 4,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(sl.ch_dat, prep.datv.data(), BS * 4,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(sl.ch_bool, prep.boolv.data(), BS * 4,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(sl.ch_sem, prep.sem8.data(), BS,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(sl.ch_pad, prep.pad.data(), BS,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(sl.ch_tgt, prep.tgt8.data(), BS,
                          cudaMemcpyHostToDevice, st));
    gemm(sl, sl.ch_col, ctx.emb.col_w, sl.etmp_col, (int)BS, kD, kDText, 0.f);
    gemm(sl, sl.ch_text, ctx.emb.text_w, sl.etmp_text, (int)BS, kD, kDText,
         0.f);
    k_embed_gather<<<(int)BS, 128, 0, st>>>(
        sl.etmp_col, sl.etmp_text, sl.ch_num, sl.ch_dat, sl.ch_bool,
        sl.ch_sem, sl.ch_pad, sl.ch_tgt, ctx.emb.col_b, ctx.emb.text_b,
        ctx.emb.scal_w, ctx.emb.scal_b, ctx.emb.norm_col, ctx.emb.norm_enc,
        ctx.emb.mask, sl.x);
  } else {
    RT_CU(cudaMemcpyAsync(sl.x, prep.x.data(), BS * kD * sizeof(float),
                          cudaMemcpyHostToDevice, st));
  }

  // ---- transformer blocks ------------------------------------------------
  const int kThreads = 256;
  auto blocks_for = [&](size_t total) {
    return (int)((total + kThreads - 1) / kThreads);
  };
  // RT_CUDA_KERNEL_PROF=1: wall split of the forward by kernel family
  // (proj GEMMs vs attention vs elementwise), printed per forward.
  static const bool kprof = std::getenv("RT_CUDA_KERNEL_PROF") != nullptr;
  cudaEvent_t ev[8];
  float sec_ms[6] = {0, 0, 0, 0, 0, 0};  // 0=proj, 1=attn, 2=elem, 3..5=attn col/feat/nbr
  if (kprof)
    for (auto& e : ev) RT_CU(cudaEventCreate(&e));
  auto mark = [&](int i) {
    if (kprof) RT_CU(cudaEventRecord(ev[i], st));
  };
  auto lap = [&](int sec, int a, int b) {
    if (!kprof) return;
    RT_CU(cudaEventSynchronize(ev[b]));
    float ms = 0;
    RT_CU(cudaEventElapsedTime(&ms, ev[a], ev[b]));
    sec_ms[sec] += ms;
  };
  // Producers write fp16 directly when the consumer projection runs on the
  // tensor-core GemmEx path — same rounding point, no separate f32->f16 pass.
  static const bool fuse_h16 = std::getenv("RT_CUDA_F16_QGEMM") == nullptr;
  for (int blk_i = 0; blk_i < kBlocks; blk_i++) {
    const BlockWeights& gw = ctx.blk[blk_i];
    for (int a = 0; a < 3; a++) {
      const bool h16a = fuse_h16 && gw.wqkvg[a].type == WType::F16;
      static const bool attn_v2_env = std::getenv("RT_CUDA_ATTN_V2") != nullptr;
      const bool h16o =
          fuse_h16 && !attn_v2_env && gw.wo[a].type == WType::F16;
      // Pre-norm + clear the attention output in one row pass. Uncovered
      // attention rows must read as zero in either output dtype.
      if (h16o) {
        k_rmsnorm_rows<true><<<(int)BS, 32, 0, st>>>(
            sl.x, sl.xn, gw.norm[a], kD, nullptr, h16a ? sl.xnh : nullptr,
            sl.atth);
      } else {
        k_rmsnorm_rows<true><<<(int)BS, 32, 0, st>>>(
            sl.x, sl.xn, gw.norm[a], kD, sl.att, h16a ? sl.xnh : nullptr);
      }
      mark(0);
      if (use_h16_qkvg)
        proj_h16(sl, gw.wqkvg[a], sl.qkvgh, (int)BS, sl.xnh);
      else
        proj(sl, sl.xn, gw.wqkvg[a], sl.qkvg, (int)BS, 0.f,
             h16a ? sl.xnh : nullptr);
      mark(1);
      lap(0, 0, 1);
      // Attention with fused QK-RMSNorm and output gating.
      // k_attn2 (lane-per-key layout) measured slower than the tiled kernel
      // on Blackwell; kept behind RT_CUDA_ATTN_V2 for future re-evaluation.
      static const bool attn_v2 = std::getenv("RT_CUDA_ATTN_V2") != nullptr;
      static const bool attn2_ok = [] {
        return cudaFuncSetAttribute(k_attn2,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kAttn2Smem) == cudaSuccess;
      }();
      if (!wflat[a].empty()) {
        if (attn_v2 && attn2_ok)
          k_attn2<<<(int)wflat[a].size(), 256, kAttn2Smem, st>>>(
              sl.qkvg, sl.att, sl.qidx[a], sl.kidx[a], sl.work[a],
              gw.head_scale[a], gw.q_norm[a], gw.k_norm[a]);
        else if (use_h16_qkvg)
          k_attn<__half><<<(int)wflat[a].size(), 256, 0, st>>>(
              sl.qkvgh, sl.att, sl.qidx[a], sl.kidx[a], sl.work[a],
              gw.head_scale[a], gw.q_norm[a], gw.k_norm[a],
              h16o ? sl.atth : nullptr);
        else
          k_attn<float><<<(int)wflat[a].size(), 256, 0, st>>>(
              sl.qkvg, sl.att, sl.qidx[a], sl.kidx[a], sl.work[a],
              gw.head_scale[a], gw.q_norm[a], gw.k_norm[a],
              h16o ? sl.atth : nullptr);
      }
      if (!mflat[a].empty()) {
        static const bool mma_ok_f32 = [] {
          return cudaFuncSetAttribute(
                     k_attn_mma<float>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     (int)kMmaSmem) == cudaSuccess;
        }();
        static const bool mma_ok_f16 = [] {
          return cudaFuncSetAttribute(
                     k_attn_mma<__half>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     (int)kMmaSmem) == cudaSuccess;
        }();
        const bool mma_ok = use_h16_qkvg ? mma_ok_f16 : mma_ok_f32;
        if (!mma_ok)
          throw std::runtime_error("rt/cuda: k_attn_mma smem opt-in failed "
                                   "(set RT_CUDA_ATTN_MMA=0)");
        if (use_h16_qkvg) {
          k_prep_kv<__half>
              <<<(int)kflat[a].size() + kMmaK, 256, 0, st>>>(
                  sl.qkvgh, sl.kabsd[a], (int)kflat[a].size(), gw.k_norm[a],
                  sl.kp, sl.vp);
          k_attn_mma<__half><<<(int)mflat[a].size(), 256, kMmaSmem, st>>>(
              sl.qkvgh, sl.att, sl.qidx[a], sl.kp, sl.vp, sl.mwork[a],
              gw.head_scale[a], gw.q_norm[a], h16o ? sl.atth : nullptr);
        } else {
          k_prep_kv<float>
              <<<(int)kflat[a].size() + kMmaK, 256, 0, st>>>(
                  sl.qkvg, sl.kabsd[a], (int)kflat[a].size(), gw.k_norm[a],
                  sl.kp, sl.vp);
          k_attn_mma<float><<<(int)mflat[a].size(), 256, kMmaSmem, st>>>(
              sl.qkvg, sl.att, sl.qidx[a], sl.kp, sl.vp, sl.mwork[a],
              gw.head_scale[a], gw.q_norm[a], h16o ? sl.atth : nullptr);
        }
      }
      if (!rflat[a].empty()) {           // flash split-K large groups
        if (use_h16_qkvg) {
          k_attn_part<__half><<<(int)pflat[a].size(), 256, 0, st>>>(
              sl.qkvgh, sl.partials, sl.qidx[a], sl.kidx[a], sl.pwork[a],
              gw.head_scale[a], gw.q_norm[a], gw.k_norm[a]);
          k_attn_reduce<__half><<<(int)rflat[a].size(), 128, 0, st>>>(
              sl.partials, sl.att, sl.qidx[a], sl.rwork[a], sl.qkvgh,
              h16o ? sl.atth : nullptr);
        } else {
          k_attn_part<float><<<(int)pflat[a].size(), 256, 0, st>>>(
              sl.qkvg, sl.partials, sl.qidx[a], sl.kidx[a], sl.pwork[a],
              gw.head_scale[a], gw.q_norm[a], gw.k_norm[a]);
          k_attn_reduce<float><<<(int)rflat[a].size(), 128, 0, st>>>(
              sl.partials, sl.att, sl.qidx[a], sl.rwork[a], sl.qkvg,
              h16o ? sl.atth : nullptr);
        }
      }
      mark(2);
      lap(1, 1, 2);
      lap(3 + a, 1, 2);
      proj(sl, sl.att, gw.wo[a], sl.x, (int)BS, 1.f,
           h16o ? sl.atth : nullptr);
      mark(3);
      lap(0, 2, 3);
    }
    // FFN: x += w2( silu(w1 xn) * w3 xn ), up-projection as one stacked GEMM
    const bool h16f = fuse_h16 && gw.w13.type == WType::F16;
    const bool h16w2 = fuse_h16 && gw.w2.type == WType::F16;
    k_rmsnorm_rows<false><<<(int)BS, 32, 0, st>>>(sl.x, sl.xn, gw.norm[3],
                                                  kD, nullptr,
                                                  h16f ? sl.xnh : nullptr);
    mark(4);
    if (packed_ffn) {
      if (use_h16_ff13)
        proj_h16(sl, gw.w13, sl.ff13h, (int)BS, sl.xnh);
      else
        proj(sl, sl.xn, gw.w13, sl.ff13, (int)BS, 0.f,
             h16f ? sl.xnh : nullptr);
      mark(5);
      lap(0, 4, 5);
      if (use_h16_ff13)
        k_swiglu_packed_h16<<<blocks_for(BS * kDFF), kThreads, 0, st>>>(
            sl.ff13h, sl.ffa, BS * kDFF, h16w2 ? sl.ffah : nullptr);
      else
        k_swiglu_packed<<<blocks_for(BS * kDFF), kThreads, 0, st>>>(
            sl.ff13, sl.ffa, BS * kDFF, h16w2 ? sl.ffah : nullptr);
    } else {
      proj(sl, sl.xn, gw.w1, sl.ffa, (int)BS, 0.f);
      proj(sl, sl.xn, gw.w3, sl.ffb, (int)BS, 0.f);
      mark(5);
      lap(0, 4, 5);
      k_swiglu<<<blocks_for(BS * kDFF), kThreads, 0, st>>>(sl.ffa, sl.ffb,
                                                           BS * kDFF);
    }
    mark(6);
    lap(2, 5, 6);
    proj(sl, sl.ffa, gw.w2, sl.x, (int)BS, 1.f,
         packed_ffn && h16w2 ? sl.ffah : nullptr);
    mark(7);
    lap(0, 6, 7);
    if (blk_i == 0 && debug_taps)
      RT_CU(cudaMemcpyAsync(sl.tap, sl.x, BS * kD * sizeof(float),
                            cudaMemcpyDeviceToDevice, st));
  }

  // ---- output norm + number head -----------------------------------------
  k_head<<<(int)BS, 32, 0, st>>>(sl.x, ctx.norm_out, ctx.dec_w, ctx.dec_b,
                                 sl.yhat);
  if (want_target_features) {
    grow(&sl.tgt, &sl.cap_tgt, BS);
    grow(&sl.tfeat, &sl.cap_tf, (size_t)B * kD);
    RT_CU(cudaMemcpyAsync(sl.tgt, out.sorted_is_target.data(), BS,
                          cudaMemcpyHostToDevice, st));
    k_target_feats<<<B, 32, 0, st>>>(sl.x, sl.tgt, ctx.norm_out, sl.tfeat,
                                     S);
    out.target_features.resize((size_t)B * kD);
    RT_CU(cudaMemcpyAsync(out.target_features.data(), sl.tfeat,
                          (size_t)B * kD * sizeof(float),
                          cudaMemcpyDeviceToHost, st));
  }

  RT_CU(cudaMemcpyAsync(out.yhat_number.data(), sl.yhat, BS * sizeof(float),
                        cudaMemcpyDeviceToHost, st));
  if (debug_taps) {
    out.x_block0.resize(BS * kD);
    RT_CU(cudaMemcpyAsync(out.x_block0.data(), sl.tap,
                          BS * kD * sizeof(float), cudaMemcpyDeviceToHost, st));
  }
  RT_CU(cudaStreamSynchronize(st));
  RT_CU(cudaGetLastError());
  if (kprof) {
    size_t nk_sum[3] = {0, 0, 0};
    for (int a = 0; a < 3; a++) {
      for (const auto& w : wflat[a]) nk_sum[a] += w.nk;
      for (const auto& w : mflat[a]) nk_sum[a] += w.nk;
      for (const auto& w : pflat[a]) nk_sum[a] += w.nk;
    }
    std::fprintf(stderr,
                 "[rt-cuda] BS=%zu proj=%.1fms attn=%.1fms elem=%.1fms | "
                 "col=%.1fms(%zuw/%zup nk=%zu) feat=%.1fms(%zuw/%zup nk=%zu) "
                 "nbr=%.1fms(%zuw/%zup nk=%zu)\n",
                 BS, sec_ms[0], sec_ms[1], sec_ms[2],
                 sec_ms[3], wflat[0].size() + mflat[0].size(), pflat[0].size(), nk_sum[0],
                 sec_ms[4], wflat[1].size() + mflat[1].size(), pflat[1].size(), nk_sum[1],
                 sec_ms[5], wflat[2].size() + mflat[2].size(), pflat[2].size(), nk_sum[2]);
    for (auto& e : ev) cudaEventDestroy(e);
  }
}

}  // namespace detail
}  // namespace rt
