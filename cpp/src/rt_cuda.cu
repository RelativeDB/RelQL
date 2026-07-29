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
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
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
                               float* __restrict__ clear) {
  int row = blockIdx.x;
  int lane = threadIdx.x;
  const float* x = in + (size_t)row * n;
  float* y = out + (size_t)row * n;
  float ss = 0.f;
  for (int i = lane; i < n; i += 32) ss += x[i] * x[i];
  ss = warp_sum(ss);
  float inv = rsqrtf(ss / n + kNormEps);
  for (int i = lane; i < n; i += 32) {
    y[i] = x[i] * inv * scale[i];
    if (CLEAR) clear[(size_t)row * n + i] = 0.f;
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
__global__ void k_attn(const float* __restrict__ qkvg, float* __restrict__ att,
                       const int* __restrict__ qidx,
                       const int* __restrict__ kidx,
                       const AttnWorkGpu* __restrict__ work,
                       const float* __restrict__ head_scale,
                       const float* __restrict__ q_norm,
                       const float* __restrict__ k_norm) {
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
      const float* q = qkvg + qrowi * kC4 + h * kHeadDim;
      float r0 = q[2 * lane], r1 = q[2 * lane + 1];
      float inv = rsqrtf(warp_sum(r0 * r0 + r1 * r1) / kHeadDim + kNormEps);
      float qscale = head_scale[h] * w.logkv / kHeadDim;
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
    for (int j = 0; j < kn; j++) {
      for (int s = 0; s < kMQ; s++) {
        int p = sg + s * 8;
        if (p >= npair) continue;
        int h = p % kHeads;
        const float* k = Kt + j * kD + h * kHeadDim;
        float score = warp_sum(q0v[s] * k[2 * lane] + q1v[s] * k[2 * lane + 1]);
        const float* v = Vt + j * kD + h * kHeadDim;
        float nm = fmaxf(mx[s], score);
        float corr = expf(mx[s] - nm);
        float wt = expf(score - nm);
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
    size_t grow = (size_t)(w.rowbase + qidx[w.qstart + r]) * kC4 + 3 * kD +
                  h * kHeadDim;
    float g0 = 2.f / (1.f + expf(-qkvg[grow + 2 * lane]));
    float g1 = 2.f / (1.f + expf(-qkvg[grow + 2 * lane + 1]));
    float* o = att + orow[s];
    o[2 * lane] = a0[s] / den[s] * g0;
    o[2 * lane + 1] = a1[s] / den[s] * g1;
  }
}

// Flash split-K attention for long key lists: same tiled streaming as
// k_attn, but only over this work item's key chunk, emitting per-
// (query, head, chunk) partials {running max m, denom l, unnormalized
// weighted-V sum o[64]} at float offset w.part + (r*8+h)*66.
__global__ void k_attn_part(const float* __restrict__ qkvg,
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
      const float* q = qkvg + qrowi * kC4 + h * kHeadDim;
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
        float corr = expf(mx[s] - nm);
        float wt = expf(score - nm);
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
__global__ void k_attn_reduce(const float* __restrict__ partials,
                              float* __restrict__ att,
                              const int* __restrict__ qidx,
                              const AttnRWorkGpu* __restrict__ work,
                              const float* __restrict__ qkvg) {
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
      float f = expf(pc[0] - M);
      l += pc[1] * f;
      o0 += pc[2 + 2 * lane] * f;
      o1 += pc[2 + 2 * lane + 1] * f;
    }
    size_t qrowi = (size_t)(w.rowbase + qidx[w.qstart + r]);
    size_t grow = qrowi * kC4 + 3 * kD + h * kHeadDim;
    float g0 = 2.f / (1.f + expf(-qkvg[grow + 2 * lane]));
    float g1 = 2.f / (1.f + expf(-qkvg[grow + 2 * lane + 1]));
    float* o = att + qrowi * kD + h * kHeadDim;
    o[2 * lane] = o0 / l * g0;
    o[2 * lane + 1] = o1 / l * g1;
  }
}

// ffa = silu(ffa) * ffb (separate w1/w3 fallback when dtypes differ).
__global__ void k_swiglu(float* __restrict__ ffa,
                         const float* __restrict__ ffb, size_t total) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  float a = ffa[gid];
  ffa[gid] = (a / (1.f + expf(-a))) * ffb[gid];
}

// ffa[row, d] = silu(ff13[row, d]) * ff13[row, kDFF + d] — SwiGLU on the
// stacked [w1; w3] GEMM output.
__global__ void k_swiglu_packed(const float* __restrict__ ff13,
                                float* __restrict__ ffa, size_t total) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  size_t row = gid / kDFF, d = gid % kDFF;
  float a = ff13[row * (2 * kDFF) + d];
  float b = ff13[row * (2 * kDFF) + kDFF + d];
  ffa[gid] = (a / (1.f + expf(-a))) * b;
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

struct CudaCtx {
  std::mutex mu;                       // serializes forwards on this model
  cublasHandle_t blas = nullptr;
  cudaStream_t stream = nullptr;
  BlockWeights blk[kBlocks] = {};
  float *norm_out = nullptr, *dec_w = nullptr;
  float dec_b = 0.f;
  // grow-on-demand activation / index buffers
  float *x = nullptr, *xn = nullptr, *qkvg = nullptr, *att = nullptr;
  float *ffa = nullptr, *ffb = nullptr, *ff13 = nullptr;
  float *yhat = nullptr, *tap = nullptr;
  int8_t* xq = nullptr;                // int8 activations for Q8 projections
  float* xqs = nullptr;                // per-row activation scales
  int *qidx[3] = {}, *kidx[3] = {};
  AttnWorkGpu* work[3] = {};
  AttnPWorkGpu* pwork[3] = {};         // flash split-K chunk items
  AttnRWorkGpu* rwork[3] = {};         // flash split-K reduce items
  float* partials = nullptr;           // split-K partial {m, l, o[64]} states
  size_t cap_bs = 0, cap_q[3] = {}, cap_k[3] = {}, cap_w[3] = {};
  size_t cap_pw[3] = {}, cap_rw[3] = {}, cap_part = 0;
  std::vector<void*> owned;            // every cudaMalloc for cleanup

  ~CudaCtx() {
    for (void* p : owned) cudaFree(p);
    for (float* p : {x, xn, qkvg, att, ffa, ffb, ff13, yhat, tap, xqs,
                     partials})
      cudaFree(p);
    cudaFree(xq);
    for (int a = 0; a < 3; a++) {
      cudaFree(qidx[a]);
      cudaFree(kidx[a]);
      cudaFree(work[a]);
      cudaFree(pwork[a]);
      cudaFree(rwork[a]);
    }
    if (blas) cublasDestroy(blas);
    if (stream) cudaStreamDestroy(stream);
  }
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
    RT_CU(cudaStreamCreate(&ctx->stream));
    RT_CUBLAS(cublasCreate(&ctx->blas));
    RT_CUBLAS(cublasSetStream(ctx->blas, ctx->stream));
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
    return ctx;
  } catch (...) {
    delete ctx;
    throw;
  }
}

// y[M,N] = x[M,K] @ W[N,K]^T (+ beta * y), all row-major, via the col-major
// transpose identity: y_cm[N,M] = W_cm^T[N,K] @ x_cm[K,M].
void gemm(CudaCtx& ctx, const float* x, const float* w, float* y, int M, int N,
          int K, float beta) {
  const float alpha = 1.f;
  RT_CUBLAS(cublasSgemm(ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w,
                        K, x, K, &beta, y, N));
}

// Projection dispatch: fp32 uses cuBLAS SGEMM; q8 quantizes activations on
// device and runs the true-int8 dp4a GEMM (mirrors the CPU SDOT path);
// f16/q4 run the dequant-in-register qgemm. Weights stay quantized-resident.
// beta is only ever 0 or 1.
void proj(CudaCtx& ctx, const float* x, const GpuWeight& w, float* y, int M,
          float beta) {
  if (w.type == WType::F32) {
    gemm(ctx, x, w.f32, y, M, w.out, w.in, beta);
    return;
  }
  const dim3 grid((unsigned)(w.out / 32), (unsigned)((M + 31) / 32));
  const dim3 block(32, 8);
  const bool acc = beta != 0.f;
  cudaStream_t st = ctx.stream;
  switch (w.type) {
    case WType::F16:
      if (acc)
        k_qgemm<1, true><<<grid, block, 0, st>>>(x, w.q, w.s, y, M, w.out,
                                                 w.in);
      else
        k_qgemm<1, false><<<grid, block, 0, st>>>(x, w.q, w.s, y, M, w.out,
                                                  w.in);
      break;
    case WType::Q8:
      k_quant_rows<<<M, 32, 0, st>>>(x, ctx.xq, ctx.xqs, w.in);
      if (acc)
        k_qgemm_i8<true><<<grid, block, 0, st>>>(ctx.xq, ctx.xqs, w.q, w.s, y,
                                                 M, w.out, w.in);
      else
        k_qgemm_i8<false><<<grid, block, 0, st>>>(ctx.xq, ctx.xqs, w.q, w.s,
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
                     bool debug_taps) {
  // ---- lazy per-model context --------------------------------------------
  static std::mutex init_mu;
  std::shared_ptr<void>& slot = m.device_ctx[(int)Device::CUDA];
  {
    std::lock_guard<std::mutex> lk(init_mu);
    if (!slot) slot.reset(make_ctx(m), [](void* p) { delete (CudaCtx*)p; });
  }
  CudaCtx& ctx = *(CudaCtx*)slot.get();

  const int B = prep.B, S = prep.S;
  const size_t BS = (size_t)B * S;

  // ---- flatten group indices / work items for the GPU --------------------
  // Pure host work on prep — runs before taking the ctx lock so concurrent
  // forwards overlap their CPU flattening with another forward's GPU time.
  // Small groups (nk <= kFlashSplit) run the single-pass tiled kernel; large
  // groups split into kFlashChunk-key chunks for attn_part -> attn_reduce.
  // Both tiled kernels take at most kMQ queries per item (per-pair softmax
  // state lives in registers), so prep's kQTile items are sub-tiled here.
  std::vector<int32_t> qflat[3], kflat[3];
  std::vector<AttnWorkGpu> wflat[3];
  std::vector<AttnPWorkGpu> pflat[3];
  std::vector<AttnRWorkGpu> rflat[3];
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
    qflat[a].reserve(q); kflat[a].reserve(k);
    for (int b = 0; b < B; b++) {
      const Groups& G = (*gsets[a])[b];
      qflat[a].insert(qflat[a].end(), G.q.begin(), G.q.end());
      kflat[a].insert(kflat[a].end(), G.k.begin(), G.k.end());
    }
    size_t poff = 0;
    for (const Work& W : prep.work[a]) {
      const Groups& G = (*gsets[a])[W.b];
      const int qs = qbase[W.b] + G.qoff[W.g] + W.q0;
      const int tq = W.q1 - W.q0;
      const int ks = kbase[W.b] + G.koff[W.g];
      const int nk = G.koff[W.g + 1] - G.koff[W.g];
      if (nk <= kFlashSplit) {
        for (int sub = 0; sub < tq; sub += kMQ)
          wflat[a].push_back({qs + sub, std::min(kMQ, tq - sub), ks, nk,
                              W.b * S, W.logkv});
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

  std::lock_guard<std::mutex> lk(ctx.mu);
  cudaStream_t st = ctx.stream;

  // ---- buffers -----------------------------------------------------------
  const bool packed_ffn = ctx.blk[0].w13.out != 0;
  if (ctx.cap_bs < BS) {
    for (float** p : {&ctx.x, &ctx.xn, &ctx.att, &ctx.tap}) {
      if (*p) RT_CU(cudaFree(*p));
      *p = nullptr;
      RT_CU(cudaMalloc(p, BS * kD * sizeof(float)));
    }
    for (float** p : {&ctx.qkvg}) {
      if (*p) RT_CU(cudaFree(*p));
      *p = nullptr;
      RT_CU(cudaMalloc(p, BS * (size_t)kC4 * sizeof(float)));
    }
    if (packed_ffn) {
      for (float** p : {&ctx.ffa}) {
        if (*p) RT_CU(cudaFree(*p));
        *p = nullptr;
        RT_CU(cudaMalloc(p, BS * (size_t)kDFF * sizeof(float)));
      }
      if (ctx.ff13) RT_CU(cudaFree(ctx.ff13));
      ctx.ff13 = nullptr;
      RT_CU(cudaMalloc(&ctx.ff13, BS * (size_t)(2 * kDFF) * sizeof(float)));
    } else {
      for (float** p : {&ctx.ffa, &ctx.ffb}) {
        if (*p) RT_CU(cudaFree(*p));
        *p = nullptr;
        RT_CU(cudaMalloc(p, BS * (size_t)kDFF * sizeof(float)));
      }
    }
    if (ctx.yhat) RT_CU(cudaFree(ctx.yhat));
    ctx.yhat = nullptr;
    RT_CU(cudaMalloc(&ctx.yhat, BS * sizeof(float)));
    if (ctx.xq) RT_CU(cudaFree(ctx.xq));
    ctx.xq = nullptr;
    RT_CU(cudaMalloc(&ctx.xq, BS * (size_t)kDFF));
    if (ctx.xqs) RT_CU(cudaFree(ctx.xqs));
    ctx.xqs = nullptr;
    RT_CU(cudaMalloc(&ctx.xqs, BS * sizeof(float)));
    ctx.cap_bs = BS;
  }
  grow(&ctx.partials, &ctx.cap_part, part_max);
  for (int a = 0; a < 3; a++) {
    grow(&ctx.qidx[a], &ctx.cap_q[a], std::max<size_t>(1, qflat[a].size()));
    grow(&ctx.kidx[a], &ctx.cap_k[a], std::max<size_t>(1, kflat[a].size()));
    grow(&ctx.work[a], &ctx.cap_w[a], std::max<size_t>(1, wflat[a].size()));
    grow(&ctx.pwork[a], &ctx.cap_pw[a], std::max<size_t>(1, pflat[a].size()));
    grow(&ctx.rwork[a], &ctx.cap_rw[a], std::max<size_t>(1, rflat[a].size()));
    RT_CU(cudaMemcpyAsync(ctx.qidx[a], qflat[a].data(), qflat[a].size() * 4,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(ctx.kidx[a], kflat[a].data(), kflat[a].size() * 4,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(ctx.work[a], wflat[a].data(),
                          wflat[a].size() * sizeof(AttnWorkGpu),
                          cudaMemcpyHostToDevice, st));
    if (!pflat[a].empty())
      RT_CU(cudaMemcpyAsync(ctx.pwork[a], pflat[a].data(),
                            pflat[a].size() * sizeof(AttnPWorkGpu),
                            cudaMemcpyHostToDevice, st));
    if (!rflat[a].empty())
      RT_CU(cudaMemcpyAsync(ctx.rwork[a], rflat[a].data(),
                            rflat[a].size() * sizeof(AttnRWorkGpu),
                            cudaMemcpyHostToDevice, st));
  }
  RT_CU(cudaMemcpyAsync(ctx.x, prep.x.data(), BS * kD * sizeof(float),
                        cudaMemcpyHostToDevice, st));

  // ---- transformer blocks ------------------------------------------------
  const int kThreads = 256;
  auto blocks_for = [&](size_t total) {
    return (int)((total + kThreads - 1) / kThreads);
  };
  for (int blk_i = 0; blk_i < kBlocks; blk_i++) {
    const BlockWeights& gw = ctx.blk[blk_i];
    for (int a = 0; a < 3; a++) {
      // Pre-norm + clear the attention output in one row pass.
      k_rmsnorm_rows<true><<<(int)BS, 32, 0, st>>>(ctx.x, ctx.xn, gw.norm[a],
                                                   kD, ctx.att);
      proj(ctx, ctx.xn, gw.wqkvg[a], ctx.qkvg, (int)BS, 0.f);
      // Attention with fused QK-RMSNorm and output gating.
      if (!wflat[a].empty())
        k_attn<<<(int)wflat[a].size(), 256, 0, st>>>(
            ctx.qkvg, ctx.att, ctx.qidx[a], ctx.kidx[a], ctx.work[a],
            gw.head_scale[a], gw.q_norm[a], gw.k_norm[a]);
      if (!rflat[a].empty()) {           // flash split-K large groups
        k_attn_part<<<(int)pflat[a].size(), 256, 0, st>>>(
            ctx.qkvg, ctx.partials, ctx.qidx[a], ctx.kidx[a], ctx.pwork[a],
            gw.head_scale[a], gw.q_norm[a], gw.k_norm[a]);
        k_attn_reduce<<<(int)rflat[a].size(), 128, 0, st>>>(
            ctx.partials, ctx.att, ctx.qidx[a], ctx.rwork[a], ctx.qkvg);
      }
      proj(ctx, ctx.att, gw.wo[a], ctx.x, (int)BS, 1.f);
    }
    // FFN: x += w2( silu(w1 xn) * w3 xn ), up-projection as one stacked GEMM
    k_rmsnorm_rows<false><<<(int)BS, 32, 0, st>>>(ctx.x, ctx.xn, gw.norm[3],
                                                  kD, nullptr);
    if (packed_ffn) {
      proj(ctx, ctx.xn, gw.w13, ctx.ff13, (int)BS, 0.f);
      k_swiglu_packed<<<blocks_for(BS * kDFF), kThreads, 0, st>>>(
          ctx.ff13, ctx.ffa, BS * kDFF);
    } else {
      proj(ctx, ctx.xn, gw.w1, ctx.ffa, (int)BS, 0.f);
      proj(ctx, ctx.xn, gw.w3, ctx.ffb, (int)BS, 0.f);
      k_swiglu<<<blocks_for(BS * kDFF), kThreads, 0, st>>>(ctx.ffa, ctx.ffb,
                                                           BS * kDFF);
    }
    proj(ctx, ctx.ffa, gw.w2, ctx.x, (int)BS, 1.f);
    if (blk_i == 0 && debug_taps)
      RT_CU(cudaMemcpyAsync(ctx.tap, ctx.x, BS * kD * sizeof(float),
                            cudaMemcpyDeviceToDevice, st));
  }

  // ---- output norm + number head -----------------------------------------
  k_head<<<(int)BS, 32, 0, st>>>(ctx.x, ctx.norm_out, ctx.dec_w, ctx.dec_b,
                                 ctx.yhat);

  RT_CU(cudaMemcpyAsync(out.yhat_number.data(), ctx.yhat, BS * sizeof(float),
                        cudaMemcpyDeviceToHost, st));
  if (debug_taps) {
    out.x_block0.resize(BS * kD);
    RT_CU(cudaMemcpyAsync(out.x_block0.data(), ctx.tap,
                          BS * kD * sizeof(float), cudaMemcpyDeviceToHost, st));
  }
  RT_CU(cudaStreamSynchronize(st));
  RT_CU(cudaGetLastError());
}

}  // namespace detail
}  // namespace rt
