// rt_cuda.cu — CUDA backend for RT-J inference (mirror of the Metal design).
//
// fp32 dense projections run as cuBLAS SGEMMs (row-major y = x W^T via the
// col-major transpose trick); wo and w2 accumulate into the residual stream
// with beta=1. f16/q8/q4 projections run a custom qgemm (32x32 output tiles,
// shared-memory-staged in-register dequant — the fp32 weight tile never
// exists in DRAM), weights uploaded quantized-resident. Custom kernels
// handle the rest:
//  - rmsnorm_rows: one warp per row (pre-norms, d=512)
//  - qknorm: in-place QK-RMSNorm per (token, head) on the fused qkvg buffer
//  - attn: one block per (group, query-tile) work item; each warp owns a
//    (query, head) pair and streams the shared key list with a single-pass
//    online softmax — the same query-group sparsity as the CPU/MPS paths,
//    with the identical bf16-rounded log(kv) query scaling
//  - gate_mul / swiglu / head: elementwise gating, SwiGLU, fused output head
// Weights are uploaded once per model; activation/index buffers grow on
// demand and are reused. Forwards on one model are serialized by the ctx.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

// out[row] = rmsnorm(in[row]) * scale, rows of length n. One warp per row.
__global__ void k_rmsnorm_rows(const float* __restrict__ in,
                               float* __restrict__ out,
                               const float* __restrict__ scale, int n) {
  int row = blockIdx.x;
  int lane = threadIdx.x;
  const float* x = in + (size_t)row * n;
  float* y = out + (size_t)row * n;
  float ss = 0.f;
  for (int i = lane; i < n; i += 32) ss += x[i] * x[i];
  ss = warp_sum(ss);
  float inv = rsqrtf(ss / n + kNormEps);
  for (int i = lane; i < n; i += 32) y[i] = x[i] * inv * scale[i];
}

// In-place QK-RMSNorm on the fused qkvg buffer (row = [q|k|v|g]).
// One warp per (token, head, q-or-k) segment of 64 floats.
__global__ void k_qknorm(float* __restrict__ qkvg,
                         const float* __restrict__ q_scale,
                         const float* __restrict__ k_scale) {
  int tg = blockIdx.x;
  int lane = threadIdx.x;
  int token = tg / 16, seg = tg % 16;
  int head = seg / 2, isk = seg % 2;
  float* x = qkvg + (size_t)token * kC4 + isk * kD + head * kHeadDim;
  const float* scale = isk ? k_scale : q_scale;
  float a = x[lane], b = x[lane + 32];
  float ss = warp_sum(a * a + b * b);
  float inv = rsqrtf(ss / kHeadDim + kNormEps);
  x[lane] = a * inv * scale[lane];
  x[lane + 32] = b * inv * scale[lane + 32];
}

// One block per work item; each warp streams the key list for a
// (query, head) pair with an online softmax. q/k already QK-normed.
__global__ void k_attn(const float* __restrict__ qkvg, float* __restrict__ att,
                       const int* __restrict__ qidx,
                       const int* __restrict__ kidx,
                       const AttnWorkGpu* __restrict__ work,
                       const float* __restrict__ head_scale) {
  const AttnWorkGpu w = work[blockIdx.x];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  int nwarp = blockDim.x / 32;
  for (int p = warp; p < w.tq * kHeads; p += nwarp) {
    int r = p / kHeads, h = p % kHeads;
    size_t qrow = (size_t)(w.rowbase + qidx[w.qstart + r]);
    const float* q = qkvg + qrow * kC4 + h * kHeadDim;
    float qscale = head_scale[h] * w.logkv / kHeadDim;
    float q0 = q[2 * lane] * qscale;
    float q1 = q[2 * lane + 1] * qscale;
    float mx = -INFINITY, den = 0.f, a0 = 0.f, a1 = 0.f;
    for (int j = 0; j < w.nk; j++) {
      size_t krow = (size_t)(w.rowbase + kidx[w.kstart + j]);
      const float* k = qkvg + krow * kC4 + kD + h * kHeadDim;
      float score = warp_sum(q0 * k[2 * lane] + q1 * k[2 * lane + 1]);
      const float* v = k + kD;
      float nm = fmaxf(mx, score);
      float corr = expf(mx - nm);
      float wt = expf(score - nm);
      den = den * corr + wt;
      a0 = a0 * corr + wt * v[2 * lane];
      a1 = a1 * corr + wt * v[2 * lane + 1];
      mx = nm;
    }
    float* o = att + qrow * kD + h * kHeadDim;
    o[2 * lane] = a0 / den;
    o[2 * lane + 1] = a1 / den;
  }
}

// att *= 2*sigmoid(g), g = qkvg[token, 3D + d].
__global__ void k_gate_mul(float* __restrict__ att,
                           const float* __restrict__ qkvg, size_t total) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  size_t token = gid / kD, d = gid % kD;
  float g = qkvg[token * kC4 + 3 * kD + d];
  att[gid] *= 2.f / (1.f + expf(-g));
}

// ffa = silu(ffa) * ffb.
__global__ void k_swiglu(float* __restrict__ ffa,
                         const float* __restrict__ ffb, size_t total) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  float a = ffa[gid];
  ffa[gid] = (a / (1.f + expf(-a))) * ffb[gid];
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
  GpuWeight w1, w2, w3;
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
  float *ffa = nullptr, *ffb = nullptr, *yhat = nullptr, *tap = nullptr;
  int *qidx[3] = {}, *kidx[3] = {};
  AttnWorkGpu* work[3] = {};
  size_t cap_bs = 0, cap_q[3] = {}, cap_k[3] = {}, cap_w[3] = {};
  std::vector<void*> owned;            // every cudaMalloc for cleanup

  ~CudaCtx() {
    for (void* p : owned) cudaFree(p);
    for (float* p : {x, xn, qkvg, att, ffa, ffb, yhat, tap}) cudaFree(p);
    for (int a = 0; a < 3; a++) {
      cudaFree(qidx[a]);
      cudaFree(kidx[a]);
      cudaFree(work[a]);
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
      g.w1 = upload_weight(ctx, blk.w1);
      g.w2 = upload_weight(ctx, blk.w2);
      g.w3 = upload_weight(ctx, blk.w3);
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

// Projection dispatch: fp32 uses cuBLAS SGEMM; f16/q8/q4 run the custom
// qgemm with the weight kept quantized-resident. beta is only ever 0 or 1.
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
      if (acc)
        k_qgemm<2, true><<<grid, block, 0, st>>>(x, w.q, w.s, y, M, w.out,
                                                 w.in);
      else
        k_qgemm<2, false><<<grid, block, 0, st>>>(x, w.q, w.s, y, M, w.out,
                                                  w.in);
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
  std::lock_guard<std::mutex> lk(ctx.mu);
  cudaStream_t st = ctx.stream;

  const int B = prep.B, S = prep.S;
  const size_t BS = (size_t)B * S;

  // ---- flatten group indices / work items for the GPU --------------------
  std::vector<int32_t> qflat[3], kflat[3];
  std::vector<AttnWorkGpu> wflat[3];
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
    wflat[a].reserve(prep.work[a].size());
    for (const Work& W : prep.work[a]) {
      const Groups& G = (*gsets[a])[W.b];
      wflat[a].push_back({qbase[W.b] + G.qoff[W.g] + W.q0, W.q1 - W.q0,
                          kbase[W.b] + G.koff[W.g],
                          G.koff[W.g + 1] - G.koff[W.g], W.b * S, W.logkv});
    }
  }

  // ---- buffers -----------------------------------------------------------
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
    for (float** p : {&ctx.ffa, &ctx.ffb}) {
      if (*p) RT_CU(cudaFree(*p));
      *p = nullptr;
      RT_CU(cudaMalloc(p, BS * (size_t)kDFF * sizeof(float)));
    }
    if (ctx.yhat) RT_CU(cudaFree(ctx.yhat));
    ctx.yhat = nullptr;
    RT_CU(cudaMalloc(&ctx.yhat, BS * sizeof(float)));
    ctx.cap_bs = BS;
  }
  for (int a = 0; a < 3; a++) {
    grow(&ctx.qidx[a], &ctx.cap_q[a], std::max<size_t>(1, qflat[a].size()));
    grow(&ctx.kidx[a], &ctx.cap_k[a], std::max<size_t>(1, kflat[a].size()));
    grow(&ctx.work[a], &ctx.cap_w[a], std::max<size_t>(1, wflat[a].size()));
    RT_CU(cudaMemcpyAsync(ctx.qidx[a], qflat[a].data(), qflat[a].size() * 4,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(ctx.kidx[a], kflat[a].data(), kflat[a].size() * 4,
                          cudaMemcpyHostToDevice, st));
    RT_CU(cudaMemcpyAsync(ctx.work[a], wflat[a].data(),
                          wflat[a].size() * sizeof(AttnWorkGpu),
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
      k_rmsnorm_rows<<<(int)BS, 32, 0, st>>>(ctx.x, ctx.xn, gw.norm[a], kD);
      proj(ctx, ctx.xn, gw.wqkvg[a], ctx.qkvg, (int)BS, 0.f);
      k_qknorm<<<(int)BS * 16, 32, 0, st>>>(ctx.qkvg, gw.q_norm[a],
                                            gw.k_norm[a]);
      RT_CU(cudaMemsetAsync(ctx.att, 0, BS * kD * sizeof(float), st));
      if (!wflat[a].empty())
        k_attn<<<(int)wflat[a].size(), 128, 0, st>>>(
            ctx.qkvg, ctx.att, ctx.qidx[a], ctx.kidx[a], ctx.work[a],
            gw.head_scale[a]);
      k_gate_mul<<<blocks_for(BS * kD), kThreads, 0, st>>>(ctx.att, ctx.qkvg,
                                                           BS * kD);
      proj(ctx, ctx.att, gw.wo[a], ctx.x, (int)BS, 1.f);
    }
    // FFN: x += w2( silu(w1 xn) * w3 xn )
    k_rmsnorm_rows<<<(int)BS, 32, 0, st>>>(ctx.x, ctx.xn, gw.norm[3], kD);
    proj(ctx, ctx.xn, gw.w1, ctx.ffa, (int)BS, 0.f);
    proj(ctx, ctx.xn, gw.w3, ctx.ffb, (int)BS, 0.f);
    k_swiglu<<<blocks_for(BS * kDFF), kThreads, 0, st>>>(ctx.ffa, ctx.ffb,
                                                         BS * kDFF);
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
