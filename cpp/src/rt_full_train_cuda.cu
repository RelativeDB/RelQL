// rt_full_train_cuda.cu -- end-to-end RT-J optimization on CUDA.
//
// Port of rt_full_train_metal.mm. Dense forward/backward/weight-gradient
// products run as cuBLAS SGEMMs; relational attention stays sparse and is
// differentiated by custom kernels over the exact query/key groups produced
// by detail::prepare -- no SxS mask is materialized. Where Metal accumulated
// into MSL 3.0 float atomics, CUDA uses native atomicAdd, which every
// supported device has, so there is no capability gate here. Where Metal
// leaned on unified memory and ARC, this file copies scalars/params
// explicitly and frees each step's activation tape from a scratch arena.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "rt_internal.hpp"
#include "rt_train.hpp"

namespace rt {
namespace {

constexpr int D = kDModel;
constexpr int H = kHeads;
constexpr int HD = kHeadDim;
constexpr float EPS = 1e-6f;

struct WorkGpu { int qs, nq, ks, nk, base; float logkv; };
struct AdamArgs { uint32_t n, step; float lr, wd, b1, b2, eps, clip; };

#define RT_CU(call)                                                     \
  do {                                                                  \
    cudaError_t e_ = (call);                                            \
    if (e_ != cudaSuccess)                                              \
      throw std::runtime_error(std::string("rt/full-train-cuda: ") +    \
                               cudaGetErrorString(e_));                 \
  } while (0)

#define RT_CUBLAS(call)                                                 \
  do {                                                                  \
    cublasStatus_t s_ = (call);                                         \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                    \
      throw std::runtime_error("rt/full-train-cuda: cublas error " +    \
                               std::to_string((int)s_));                \
  } while (0)

__device__ inline float wsum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// One warp per row.
__global__ void k_rms_fwd(const float* __restrict__ x,
                          const float* __restrict__ s, float* __restrict__ y,
                          uint32_t n) {
  uint32_t row = blockIdx.x, lane = threadIdx.x;
  x += (size_t)row * n; y += (size_t)row * n;
  float ss = 0;
  for (uint32_t i = lane; i < n; i += 32) ss += x[i] * x[i];
  float inv = rsqrtf(wsum(ss) / (float)n + EPS);
  for (uint32_t i = lane; i < n; i += 32) y[i] = x[i] * inv * s[i];
}

__global__ void k_rms_dx(const float* __restrict__ x,
                         const float* __restrict__ s,
                         const float* __restrict__ dy, float* __restrict__ dx,
                         uint32_t n) {
  uint32_t row = blockIdx.x, lane = threadIdx.x;
  x += (size_t)row * n; dy += (size_t)row * n; dx += (size_t)row * n;
  float ss = 0, dot = 0;
  for (uint32_t i = lane; i < n; i += 32) {
    ss += x[i] * x[i];
    dot += dy[i] * s[i] * x[i];
  }
  ss = wsum(ss); dot = wsum(dot);
  float inv = rsqrtf(ss / (float)n + EPS);
  float c = dot * inv * inv / (float)n;
  for (uint32_t i = lane; i < n; i += 32) dx[i] += inv * (dy[i] * s[i] - x[i] * c);
}

__global__ void k_rms_ds(const float* __restrict__ y,
                         const float* __restrict__ s,
                         const float* __restrict__ dy, float* __restrict__ ds,
                         uint32_t rows, uint32_t n) {
  uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= n) return;
  float z = 0, invs = 1.0f / s[d];
  for (uint32_t r = 0; r < rows; r++)
    z += dy[(size_t)r * n + d] * y[(size_t)r * n + d] * invs;
  ds[d] += z;
}

// One warp per 64-float (token, head) segment.
__global__ void k_qnorm_fwd(const float* __restrict__ x,
                            const float* __restrict__ s,
                            float* __restrict__ y) {
  uint32_t seg = blockIdx.x, lane = threadIdx.x;
  const float* a = x + (size_t)seg * 64;
  float* b = y + (size_t)seg * 64;
  float u = a[lane], v = a[lane + 32];
  float inv = rsqrtf(wsum(u * u + v * v) / 64.0f + EPS);
  b[lane] = u * inv * s[lane];
  b[lane + 32] = v * inv * s[lane + 32];
}

__global__ void k_qnorm_dx(const float* __restrict__ x,
                           const float* __restrict__ s,
                           const float* __restrict__ dy,
                           float* __restrict__ dx) {
  uint32_t seg = blockIdx.x, lane = threadIdx.x;
  x += (size_t)seg * 64; dy += (size_t)seg * 64; dx += (size_t)seg * 64;
  float u = x[lane], v = x[lane + 32];
  float inv = rsqrtf(wsum(u * u + v * v) / 64.0f + EPS);
  float dot = wsum(dy[lane] * s[lane] * u + dy[lane + 32] * s[lane + 32] * v);
  float c = dot * inv * inv / 64.0f;
  dx[lane] += inv * (dy[lane] * s[lane] - u * c);
  dx[lane + 32] += inv * (dy[lane + 32] * s[lane + 32] - v * c);
}

__global__ void k_qnorm_ds(const float* __restrict__ y,
                           const float* __restrict__ s,
                           const float* __restrict__ dy,
                           float* __restrict__ ds, uint32_t nseg) {
  uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= 64) return;
  float z = 0, invs = 1.0f / s[d];
  for (uint32_t seg = 0; seg < nseg; seg++)
    z += dy[(size_t)seg * 64 + d] * y[(size_t)seg * 64 + d] * invs;
  ds[d] += z;
}

// One warp per work item, streaming pairs like the Metal training kernel.
__global__ void k_attn_fwd(const float* __restrict__ q,
                           const float* __restrict__ k,
                           const float* __restrict__ v, float* __restrict__ out,
                           const int* __restrict__ qi,
                           const int* __restrict__ ki,
                           const WorkGpu* __restrict__ works,
                           const float* __restrict__ hs) {
  WorkGpu w = works[blockIdx.x];
  uint32_t lane = threadIdx.x;
  for (uint32_t p = 0; p < (uint32_t)w.nq * 8; p++) {
    uint32_t r = p / 8, h = p % 8, qr = w.base + qi[w.qs + r];
    float sc = hs[h] * w.logkv / 64.0f;
    const float* qq = q + (size_t)qr * 512 + h * 64;
    float q0 = qq[2 * lane] * sc, q1 = qq[2 * lane + 1] * sc;
    float m = -INFINITY, z = 0, o0 = 0, o1 = 0;
    for (int j = 0; j < w.nk; j++) {
      uint32_t kr = w.base + ki[w.ks + j];
      const float* kk = k + (size_t)kr * 512 + h * 64;
      float score = wsum(q0 * kk[2 * lane] + q1 * kk[2 * lane + 1]);
      float nm = fmaxf(m, score), a = expf(m - nm), b = expf(score - nm);
      z = z * a + b;
      const float* vv = v + (size_t)kr * 512 + h * 64;
      o0 = o0 * a + b * vv[2 * lane];
      o1 = o1 * a + b * vv[2 * lane + 1];
      m = nm;
    }
    float* oo = out + (size_t)qr * 512 + h * 64;
    oo[2 * lane] = o0 / z;
    oo[2 * lane + 1] = o1 / z;
  }
}

__global__ void k_attn_bwd(const float* __restrict__ q,
                           const float* __restrict__ k,
                           const float* __restrict__ v,
                           const float* __restrict__ out,
                           const float* __restrict__ dout,
                           float* __restrict__ dq, float* __restrict__ dk,
                           float* __restrict__ dv,
                           const int* __restrict__ qi,
                           const int* __restrict__ ki,
                           const WorkGpu* __restrict__ works,
                           const float* __restrict__ hs,
                           float* __restrict__ dhs) {
  WorkGpu w = works[blockIdx.x];
  uint32_t lane = threadIdx.x;
  for (uint32_t p = 0; p < (uint32_t)w.nq * 8; p++) {
    uint32_t r = p / 8, h = p % 8, qr = w.base + qi[w.qs + r];
    float sc = hs[h] * w.logkv / 64.0f;
    const float* qq = q + (size_t)qr * 512 + h * 64;
    const float* go = dout + (size_t)qr * 512 + h * 64;
    const float* oo = out + (size_t)qr * 512 + h * 64;
    float q0 = qq[2 * lane], q1 = qq[2 * lane + 1], mx = -INFINITY, z = 0;
    for (int j = 0; j < w.nk; j++) {
      uint32_t kr = w.base + ki[w.ks + j];
      const float* kk = k + (size_t)kr * 512 + h * 64;
      mx = fmaxf(mx, wsum(q0 * sc * kk[2 * lane] + q1 * sc * kk[2 * lane + 1]));
    }
    for (int j = 0; j < w.nk; j++) {
      uint32_t kr = w.base + ki[w.ks + j];
      const float* kk = k + (size_t)kr * 512 + h * 64;
      z += expf(wsum(q0 * sc * kk[2 * lane] + q1 * sc * kk[2 * lane + 1]) - mx);
    }
    float gq0 = 0, gq1 = 0, gh = 0;
    for (int j = 0; j < w.nk; j++) {
      uint32_t kr = w.base + ki[w.ks + j];
      const float* kk = k + (size_t)kr * 512 + h * 64;
      const float* vv = v + (size_t)kr * 512 + h * 64;
      float dotq = wsum(q0 * kk[2 * lane] + q1 * kk[2 * lane + 1]);
      float prob = expf(dotq * sc - mx) / z;
      float ds = prob * wsum(go[2 * lane] * (vv[2 * lane] - oo[2 * lane]) +
                             go[2 * lane + 1] * (vv[2 * lane + 1] - oo[2 * lane + 1]));
      gq0 += ds * sc * kk[2 * lane];
      gq1 += ds * sc * kk[2 * lane + 1];
      atomicAdd(dk + (size_t)kr * 512 + h * 64 + 2 * lane, ds * sc * q0);
      atomicAdd(dk + (size_t)kr * 512 + h * 64 + 2 * lane + 1, ds * sc * q1);
      atomicAdd(dv + (size_t)kr * 512 + h * 64 + 2 * lane, prob * go[2 * lane]);
      atomicAdd(dv + (size_t)kr * 512 + h * 64 + 2 * lane + 1,
                prob * go[2 * lane + 1]);
      gh += ds * dotq * w.logkv / 64.0f;
    }
    float* qg = dq + (size_t)qr * 512 + h * 64;
    qg[2 * lane] = gq0;
    qg[2 * lane + 1] = gq1;
    if (lane == 0) atomicAdd(dhs + h, gh);
  }
}

__global__ void k_gate_fwd(const float* __restrict__ a,
                           const float* __restrict__ g, float* __restrict__ y,
                           uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * (2.0f / (1.0f + expf(-g[i])));
}

__global__ void k_gate_bwd(const float* __restrict__ a,
                           const float* __restrict__ g,
                           const float* __restrict__ dy, float* __restrict__ da,
                           float* __restrict__ dg, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float s = 1.0f / (1.0f + expf(-g[i]));
    da[i] = dy[i] * 2 * s;
    dg[i] = dy[i] * a[i] * 2 * s * (1 - s);
  }
}

__global__ void k_swiglu_fwd(const float* __restrict__ a,
                             const float* __restrict__ b, float* __restrict__ y,
                             uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float s = 1.0f / (1.0f + expf(-a[i]));
    y[i] = a[i] * s * b[i];
  }
}

__global__ void k_swiglu_bwd(const float* __restrict__ a,
                             const float* __restrict__ b,
                             const float* __restrict__ dy,
                             float* __restrict__ da, float* __restrict__ db,
                             uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float s = 1.0f / (1.0f + expf(-a[i]));
    da[i] = dy[i] * b[i] * s * (1 + a[i] * (1 - s));
    db[i] = dy[i] * a[i] * s;
  }
}

__global__ void k_huber_targets(const float* __restrict__ pred,
                                const float* __restrict__ truth,
                                const uint8_t* __restrict__ target,
                                float* __restrict__ dp, float* __restrict__ loss,
                                uint32_t B, uint32_t S) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t n = B * S;
  if (i >= n) return;
  dp[i] = 0;
  if (!target[i]) return;
  float e = pred[i] - truth[i], ae = fabsf(e);
  float l = ae < 1 ? 0.5f * e * e : ae - 0.5f;
  float inv = 1.0f / (float)B;
  dp[i] = (ae < 1 ? e : copysignf(1.0f, e)) * inv;
  atomicAdd(loss, l * inv);
}

__global__ void k_bias_grad(const float* __restrict__ dy, float* __restrict__ db,
                            uint32_t rows, uint32_t cols) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  float z = 0;
  for (uint32_t r = 0; r < rows; r++) z += dy[(size_t)r * cols + c];
  db[c] += z;
}

__global__ void k_add_bias(float* __restrict__ y, const float* __restrict__ b,
                           uint32_t rows, uint32_t cols) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * cols) y[i] += b[i % cols];
}

__global__ void k_mask_rows(const float* __restrict__ x,
                            const uint8_t* __restrict__ mask,
                            float* __restrict__ y, uint32_t rows,
                            uint32_t cols) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * cols) y[i] = mask[i / cols] ? x[i] : 0.0f;
}

__global__ void k_mask_embedding_grad(const float* __restrict__ dx,
                                      const uint8_t* __restrict__ target_type,
                                      float* __restrict__ grad, uint32_t rows,
                                      uint32_t cols) {
  uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= cols) return;
  float z = 0;
  for (uint32_t r = 0; r < rows; r++)
    if (target_type[r]) z += dx[(size_t)r * cols + d];
  grad[d] += z;
}

__global__ void k_grad_square(const float* __restrict__ g,
                              float* __restrict__ sum, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(sum, g[i] * g[i]);
}

__global__ void k_adamw(float* __restrict__ w, const float* __restrict__ g,
                        float* __restrict__ m, float* __restrict__ v,
                        AdamArgs a) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= a.n) return;
  float gg = g[i] * a.clip;
  float mm = a.b1 * m[i] + (1 - a.b1) * gg;
  float vv = a.b2 * v[i] + (1 - a.b2) * gg * gg;
  m[i] = mm; v[i] = vv;
  float mh = mm / (1 - powf(a.b1, (float)a.step));
  float vh = vv / (1 - powf(a.b2, (float)a.step));
  w[i] -= a.lr * (mh / (sqrtf(vh) + a.eps) + a.wd * w[i]);
}

// ---- host-side context ----------------------------------------------------

struct Param {
  std::string key;
  float *w = nullptr, *g = nullptr, *m = nullptr, *v = nullptr;
  size_t n = 0;
  bool used = false;
};

struct FullCtx {
  std::mutex mu;
  cublasHandle_t blas = nullptr;
  cudaStream_t stream = nullptr;
  std::unordered_map<std::string, Param> params;
  std::vector<void*> scratch;          // per-step activations/tapes/uploads
  uint64_t step = 0;
  uint32_t accumulated_microbatches = 0;

  ~FullCtx() {
    release_scratch();
    for (auto& kv : params)
      for (float* p : {kv.second.w, kv.second.g, kv.second.m, kv.second.v})
        cudaFree(p);
    if (blas) cublasDestroy(blas);
    if (stream) cudaStreamDestroy(stream);
  }
  void release_scratch() {
    for (void* p : scratch) cudaFreeAsync(p, stream);
    scratch.clear();
    cudaStreamSynchronize(stream);
  }
};

// Scratch is stream-ordered (cudaMallocAsync/cudaFreeAsync): the pool reuses
// a freed buffer for a later allocation on the same stream without the GPU
// having to catch up first. Plain cudaMalloc here would OOM — the CPU
// enqueues the whole step ahead of the GPU, so synchronous allocations pile
// up long before any queued free executes.
float* buffer(FullCtx& c, size_t bytes) {
  void* p = nullptr;
  RT_CU(cudaMallocAsync(&p, bytes, c.stream));
  c.scratch.push_back(p);
  return (float*)p;
}

// Scoped scratch release. Metal keeps every tape buffer of a step alive
// (unified memory, command-buffer retention) — on a 12GB card that OOMs
// around 4k cells, so each block's temporaries are freed as soon as they are
// consumed. cudaFreeAsync is stream-ordered: the free happens after all
// already-enqueued kernels using the buffer, no sync needed.
size_t mark(FullCtx& c) { return c.scratch.size(); }

// Free every allocation made since `mk` except `keep` (a buffer that must
// outlive the scope, e.g. a block's output activation).
void release_to(FullCtx& c, size_t mk, const float* keep = nullptr) {
  bool kept = false;
  for (size_t i = mk; i < c.scratch.size(); i++) {
    if (c.scratch[i] == (const void*)keep) {
      c.scratch[mk] = c.scratch[i];   // survivor slides down into the scope
      kept = true;
      continue;
    }
    RT_CU(cudaFreeAsync(c.scratch[i], c.stream));
  }
  c.scratch.resize(mk + (kept ? 1 : 0));
}

float* upload(FullCtx& c, const void* src, size_t bytes) {
  float* p = buffer(c, bytes);
  RT_CU(cudaMemcpyAsync(p, src, bytes, cudaMemcpyHostToDevice, c.stream));
  return p;
}

float* param_upload(FullCtx& c, const void* src, size_t bytes, bool zeroed) {
  void* p = nullptr;
  RT_CU(cudaMalloc(&p, bytes));
  if (zeroed)
    RT_CU(cudaMemset(p, 0, bytes));
  else
    RT_CU(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice));
  return (float*)p;
}

FullCtx* make_full_ctx(Model& model) {
  auto* c = new FullCtx;
  try {
    RT_CU(cudaStreamCreate(&c->stream));
    RT_CUBLAS(cublasCreate(&c->blas));
    RT_CUBLAS(cublasSetStream(c->blas, c->stream));
    for (auto& kv : model.store) {
      const Tensor& t = kv.second;
      if (t.qtype != (uint8_t)WType::F32)
        throw std::runtime_error(
            "full-model fine-tuning requires an unquantized checkpoint: " +
            kv.first);
      Param p;
      p.key = kv.first;
      p.n = t.data.size();
      p.w = param_upload(*c, t.data.data(), p.n * 4, false);
      p.g = param_upload(*c, nullptr, p.n * 4, true);
      p.m = param_upload(*c, nullptr, p.n * 4, true);
      p.v = param_upload(*c, nullptr, p.n * 4, true);
      c->params.emplace(kv.first, std::move(p));
    }
    return c;
  } catch (...) {
    delete c;
    throw;
  }
}

Param& P(FullCtx& c, const std::string& key) {
  auto it = c.params.find(key);
  if (it == c.params.end())
    throw std::runtime_error("missing train parameter " + key);
  it->second.used = true;
  return it->second;
}

void flush(FullCtx& c) {
  RT_CU(cudaStreamSynchronize(c.stream));
  RT_CU(cudaGetLastError());
}

inline unsigned nb(size_t n) { return (unsigned)((n + 255) / 256); }

void zero(FullCtx& c, float* x, size_t n) {
  RT_CU(cudaMemsetAsync(x, 0, n * 4, c.stream));
}
void copy(FullCtx& c, const float* a, float* b, size_t n) {
  RT_CU(cudaMemcpyAsync(b, a, n * 4, cudaMemcpyDeviceToDevice, c.stream));
}

// Row-major C[M,N] = op_ta(A) * op_tb(B) + beta*C via the col-major identity
// C' = op_tb(B)' * op_ta(A)'.
void gemm(FullCtx& c, const float* a, const float* b, float* o, int M, int N,
          int K, bool ta, bool tb, float beta = 0) {
  const float alpha = 1.f;
  RT_CUBLAS(cublasSgemm(c.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N,
                        ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, b,
                        tb ? K : N, a, ta ? M : K, &beta, o, N));
}

struct GroupBuffers {
  int *qi = nullptr, *ki = nullptr;
  WorkGpu* w = nullptr;
  size_t nw = 0;
};

GroupBuffers groups(FullCtx& c, const detail::Prepared& prep, int which) {
  const std::vector<detail::Groups>* all[3] = {&prep.g_col, &prep.g_feat,
                                               &prep.g_nbr};
  std::vector<int32_t> q, k;
  std::vector<WorkGpu> w;
  for (int b = 0; b < prep.B; b++) {
    const auto& G = (*all[which])[b];
    int qb = (int)q.size(), kb = (int)k.size();
    q.insert(q.end(), G.q.begin(), G.q.end());
    k.insert(k.end(), G.k.begin(), G.k.end());
    for (int g = 0; g < G.n(); g++) {
      int nq = G.qoff[g + 1] - G.qoff[g], nk = G.koff[g + 1] - G.koff[g];
      for (int q0 = 0; q0 < nq; q0 += detail::kQTile)
        w.push_back({qb + G.qoff[g] + q0, std::min(detail::kQTile, nq - q0),
                     kb + G.koff[g], nk, b * prep.S,
                     std::log(std::max(1.f, detail::bf16_round((float)nk)))});
    }
  }
  GroupBuffers gb;
  gb.qi = (int*)upload(c, q.data(), std::max<size_t>(1, q.size()) * 4);
  gb.ki = (int*)upload(c, k.data(), std::max<size_t>(1, k.size()) * 4);
  gb.w = (WorkGpu*)upload(c, w.data(),
                          std::max<size_t>(1, w.size()) * sizeof(WorkGpu));
  gb.nw = w.size();
  return gb;
}

struct AttnTape { float *x, *xn, *q, *k, *v, *g, *qn, *kn, *a, *y, *out; };
struct FfnTape { float *x, *xn, *a, *b, *y, *out; };

void rms_forward(FullCtx& c, const float* x, Param& s, float* y, size_t rows,
                 uint32_t n) {
  k_rms_fwd<<<(unsigned)rows, 32, 0, c.stream>>>(x, s.w, y, n);
}
void rms_backward(FullCtx& c, const float* x, const float* y, Param& s,
                  const float* dy, float* dx, size_t rows, uint32_t n) {
  k_rms_dx<<<(unsigned)rows, 32, 0, c.stream>>>(x, s.w, dy, dx, n);
  k_rms_ds<<<nb(n), 256, 0, c.stream>>>(y, s.w, dy, s.g, (uint32_t)rows, n);
}

std::string ap(int b, int a) {
  static const char* n[] = {"col", "feat", "nbr"};
  return "blocks." + std::to_string(b) + ".attns." + n[a] + ".";
}
std::string np(int b, int a) {
  static const char* n[] = {"col", "feat", "nbr", "ffn"};
  return "blocks." + std::to_string(b) + ".norms." + n[a] + ".scale";
}

AttnTape attention_forward(FullCtx& c, int b, int a, float* x, size_t rows,
                           const GroupBuffers& gb) {
  const size_t n = rows * D, bytes = n * 4;
  AttnTape t;
  t.x = x;
  t.xn = buffer(c, bytes);
  rms_forward(c, x, P(c, np(b, a)), t.xn, rows, D);
  t.q = buffer(c, bytes); t.k = buffer(c, bytes);
  t.v = buffer(c, bytes); t.g = buffer(c, bytes);
  std::string p = ap(b, a);
  gemm(c, t.xn, P(c, p + "wq.weight").w, t.q, (int)rows, D, D, false, true);
  gemm(c, t.xn, P(c, p + "wk.weight").w, t.k, (int)rows, D, D, false, true);
  gemm(c, t.xn, P(c, p + "wv.weight").w, t.v, (int)rows, D, D, false, true);
  gemm(c, t.xn, P(c, p + "wg.weight").w, t.g, (int)rows, D, D, false, true);
  t.qn = buffer(c, bytes); t.kn = buffer(c, bytes);
  Param& qns = P(c, p + "q_norm.scale");
  Param& kns = P(c, p + "k_norm.scale");
  k_qnorm_fwd<<<(unsigned)(rows * H), 32, 0, c.stream>>>(t.q, qns.w, t.qn);
  k_qnorm_fwd<<<(unsigned)(rows * H), 32, 0, c.stream>>>(t.k, kns.w, t.kn);
  t.a = buffer(c, bytes);
  zero(c, t.a, n);
  Param& hs = P(c, p + "scale");
  if (gb.nw)
    k_attn_fwd<<<(unsigned)gb.nw, 32, 0, c.stream>>>(t.qn, t.kn, t.v, t.a,
                                                     gb.qi, gb.ki, gb.w, hs.w);
  t.y = buffer(c, bytes);
  k_gate_fwd<<<nb(n), 256, 0, c.stream>>>(t.a, t.g, t.y, (uint32_t)n);
  t.out = buffer(c, bytes);
  copy(c, x, t.out, n);
  gemm(c, t.y, P(c, p + "wo.weight").w, t.out, (int)rows, D, D, false, true, 1);
  return t;
}

FfnTape ffn_forward(FullCtx& c, int b, float* x, size_t rows) {
  const size_t nd = rows * D, nf = rows * kDFF;
  FfnTape t;
  t.x = x;
  t.xn = buffer(c, nd * 4);
  rms_forward(c, x, P(c, np(b, 3)), t.xn, rows, D);
  std::string p = "blocks." + std::to_string(b) + ".ffn.";
  t.a = buffer(c, nf * 4); t.b = buffer(c, nf * 4); t.y = buffer(c, nf * 4);
  gemm(c, t.xn, P(c, p + "w1.weight").w, t.a, (int)rows, kDFF, D, false, true);
  gemm(c, t.xn, P(c, p + "w3.weight").w, t.b, (int)rows, kDFF, D, false, true);
  k_swiglu_fwd<<<nb(nf), 256, 0, c.stream>>>(t.a, t.b, t.y, (uint32_t)nf);
  t.out = buffer(c, nd * 4);
  copy(c, x, t.out, nd);
  gemm(c, t.y, P(c, p + "w2.weight").w, t.out, (int)rows, D, kDFF, false, true,
       1);
  return t;
}

float* block_forward(FullCtx& c, int b, float* x, size_t rows,
                     const GroupBuffers gb[3]) {
  for (int a = 0; a < 3; a++) x = attention_forward(c, b, a, x, rows, gb[a]).out;
  return ffn_forward(c, b, x, rows).out;
}

void linear_backward(FullCtx& c, const float* x, Param& w, const float* dy,
                     float* dx, int M, int N, int K) {
  gemm(c, dy, w.w, dx, M, K, N, false, false, 1);
  gemm(c, dy, x, w.g, N, K, M, true, false, 1);
}

float* ffn_backward(FullCtx& c, int b, const FfnTape& t, const float* dout,
                    size_t rows) {
  size_t nd = rows * D, nf = rows * kDFF;
  std::string p = "blocks." + std::to_string(b) + ".ffn.";
  float* dx = buffer(c, nd * 4);
  copy(c, dout, dx, nd);
  float* dy = buffer(c, nf * 4);
  zero(c, dy, nf);
  linear_backward(c, t.y, P(c, p + "w2.weight"), dout, dy, (int)rows, D, kDFF);
  float* da = buffer(c, nf * 4);
  float* db = buffer(c, nf * 4);
  k_swiglu_bwd<<<nb(nf), 256, 0, c.stream>>>(t.a, t.b, dy, da, db,
                                             (uint32_t)nf);
  float* dxn = buffer(c, nd * 4);
  zero(c, dxn, nd);
  linear_backward(c, t.xn, P(c, p + "w1.weight"), da, dxn, (int)rows, kDFF, D);
  linear_backward(c, t.xn, P(c, p + "w3.weight"), db, dxn, (int)rows, kDFF, D);
  rms_backward(c, t.x, t.xn, P(c, np(b, 3)), dxn, dx, rows, D);
  return dx;
}

float* attention_backward(FullCtx& c, int b, int a, const AttnTape& t,
                          const float* dout, size_t rows,
                          const GroupBuffers& gb) {
  size_t n = rows * D, bytes = n * 4;
  std::string p = ap(b, a);
  float* dx = buffer(c, bytes);
  copy(c, dout, dx, n);
  float* dy = buffer(c, bytes);
  zero(c, dy, n);
  linear_backward(c, t.y, P(c, p + "wo.weight"), dout, dy, (int)rows, D, D);
  float* da = buffer(c, bytes);
  float* dg = buffer(c, bytes);
  k_gate_bwd<<<nb(n), 256, 0, c.stream>>>(t.a, t.g, dy, da, dg, (uint32_t)n);
  float* dqn = buffer(c, bytes);
  float* dkn = buffer(c, bytes);
  float* dv = buffer(c, bytes);
  zero(c, dqn, n); zero(c, dkn, n); zero(c, dv, n);
  Param& hs = P(c, p + "scale");
  if (gb.nw)
    k_attn_bwd<<<(unsigned)gb.nw, 32, 0, c.stream>>>(
        t.qn, t.kn, t.v, t.a, da, dqn, dkn, dv, gb.qi, gb.ki, gb.w, hs.w,
        hs.g);
  float* dq = buffer(c, bytes);
  float* dk = buffer(c, bytes);
  zero(c, dq, n); zero(c, dk, n);
  Param& qns = P(c, p + "q_norm.scale");
  Param& kns = P(c, p + "k_norm.scale");
  uint32_t nseg = (uint32_t)(rows * H);
  k_qnorm_dx<<<nseg, 32, 0, c.stream>>>(t.q, qns.w, dqn, dq);
  k_qnorm_dx<<<nseg, 32, 0, c.stream>>>(t.k, kns.w, dkn, dk);
  k_qnorm_ds<<<nb(HD), 256, 0, c.stream>>>(t.qn, qns.w, dqn, qns.g, nseg);
  k_qnorm_ds<<<nb(HD), 256, 0, c.stream>>>(t.kn, kns.w, dkn, kns.g, nseg);
  float* dxn = buffer(c, bytes);
  zero(c, dxn, n);
  linear_backward(c, t.xn, P(c, p + "wq.weight"), dq, dxn, (int)rows, D, D);
  linear_backward(c, t.xn, P(c, p + "wk.weight"), dk, dxn, (int)rows, D, D);
  linear_backward(c, t.xn, P(c, p + "wv.weight"), dv, dxn, (int)rows, D, D);
  linear_backward(c, t.xn, P(c, p + "wg.weight"), dg, dxn, (int)rows, D, D);
  rms_backward(c, t.x, t.xn, P(c, np(b, a)), dxn, dx, rows, D);
  return dx;
}

float* block_backward(FullCtx& c, int b, float* x0, float* dout, size_t rows,
                      const GroupBuffers gb[3]) {
  AttnTape at[3];
  float* x = x0;
  for (int a = 0; a < 3; a++) {
    at[a] = attention_forward(c, b, a, x, rows, gb[a]);
    x = at[a].out;
  }
  FfnTape ft = ffn_forward(c, b, x, rows);
  float* dx = ffn_backward(c, b, ft, dout, rows);
  for (int a = 2; a >= 0; a--)
    dx = attention_backward(c, b, a, at[a], dx, rows, gb[a]);
  return dx;
}

void encoder_backward(FullCtx& c, const char* name, const float* input, int in,
                      const uint8_t* mask, const float* dx, size_t rows) {
  const size_t nd = rows * D;
  std::string base = "enc_dict." + std::string(name);
  Param& w = P(c, base + ".weight");
  Param& b = P(c, base + ".bias");
  Param& scale = P(c, "norm_dict." + std::string(name) + ".scale");
  float* raw = buffer(c, nd * 4);
  gemm(c, input, w.w, raw, (int)rows, D, in, false, true);
  k_add_bias<<<nb(nd), 256, 0, c.stream>>>(raw, b.w, (uint32_t)rows, D);
  float* dy = buffer(c, nd * 4);
  k_mask_rows<<<nb(nd), 256, 0, c.stream>>>(dx, mask, dy, (uint32_t)rows, D);
  float* normed = buffer(c, nd * 4);
  rms_forward(c, raw, scale, normed, rows, D);
  float* draw = buffer(c, nd * 4);
  zero(c, draw, nd);
  rms_backward(c, raw, normed, scale, dy, draw, rows, D);
  // Only parameter gradients are needed at the input boundary.
  gemm(c, draw, input, w.g, D, in, (int)rows, true, false, 1);
  k_bias_grad<<<nb(D), 256, 0, c.stream>>>(draw, b.g, (uint32_t)rows, D);
}

void embedding_backward(FullCtx& c, const Batch& batch, const Output& meta,
                        const float* dx, size_t rows) {
  const size_t BS = rows;
  std::vector<float> col(BS * kDText), text(BS * kDText);
  std::vector<float> number(BS), datetime(BS), booleanv(BS);
  std::vector<uint8_t> colmask(BS), masks[4], targets[4];
  for (int t = 0; t < 4; t++) {
    masks[t].resize(BS);
    targets[t].resize(BS);
  }
  for (int b = 0; b < batch.B; b++)
    for (int s = 0; s < batch.S; s++) {
      size_t dst = (size_t)b * batch.S + s;
      size_t src = (size_t)b * batch.S + meta.sort_idxs[dst];
      bool valid = !batch.is_padding[src];
      colmask[dst] = valid;
      std::memcpy(col.data() + dst * kDText,
                  batch.col_name_v.data() + src * kDText, kDText * 4);
      std::memcpy(text.data() + dst * kDText,
                  batch.text_v.data() + src * kDText, kDText * 4);
      number[dst] = batch.number_v[src];
      datetime[dst] = batch.datetime_v[src];
      booleanv[dst] = batch.boolean_v[src];
      int sem = (int)batch.sem_types[src];
      if (sem >= 0 && sem < 4) {
        masks[sem][dst] = valid && !batch.is_target[src];
        targets[sem][dst] = valid && batch.is_target[src];
      }
    }
  float* bcol = upload(c, col.data(), col.size() * 4);
  float* btext = upload(c, text.data(), text.size() * 4);
  float* bn = upload(c, number.data(), number.size() * 4);
  float* bd = upload(c, datetime.data(), datetime.size() * 4);
  float* bb = upload(c, booleanv.data(), booleanv.size() * 4);
  encoder_backward(c, "col_name", bcol, kDText,
                   (uint8_t*)upload(c, colmask.data(), BS), dx, BS);
  float* inputs[4] = {bn, btext, bd, bb};
  int widths[4] = {1, kDText, 1, 1};
  static const char* names[] = {"number", "text", "datetime", "boolean"};
  for (int t = 0; t < 4; t++) {
    encoder_backward(c, names[t], inputs[t], widths[t],
                     (uint8_t*)upload(c, masks[t].data(), BS), dx, BS);
    Param& me = P(c, "mask_embs." + std::string(names[t]));
    uint8_t* tm = (uint8_t*)upload(c, targets[t].data(), BS);
    k_mask_embedding_grad<<<nb(D), 256, 0, c.stream>>>(dx, tm, me.g,
                                                       (uint32_t)BS, D);
  }
}

float read_scalar(FullCtx& c, const float* dev) {
  float v = 0.f;
  flush(c);
  RT_CU(cudaMemcpy(&v, dev, 4, cudaMemcpyDeviceToHost));
  return v;
}

float gradient_norm(FullCtx& c, float scale) {
  float* sum = buffer(c, 4);
  zero(c, sum, 1);
  for (auto& kv : c.params) {
    Param& p = kv.second;
    if (p.used)
      k_grad_square<<<nb(p.n), 256, 0, c.stream>>>(p.g, sum, (uint32_t)p.n);
  }
  // The clamp exists to absorb a tiny negative sum from float error. It must
  // not absorb a NaN: std::max(0.f, NaN) returns 0.f, so a NaN gradient would
  // report a norm of zero and sail into the weights (see the Metal twin).
  float total = read_scalar(c, sum);
  if (!std::isfinite(total)) return total;
  return std::sqrt(std::max(0.f, total)) * scale;
}

float update_parameters(FullCtx& c, const FullFineTuneOptions& o,
                        uint32_t microbatches) {
  const float average = 1.f / std::max(1u, microbatches);
  float norm = gradient_norm(c, average);
  float clip = average * ((o.grad_clip_norm > 0 && norm > o.grad_clip_norm)
                              ? o.grad_clip_norm / norm
                              : 1.f);
  c.step++;
  for (auto& kv : c.params) {
    Param& p = kv.second;
    if (!p.used) continue;
    // Matrices decay; biases and norm/scale vectors do not.
    bool matrix = p.key.size() > 7 &&
                  p.key.compare(p.key.size() - 7, 7, ".weight") == 0 &&
                  p.n > (size_t)D;
    AdamArgs a{(uint32_t)p.n, (uint32_t)c.step, o.learning_rate,
               matrix ? o.weight_decay : 0.f, o.beta1, o.beta2, o.epsilon,
               clip};
    k_adamw<<<nb(p.n), 256, 0, c.stream>>>(p.w, p.g, p.m, p.v, a);
  }
  flush(c);
  return norm;
}

void sync_model(Model& model, FullCtx& c) {
  flush(c);
  for (auto& kv : c.params) {
    Param& p = kv.second;
    RT_CU(cudaMemcpy(model.store.at(p.key).data.data(), p.w, p.n * 4,
                     cudaMemcpyDeviceToHost));
  }
  // Fused qkvg storage mirrors the four source matrices (see Metal twin).
  static const char* an[] = {"col", "feat", "nbr"};
  for (int b = 0; b < kBlocks; b++)
    for (int a = 0; a < 3; a++) {
      std::string p = "blocks." + std::to_string(b) + ".attns." + an[a] + ".";
      for (int j = 0; j < 4; j++) {
        static const char* wn[] = {"wq", "wk", "wv", "wg"};
        auto& src = model.store.at(p + wn[j] + ".weight").data;
        std::memcpy(model.blocks[b].attn[a].wqkvg_f32.data() + (size_t)j * D * D,
                    src.data(), src.size() * 4);
      }
    }
  for (auto& slot : model.device_ctx) slot.reset();
}

float model_loss_locked(FullCtx& c, Model& model, const Batch& batch) {
  Output meta;
  detail::Prepared prep = detail::prepare(model, batch, meta, false);
  size_t rows = (size_t)prep.B * prep.S, n = rows * D;
  GroupBuffers gb[3] = {groups(c, prep, 0), groups(c, prep, 1),
                        groups(c, prep, 2)};
  float* x = upload(c, prep.x.data(), n * 4);
  for (int b = 0; b < kBlocks; b++) {
    size_t mk = mark(c);
    x = block_forward(c, b, x, rows, gb);
    release_to(c, mk, x);
  }
  float* xn = buffer(c, n * 4);
  rms_forward(c, x, P(c, "norm_out.scale"), xn, rows, D);
  Param& dw = P(c, "dec_dict.number.weight");
  Param& db = P(c, "dec_dict.number.bias");
  float* pred = buffer(c, rows * 4);
  gemm(c, xn, dw.w, pred, (int)rows, 1, D, false, true);
  k_add_bias<<<nb(rows), 256, 0, c.stream>>>(pred, db.w, (uint32_t)rows, 1);
  std::vector<float> truth(rows);
  for (int b = 0; b < batch.B; b++)
    for (int s = 0; s < batch.S; s++) {
      size_t dst = (size_t)b * batch.S + s;
      size_t src = (size_t)b * batch.S + meta.sort_idxs[dst];
      truth[dst] = batch.number_v[src];
      if (meta.sorted_is_target[dst] && batch.sem_types[src] != kNumber)
        throw std::runtime_error(
            "native full fine-tuning currently requires number/bool-as-number "
            "targets");
    }
  float* bt = upload(c, truth.data(), rows * 4);
  uint8_t* bm = (uint8_t*)upload(c, meta.sorted_is_target.data(), rows);
  float* dp = buffer(c, rows * 4);
  float* loss = buffer(c, 4);
  zero(c, loss, 1);
  k_huber_targets<<<nb(rows), 256, 0, c.stream>>>(pred, bt, bm, dp, loss,
                                                  (uint32_t)batch.B,
                                                  (uint32_t)batch.S);
  return read_scalar(c, loss);
}

// Name the side that went bad, and if it is the gradients, the parameters
// carrying the NaN -- the first few identify the backward stage.
std::string describe_non_finite(FullCtx& c, float loss_value, float grad_norm) {
  std::string msg = "non-finite ";
  bool bad_loss = !std::isfinite(loss_value);
  bool bad_grad = !std::isfinite(grad_norm);
  msg += bad_loss && bad_grad ? "loss and gradient" : bad_loss ? "loss"
                                                               : "gradient";
  msg += " (loss=" + std::to_string(loss_value) +
         ", grad_norm=" + std::to_string(grad_norm) + ")";
  if (!bad_grad) return msg;
  flush(c);
  std::vector<std::string> named;
  size_t n_bad = 0, n_used = 0;
  std::vector<float> g;
  for (auto& kv : c.params) {
    Param& p = kv.second;
    if (!p.used) continue;
    n_used++;
    g.resize(p.n);
    RT_CU(cudaMemcpy(g.data(), p.g, p.n * 4, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < p.n; i++)
      if (!std::isfinite(g[i])) { ok = false; break; }
    if (ok) continue;
    n_bad++;
    if (named.size() < 6) named.push_back(p.key);
  }
  msg += "; " + std::to_string(n_bad) + " of " + std::to_string(n_used) +
         " used parameters have non-finite gradients";
  if (!named.empty()) {
    msg += ", including:";
    for (auto& k : named) msg += " " + k;
  }
  return msg;
}

}  // namespace

bool full_finetune_cuda_available() {
  static bool ok = [] {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }();
  return ok;
}

FullFineTuneStep fit_model_cuda_step(Model& model, const Batch& batch,
                                     const FullFineTuneOptions& opts) {
  if (!full_finetune_cuda_available())
    throw std::runtime_error("rt/full-train-cuda: no CUDA device");
  if (batch.B <= 0 || batch.S <= 0)
    throw std::runtime_error("full-model fine-tuning batch is empty");
  if (!(opts.learning_rate > 0) || opts.weight_decay < 0 || opts.beta1 < 0 ||
      opts.beta1 >= 1 || opts.beta2 < 0 || opts.beta2 >= 1 ||
      !(opts.epsilon > 0))
    throw std::runtime_error("invalid full-model fine-tuning options");
  if (!model.training_ctx)
    model.training_ctx.reset(make_full_ctx(model),
                             [](void* p) { delete (FullCtx*)p; });
  FullCtx& c = *(FullCtx*)model.training_ctx.get();
  std::lock_guard<std::mutex> lock(c.mu);
  auto start = std::chrono::steady_clock::now();
  if (c.accumulated_microbatches == 0)
    for (auto& kv : c.params) {
      kv.second.used = false;
      zero(c, kv.second.g, kv.second.n);
    }
  Output meta;
  detail::Prepared prep = detail::prepare(model, batch, meta, false);
  size_t rows = (size_t)prep.B * prep.S, n = rows * D;
  GroupBuffers gb[3] = {groups(c, prep, 0), groups(c, prep, 1),
                        groups(c, prep, 2)};
  float* x = upload(c, prep.x.data(), n * 4);
  std::vector<float*> boundary(kBlocks + 1);
  boundary[0] = x;
  for (int b = 0; b < kBlocks; b++) {
    size_t mk = mark(c);
    boundary[b + 1] = block_forward(c, b, boundary[b], rows, gb);
    release_to(c, mk, boundary[b + 1]);   // keep only the block boundary
  }
  float* xn = buffer(c, n * 4);
  rms_forward(c, boundary[kBlocks], P(c, "norm_out.scale"), xn, rows, D);
  Param& dw = P(c, "dec_dict.number.weight");
  Param& db = P(c, "dec_dict.number.bias");
  float* pred = buffer(c, rows * 4);
  gemm(c, xn, dw.w, pred, (int)rows, 1, D, false, true);
  k_add_bias<<<nb(rows), 256, 0, c.stream>>>(pred, db.w, (uint32_t)rows, 1);
  // Sorted numeric labels/target flags.
  std::vector<float> truth(rows);
  for (int b = 0; b < batch.B; b++)
    for (int s = 0; s < batch.S; s++) {
      size_t dst = (size_t)b * batch.S + s;
      size_t src = (size_t)b * batch.S + meta.sort_idxs[dst];
      truth[dst] = batch.number_v[src];
      if (meta.sorted_is_target[dst] && batch.sem_types[src] != kNumber)
        throw std::runtime_error(
            "native full fine-tuning currently requires number/bool-as-number "
            "targets");
    }
  float* bt = upload(c, truth.data(), rows * 4);
  uint8_t* bm = (uint8_t*)upload(c, meta.sorted_is_target.data(), rows);
  float* dp = buffer(c, rows * 4);
  float* loss = buffer(c, 4);
  zero(c, loss, 1);
  k_huber_targets<<<nb(rows), 256, 0, c.stream>>>(pred, bt, bm, dp, loss,
                                                  (uint32_t)batch.B,
                                                  (uint32_t)batch.S);
  float* dxn = buffer(c, n * 4);
  zero(c, dxn, n);
  linear_backward(c, xn, dw, dp, dxn, (int)rows, 1, D);
  k_bias_grad<<<1, 1, 0, c.stream>>>(dp, db.g, (uint32_t)rows, 1);
  float* dx = buffer(c, n * 4);
  zero(c, dx, n);
  rms_backward(c, boundary[kBlocks], xn, P(c, "norm_out.scale"), dxn, dx, rows,
               D);
  for (int b = kBlocks - 1; b >= 0; b--) {
    size_t mk = mark(c);
    dx = block_backward(c, b, boundary[b], dx, rows, gb);
    release_to(c, mk, dx);                // recomputed tape dies with the block
  }
  embedding_backward(c, batch, meta, dx, rows);
  float loss_value = read_scalar(c, loss);
  c.accumulated_microbatches++;
  float grad_norm = gradient_norm(c, 1.f / c.accumulated_microbatches);
  bool updated = opts.apply_update;
  uint32_t accumulated = c.accumulated_microbatches;
  if (updated) {
    grad_norm = update_parameters(c, opts, c.accumulated_microbatches);
    sync_model(model, c);
    c.accumulated_microbatches = 0;
  }
  if (!std::isfinite(loss_value) || !std::isfinite(grad_norm)) {
    std::string msg = describe_non_finite(c, loss_value, grad_norm);
    c.release_scratch();
    throw std::runtime_error("rt/full-train-cuda: " + msg);
  }
  c.release_scratch();
  auto stop = std::chrono::steady_clock::now();
  return {loss_value, grad_norm, c.step,
          std::chrono::duration<double>(stop - start).count(), accumulated,
          updated};
}

void reset_model_cuda_optimizer(Model& model) { model.training_ctx.reset(); }

void save_model_cuda_optimizer(Model& model, const std::string& path) {
  if (!model.training_ctx)
    model.training_ctx.reset(make_full_ctx(model),
                             [](void* p) { delete (FullCtx*)p; });
  FullCtx& c = *(FullCtx*)model.training_ctx.get();
  std::lock_guard<std::mutex> lock(c.mu);
  flush(c);
  if (c.accumulated_microbatches != 0)
    throw std::runtime_error(
        "optimizer state can only be saved at an update boundary");
  std::vector<std::string> keys;
  keys.reserve(c.params.size());
  for (auto& kv : c.params) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot create optimizer state " + path);
  const char magic[8] = {'R', 'T', 'O', 'P', 'T', '0', '0', '2'};
  out.write(magic, 8);
  uint64_t step = c.step, nkeys = keys.size();
  out.write((char*)&step, 8);
  out.write((char*)&nkeys, 8);
  std::vector<float> host;
  for (const auto& key : keys) {
    Param& p = c.params.at(key);
    uint32_t len = (uint32_t)key.size();
    uint64_t nn = p.n;
    out.write((char*)&len, 4);
    out.write(key.data(), len);
    out.write((char*)&nn, 8);
    host.resize(p.n);
    RT_CU(cudaMemcpy(host.data(), p.m, p.n * 4, cudaMemcpyDeviceToHost));
    out.write((char*)host.data(), p.n * 4);
    RT_CU(cudaMemcpy(host.data(), p.v, p.n * 4, cudaMemcpyDeviceToHost));
    out.write((char*)host.data(), p.n * 4);
  }
  if (!out) throw std::runtime_error("failed writing optimizer state " + path);
}

void load_model_cuda_optimizer(Model& model, const std::string& path) {
  if (!model.training_ctx)
    model.training_ctx.reset(make_full_ctx(model),
                             [](void* p) { delete (FullCtx*)p; });
  FullCtx& c = *(FullCtx*)model.training_ctx.get();
  std::lock_guard<std::mutex> lock(c.mu);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open optimizer state " + path);
  char magic[8];
  in.read(magic, 8);
  if (std::memcmp(magic, "RTOPT002", 8) != 0)
    throw std::runtime_error("bad optimizer state magic");
  uint64_t step = 0, nkeys = 0;
  in.read((char*)&step, 8);
  in.read((char*)&nkeys, 8);
  if (nkeys != c.params.size())
    throw std::runtime_error("optimizer state parameter count mismatch");
  std::vector<float> host;
  for (uint64_t i = 0; i < nkeys; i++) {
    uint32_t len = 0;
    uint64_t nn = 0;
    in.read((char*)&len, 4);
    std::string key(len, '\0');
    in.read(key.data(), len);
    in.read((char*)&nn, 8);
    auto it = c.params.find(key);
    if (it == c.params.end() || it->second.n != nn)
      throw std::runtime_error("optimizer state tensor mismatch: " + key);
    Param& p = it->second;
    p.used = false;
    RT_CU(cudaMemset(p.g, 0, nn * 4));
    host.resize(nn);
    in.read((char*)host.data(), nn * 4);
    RT_CU(cudaMemcpy(p.m, host.data(), nn * 4, cudaMemcpyHostToDevice));
    in.read((char*)host.data(), nn * 4);
    RT_CU(cudaMemcpy(p.v, host.data(), nn * 4, cudaMemcpyHostToDevice));
  }
  if (!in) throw std::runtime_error("truncated optimizer state " + path);
  c.step = step;
  c.accumulated_microbatches = 0;
}

FullGradientCheck check_model_cuda_gradients(Model& model, const Batch& batch,
                                             float epsilon) {
  if (!(epsilon > 0))
    throw std::runtime_error("gradient-check epsilon must be positive");
  reset_model_cuda_optimizer(model);
  FullFineTuneOptions opts;
  opts.apply_update = false;
  (void)fit_model_cuda_step(model, batch, opts);
  FullCtx& c = *(FullCtx*)model.training_ctx.get();
  std::lock_guard<std::mutex> lock(c.mu);
  flush(c);
  const std::vector<std::pair<std::string, size_t>> probes = {
      {"enc_dict.number.weight", 17},
      {"mask_embs.number", 23},
      {"blocks.0.attns.col.wq.weight", 1000},
      {"blocks.0.attns.col.wv.weight", 2000},
      {"blocks.0.attns.col.q_norm.scale", 13},
      {"blocks.0.attns.col.scale", 0},
      {"blocks.11.ffn.w2.weight", 12345},
      {"norm_out.scale", 31},
      {"dec_dict.number.weight", 47},
      {"dec_dict.number.bias", 0}};
  FullGradientCheck out;
  auto poke = [&](Param& p, size_t idx, float value) {
    RT_CU(cudaMemcpy(p.w + idx, &value, 4, cudaMemcpyHostToDevice));
  };
  for (const auto& probe : probes) {
    auto it = c.params.find(probe.first);
    if (it == c.params.end() || it->second.n == 0) continue;
    Param& p = it->second;
    size_t idx = std::min(probe.second, p.n - 1);
    float original = 0.f, analytic = 0.f;
    RT_CU(cudaMemcpy(&original, p.w + idx, 4, cudaMemcpyDeviceToHost));
    RT_CU(cudaMemcpy(&analytic, p.g + idx, 4, cudaMemcpyDeviceToHost));
    // prepare() computes the embeddings from the HOST model, so each poke of
    // the device weight must sync back before the loss is evaluated.
    poke(p, idx, original + epsilon);
    sync_model(model, c);
    float plus = model_loss_locked(c, model, batch);
    c.release_scratch();
    poke(p, idx, original - epsilon);
    sync_model(model, c);
    float minus = model_loss_locked(c, model, batch);
    c.release_scratch();
    poke(p, idx, original);
    sync_model(model, c);
    float numeric = (plus - minus) / (2 * epsilon);
    float ae = std::abs(analytic - numeric);
    float re = ae / std::max(1e-3f, std::abs(analytic) + std::abs(numeric));
    out.max_absolute_error = std::max(out.max_absolute_error, ae);
    out.max_relative_error = std::max(out.max_relative_error, re);
    out.checked++;
  }
  c.accumulated_microbatches = 0;
  for (auto& kv : c.params) {
    kv.second.used = false;
    RT_CU(cudaMemset(kv.second.g, 0, kv.second.n * 4));
  }
  sync_model(model, c);
  return out;
}

}  // namespace rt
