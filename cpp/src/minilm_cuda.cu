// minilm_cuda.cu — the pinned MiniLM-L12-v2 text encoder on CUDA.
//
// Texts are tokenized on the host and bucketed by token count; each bucket
// runs as one batch of identical-length sequences, so no padding and no
// attention mask exist — per text the math is the CPU encoder's, only fp
// reduction order differs (absorbed by the downstream bf16 rounding of
// every embedding channel). Weights upload once per process; activation
// buffers grow to the largest bucket chunk.
//
// This exists because text embedding is the wall clock of any task that
// meets fresh cell text: a rented GPU box often pairs a serious device with
// a cgroup-capped sliver of an old CPU, where a 12-block BERT over long
// texts costs ~100ms each. The device does it in microseconds.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "minilm.hpp"
#include "minilm_internal.hpp"

namespace minilm {
namespace detail {
namespace {

constexpr int D = 384, H = 12, HD = 32, FF = 1536, L = 12;
constexpr size_t kMaxChunkTokens = 1 << 16;   // activation-buffer cap

#define ML_CU(call)                                                       \
  do {                                                                    \
    cudaError_t e_ = (call);                                              \
    if (e_ != cudaSuccess)                                                \
      throw std::runtime_error(std::string("minilm/cuda: ") +             \
                               cudaGetErrorString(e_));                   \
  } while (0)
#define ML_CUBLAS(call)                                                   \
  do {                                                                    \
    cublasStatus_t s_ = (call);                                           \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                      \
      throw std::runtime_error("minilm/cuda: cublas error " +             \
                               std::to_string((int)s_));                  \
  } while (0)

struct DevLayer {
  float *q_w, *q_b, *k_w, *k_b, *v_w, *v_b;
  float *ao_w, *ao_b, *aln_w, *aln_b;
  float *i_w, *i_b, *o_w, *o_b, *oln_w, *oln_b;
};

struct Ctx {
  cublasHandle_t blas = nullptr;
  cudaStream_t st = nullptr;
  float *word = nullptr, *pos = nullptr, *type = nullptr;
  float *eln_w = nullptr, *eln_b = nullptr;
  DevLayer lay[L] = {};
  std::vector<void*> owned;
  // grow-on-demand activations (nS tokens)
  int32_t* ids = nullptr;
  float *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr;
  float *att = nullptr, *ffn = nullptr, *y = nullptr, *pooled = nullptr;
  size_t cap_ns = 0, cap_n = 0;

  float* up(const float* p, size_t n) {
    float* d = nullptr;
    ML_CU(cudaMalloc(&d, n * sizeof(float)));
    ML_CU(cudaMemcpy(d, p, n * sizeof(float), cudaMemcpyHostToDevice));
    owned.push_back(d);
    return d;
  }
};

// ---- kernels ---------------------------------------------------------------

__global__ void k_seed(const int32_t* __restrict__ ids,
                       const float* __restrict__ word,
                       const float* __restrict__ pos,
                       const float* __restrict__ type, float* __restrict__ x,
                       int S) {
  const size_t tok = blockIdx.x;          // n*S tokens
  const int s = (int)(tok % S);
  const int32_t id = ids[tok];
  float* xr = x + tok * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    xr[i] = word[(size_t)id * D + i] + pos[(size_t)s * D + i] + type[i];
}

// LayerNorm rows in place (eps 1e-12, matches the CPU encoder).
__global__ void k_ln(float* __restrict__ x, const float* __restrict__ w,
                     const float* __restrict__ b) {
  float* row = x + (size_t)blockIdx.x * D;
  const int tid = threadIdx.x;            // 128
  __shared__ float red[128];
  float acc = 0.f;
  for (int i = tid; i < D; i += 128) acc += row[i];
  red[tid] = acc;
  __syncthreads();
  for (int wd = 64; wd > 0; wd >>= 1) {
    if (tid < wd) red[tid] += red[tid + wd];
    __syncthreads();
  }
  const float mean = red[0] / D;
  __syncthreads();
  acc = 0.f;
  for (int i = tid; i < D; i += 128) {
    const float d = row[i] - mean;
    acc += d * d;
  }
  red[tid] = acc;
  __syncthreads();
  for (int wd = 64; wd > 0; wd >>= 1) {
    if (tid < wd) red[tid] += red[tid + wd];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / D + 1e-12f);
  __syncthreads();
  for (int i = tid; i < D; i += 128)
    row[i] = (row[i] - mean) * inv * w[i] + b[i];
}

__global__ void k_bias(float* __restrict__ y, const float* __restrict__ b,
                       int width, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += b[i % width];
}

__global__ void k_bias_gelu(float* __restrict__ y, const float* __restrict__ b,
                            int width, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = y[i] + b[i % width];
  y[i] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

__global__ void k_bias_add(float* __restrict__ y, const float* __restrict__ b,
                           const float* __restrict__ res, int width,
                           size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += b[i % width] + res[i];
}

// One (text, head) per block; each thread owns query rows a = tid, tid+blk…
// K/V head tiles stage in shared memory (S <= 128 -> 16KB each half).
__global__ void k_attn(const float* __restrict__ q,
                       const float* __restrict__ k,
                       const float* __restrict__ v, float* __restrict__ att,
                       int S) {
  const int t = blockIdx.x / H, h = blockIdx.x % H;
  const size_t base = (size_t)t * S * D + h * HD;
  extern __shared__ float smem[];        // [S,HD] keys | [S,HD] values
  float* ks = smem;
  float* vs = smem + (size_t)S * HD;
  for (int i = threadIdx.x; i < S * HD; i += blockDim.x) {
    const int s = i / HD, d = i % HD;
    ks[i] = k[base + (size_t)s * D + d];
    vs[i] = v[base + (size_t)s * D + d];
  }
  __syncthreads();
  const float scale = rsqrtf((float)HD);
  for (int a = threadIdx.x; a < S; a += blockDim.x) {
    float qa[HD];
    const float* qr = q + base + (size_t)a * D;
    for (int d = 0; d < HD; ++d) qa[d] = qr[d];
    float sc[128];                       // S <= kMaxTokens = 128
    float mx = -1e30f;
    for (int s = 0; s < S; ++s) {
      float acc = 0.f;
      const float* kr = ks + (size_t)s * HD;
      for (int d = 0; d < HD; ++d) acc += qa[d] * kr[d];
      sc[s] = acc * scale;
      mx = fmaxf(mx, sc[s]);
    }
    float sum = 0.f;
    for (int s = 0; s < S; ++s) {
      sc[s] = expf(sc[s] - mx);
      sum += sc[s];
    }
    const float inv = 1.0f / sum;
    float o[HD];
    for (int d = 0; d < HD; ++d) o[d] = 0.f;
    for (int s = 0; s < S; ++s) {
      const float p = sc[s] * inv;
      const float* vr = vs + (size_t)s * HD;
      for (int d = 0; d < HD; ++d) o[d] += p * vr[d];
    }
    float* orow = att + base + (size_t)a * D;
    for (int d = 0; d < HD; ++d) orow[d] = o[d];
  }
}

// Mean-pool each text's rows, then L2-normalize (the pipeline's Normalize
// module — output is unit-norm exactly like the CPU path).
__global__ void k_pool(const float* __restrict__ x, float* __restrict__ out,
                       int S) {
  const size_t t = blockIdx.x;
  const int tid = threadIdx.x;           // 128
  __shared__ float red[128];
  float acc3[D / 128];
  for (int c = 0, i = tid; i < D; i += 128, c++) {
    float acc = 0.f;
    for (int s = 0; s < S; ++s) acc += x[((size_t)t * S + s) * D + i];
    acc3[c] = acc / S;
  }
  float ss = 0.f;
  for (int c = 0; c < D / 128; ++c) ss += acc3[c] * acc3[c];
  red[tid] = ss;
  __syncthreads();
  for (int wd = 64; wd > 0; wd >>= 1) {
    if (tid < wd) red[tid] += red[tid + wd];
    __syncthreads();
  }
  const float inv = 1.0f / fmaxf(sqrtf(red[0]), 1e-12f);
  for (int c = 0, i = tid; i < D; i += 128, c++)
    out[t * D + i] = acc3[c] * inv;
}

// y[M,out] = x[M,in] @ W[out,in]^T (row-major via the col-major transpose).
void gemm(Ctx& c, const float* x, const float* w, float* y, int M, int out,
          int in) {
  const float alpha = 1.f, beta = 0.f;
  ML_CUBLAS(cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, out, M, in, &alpha,
                        w, in, x, in, &beta, y, out));
}

Ctx* make_ctx(const HostWeights& w) {
  auto* c = new Ctx();
  try {
    ML_CU(cudaStreamCreate(&c->st));
    ML_CUBLAS(cublasCreate(&c->blas));
    ML_CUBLAS(cublasSetStream(c->blas, c->st));
    c->word = c->up(w.word_emb, (size_t)w.vocab_rows * D);
    c->pos = c->up(w.pos_emb, (size_t)kMaxTokens * D);
    c->type = c->up(w.type_emb, D);
    c->eln_w = c->up(w.eln_w, D);
    c->eln_b = c->up(w.eln_b, D);
    for (int l = 0; l < L; ++l) {
      const LayerW& s = w.layers[l];
      DevLayer& d = c->lay[l];
      d.q_w = c->up(s.q_w, (size_t)D * D);
      d.q_b = c->up(s.q_b, D);
      d.k_w = c->up(s.k_w, (size_t)D * D);
      d.k_b = c->up(s.k_b, D);
      d.v_w = c->up(s.v_w, (size_t)D * D);
      d.v_b = c->up(s.v_b, D);
      d.ao_w = c->up(s.ao_w, (size_t)D * D);
      d.ao_b = c->up(s.ao_b, D);
      d.aln_w = c->up(s.aln_w, D);
      d.aln_b = c->up(s.aln_b, D);
      d.i_w = c->up(s.i_w, (size_t)FF * D);
      d.i_b = c->up(s.i_b, FF);
      d.o_w = c->up(s.o_w, (size_t)D * FF);
      d.o_b = c->up(s.o_b, D);
      d.oln_w = c->up(s.oln_w, D);
      d.oln_b = c->up(s.oln_b, D);
    }
    return c;
  } catch (...) {
    for (void* p : c->owned) cudaFree(p);
    delete c;
    throw;
  }
}

void grow_bufs(Ctx& c, size_t nS, size_t n) {
  if (c.cap_ns < nS) {
    for (float** p : {&c.x, &c.q, &c.k, &c.v, &c.att, &c.y}) {
      if (*p) ML_CU(cudaFree(*p));
      *p = nullptr;
      ML_CU(cudaMalloc(p, nS * D * sizeof(float)));
    }
    if (c.ffn) ML_CU(cudaFree(c.ffn));
    c.ffn = nullptr;
    ML_CU(cudaMalloc(&c.ffn, nS * FF * sizeof(float)));
    if (c.ids) ML_CU(cudaFree(c.ids));
    c.ids = nullptr;
    ML_CU(cudaMalloc(&c.ids, nS * sizeof(int32_t)));
    c.cap_ns = nS;
  }
  if (c.cap_n < n) {
    if (c.pooled) ML_CU(cudaFree(c.pooled));
    c.pooled = nullptr;
    ML_CU(cudaMalloc(&c.pooled, n * D * sizeof(float)));
    c.cap_n = n;
  }
}

void encode_chunk(Ctx& c, const int32_t* ids_h, int n, int S, float* out_h) {
  const size_t nS = (size_t)n * S;
  grow_bufs(c, nS, n);
  ML_CU(cudaMemcpyAsync(c.ids, ids_h, nS * sizeof(int32_t),
                        cudaMemcpyHostToDevice, c.st));
  k_seed<<<(int)nS, 128, 0, c.st>>>(c.ids, c.word, c.pos, c.type, c.x, S);
  k_ln<<<(int)nS, 128, 0, c.st>>>(c.x, c.eln_w, c.eln_b);
  const int eb = (int)((nS * D + 255) / 256);
  const int fb = (int)((nS * FF + 255) / 256);
  const size_t smem = 2ull * S * HD * sizeof(float);
  for (int l = 0; l < L; ++l) {
    const DevLayer& w = c.lay[l];
    gemm(c, c.x, w.q_w, c.q, (int)nS, D, D);
    k_bias<<<eb, 256, 0, c.st>>>(c.q, w.q_b, D, nS * D);
    gemm(c, c.x, w.k_w, c.k, (int)nS, D, D);
    k_bias<<<eb, 256, 0, c.st>>>(c.k, w.k_b, D, nS * D);
    gemm(c, c.x, w.v_w, c.v, (int)nS, D, D);
    k_bias<<<eb, 256, 0, c.st>>>(c.v, w.v_b, D, nS * D);
    k_attn<<<n * H, 128, smem, c.st>>>(c.q, c.k, c.v, c.att, S);
    gemm(c, c.att, w.ao_w, c.y, (int)nS, D, D);
    k_bias_add<<<eb, 256, 0, c.st>>>(c.y, w.ao_b, c.x, D, nS * D);
    k_ln<<<(int)nS, 128, 0, c.st>>>(c.y, w.aln_w, w.aln_b);
    gemm(c, c.y, w.i_w, c.ffn, (int)nS, FF, D);
    k_bias_gelu<<<fb, 256, 0, c.st>>>(c.ffn, w.i_b, FF, nS * FF);
    gemm(c, c.ffn, w.o_w, c.x, (int)nS, D, FF);
    k_bias_add<<<eb, 256, 0, c.st>>>(c.x, w.o_b, c.y, D, nS * D);
    k_ln<<<(int)nS, 128, 0, c.st>>>(c.x, w.oln_w, w.oln_b);
  }
  k_pool<<<n, 128, 0, c.st>>>(c.x, c.pooled, S);
  ML_CU(cudaMemcpyAsync(out_h, c.pooled, (size_t)n * D * sizeof(float),
                        cudaMemcpyDeviceToHost, c.st));
  ML_CU(cudaStreamSynchronize(c.st));
  ML_CU(cudaGetLastError());
}

std::mutex g_mu;
Ctx* g_ctx = nullptr;
const float* g_ctx_key = nullptr;

}  // namespace

bool cuda_available() {
  static bool ok = [] {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }();
  return ok;
}

bool cuda_encode(const HostWeights& w,
                 const std::vector<std::vector<int32_t>>& ids, float* out,
                 std::string* err) {
  try {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_ctx == nullptr || g_ctx_key != w.word_emb) {
      // one snapshot per process in practice; a second one replaces it
      delete g_ctx;
      g_ctx = nullptr;
      g_ctx = make_ctx(w);
      g_ctx_key = w.word_emb;
    }
    // bucket by token count, chunk each bucket under the activation cap
    std::map<int, std::vector<size_t>> buckets;
    for (size_t t = 0; t < ids.size(); ++t)
      buckets[(int)ids[t].size()].push_back(t);
    std::vector<int32_t> flat;
    std::vector<float> res;
    for (const auto& [S, members] : buckets) {
      const size_t per = std::max<size_t>(1, kMaxChunkTokens / S);
      for (size_t at = 0; at < members.size(); at += per) {
        const size_t n = std::min(per, members.size() - at);
        flat.resize(n * S);
        for (size_t i = 0; i < n; ++i)
          std::copy(ids[members[at + i]].begin(), ids[members[at + i]].end(),
                    flat.begin() + i * S);
        res.resize(n * D);
        encode_chunk(*g_ctx, flat.data(), (int)n, S, res.data());
        for (size_t i = 0; i < n; ++i)
          std::copy(res.begin() + i * D, res.begin() + (i + 1) * D,
                    out + members[at + i] * D);
      }
    }
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

}  // namespace detail
}  // namespace minilm
