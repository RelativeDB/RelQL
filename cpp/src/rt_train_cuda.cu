// rt_train_cuda.cu — CUDA optimizer for frozen-backbone RT-J task heads.
//
// Mirror of rt_train_metal.mm: the backbone stays frozen, features are the
// [N, 512] output-normalized target-cell states, and a compact linear head
// trains with AdamW entirely on device. Losses: binary sigmoid CE, squared
// error, C-way softmax CE, grouped listwise softmax CE (ranking). Buffers
// live for one fit call; the final head copies back into the host struct.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "rt_train.hpp"

namespace rt {
namespace {

struct TrainArgs {
  uint32_t N, C, D, G, step, task;
  float lr, weight_decay, beta1, beta2, epsilon;
};

__device__ inline float wsum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ inline float wmax(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

// logits[row, c] = x[row] . w[c] + b[c]. One warp per (row, c).
__global__ void k_logits(const float* __restrict__ x,
                         const float* __restrict__ w,
                         const float* __restrict__ b,
                         float* __restrict__ logits, TrainArgs a) {
  uint32_t row = blockIdx.x / a.C, c = blockIdx.x % a.C;
  uint32_t lane = threadIdx.x;
  float v = 0.f;
  for (uint32_t d = lane; d < a.D; d += 32)
    v += x[(size_t)row * a.D + d] * w[(size_t)c * a.D + d];
  v = wsum(v);
  if (lane == 0) logits[(size_t)row * a.C + c] = v + b[c];
}

__global__ void k_delta_multiclass(const float* __restrict__ logits,
                                   const float* __restrict__ labels,
                                   float* __restrict__ delta,
                                   float* __restrict__ loss, TrainArgs a) {
  uint32_t row = blockIdx.x, lane = threadIdx.x;
  const float* z = logits + (size_t)row * a.C;
  float lm = -INFINITY;
  for (uint32_t c = lane; c < a.C; c += 32) lm = fmaxf(lm, z[c]);
  float mx = wmax(lm);
  float ls = 0.f;
  for (uint32_t c = lane; c < a.C; c += 32) ls += expf(z[c] - mx);
  float sum = wsum(ls);
  uint32_t y = (uint32_t)labels[row];
  for (uint32_t c = lane; c < a.C; c += 32)
    delta[(size_t)row * a.C + c] =
        (expf(z[c] - mx) / sum - (c == y ? 1.f : 0.f)) / (float)a.N;
  if (lane == 0) loss[row] = (logf(sum) + mx - z[y]) / (float)a.N;
}

__global__ void k_delta_scalar(const float* __restrict__ logits,
                               const float* __restrict__ labels,
                               float* __restrict__ delta,
                               float* __restrict__ loss, TrainArgs a) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= a.N) return;
  float z = logits[i], y = labels[i];
  if (a.task == 0) {
    float p = 1.f / (1.f + expf(-z));
    delta[i] = (p - y) / (float)a.N;
    loss[i] = (fmaxf(z, 0.f) - z * y + logf(1.f + expf(-fabsf(z)))) / (float)a.N;
  } else {
    float e = z - y;
    delta[i] = e / (float)a.N;
    loss[i] = 0.5f * e * e / (float)a.N;
  }
}

__global__ void k_delta_ranking(const float* __restrict__ logits,
                                const float* __restrict__ relevance,
                                const int* __restrict__ offsets,
                                float* __restrict__ delta,
                                float* __restrict__ loss, TrainArgs a) {
  uint32_t g = blockIdx.x, lane = threadIdx.x;
  uint32_t lo = (uint32_t)offsets[g], hi = (uint32_t)offsets[g + 1];
  float lm = -INFINITY, lr = 0.f;
  for (uint32_t i = lo + lane; i < hi; i += 32) {
    lm = fmaxf(lm, logits[i]);
    lr += relevance[i];
  }
  float mx = wmax(lm);
  float rsum = wsum(lr);
  float lz = 0.f;
  for (uint32_t i = lo + lane; i < hi; i += 32) lz += expf(logits[i] - mx);
  float zsum = wsum(lz);
  float ll = 0.f;
  for (uint32_t i = lo + lane; i < hi; i += 32) {
    float q = relevance[i] / rsum;
    float p = expf(logits[i] - mx) / zsum;
    delta[i] = (p - q) / (float)a.G;
    if (q > 0.f) ll -= q * logf(fmaxf(p, 1e-30f));
  }
  ll = wsum(ll);
  if (lane == 0) loss[g] = ll / (float)a.G;
}

__global__ void k_adam_weight(const float* __restrict__ x,
                              const float* __restrict__ delta,
                              float* __restrict__ w, float* __restrict__ m,
                              float* __restrict__ v, TrainArgs a) {
  uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = a.C * a.D;
  if (p >= total) return;
  uint32_t c = p / a.D, d = p % a.D;
  float grad = 0.f;
  for (uint32_t n = 0; n < a.N; n++)
    grad += delta[(size_t)n * a.C + c] * x[(size_t)n * a.D + d];
  float nm = a.beta1 * m[p] + (1.f - a.beta1) * grad;
  float nv = a.beta2 * v[p] + (1.f - a.beta2) * grad * grad;
  m[p] = nm; v[p] = nv;
  float mh = nm / (1.f - powf(a.beta1, (float)a.step));
  float vh = nv / (1.f - powf(a.beta2, (float)a.step));
  w[p] -= a.lr * (mh / (sqrtf(vh) + a.epsilon) + a.weight_decay * w[p]);
}

__global__ void k_adam_bias(const float* __restrict__ delta,
                            float* __restrict__ b, float* __restrict__ m,
                            float* __restrict__ v, TrainArgs a) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= a.C) return;
  float grad = 0.f;
  for (uint32_t n = 0; n < a.N; n++) grad += delta[(size_t)n * a.C + c];
  float nm = a.beta1 * m[c] + (1.f - a.beta1) * grad;
  float nv = a.beta2 * v[c] + (1.f - a.beta2) * grad * grad;
  m[c] = nm; v[c] = nv;
  float mh = nm / (1.f - powf(a.beta1, (float)a.step));
  float vh = nv / (1.f - powf(a.beta2, (float)a.step));
  b[c] -= a.lr * mh / (sqrtf(vh) + a.epsilon);
}

#define RT_CU(call)                                                     \
  do {                                                                  \
    cudaError_t e_ = (call);                                            \
    if (e_ != cudaSuccess)                                              \
      throw std::runtime_error(std::string("rt/train-cuda: ") +         \
                               cudaGetErrorString(e_));                 \
  } while (0)

// Every device allocation of one fit call, freed on scope exit.
struct DevArena {
  std::vector<void*> owned;
  ~DevArena() { for (void* p : owned) cudaFree(p); }
  float* upload(const void* p, size_t bytes) {
    void* d = nullptr;
    RT_CU(cudaMalloc(&d, bytes));
    RT_CU(cudaMemcpy(d, p, bytes, cudaMemcpyHostToDevice));
    owned.push_back(d);
    return (float*)d;
  }
  float* zeros(size_t bytes) {
    void* d = nullptr;
    RT_CU(cudaMalloc(&d, bytes));
    RT_CU(cudaMemset(d, 0, bytes));
    owned.push_back(d);
    return (float*)d;
  }
};

std::mutex& train_mu() {
  static std::mutex mu;
  return mu;
}

}  // namespace

FineTuneResult fit_head_cuda(FineTuneHead& head, const float* features,
                             const float* labels, int N,
                             const int32_t* group_offsets, int n_groups,
                             const FineTuneOptions& opts) {
  detail::check_head_inputs(head, features, labels, N, group_offsets, n_groups,
                            opts);
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0)
    throw std::runtime_error("rt/train-cuda: no CUDA device");
  std::lock_guard<std::mutex> lock(train_mu());
  const uint32_t C = (uint32_t)head.outputs;
  const uint32_t G = head.task == FineTuneTask::Ranking ? (uint32_t)n_groups
                                                        : (uint32_t)N;
  const size_t P = (size_t)C * kDModel;
  DevArena a;
  float* bx = a.upload(features, (size_t)N * kDModel * sizeof(float));
  float* by = a.upload(labels, (size_t)N * sizeof(float));
  int* boff = nullptr;
  if (head.task == FineTuneTask::Ranking)
    boff = (int*)a.upload(group_offsets, (size_t)(n_groups + 1) * sizeof(int32_t));
  float* bw = a.upload(head.weight.data(), P * sizeof(float));
  float* bb = a.upload(head.bias.data(), C * sizeof(float));
  float* logits = a.zeros((size_t)N * C * sizeof(float));
  float* delta = a.zeros((size_t)N * C * sizeof(float));
  const int nloss = head.task == FineTuneTask::Ranking ? n_groups : N;
  float* loss = a.zeros((size_t)std::max(N, nloss) * sizeof(float));
  float* mw = a.zeros(P * sizeof(float));
  float* vw = a.zeros(P * sizeof(float));
  float* mb = a.zeros(C * sizeof(float));
  float* vb = a.zeros(C * sizeof(float));
  std::vector<float> loss_h(nloss);

  auto epoch_loss = [&](uint32_t step, bool update) {
    TrainArgs targs{(uint32_t)N, C, (uint32_t)kDModel, G, step,
                    (uint32_t)head.task, opts.learning_rate, opts.weight_decay,
                    opts.beta1, opts.beta2, opts.epsilon};
    k_logits<<<(unsigned)((size_t)N * C), 32>>>(bx, bw, bb, logits, targs);
    if (head.task == FineTuneTask::Multiclass)
      k_delta_multiclass<<<N, 32>>>(logits, by, delta, loss, targs);
    else if (head.task == FineTuneTask::Ranking)
      k_delta_ranking<<<G, 32>>>(logits, by, boff, delta, loss, targs);
    else
      k_delta_scalar<<<(N + 255) / 256, 256>>>(logits, by, delta, loss, targs);
    if (update) {
      k_adam_weight<<<(unsigned)((P + 255) / 256), 256>>>(bx, delta, bw, mw,
                                                          vw, targs);
      k_adam_bias<<<(C + 255) / 256, std::min<unsigned>(C, 256)>>>(delta, bb,
                                                                   mb, vb,
                                                                   targs);
    }
    RT_CU(cudaMemcpy(loss_h.data(), loss, nloss * sizeof(float),
                     cudaMemcpyDeviceToHost));
    RT_CU(cudaGetLastError());
    float total = 0.f;
    for (int i = 0; i < nloss; i++) total += loss_h[i];
    return total;
  };

  FineTuneResult result;
  const auto start = std::chrono::steady_clock::now();
  for (int e = 0; e < opts.epochs; e++) {
    float l = epoch_loss((uint32_t)e + 1, true);
    if (e == 0) result.initial_loss = l;
    if (!std::isfinite(l)) throw std::runtime_error("fine-tuning loss diverged");
  }
  result.final_loss = epoch_loss((uint32_t)opts.epochs, false);
  const auto stop = std::chrono::steady_clock::now();
  result.epochs = opts.epochs;
  result.seconds = std::chrono::duration<double>(stop - start).count();
  RT_CU(cudaMemcpy(head.weight.data(), bw, P * sizeof(float),
                   cudaMemcpyDeviceToHost));
  RT_CU(cudaMemcpy(head.bias.data(), bb, C * sizeof(float),
                   cudaMemcpyDeviceToHost));
  return result;
}

}  // namespace rt
