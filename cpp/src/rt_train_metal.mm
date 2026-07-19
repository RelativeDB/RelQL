// rt_train_metal.mm — Metal optimizer for frozen-backbone RT-J task heads.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "rt_train.hpp"

namespace rt {
namespace {

const char* kTrainMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct TrainArgs {
  uint N, C, D, G, step, task;
  float lr, weight_decay, beta1, beta2, epsilon;
};

kernel void train_logits(device const float* x [[buffer(0)]],
                         device const float* w [[buffer(1)]],
                         device const float* b [[buffer(2)]],
                         device float* logits [[buffer(3)]],
                         constant TrainArgs& a [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  uint row = tg / a.C, c = tg % a.C;
  float v = 0.0f;
  for (uint d = lane; d < a.D; d += 32)
    v += x[(ulong)row * a.D + d] * w[(ulong)c * a.D + d];
  v = simd_sum(v);
  if (lane == 0) logits[(ulong)row * a.C + c] = v + b[c];
}

kernel void delta_multiclass(device const float* logits [[buffer(0)]],
                             device const float* labels [[buffer(1)]],
                             device float* delta [[buffer(2)]],
                             device float* loss [[buffer(3)]],
                             constant TrainArgs& a [[buffer(4)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
  device const float* z = logits + (ulong)row * a.C;
  float lm = -INFINITY;
  for (uint c = lane; c < a.C; c += 32) lm = max(lm, z[c]);
  float mx = simd_max(lm);
  float ls = 0.0f;
  for (uint c = lane; c < a.C; c += 32) ls += exp(z[c] - mx);
  float sum = simd_sum(ls);
  uint y = uint(labels[row]);
  for (uint c = lane; c < a.C; c += 32)
    delta[(ulong)row * a.C + c] =
        (exp(z[c] - mx) / sum - (c == y ? 1.0f : 0.0f)) / float(a.N);
  if (lane == 0) loss[row] = (log(sum) + mx - z[y]) / float(a.N);
}

kernel void delta_scalar(device const float* logits [[buffer(0)]],
                         device const float* labels [[buffer(1)]],
                         device float* delta [[buffer(2)]],
                         device float* loss [[buffer(3)]],
                         constant TrainArgs& a [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= a.N) return;
  float z = logits[i], y = labels[i];
  if (a.task == 0) {
    float p = 1.0f / (1.0f + exp(-z));
    delta[i] = (p - y) / float(a.N);
    loss[i] = (max(z, 0.0f) - z * y + log(1.0f + exp(-abs(z)))) / float(a.N);
  } else {
    float e = z - y;
    delta[i] = e / float(a.N);
    loss[i] = 0.5f * e * e / float(a.N);
  }
}

kernel void delta_ranking(device const float* logits [[buffer(0)]],
                          device const float* relevance [[buffer(1)]],
                          device const int* offsets [[buffer(2)]],
                          device float* delta [[buffer(3)]],
                          device float* loss [[buffer(4)]],
                          constant TrainArgs& a [[buffer(5)]],
                          uint g [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  uint lo = uint(offsets[g]), hi = uint(offsets[g + 1]);
  float lm = -INFINITY, lr = 0.0f;
  for (uint i = lo + lane; i < hi; i += 32) {
    lm = max(lm, logits[i]);
    lr += relevance[i];
  }
  float mx = simd_max(lm);
  float rsum = simd_sum(lr);
  float lz = 0.0f;
  for (uint i = lo + lane; i < hi; i += 32) lz += exp(logits[i] - mx);
  float zsum = simd_sum(lz);
  float ll = 0.0f;
  for (uint i = lo + lane; i < hi; i += 32) {
    float q = relevance[i] / rsum;
    float p = exp(logits[i] - mx) / zsum;
    delta[i] = (p - q) / float(a.G);
    if (q > 0.0f) ll -= q * log(max(p, 1e-30f));
  }
  ll = simd_sum(ll);
  if (lane == 0) loss[g] = ll / float(a.G);
}

kernel void adam_weight(device const float* x [[buffer(0)]],
                        device const float* delta [[buffer(1)]],
                        device float* w [[buffer(2)]],
                        device float* m [[buffer(3)]],
                        device float* v [[buffer(4)]],
                        constant TrainArgs& a [[buffer(5)]],
                        uint p [[thread_position_in_grid]]) {
  uint total = a.C * a.D;
  if (p >= total) return;
  uint c = p / a.D, d = p % a.D;
  float grad = 0.0f;
  for (uint n = 0; n < a.N; n++)
    grad += delta[(ulong)n * a.C + c] * x[(ulong)n * a.D + d];
  float nm = a.beta1 * m[p] + (1.0f - a.beta1) * grad;
  float nv = a.beta2 * v[p] + (1.0f - a.beta2) * grad * grad;
  m[p] = nm; v[p] = nv;
  float mh = nm / (1.0f - pow(a.beta1, float(a.step)));
  float vh = nv / (1.0f - pow(a.beta2, float(a.step)));
  w[p] -= a.lr * (mh / (sqrt(vh) + a.epsilon) + a.weight_decay * w[p]);
}

kernel void adam_bias(device const float* delta [[buffer(0)]],
                      device float* b [[buffer(1)]],
                      device float* m [[buffer(2)]],
                      device float* v [[buffer(3)]],
                      constant TrainArgs& a [[buffer(4)]],
                      uint c [[thread_position_in_grid]]) {
  if (c >= a.C) return;
  float grad = 0.0f;
  for (uint n = 0; n < a.N; n++) grad += delta[(ulong)n * a.C + c];
  float nm = a.beta1 * m[c] + (1.0f - a.beta1) * grad;
  float nv = a.beta2 * v[c] + (1.0f - a.beta2) * grad * grad;
  m[c] = nm; v[c] = nv;
  float mh = nm / (1.0f - pow(a.beta1, float(a.step)));
  float vh = nv / (1.0f - pow(a.beta2, float(a.step)));
  b[c] -= a.lr * mh / (sqrt(vh) + a.epsilon);
}
)MSL";

struct TrainArgsHost {
  uint32_t N, C, D, G, step, task;
  float lr, weight_decay, beta1, beta2, epsilon;
};

struct TrainMetalCtx {
  std::mutex mu;
  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> logits, multiclass, scalar, ranking, weight, bias;
};

TrainMetalCtx& train_ctx() {
  static TrainMetalCtx ctx;
  static std::once_flag once;
  std::call_once(once, [&] {
    ctx.dev = MTLCreateSystemDefaultDevice();
    if (!ctx.dev) throw std::runtime_error("rt/train: no Metal device");
    ctx.queue = [ctx.dev newCommandQueue];
    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opts.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [ctx.dev newLibraryWithSource:@(kTrainMsl)
                                               options:opts error:&err];
    if (!lib)
      throw std::runtime_error(
          std::string("rt/train: Metal shader compile failed: ") +
          (err ? err.localizedDescription.UTF8String : "?"));
    auto make = [&](NSString* name) {
      id<MTLFunction> fn = [lib newFunctionWithName:name];
      NSError* pe = nil;
      id<MTLComputePipelineState> p =
          [ctx.dev newComputePipelineStateWithFunction:fn error:&pe];
      if (!p)
        throw std::runtime_error(std::string("rt/train: pipeline failed: ") +
                                 name.UTF8String);
      return p;
    };
    ctx.logits = make(@"train_logits");
    ctx.multiclass = make(@"delta_multiclass");
    ctx.scalar = make(@"delta_scalar");
    ctx.ranking = make(@"delta_ranking");
    ctx.weight = make(@"adam_weight");
    ctx.bias = make(@"adam_bias");
  });
  return ctx;
}

void check_inputs(const FineTuneHead& h, const float* x, const float* y, int N,
                  const int32_t* offsets, int G, const FineTuneOptions& o) {
  if (!x || !y || N <= 0) throw std::runtime_error("fine-tuning data is empty");
  if (h.outputs <= 0 || h.weight.size() != (size_t)h.outputs * kDModel ||
      h.bias.size() != (size_t)h.outputs)
    throw std::runtime_error("fine-tune head tensor shape mismatch");
  if (h.task != FineTuneTask::Multiclass && h.outputs != 1)
    throw std::runtime_error("scalar task head must have one output");
  if (o.epochs <= 0 || !(o.learning_rate > 0.f) || o.weight_decay < 0.f ||
      !(o.beta1 >= 0.f && o.beta1 < 1.f) ||
      !(o.beta2 >= 0.f && o.beta2 < 1.f) || !(o.epsilon > 0.f))
    throw std::runtime_error("invalid fine-tuning options");
  for (int i = 0; i < N; i++) {
    if (!std::isfinite(y[i])) throw std::runtime_error("label is not finite");
    if (h.task == FineTuneTask::Binary && (y[i] < 0.f || y[i] > 1.f))
      throw std::runtime_error("binary label must be in [0,1]");
    if (h.task == FineTuneTask::Multiclass &&
        (y[i] < 0.f || y[i] >= h.outputs || y[i] != std::floor(y[i])))
      throw std::runtime_error("multiclass label is out of range");
    if (h.task == FineTuneTask::Ranking && y[i] < 0.f)
      throw std::runtime_error("ranking relevance must be non-negative");
  }
  if (h.task == FineTuneTask::Ranking) {
    if (!offsets || G <= 0 || offsets[0] != 0 || offsets[G] != N)
      throw std::runtime_error("ranking group offsets must span [0,N]");
    for (int g = 0; g < G; g++) {
      if (offsets[g] >= offsets[g + 1])
        throw std::runtime_error("ranking groups must be non-empty");
      float rel = 0.f;
      for (int i = offsets[g]; i < offsets[g + 1]; i++) rel += y[i];
      if (!(rel > 0.f))
        throw std::runtime_error("every ranking group needs positive relevance");
    }
  }
}

}  // namespace

FineTuneResult fit_head_metal(FineTuneHead& head, const float* features,
                              const float* labels, int N,
                              const int32_t* group_offsets, int n_groups,
                              const FineTuneOptions& opts) {
  @autoreleasepool {
    check_inputs(head, features, labels, N, group_offsets, n_groups, opts);
    TrainMetalCtx& ctx = train_ctx();
    std::lock_guard<std::mutex> lock(ctx.mu);
    const uint32_t C = (uint32_t)head.outputs;
    const uint32_t G = head.task == FineTuneTask::Ranking
                           ? (uint32_t)n_groups
                           : (uint32_t)N;
    const size_t P = (size_t)C * kDModel;
    auto bytes = [&](const void* p, size_t n) {
      return [ctx.dev newBufferWithBytes:p length:n
                                  options:MTLResourceStorageModeShared];
    };
    auto empty = [&](size_t n) {
      return [ctx.dev newBufferWithLength:n
                                   options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> bx = bytes(features, (size_t)N * kDModel * sizeof(float));
    id<MTLBuffer> by = bytes(labels, (size_t)N * sizeof(float));
    id<MTLBuffer> boff = nil;
    if (head.task == FineTuneTask::Ranking)
      boff = bytes(group_offsets, (size_t)(n_groups + 1) * sizeof(int32_t));
    id<MTLBuffer> bw = bytes(head.weight.data(), P * sizeof(float));
    id<MTLBuffer> bb = bytes(head.bias.data(), C * sizeof(float));
    id<MTLBuffer> logits = empty((size_t)N * C * sizeof(float));
    id<MTLBuffer> delta = empty((size_t)N * C * sizeof(float));
    id<MTLBuffer> loss = empty((size_t)std::max<uint32_t>(N, G) * sizeof(float));
    id<MTLBuffer> mw = empty(P * sizeof(float));
    id<MTLBuffer> vw = empty(P * sizeof(float));
    id<MTLBuffer> mb = empty(C * sizeof(float));
    id<MTLBuffer> vb = empty(C * sizeof(float));
    std::memset(mw.contents, 0, P * sizeof(float));
    std::memset(vw.contents, 0, P * sizeof(float));
    std::memset(mb.contents, 0, C * sizeof(float));
    std::memset(vb.contents, 0, C * sizeof(float));

    auto encode_loss = [&](uint32_t step, bool update) {
      TrainArgsHost a{(uint32_t)N, C, (uint32_t)kDModel, G, step,
                      (uint32_t)head.task, opts.learning_rate,
                      opts.weight_decay, opts.beta1, opts.beta2, opts.epsilon};
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.logits];
      [enc setBuffer:bx offset:0 atIndex:0];
      [enc setBuffer:bw offset:0 atIndex:1];
      [enc setBuffer:bb offset:0 atIndex:2];
      [enc setBuffer:logits offset:0 atIndex:3];
      [enc setBytes:&a length:sizeof(a) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((size_t)N * C, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      if (head.task == FineTuneTask::Multiclass) {
        [enc setComputePipelineState:ctx.multiclass];
        [enc setBuffer:logits offset:0 atIndex:0];
        [enc setBuffer:by offset:0 atIndex:1];
        [enc setBuffer:delta offset:0 atIndex:2];
        [enc setBuffer:loss offset:0 atIndex:3];
        [enc setBytes:&a length:sizeof(a) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      } else if (head.task == FineTuneTask::Ranking) {
        [enc setComputePipelineState:ctx.ranking];
        [enc setBuffer:logits offset:0 atIndex:0];
        [enc setBuffer:by offset:0 atIndex:1];
        [enc setBuffer:boff offset:0 atIndex:2];
        [enc setBuffer:delta offset:0 atIndex:3];
        [enc setBuffer:loss offset:0 atIndex:4];
        [enc setBytes:&a length:sizeof(a) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(G, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      } else {
        [enc setComputePipelineState:ctx.scalar];
        [enc setBuffer:logits offset:0 atIndex:0];
        [enc setBuffer:by offset:0 atIndex:1];
        [enc setBuffer:delta offset:0 atIndex:2];
        [enc setBuffer:loss offset:0 atIndex:3];
        [enc setBytes:&a length:sizeof(a) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      if (update) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:ctx.weight];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:delta offset:0 atIndex:1];
        [enc setBuffer:bw offset:0 atIndex:2];
        [enc setBuffer:mw offset:0 atIndex:3];
        [enc setBuffer:vw offset:0 atIndex:4];
        [enc setBytes:&a length:sizeof(a) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(P, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc setComputePipelineState:ctx.bias];
        [enc setBuffer:delta offset:0 atIndex:0];
        [enc setBuffer:bb offset:0 atIndex:1];
        [enc setBuffer:mb offset:0 atIndex:2];
        [enc setBuffer:vb offset:0 atIndex:3];
        [enc setBytes:&a length:sizeof(a) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(C, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(C, 256), 1, 1)];
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError)
        throw std::runtime_error(
            std::string("rt/train: command buffer failed: ") +
            (cb.error ? cb.error.localizedDescription.UTF8String : "?"));
      const int nloss = head.task == FineTuneTask::Ranking ? n_groups : N;
      const float* lp = static_cast<const float*>(loss.contents);
      float total = 0.f;
      for (int i = 0; i < nloss; i++) total += lp[i];
      return total;
    };

    FineTuneResult result;
    const auto start = std::chrono::steady_clock::now();
    for (int e = 0; e < opts.epochs; e++) {
      float l = encode_loss((uint32_t)e + 1, true);
      if (e == 0) result.initial_loss = l;
      if (!std::isfinite(l)) throw std::runtime_error("fine-tuning loss diverged");
    }
    result.final_loss = encode_loss((uint32_t)opts.epochs, false);
    const auto stop = std::chrono::steady_clock::now();
    result.epochs = opts.epochs;
    result.seconds = std::chrono::duration<double>(stop - start).count();
    std::memcpy(head.weight.data(), bw.contents, P * sizeof(float));
    std::memcpy(head.bias.data(), bb.contents, C * sizeof(float));
    return result;
  }
}

}  // namespace rt
