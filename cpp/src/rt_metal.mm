// rt_metal.mm — Metal/MPS backend for RT-J inference.
//
// The dense projections (qkvg / wo / ffn) run as MPSMatrixMultiplication
// GEMMs; wo and w2 accumulate straight into the residual stream with beta=1
// so no separate add pass exists. Everything else is custom Metal kernels:
//  - rmsnorm_rows: one simdgroup per row (pre-norms, d=512)
//  - qknorm: in-place QK-RMSNorm per (token, head) on the fused qkvg buffer
//  - attn: one threadgroup per (group, query-tile) work item; each simdgroup
//    owns a (query, head) pair and streams the shared key list with a
//    single-pass online softmax — no S x S scores, exactly the query-group
//    sparsity the CPU path uses, with the same bf16-rounded log(kv) scaling.
//    Groups with >512 keys are flash split-K'd (attn_part -> attn_reduce):
//    the key list is chunked across independent threadgroups whose partial
//    online-softmax states are merged, so a single huge reverse-FK group is
//    no longer one latency-bound key stream
//  - dense projections: fp32/f16 -> MPSMatrixMultiplication (mixed fp32
//    activations/results with native f16 weights); q8/q4 -> custom qgemm
//    (adaptive 32/64-row x 32-column tiles, threadgroup-staged in-register
//    dequant, simdgroup MMA), weights uploaded quantized-resident
//  - gate_mul / swiglu / head: elementwise gating, SwiGLU, and the fused
//    output-norm + number-head dot
// Weights are uploaded once per model (fp32, unified memory); activation and
// index buffers grow on demand and are reused across forwards. Forwards on
// one model are serialized (the ctx owns the scratch buffers); the CPU path
// stays reentrant.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "rt_internal.hpp"

namespace rt {
namespace detail {
namespace {

constexpr int kD = kDModel;          // 512
constexpr int kC4 = 4 * kDModel;     // fused qkvg row stride

const char* kMsl = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant float kEps = 1e-6f;

// Weight dtype baked per pipeline: 1 = F16, 2 = Q8, 3 = Q4 (rt::WType).
constant int WT [[function_constant(0)]];
// Overwrite and residual-accumulate projections use separate pipelines so
// overwrite GEMMs never read the destination buffer.
constant bool ACCUMULATE [[function_constant(1)]];
// 32 rows favors latency/tails; 64 rows reuses each staged weight tile across
// twice as many activation rows for throughput-oriented shapes.
constant ushort TM [[function_constant(2)]];

struct QGemmArgs {
  uint M, N, K;
};

// y[M,N] = x[M,K] @ W[N,K]^T (+ residual) with W quantized-resident.
// One threadgroup computes a TMx32 output tile: TM=32 (128 threads) favors
// latency/tails, while TM=64 (256 threads) reuses each staged weight tile
// across twice as many rows. The K loop stages a TMx32 x-tile and a 32x32
// *dequantized* W-tile in threadgroup memory (dequant happens on the load from
// DRAM — the fp32 tile never exists outside on-chip memory), then each
// simdgroup accumulates an 8x32 strip with simdgroup_float8x8 MMA. K and N are
// multiples of 32 for every RT-J projection; M (tokens) is edge-guarded.
// Q4 layout note: K-chunks of 32 align exactly with Q4's group size, so each
// staged W-row chunk touches one (scale, min) pair.
kernel void qgemm(device const float* x [[buffer(0)]],
                  device const uchar* w [[buffer(1)]],
                  device const uchar* ws [[buffer(2)]],
                  device float* y [[buffer(3)]],
                  constant QGemmArgs& args [[buffer(4)]],
                  threadgroup float* Xt [[threadgroup(0)]],
                  threadgroup float* Wt [[threadgroup(1)]],
                  threadgroup float* Ct [[threadgroup(2)]],
                  uint2 tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint n0 = tg.x * 32, m0 = tg.y * TM;
  const uint K = args.K;
  simdgroup_float8x8 C[4];
  for (int j = 0; j < 4; j++) C[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint k0 = 0; k0 < K; k0 += 32) {
    for (uint i = tid; i < TM * 32; i += TM * 4) {       // stage x tile
      uint r = i / 32, c = i % 32;
      Xt[r * 33 + c] =
          (m0 + r < args.M) ? x[(ulong)(m0 + r) * K + k0 + c] : 0.0f;
    }
    if (WT == 3) {
      // One byte contains two adjacent Q4 weights. Have one thread unpack and
      // dequantize both values, halving payload/scale reads and loop work.
      for (uint i = tid; i < 32 * 16; i += TM * 4) {
        uint n = i / 16, p = i % 16;
        ulong gn = n0 + n;
        device const half* sh =
            ((device const half*)ws) + (gn * (K >> 5) + (k0 >> 5)) * 2;
        uchar b = w[gn * (K >> 1) + (k0 >> 1) + p];
        float scale = float(sh[0]), bias = float(sh[1]);
        Wt[n * 33 + 2 * p] = scale * float(b & 0xf) + bias;
        Wt[n * 33 + 2 * p + 1] = scale * float(b >> 4) + bias;
      }
    } else {
      for (uint i = tid; i < 32 * 32; i += TM * 4) {     // stage + convert W
        uint n = i / 32, kk = i % 32;
        ulong gn = n0 + n;
        float v;
        if (WT == 1)
          v = float(((device const half*)w)[gn * K + k0 + kk]);
        else
          v = ((device const float*)ws)[gn] *
              float(((device const char*)w)[gn * K + k0 + kk]);
        Wt[n * 33 + kk] = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32; kk += 8) {
      simdgroup_float8x8 A;
      simdgroup_load(A, Xt + (sg * 8) * 33 + kk, 33);
      for (int j = 0; j < 4; j++) {
        simdgroup_float8x8 B;
        simdgroup_load(B, Wt + (j * 8) * 33 + kk, 33, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(C[j], A, B, C[j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Most projections overwrite a fresh output. When M is 8-row aligned,
  // store each simdgroup tile straight to device memory and skip the output
  // scratch tile, threadgroup barrier, and scalar copy-out entirely.
  if (!ACCUMULATE && (args.M & 7) == 0) {
    if (m0 + sg * 8 < args.M)
      for (int j = 0; j < 4; j++)
        simdgroup_store(C[j], y + (ulong)(m0 + sg * 8) * args.N + n0 + j * 8,
                        args.N);
    return;
  }

  for (int j = 0; j < 4; j++)
    simdgroup_store(C[j], Ct + (sg * 8) * 33 + j * 8, 33);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < TM * 32; i += TM * 4) {
    uint r = i / 32, c = i % 32;
    if (m0 + r < args.M) {
      device float* d = y + (ulong)(m0 + r) * args.N + n0 + c;
      if (ACCUMULATE)
        *d += Ct[r * 33 + c];
      else
        *d = Ct[r * 33 + c];
    }
  }
}

// out[row] = rmsnorm(in[row]) * scale, rows of length n (multiple of 32).
// One simdgroup (32 lanes) per row.
kernel void rmsnorm_rows(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         device const float* scale [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  device const float* x = in + (ulong)row * n;
  device float* y = out + (ulong)row * n;
  float ss = 0.0f;
  for (uint i = lane; i < n; i += 32) ss += x[i] * x[i];
  ss = simd_sum(ss);
  float inv = 1.0f / sqrt(ss / float(n) + kEps);
  for (uint i = lane; i < n; i += 32) y[i] = x[i] * inv * scale[i];
}

// Attention pre-norm plus zeroing of the attention output row. Folding the
// clear into this already row-coalesced pass avoids a separate blit encoder.
kernel void rmsnorm_rows_clear(device const float* in [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               device const float* scale [[buffer(2)]],
                               device float* clear [[buffer(3)]],
                               constant uint& n [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
  device const float* x = in + (ulong)row * n;
  device float* y = out + (ulong)row * n;
  device float* z = clear + (ulong)row * n;
  float ss = 0.0f;
  for (uint i = lane; i < n; i += 32) ss += x[i] * x[i];
  ss = simd_sum(ss);
  float inv = 1.0f / sqrt(ss / float(n) + kEps);
  for (uint i = lane; i < n; i += 32) {
    y[i] = x[i] * inv * scale[i];
    z[i] = 0.0f;
  }
}

// In-place QK-RMSNorm on the fused qkvg buffer (row = [q|k|v|g], stride 2048).
// One simdgroup per (token, head, q-or-k) segment of 64 floats.
kernel void qknorm(device float* qkvg [[buffer(0)]],
                   device const float* q_scale [[buffer(1)]],
                   device const float* k_scale [[buffer(2)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint lane [[thread_index_in_simdgroup]]) {
  uint token = tg / 16, seg = tg % 16;
  uint head = seg / 2, isk = seg % 2;
  device float* x = qkvg + (ulong)token * 2048 + isk * 512 + head * 64;
  device const float* scale = isk ? k_scale : q_scale;
  float a = x[lane], b = x[lane + 32];
  float ss = simd_sum(a * a + b * b);
  float inv = 1.0f / sqrt(ss / 64.0f + kEps);
  x[lane] = a * inv * scale[lane];
  x[lane + 32] = b * inv * scale[lane + 32];
}

struct AttnWork {
  int qstart;    // into qidx
  int tq;        // queries in this tile
  int kstart;    // into kidx
  int nk;        // shared key count
  int rowbase;   // batch row * S
  float logkv;   // log(clamp_min(bf16(nk), 1))
};

// One threadgroup per work item; each simdgroup takes (query, head) pairs and
// streams the key list with an online softmax. q/k in qkvg are already
// QK-normed; v sits at +1024. Output rows in att (query scaling folds
// head_scale * logkv / head_dim into q).
kernel void attn(device const float* qkvg [[buffer(0)]],
                 device float* att [[buffer(1)]],
                 device const int* qidx [[buffer(2)]],
                 device const int* kidx [[buffer(3)]],
                 device const AttnWork* work [[buffer(4)]],
                 device const float* head_scale [[buffer(5)]],
                 uint tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint nsg [[simdgroups_per_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
  const AttnWork w = work[tg];
  for (uint p = sg; p < uint(w.tq) * 8; p += nsg) {
    uint r = p / 8, h = p % 8;
    uint qrowi = uint(w.rowbase + qidx[w.qstart + int(r)]);
    device const float* q = qkvg + (ulong)qrowi * 2048 + h * 64;
    float qscale = head_scale[h] * w.logkv / 64.0f;
    float q0 = q[2 * lane] * qscale;
    float q1 = q[2 * lane + 1] * qscale;
    float mx = -INFINITY, den = 0.0f, a0 = 0.0f, a1 = 0.0f;
    for (int j = 0; j < w.nk; j++) {
      uint krowi = uint(w.rowbase + kidx[w.kstart + j]);
      device const float* k = qkvg + (ulong)krowi * 2048 + 512 + h * 64;
      float score = simd_sum(q0 * k[2 * lane] + q1 * k[2 * lane + 1]);
      device const float* v = k + 512;
      float nm = max(mx, score);
      float corr = exp(mx - nm);
      float wt = exp(score - nm);
      den = den * corr + wt;
      a0 = a0 * corr + wt * v[2 * lane];
      a1 = a1 * corr + wt * v[2 * lane + 1];
      mx = nm;
    }
    device float* o = att + (ulong)qrowi * 512 + h * 64;
    o[2 * lane] = a0 / den;
    o[2 * lane + 1] = a1 / den;
  }
}

// ---- flash-style split-K attention for long key lists --------------------
// Work items whose key list exceeds kFlashK keys are split into key chunks
// processed by independent threadgroups (flash-decoding style): attn_part
// emits per-(query, head, chunk) partials {running max m, denom l,
// unnormalized weighted-V sum o[64]}, and attn_reduce merges the chunks
// with the online-softmax identity  l = sum l_c*exp(m_c-M),
// o = sum o_c*exp(m_c-M). This turns the latency-bound single-simdgroup
// key stream of e.g. a 1500-key reverse-FK group into nk/kFlashK parallel
// streams, which is what the B=1 long-context case needs.
// Partials layout: item base + ((chunk*tq + r)*8 + h) * 66.

struct AttnPWork {
  int qstart;    // into qidx
  int tq;        // queries in this tile
  int kstart;    // into kidx (start of THIS chunk)
  int nk;        // keys in this chunk
  int rowbase;   // batch row * S
  float logkv;   // log of the FULL key count (query scaling is per-group)
  int part;      // float offset of this chunk's partials
};

kernel void attn_part(device const float* qkvg [[buffer(0)]],
                      device float* partials [[buffer(1)]],
                      device const int* qidx [[buffer(2)]],
                      device const int* kidx [[buffer(3)]],
                      device const AttnPWork* work [[buffer(4)]],
                      device const float* head_scale [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint nsg [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const AttnPWork w = work[tg];
  for (uint p = sg; p < uint(w.tq) * 8; p += nsg) {
    uint r = p / 8, h = p % 8;
    uint qrowi = uint(w.rowbase + qidx[w.qstart + int(r)]);
    device const float* q = qkvg + (ulong)qrowi * 2048 + h * 64;
    float qscale = head_scale[h] * w.logkv / 64.0f;
    float q0 = q[2 * lane] * qscale;
    float q1 = q[2 * lane + 1] * qscale;
    float mx = -INFINITY, den = 0.0f, a0 = 0.0f, a1 = 0.0f;
    for (int j = 0; j < w.nk; j++) {
      uint krowi = uint(w.rowbase + kidx[w.kstart + j]);
      device const float* k = qkvg + (ulong)krowi * 2048 + 512 + h * 64;
      float score = simd_sum(q0 * k[2 * lane] + q1 * k[2 * lane + 1]);
      device const float* v = k + 512;
      float nm = max(mx, score);
      float corr = exp(mx - nm);
      float wt = exp(score - nm);
      den = den * corr + wt;
      a0 = a0 * corr + wt * v[2 * lane];
      a1 = a1 * corr + wt * v[2 * lane + 1];
      mx = nm;
    }
    device float* o = partials + w.part + (ulong)(r * 8 + h) * 66;
    if (lane == 0) { o[0] = mx; o[1] = den; }
    o[2 + 2 * lane] = a0;
    o[2 + 2 * lane + 1] = a1;
  }
}

struct AttnRWork {
  int qstart;    // into qidx
  int tq;
  int rowbase;
  int part;      // float offset of chunk 0's partials
  int nchunks;   // chunk stride = tq*8*66 floats
};

kernel void attn_reduce(device const float* partials [[buffer(0)]],
                        device float* att [[buffer(1)]],
                        device const int* qidx [[buffer(2)]],
                        device const AttnRWork* work [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint nsg [[simdgroups_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const AttnRWork w = work[tg];
  const ulong cstride = (ulong)w.tq * 8 * 66;
  for (uint p = sg; p < uint(w.tq) * 8; p += nsg) {
    uint r = p / 8, h = p % 8;
    device const float* base = partials + w.part + (ulong)(r * 8 + h) * 66;
    float M = -INFINITY;
    for (int c = 0; c < w.nchunks; c++) M = max(M, base[c * cstride]);
    float l = 0.0f, o0 = 0.0f, o1 = 0.0f;
    for (int c = 0; c < w.nchunks; c++) {
      device const float* pc = base + c * cstride;
      float f = exp(pc[0] - M);
      l += pc[1] * f;
      o0 += pc[2 + 2 * lane] * f;
      o1 += pc[2 + 2 * lane + 1] * f;
    }
    uint qrowi = uint(w.rowbase + qidx[w.qstart + int(r)]);
    device float* o = att + (ulong)qrowi * 512 + h * 64;
    o[2 * lane] = o0 / l;
    o[2 * lane + 1] = o1 / l;
  }
}

// att *= 2*sigmoid(g), g = qkvg[token, 1536 + d]. Grid: BS*512 threads.
kernel void gate_mul(device float* att [[buffer(0)]],
                     device const float* qkvg [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  ulong token = gid / 512, d = gid % 512;
  float g = qkvg[token * 2048 + 1536 + d];
  att[gid] *= 2.0f / (1.0f + exp(-g));
}

// ffa = silu(ffa) * ffb. Grid: BS*2048 threads.
kernel void swiglu(device float* ffa [[buffer(0)]],
                   device const float* ffb [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
  float a = ffa[gid];
  ffa[gid] = (a / (1.0f + exp(-a))) * ffb[gid];
}

// SwiGLU from the row-interleaved output of a stacked native-MPS [w1; w3]
// GEMM. Keeping the two projections together removes one launch per block.
kernel void swiglu_packed(device const float* ff13 [[buffer(0)]],
                          device float* ffa [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
  ulong token = gid / 2048, d = gid % 2048;
  device const float* row = ff13 + token * 4096;
  float a = row[d];
  ffa[gid] = (a / (1.0f + exp(-a))) * row[2048 + d];
}

// yhat[row] = dec_b + dot(rmsnorm(x[row]) * norm_scale, dec_w).
// One simdgroup per row (n = 512).
kernel void head(device const float* x [[buffer(0)]],
                 device const float* norm_scale [[buffer(1)]],
                 device const float* dec_w [[buffer(2)]],
                 constant float& dec_b [[buffer(3)]],
                 device float* yhat [[buffer(4)]],
                 uint row [[threadgroup_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]]) {
  device const float* xr = x + (ulong)row * 512;
  float ss = 0.0f;
  for (uint i = lane; i < 512; i += 32) ss += xr[i] * xr[i];
  ss = simd_sum(ss);
  float inv = 1.0f / sqrt(ss / 512.0f + kEps);
  float d = 0.0f;
  for (uint i = lane; i < 512; i += 32)
    d += xr[i] * inv * norm_scale[i] * dec_w[i];
  d = simd_sum(d);
  if (lane == 0) yhat[row] = dec_b + d;
}
)MSL";

struct AttnWorkGpu {
  int qstart, tq, kstart, nk, rowbase;
  float logkv;
};

// Host mirrors of the flash split-K structs (see the MSL above).
struct AttnPWorkGpu {
  int qstart, tq, kstart, nk, rowbase;
  float logkv;
  int part;
};
struct AttnRWorkGpu {
  int qstart, tq, rowbase, part, nchunks;
};

// A group whose shared key list exceeds this is split into chunks of this
// size, each streamed by an independent threadgroup (flash-decoding).
constexpr int kFlashChunk = 256;
constexpr int kFlashSplit = 512;   // only split when nk exceeds this

// Weight on the GPU: fp32 buffer (MPS GEMM path) or quantized payload +
// scales (custom qgemm path). Uploaded once per model.
struct GpuWeight {
  WType type{};
  id<MTLBuffer> w;                     // f32 or quantized payload
  id<MTLBuffer> s;                     // Q8/Q4 scales (nil otherwise)
  int out = 0, in = 0;
};

struct BlockWeights {
  GpuWeight wqkvg[3], wo[3];           // per attention type (col, feat, nbr)
  GpuWeight w1, w2, w3, w13;           // w13 = stacked MPS [w1; w3]
  id<MTLBuffer> norm[4];
  id<MTLBuffer> q_norm[3], k_norm[3], head_scale[3];
};

struct QGemmArgsHost {
  uint32_t M, N, K;
};

struct MetalCtx {
  std::mutex mu;                       // serializes forwards on this model
  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> p_rms, p_rms_clear, p_qknorm, p_attn, p_gate;
  id<MTLComputePipelineState> p_swiglu, p_swiglu_packed, p_head;
  id<MTLComputePipelineState> p_attn_part, p_attn_reduce;
  // Indexed by [WType][overwrite/accumulate][32/64-row tile].
  id<MTLComputePipelineState> p_qgemm[4][2][2];
  BlockWeights blk[kBlocks];
  id<MTLBuffer> norm_out, dec_w;
  float dec_b = 0.f;
  // GEMM kernel cache: (M, N, K, beta) with transposeRight always true
  std::map<std::tuple<int, int, int, int>, MPSMatrixMultiplication*> gemms;
  // reusable activation / index buffers (grow-on-demand)
  id<MTLBuffer> x, xn, qkvg, att, ffa, ffb, ff13, yhat, tap;
  id<MTLBuffer> qidx[3], kidx[3], work[3];
  // flash split-K attention (large key lists): partials + part/reduce work
  id<MTLBuffer> pwork[3], rwork[3], partials;
};

id<MTLDevice> pick_device() {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (dev) return dev;
  NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
  return all.count ? all[0] : nil;
}

id<MTLBuffer> upload(id<MTLDevice> dev, const float* p, size_t n) {
  return [dev newBufferWithBytes:p
                          length:n * sizeof(float)
                         options:MTLResourceStorageModeShared];
}

void ensure(id<MTLDevice> dev, id<MTLBuffer> __strong& buf, size_t bytes) {
  if (!buf || buf.length < bytes)
    buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

MetalCtx* make_ctx(const Model& m) {
  auto* ctx = new MetalCtx();
  ctx->dev = pick_device();
  if (!ctx->dev) throw std::runtime_error("rt/metal: no Metal device");
  ctx->queue = [ctx->dev newCommandQueue];

  NSError* err = nil;
  MTLCompileOptions* opts = [MTLCompileOptions new];
  // keep exp/rsqrt at fp32 precision
  if (@available(macOS 15.0, *)) {
    opts.mathMode = MTLMathModeSafe;
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
  }
  id<MTLLibrary> lib =
      [ctx->dev newLibraryWithSource:@(kMsl) options:opts error:&err];
  if (!lib)
    throw std::runtime_error(
        std::string("rt/metal: shader compile failed: ") +
        (err ? err.localizedDescription.UTF8String : "?"));
  auto pipe = [&](NSString* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    NSError* perr = nil;
    id<MTLComputePipelineState> ps =
        [ctx->dev newComputePipelineStateWithFunction:fn error:&perr];
    if (!ps)
      throw std::runtime_error(std::string("rt/metal: pipeline failed: ") +
                               name.UTF8String);
    return ps;
  };
  ctx->p_rms = pipe(@"rmsnorm_rows");
  ctx->p_rms_clear = pipe(@"rmsnorm_rows_clear");
  ctx->p_qknorm = pipe(@"qknorm");
  ctx->p_attn = pipe(@"attn");
  ctx->p_attn_part = pipe(@"attn_part");
  ctx->p_attn_reduce = pipe(@"attn_reduce");
  ctx->p_gate = pipe(@"gate_mul");
  ctx->p_swiglu = pipe(@"swiglu");
  ctx->p_swiglu_packed = pipe(@"swiglu_packed");
  ctx->p_head = pipe(@"head");
  for (int wt : {(int)WType::Q8, (int)WType::Q4})
    for (int accumulate = 0; accumulate < 2; accumulate++)
      for (int tile = 0; tile < 2; tile++) {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        [cv setConstantValue:&wt type:MTLDataTypeInt atIndex:0];
        bool acc = accumulate;
        [cv setConstantValue:&acc type:MTLDataTypeBool atIndex:1];
        uint16_t tile_m = tile ? 64 : 32;
        [cv setConstantValue:&tile_m type:MTLDataTypeUShort atIndex:2];
        NSError* ferr = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:@"qgemm"
                                       constantValues:cv
                                                error:&ferr];
        NSError* perr = nil;
        ctx->p_qgemm[wt][accumulate][tile] =
            fn ? [ctx->dev newComputePipelineStateWithFunction:fn error:&perr]
               : nil;
        if (!ctx->p_qgemm[wt][accumulate][tile])
          throw std::runtime_error("rt/metal: qgemm pipeline failed");
      }

  auto gw = [&](const Weight& w) {
    GpuWeight g;
    g.type = w.type;
    g.out = w.out;
    g.in = w.in;
    if (w.type == WType::F32) {
      g.w = upload(ctx->dev, w.f32, (size_t)w.out * w.in);
    } else {
      g.w = [ctx->dev newBufferWithBytes:w.q
                                  length:row_bytes(w.type, w.in) * w.out
                                 options:MTLResourceStorageModeShared];
      size_t sb = scale_bytes(w.type, w.in) * w.out;
      g.s = sb ? [ctx->dev newBufferWithBytes:w.qs
                                       length:sb
                                      options:MTLResourceStorageModeShared]
               : nil;
    }
    return g;
  };
  for (int b = 0; b < kBlocks; b++) {
    const Block& blk = m.blocks[b];
    BlockWeights& g = ctx->blk[b];
    for (int a = 0; a < 3; a++) {
      g.wqkvg[a] = gw(blk.attn[a].wqkvg);
      g.wo[a] = gw(blk.attn[a].wo);
      g.q_norm[a] = upload(ctx->dev, blk.attn[a].q_norm, kHeadDim);
      g.k_norm[a] = upload(ctx->dev, blk.attn[a].k_norm, kHeadDim);
      g.head_scale[a] = upload(ctx->dev, blk.attn[a].head_scale, kHeads);
      g.norm[a] = upload(ctx->dev, blk.norm[a], kD);
    }
    g.norm[3] = upload(ctx->dev, blk.norm[3], kD);
    g.w2 = gw(blk.w2);
    if (blk.w1.type == blk.w3.type &&
        (blk.w1.type == WType::F32 || blk.w1.type == WType::F16)) {
      g.w13.type = blk.w1.type;
      g.w13.out = 2 * kDFF;
      g.w13.in = kD;
      const size_t one_bytes = (size_t)kDFF * row_bytes(blk.w1.type, kD);
      g.w13.w = [ctx->dev newBufferWithLength:2 * one_bytes
                                      options:MTLResourceStorageModeShared];
      const void* w1 = blk.w1.type == WType::F32 ? (const void*)blk.w1.f32
                                                 : (const void*)blk.w1.q;
      const void* w3 = blk.w3.type == WType::F32 ? (const void*)blk.w3.f32
                                                 : (const void*)blk.w3.q;
      std::memcpy(g.w13.w.contents, w1, one_bytes);
      std::memcpy((char*)g.w13.w.contents + one_bytes, w3, one_bytes);
    } else {
      g.w1 = gw(blk.w1);
      g.w3 = gw(blk.w3);
    }
  }
  ctx->norm_out = upload(ctx->dev, m.norm_out, kD);
  ctx->dec_w = upload(ctx->dev, m.dec_number.w, kD);
  ctx->dec_b = m.dec_number.b[0];
  return ctx;
}

}  // namespace

bool metal_available() {
  static bool ok = [] {
    @autoreleasepool {
      return pick_device() != nil;
    }
  }();
  return ok;
}

void run_blocks_metal(const Model& m, Prepared& prep, Output& out,
                      bool debug_taps) {
  @autoreleasepool {
    // ---- lazy per-model context ------------------------------------------
    static std::mutex init_mu;
    std::shared_ptr<void>& slot = m.device_ctx[(int)Device::MPS];
    {
      std::lock_guard<std::mutex> lk(init_mu);
      if (!slot) slot.reset(make_ctx(m), [](void* p) { delete (MetalCtx*)p; });
    }
    MetalCtx& ctx = *(MetalCtx*)slot.get();
    std::lock_guard<std::mutex> lk(ctx.mu);

    const int B = prep.B, S = prep.S;
    const size_t BS = (size_t)B * S;

    // ---- flatten group indices / work items for the GPU ------------------
    // Small groups (nk <= kFlashSplit) run the single-pass `attn` kernel;
    // large groups are split into kFlashChunk-key chunks handled by the
    // flash split-K pair (attn_part -> attn_reduce).
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
          wflat[a].push_back({qs, tq, ks, nk, W.b * S, W.logkv});
          continue;
        }
        const int nchunks = (nk + kFlashChunk - 1) / kFlashChunk;
        const int item_base = (int)poff;
        for (int c = 0; c < nchunks; c++) {
          const int c0 = c * kFlashChunk;
          const int cnk = std::min(kFlashChunk, nk - c0);
          pflat[a].push_back({qs, tq, ks + c0, cnk, W.b * S, W.logkv,
                              item_base + c * (tq * 8 * 66)});
        }
        rflat[a].push_back({qs, tq, W.b * S, item_base, nchunks});
        poff += (size_t)nchunks * tq * 8 * 66;
      }
      part_floats[a] = poff;
    }
    const size_t part_max =
        std::max({part_floats[0], part_floats[1], part_floats[2], (size_t)1});

    // ---- buffers ----------------------------------------------------------
    ensure(ctx.dev, ctx.x, BS * kD * 4);
    ensure(ctx.dev, ctx.xn, BS * kD * 4);
    ensure(ctx.dev, ctx.qkvg, BS * (size_t)kC4 * 4);
    ensure(ctx.dev, ctx.att, BS * kD * 4);
    ensure(ctx.dev, ctx.ffa, BS * (size_t)kDFF * 4);
    if (ctx.blk[0].w13.w)
      ensure(ctx.dev, ctx.ff13, BS * (size_t)(2 * kDFF) * 4);
    else
      ensure(ctx.dev, ctx.ffb, BS * (size_t)kDFF * 4);
    ensure(ctx.dev, ctx.yhat, BS * 4);
    if (debug_taps) ensure(ctx.dev, ctx.tap, BS * kD * 4);
    ensure(ctx.dev, ctx.partials, part_max * 4);
    for (int a = 0; a < 3; a++) {
      ensure(ctx.dev, ctx.qidx[a], std::max<size_t>(1, qflat[a].size()) * 4);
      ensure(ctx.dev, ctx.kidx[a], std::max<size_t>(1, kflat[a].size()) * 4);
      ensure(ctx.dev, ctx.work[a],
             std::max<size_t>(1, wflat[a].size()) * sizeof(AttnWorkGpu));
      std::memcpy(ctx.qidx[a].contents, qflat[a].data(), qflat[a].size() * 4);
      std::memcpy(ctx.kidx[a].contents, kflat[a].data(), kflat[a].size() * 4);
      std::memcpy(ctx.work[a].contents, wflat[a].data(),
                  wflat[a].size() * sizeof(AttnWorkGpu));
      ensure(ctx.dev, ctx.pwork[a],
             std::max<size_t>(1, pflat[a].size()) * sizeof(AttnPWorkGpu));
      ensure(ctx.dev, ctx.rwork[a],
             std::max<size_t>(1, rflat[a].size()) * sizeof(AttnRWorkGpu));
      if (!pflat[a].empty())
        std::memcpy(ctx.pwork[a].contents, pflat[a].data(),
                    pflat[a].size() * sizeof(AttnPWorkGpu));
      if (!rflat[a].empty())
        std::memcpy(ctx.rwork[a].contents, rflat[a].data(),
                    rflat[a].size() * sizeof(AttnRWorkGpu));
    }
    std::memcpy(ctx.x.contents, prep.x.data(), BS * kD * 4);

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];

    auto gemm = [&](id<MTLBuffer> a, id<MTLBuffer> w, id<MTLBuffer> c, int M,
                    int N, int K, float beta, MPSDataType wtype) {
      auto key = std::make_tuple(M, N, K, (int)beta);
      MPSMatrixMultiplication* __strong& mm = ctx.gemms[key];
      if (!mm)
        mm = [[MPSMatrixMultiplication alloc] initWithDevice:ctx.dev
                                               transposeLeft:false
                                              transposeRight:true
                                                  resultRows:M
                                               resultColumns:N
                                             interiorColumns:K
                                                       alpha:1.0
                                                        beta:beta];
      auto desc = [&](int r, int cc, MPSDataType type) {
        const size_t elem = type == MPSDataTypeFloat16 ? 2 : 4;
        return [MPSMatrixDescriptor matrixDescriptorWithRows:r
                                                     columns:cc
                                                    rowBytes:(size_t)cc * elem
                                                    dataType:type];
      };
      MPSMatrix* A = [[MPSMatrix alloc]
          initWithBuffer:a descriptor:desc(M, K, MPSDataTypeFloat32)];
      MPSMatrix* W = [[MPSMatrix alloc] initWithBuffer:w
                                           descriptor:desc(N, K, wtype)];
      MPSMatrix* C = [[MPSMatrix alloc]
          initWithBuffer:c descriptor:desc(M, N, MPSDataTypeFloat32)];
      [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:W resultMatrix:C];
    };

    // Keep adjacent custom kernels in one encoder. Quantized forwards issue
    // over a hundred small compute dispatches, and creating an encoder for
    // every projection is measurable at latency-oriented shapes.
    id<MTLComputeCommandEncoder> enc = nil;
    auto compute = [&]() -> id<MTLComputeCommandEncoder> {
      if (!enc) enc = [cb computeCommandEncoder];
      return enc;
    };
    auto end_compute = [&] {
      if (enc) {
        [enc endEncoding];
        enc = nil;
      }
    };
    auto buffer_barrier = [&] {
      if (enc) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };

    // Projection dispatch: fp32/f16 use native MPS GEMM (MPS accepts mixed
    // fp32 activations/results with f16 weights); q8/q4 run custom qgemm.
    auto proj = [&](id<MTLBuffer> a, const GpuWeight& w, id<MTLBuffer> c,
                    int M, float beta) {
      if (w.type == WType::F32 || w.type == WType::F16) {
        end_compute();
        gemm(a, w.w, c, M, w.out, w.in, beta,
             w.type == WType::F16 ? MPSDataTypeFloat16
                                  : MPSDataTypeFloat32);
        return;
      }
      id<MTLComputeCommandEncoder> qenc = compute();
      const int accumulate = beta != 0.f;
      const int tile = M >= 128;
      const int tile_m = tile ? 64 : 32;
      [qenc setComputePipelineState:ctx.p_qgemm[(int)w.type][accumulate][tile]];
      [qenc setBuffer:a offset:0 atIndex:0];
      [qenc setBuffer:w.w offset:0 atIndex:1];
      [qenc setBuffer:(w.s ? w.s : w.w) offset:0 atIndex:2];
      [qenc setBuffer:c offset:0 atIndex:3];
      QGemmArgsHost args{(uint32_t)M, (uint32_t)w.out, (uint32_t)w.in};
      [qenc setBytes:&args length:sizeof(args) atIndex:4];
      [qenc setThreadgroupMemoryLength:(size_t)tile_m * 33 * sizeof(float)
                               atIndex:0];
      [qenc setThreadgroupMemoryLength:32 * 33 * sizeof(float) atIndex:1];
      const size_t output_scratch =
          !accumulate && M % 8 == 0
              ? 0
              : (size_t)tile_m * 33 * sizeof(float);
      [qenc setThreadgroupMemoryLength:output_scratch atIndex:2];
      [qenc dispatchThreadgroups:MTLSizeMake((w.out + 31) / 32,
                                             (M + tile_m - 1) / tile_m, 1)
             threadsPerThreadgroup:MTLSizeMake(tile_m * 4, 1, 1)];
    };

    auto simdrows = [&](id<MTLComputePipelineState> ps, size_t rows,
                        id<MTLComputeCommandEncoder> enc) {
      [enc setComputePipelineState:ps];
      [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    };

    // ---- transformer blocks ----------------------------------------------
    for (int blk_i = 0; blk_i < kBlocks; blk_i++) {
      const BlockWeights& gw = ctx.blk[blk_i];
      for (int a = 0; a < 3; a++) {
        // Pre-norm and clear attention in one row pass. The barrier also
        // makes the preceding residual projection visible when this remains
        // in the same quantized compute encoder.
        buffer_barrier();
        id<MTLComputeCommandEncoder> cenc = compute();
        uint32_t n = kD;
        [cenc setBuffer:ctx.x offset:0 atIndex:0];
        [cenc setBuffer:ctx.xn offset:0 atIndex:1];
        [cenc setBuffer:gw.norm[a] offset:0 atIndex:2];
        [cenc setBuffer:ctx.att offset:0 atIndex:3];
        [cenc setBytes:&n length:4 atIndex:4];
        simdrows(ctx.p_rms_clear, BS, cenc);
        buffer_barrier();
        // fused qkvg projection
        proj(ctx.xn, gw.wqkvg[a], ctx.qkvg, (int)BS, 0.f);
        buffer_barrier();
        // qk-norm + attention + gating
        cenc = compute();
        [cenc setComputePipelineState:ctx.p_qknorm];
        [cenc setBuffer:ctx.qkvg offset:0 atIndex:0];
        [cenc setBuffer:gw.q_norm[a] offset:0 atIndex:1];
        [cenc setBuffer:gw.k_norm[a] offset:0 atIndex:2];
        [cenc dispatchThreadgroups:MTLSizeMake(BS * 16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        buffer_barrier();
        if (!wflat[a].empty()) {          // single-pass small groups
          [enc setComputePipelineState:ctx.p_attn];
          [enc setBuffer:ctx.qkvg offset:0 atIndex:0];
          [enc setBuffer:ctx.att offset:0 atIndex:1];
          [enc setBuffer:ctx.qidx[a] offset:0 atIndex:2];
          [enc setBuffer:ctx.kidx[a] offset:0 atIndex:3];
          [enc setBuffer:ctx.work[a] offset:0 atIndex:4];
          [enc setBuffer:gw.head_scale[a] offset:0 atIndex:5];
          [enc dispatchThreadgroups:MTLSizeMake(wflat[a].size(), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * 4, 1, 1)];
        }
        if (!rflat[a].empty()) {          // flash split-K large groups
          // serial encoder: part -> reduce is implicitly ordered
          [enc setComputePipelineState:ctx.p_attn_part];
          [enc setBuffer:ctx.qkvg offset:0 atIndex:0];
          [enc setBuffer:ctx.partials offset:0 atIndex:1];
          [enc setBuffer:ctx.qidx[a] offset:0 atIndex:2];
          [enc setBuffer:ctx.kidx[a] offset:0 atIndex:3];
          [enc setBuffer:ctx.pwork[a] offset:0 atIndex:4];
          [enc setBuffer:gw.head_scale[a] offset:0 atIndex:5];
          [enc dispatchThreadgroups:MTLSizeMake(pflat[a].size(), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * 4, 1, 1)];
          buffer_barrier();
          [enc setComputePipelineState:ctx.p_attn_reduce];
          [enc setBuffer:ctx.partials offset:0 atIndex:0];
          [enc setBuffer:ctx.att offset:0 atIndex:1];
          [enc setBuffer:ctx.qidx[a] offset:0 atIndex:2];
          [enc setBuffer:ctx.rwork[a] offset:0 atIndex:3];
          [enc dispatchThreadgroups:MTLSizeMake(rflat[a].size(), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * 4, 1, 1)];
        }
        buffer_barrier();
        [enc setComputePipelineState:ctx.p_gate];
        [enc setBuffer:ctx.att offset:0 atIndex:0];
        [enc setBuffer:ctx.qkvg offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(BS * kD, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        buffer_barrier();
        // x += att @ wo^T
        proj(ctx.att, gw.wo[a], ctx.x, (int)BS, 1.f);
      }
      // FFN: x += w2( silu(w1 xn) * w3 xn )
      buffer_barrier();
      id<MTLComputeCommandEncoder> fenc = compute();
      uint32_t n = kD;
      [fenc setBuffer:ctx.x offset:0 atIndex:0];
      [fenc setBuffer:ctx.xn offset:0 atIndex:1];
      [fenc setBuffer:gw.norm[3] offset:0 atIndex:2];
      [fenc setBytes:&n length:4 atIndex:3];
      simdrows(ctx.p_rms, BS, fenc);
      buffer_barrier();
      if (gw.w13.w) {
        proj(ctx.xn, gw.w13, ctx.ff13, (int)BS, 0.f);
        buffer_barrier();
        fenc = compute();
        [fenc setComputePipelineState:ctx.p_swiglu_packed];
        [fenc setBuffer:ctx.ff13 offset:0 atIndex:0];
        [fenc setBuffer:ctx.ffa offset:0 atIndex:1];
      } else {
        proj(ctx.xn, gw.w1, ctx.ffa, (int)BS, 0.f);
        proj(ctx.xn, gw.w3, ctx.ffb, (int)BS, 0.f);
        buffer_barrier();
        fenc = compute();
        [fenc setComputePipelineState:ctx.p_swiglu];
        [fenc setBuffer:ctx.ffa offset:0 atIndex:0];
        [fenc setBuffer:ctx.ffb offset:0 atIndex:1];
      }
      [fenc dispatchThreads:MTLSizeMake(BS * kDFF, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      buffer_barrier();
      proj(ctx.ffa, gw.w2, ctx.x, (int)BS, 1.f);
      if (blk_i == 0 && debug_taps) {
        end_compute();
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:ctx.x
                sourceOffset:0
                    toBuffer:ctx.tap
           destinationOffset:0
                        size:BS * kD * 4];
        [blit endEncoding];
      }
    }

    // ---- output norm + number head ---------------------------------------
    buffer_barrier();
    id<MTLComputeCommandEncoder> henc = compute();
    [henc setComputePipelineState:ctx.p_head];
    [henc setBuffer:ctx.x offset:0 atIndex:0];
    [henc setBuffer:ctx.norm_out offset:0 atIndex:1];
    [henc setBuffer:ctx.dec_w offset:0 atIndex:2];
    [henc setBytes:&ctx.dec_b length:4 atIndex:3];
    [henc setBuffer:ctx.yhat offset:0 atIndex:4];
    [henc dispatchThreadgroups:MTLSizeMake(BS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    end_compute();

    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError)
      throw std::runtime_error(
          std::string("rt/metal: command buffer failed: ") +
          (cb.error ? cb.error.localizedDescription.UTF8String : "?"));

    std::memcpy(out.yhat_number.data(), ctx.yhat.contents, BS * 4);
    if (debug_taps) {
      out.x_block0.resize(BS * kD);
      std::memcpy(out.x_block0.data(), ctx.tap.contents, BS * kD * 4);
    }
  }
}

}  // namespace detail
}  // namespace rt
