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
//    sparsity the CPU path uses, with the same bf16-rounded log(kv) scaling
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

struct QGemmArgs {
  uint M, N, K;
  float beta;
};

// y[M,N] = x[M,K] @ W[N,K]^T (+ beta*y) with W quantized-resident.
// One threadgroup (128 threads / 4 simdgroups) computes a 32x32 output tile:
// the K loop stages a 32x32 x-tile and a 32x32 *dequantized* W-tile in
// threadgroup memory (dequant happens on the load from DRAM — the fp32 tile
// never exists outside on-chip memory), then each simdgroup accumulates an
// 8x32 strip with simdgroup_float8x8 MMA. K and N are multiples of 32 for
// every RT-J projection; M (tokens) is edge-guarded.
// Q4 layout note: K-chunks of 32 align exactly with Q4's group size, so each
// staged W-row chunk touches one (scale, min) pair.
kernel void qgemm(device const float* x [[buffer(0)]],
                  device const uchar* w [[buffer(1)]],
                  device const uchar* ws [[buffer(2)]],
                  device float* y [[buffer(3)]],
                  constant QGemmArgs& args [[buffer(4)]],
                  uint2 tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint n0 = tg.x * 32, m0 = tg.y * 32;
  const uint K = args.K;
  threadgroup float Xt[32 * 33];
  threadgroup float Wt[32 * 33];
  threadgroup float Ct[32 * 33];
  simdgroup_float8x8 C[4];
  for (int j = 0; j < 4; j++) C[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint k0 = 0; k0 < K; k0 += 32) {
    for (uint i = tid; i < 32 * 32; i += 128) {          // stage x tile
      uint r = i / 32, c = i % 32;
      Xt[r * 33 + c] = (m0 + r < args.M) ? x[(ulong)(m0 + r) * K + k0 + c] : 0.0f;
    }
    for (uint i = tid; i < 32 * 32; i += 128) {          // stage + dequant W
      uint n = i / 32, kk = i % 32;
      ulong gn = n0 + n;
      float v;
      if (WT == 1) {
        v = float(((device const half*)w)[gn * K + k0 + kk]);
      } else if (WT == 2) {
        v = ((device const float*)ws)[gn] *
            float(((device const char*)w)[gn * K + k0 + kk]);
      } else {
        device const half* sh =
            ((device const half*)ws) + (gn * (K >> 5) + ((k0 + kk) >> 5)) * 2;
        uchar b = w[gn * (K >> 1) + ((k0 + kk) >> 1)];
        uint nib = (kk & 1) ? (b >> 4) : (b & 0xf);
        v = float(sh[0]) * float(nib) + float(sh[1]);
      }
      Wt[n * 33 + kk] = v;
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

  for (int j = 0; j < 4; j++)
    simdgroup_store(C[j], Ct + (sg * 8) * 33 + j * 8, 33);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 32 * 32; i += 128) {
    uint r = i / 32, c = i % 32;
    if (m0 + r < args.M) {
      device float* d = y + (ulong)(m0 + r) * args.N + n0 + c;
      *d = args.beta * *d + Ct[r * 33 + c];
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
  GpuWeight w1, w2, w3;
  id<MTLBuffer> norm[4];
  id<MTLBuffer> q_norm[3], k_norm[3], head_scale[3];
};

struct QGemmArgsHost {
  uint32_t M, N, K;
  float beta;
};

struct MetalCtx {
  std::mutex mu;                       // serializes forwards on this model
  id<MTLDevice> dev;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> p_rms, p_qknorm, p_attn, p_gate, p_swiglu, p_head;
  id<MTLComputePipelineState> p_qgemm[4];  // indexed by WType (F16/Q8/Q4)
  BlockWeights blk[kBlocks];
  id<MTLBuffer> norm_out, dec_w;
  float dec_b = 0.f;
  // GEMM kernel cache: (M, N, K, beta) with transposeRight always true
  std::map<std::tuple<int, int, int, int>, MPSMatrixMultiplication*> gemms;
  // reusable activation / index buffers (grow-on-demand)
  id<MTLBuffer> x, xn, qkvg, att, ffa, ffb, yhat, tap;
  id<MTLBuffer> qidx[3], kidx[3], work[3];
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
  ctx->p_qknorm = pipe(@"qknorm");
  ctx->p_attn = pipe(@"attn");
  ctx->p_gate = pipe(@"gate_mul");
  ctx->p_swiglu = pipe(@"swiglu");
  ctx->p_head = pipe(@"head");
  for (int wt : {(int)WType::F16, (int)WType::Q8, (int)WType::Q4}) {
    MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wt type:MTLDataTypeInt atIndex:0];
    NSError* ferr = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"qgemm"
                                   constantValues:cv
                                            error:&ferr];
    NSError* perr = nil;
    ctx->p_qgemm[wt] = fn ? [ctx->dev newComputePipelineStateWithFunction:fn
                                                                    error:&perr]
                          : nil;
    if (!ctx->p_qgemm[wt])
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
    g.w1 = gw(blk.w1);
    g.w2 = gw(blk.w2);
    g.w3 = gw(blk.w3);
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

    // ---- buffers ----------------------------------------------------------
    ensure(ctx.dev, ctx.x, BS * kD * 4);
    ensure(ctx.dev, ctx.xn, BS * kD * 4);
    ensure(ctx.dev, ctx.qkvg, BS * (size_t)kC4 * 4);
    ensure(ctx.dev, ctx.att, BS * kD * 4);
    ensure(ctx.dev, ctx.ffa, BS * (size_t)kDFF * 4);
    ensure(ctx.dev, ctx.ffb, BS * (size_t)kDFF * 4);
    ensure(ctx.dev, ctx.yhat, BS * 4);
    if (debug_taps) ensure(ctx.dev, ctx.tap, BS * kD * 4);
    for (int a = 0; a < 3; a++) {
      ensure(ctx.dev, ctx.qidx[a], std::max<size_t>(1, qflat[a].size()) * 4);
      ensure(ctx.dev, ctx.kidx[a], std::max<size_t>(1, kflat[a].size()) * 4);
      ensure(ctx.dev, ctx.work[a],
             std::max<size_t>(1, wflat[a].size()) * sizeof(AttnWorkGpu));
      std::memcpy(ctx.qidx[a].contents, qflat[a].data(), qflat[a].size() * 4);
      std::memcpy(ctx.kidx[a].contents, kflat[a].data(), kflat[a].size() * 4);
      std::memcpy(ctx.work[a].contents, wflat[a].data(),
                  wflat[a].size() * sizeof(AttnWorkGpu));
    }
    std::memcpy(ctx.x.contents, prep.x.data(), BS * kD * 4);

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];

    auto gemm = [&](id<MTLBuffer> a, id<MTLBuffer> w, id<MTLBuffer> c, int M,
                    int N, int K, float beta) {
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
      auto desc = [&](int r, int cc) {
        return [MPSMatrixDescriptor matrixDescriptorWithRows:r
                                                     columns:cc
                                                    rowBytes:(size_t)cc * 4
                                                    dataType:MPSDataTypeFloat32];
      };
      MPSMatrix* A = [[MPSMatrix alloc] initWithBuffer:a descriptor:desc(M, K)];
      MPSMatrix* W = [[MPSMatrix alloc] initWithBuffer:w descriptor:desc(N, K)];
      MPSMatrix* C = [[MPSMatrix alloc] initWithBuffer:c descriptor:desc(M, N)];
      [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:W resultMatrix:C];
    };

    // Projection dispatch: fp32 weights keep the MPS GEMM; quantized weights
    // run the custom qgemm (threadgroup-staged in-register dequant + MMA).
    auto proj = [&](id<MTLBuffer> a, const GpuWeight& w, id<MTLBuffer> c,
                    int M, float beta) {
      if (w.type == WType::F32) {
        gemm(a, w.w, c, M, w.out, w.in, beta);
        return;
      }
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.p_qgemm[(int)w.type]];
      [enc setBuffer:a offset:0 atIndex:0];
      [enc setBuffer:w.w offset:0 atIndex:1];
      [enc setBuffer:(w.s ? w.s : w.w) offset:0 atIndex:2];
      [enc setBuffer:c offset:0 atIndex:3];
      QGemmArgsHost args{(uint32_t)M, (uint32_t)w.out, (uint32_t)w.in, beta};
      [enc setBytes:&args length:sizeof(args) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((w.out + 31) / 32, (M + 31) / 32, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [enc endEncoding];
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
        // pre-norm
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        uint32_t n = kD;
        [enc setBuffer:ctx.x offset:0 atIndex:0];
        [enc setBuffer:ctx.xn offset:0 atIndex:1];
        [enc setBuffer:gw.norm[a] offset:0 atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        simdrows(ctx.p_rms, BS, enc);
        [enc endEncoding];
        // fused qkvg projection
        proj(ctx.xn, gw.wqkvg[a], ctx.qkvg, (int)BS, 0.f);
        // zero att accumulator (tokens outside every group must stay 0)
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:ctx.att range:NSMakeRange(0, BS * kD * 4) value:0];
        [blit endEncoding];
        // qk-norm + attention + gating
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx.p_qknorm];
        [enc setBuffer:ctx.qkvg offset:0 atIndex:0];
        [enc setBuffer:gw.q_norm[a] offset:0 atIndex:1];
        [enc setBuffer:gw.k_norm[a] offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(BS * 16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        if (!wflat[a].empty()) {
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
        [enc setComputePipelineState:ctx.p_gate];
        [enc setBuffer:ctx.att offset:0 atIndex:0];
        [enc setBuffer:ctx.qkvg offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(BS * kD, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        // x += att @ wo^T
        proj(ctx.att, gw.wo[a], ctx.x, (int)BS, 1.f);
      }
      // FFN: x += w2( silu(w1 xn) * w3 xn )
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      uint32_t n = kD;
      [enc setBuffer:ctx.x offset:0 atIndex:0];
      [enc setBuffer:ctx.xn offset:0 atIndex:1];
      [enc setBuffer:gw.norm[3] offset:0 atIndex:2];
      [enc setBytes:&n length:4 atIndex:3];
      simdrows(ctx.p_rms, BS, enc);
      [enc endEncoding];
      proj(ctx.xn, gw.w1, ctx.ffa, (int)BS, 0.f);
      proj(ctx.xn, gw.w3, ctx.ffb, (int)BS, 0.f);
      enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ctx.p_swiglu];
      [enc setBuffer:ctx.ffa offset:0 atIndex:0];
      [enc setBuffer:ctx.ffb offset:0 atIndex:1];
      [enc dispatchThreads:MTLSizeMake(BS * kDFF, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
      proj(ctx.ffa, gw.w2, ctx.x, (int)BS, 1.f);
      if (blk_i == 0 && debug_taps) {
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
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx.p_head];
    [enc setBuffer:ctx.x offset:0 atIndex:0];
    [enc setBuffer:ctx.norm_out offset:0 atIndex:1];
    [enc setBuffer:ctx.dec_w offset:0 atIndex:2];
    [enc setBytes:&ctx.dec_b length:4 atIndex:3];
    [enc setBuffer:ctx.yhat offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(BS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];

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
