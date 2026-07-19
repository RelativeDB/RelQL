// rt.hpp — Relational Transformer (RT-J) inference in C++.
//
// Faithful port of rt/model.py (stanford-star/relational-transformer, main):
// 12 blocks x [col | feat | nbr masked attention + SwiGLU FFN], RMSNorm
// everywhere, QK-RMSNorm, learnable per-head scale x log(kv_count) query
// scaling, sigmoid output gating x2, per-sem-type encoders/decoders, no
// positional encodings. Scale on attention scores is 1/head_dim (not rsqrt).
//
// Optimization notes (idioms from llama.cpp / vllm):
//  - all dense projections are Accelerate cblas_sgemm over (B*S, d) panels;
//    wq/wk/wv/wg are stacked into one [4d, d] weight so QKV+gate is one GEMM
//  - masked attention never materializes S x S: queries sharing a key list
//    (column groups / (node, FK-set) groups / reverse-FK lists) are batched,
//    so scores and output run as per-head GEMMs over query tiles, with a
//    numerically-stable two-pass softmax (exp via vvexpf) between them
//  - weights converted bf16 -> fp32 once at load, contiguous row-major
//  - attention parallelized across (batch, group, query-tile) work items on a
//    persistent thread pool (workers park between jobs, no per-pass spawns)
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rt_quant.hpp"

namespace rt {

constexpr int kDModel = 512;
constexpr int kHeads = 8;
constexpr int kHeadDim = kDModel / kHeads;   // 64
constexpr int kDFF = 2048;
constexpr int kBlocks = 12;
constexpr int kDText = 384;
constexpr int kMaxF2p = 5;
constexpr float kEps = 1e-6f;
enum SemType { kNumber = 0, kText = 1, kDatetime = 2, kBoolean = 3 };

// ---- devices -------------------------------------------------------------
// CPU: Accelerate on Apple, register-blocked portable SIMD kernels elsewhere.
// MPS: Metal GPU — MPSMatrixMultiplication GEMMs + custom kernels (macOS).
// CUDA: cuBLAS GEMMs + custom kernels (builds with -DRT_CUDA=ON).
enum class Device { CPU = 0, MPS = 1, CUDA = 2 };

// True if the backend is compiled in and a usable device is present.
bool device_available(Device d);
const char* device_name(Device d);

// ---- tensors -------------------------------------------------------------
struct Tensor {                       // dense row-major
  std::vector<int64_t> shape;         // logical shape (Q4: unpacked)
  std::vector<float> data;            // fp32 payload (empty when quantized)
  // quantized payload, kept resident — compute paths dequantize in-kernel.
  uint8_t qtype = 0;                  // (uint8_t)WType; 0 = fp32
  std::vector<uint8_t> qdata;         // F16/Q8/Q4 raw payload
  std::vector<uint8_t> qscale;        // Q8: f32[out]; Q4: f16[out*in/32]
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

// Parsed safetensors file (bf16 -> fp32 at load; F16/I8/Q4 kept quantized).
std::unordered_map<std::string, Tensor> load_safetensors(const std::string& path);

// ---- model ---------------------------------------------------------------
struct Linear {                        // y = x W^T + b   (W: [out, in])
  const float* w = nullptr;
  const float* b = nullptr;            // nullable
  int out = 0, in = 0;
};

// Quantization-aware weight matrix [out, in] (no bias — block projections).
// F32 uses f32; F16/Q8/Q4 stay quantized in memory and are dequantized
// inside the GEMM kernels (see rt_quant.hpp for the payload layouts).
struct Weight {
  WType type{};                        // WType::F32 by default
  const float* f32 = nullptr;          // F32 payload
  const uint8_t* q = nullptr;          // F16/Q8/Q4 payload
  const uint8_t* qs = nullptr;         // Q8/Q4 scales
  int out = 0, in = 0;
  // Q8 only: lazily-built i8mm (SMMLA) panel repacking of `q`, cached on
  // first use of the CPU SMMLA kernel. `q` itself stays row-major so the GPU
  // backend and the SDOT/tile-dequant paths keep reading the canonical layout.
  mutable std::shared_ptr<std::vector<int8_t>> q8_smmla;
};

struct Attn {
  // fused wq/wk/wv/wg [4*d, d] — one GEMM; backing storage per dtype
  std::vector<float> wqkvg_f32;
  std::vector<uint8_t> wqkvg_q, wqkvg_s;
  Weight wqkvg;
  Weight wo;
  const float* q_norm = nullptr;       // [head_dim]
  const float* k_norm = nullptr;       // [head_dim]
  const float* head_scale = nullptr;   // [heads]
};

struct Block {
  Attn attn[3];                        // col, feat, nbr
  const float* norm[4] = {};           // col, feat, nbr, ffn   [d]
  Weight w1, w2, w3;                   // SwiGLU
};

struct Model {
  std::unordered_map<std::string, Tensor> store;  // owns all weights
  Linear enc[4];                       // number, text, datetime, boolean
  Linear enc_col_name;
  const float* norm_enc[4] = {};       // value-type input norms [d]
  const float* norm_col_name = nullptr;
  const float* mask_emb[4] = {};       // [d]
  Block blocks[kBlocks];
  const float* norm_out = nullptr;
  Linear dec_number;                   // classification score head (bool_as_num)

  // Lazily-created per-device state (weight uploads, pipelines, streams),
  // indexed by Device. Created on first forward for that device; shared by
  // copies of the Model. Opaque — owned by the backend that created it.
  mutable std::shared_ptr<void> device_ctx[3];

  static Model load(const std::string& safetensors_path);
};

// ---- batch ---------------------------------------------------------------
struct Batch {                         // pre-sort order, mirrors rt/data.py
  int B = 0, S = 0;
  std::vector<int64_t> node_idxs;      // [B,S]
  std::vector<int64_t> f2p;            // [B,S,5]  (-1 = none)
  std::vector<int64_t> col_idxs;       // [B,S]
  std::vector<int64_t> table_idxs;     // [B,S]
  std::vector<uint8_t> is_padding;     // [B,S]
  std::vector<int64_t> sem_types;      // [B,S]
  std::vector<uint8_t> is_target;      // [B,S]
  std::vector<float> number_v;         // [B,S]
  std::vector<float> datetime_v;       // [B,S]
  std::vector<float> boolean_v;        // [B,S]
  std::vector<float> text_v;           // [B,S,384]
  std::vector<float> col_name_v;       // [B,S,384]
};

struct Output {
  int B = 0, S = 0;
  std::vector<float> yhat_number;      // [B,S]  post-sort order
  std::vector<uint8_t> sorted_is_target;  // [B,S]
  std::vector<int64_t> sort_idxs;      // [B,S]
  std::vector<float> x_embed;          // [B,S,d] debug tap (block-0 input)
  std::vector<float> x_block0;         // [B,S,d] debug tap (block-0 output)
};

struct ForwardOpts {
  Device device = Device::CPU;
  int n_threads = 0;                   // <=0: hardware concurrency (CPU only)
  bool debug_taps = true;              // fill x_embed/x_block0 (off for bench)
};

Output forward(const Model& m, const Batch& batch, const ForwardOpts& opts);

// Legacy CPU-only entry point (kept for existing callers).
Output forward(const Model& m, const Batch& batch, int n_threads = 0,
               bool debug_taps = true);

}  // namespace rt
