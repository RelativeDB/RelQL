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
#include <string>
#include <unordered_map>
#include <vector>

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

// ---- tensors -------------------------------------------------------------
struct Tensor {                       // dense row-major fp32
  std::vector<int64_t> shape;
  std::vector<float> data;
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

// Parsed safetensors file (bf16 -> fp32 at load).
std::unordered_map<std::string, Tensor> load_safetensors(const std::string& path);

// ---- model ---------------------------------------------------------------
struct Linear {                        // y = x W^T + b   (W: [out, in])
  const float* w = nullptr;
  const float* b = nullptr;            // nullable
  int out = 0, in = 0;
};

struct Attn {
  std::vector<float> wqkvg;            // [4*d, d] — wq/wk/wv/wg stacked, one GEMM
  Linear wo;
  const float* q_norm = nullptr;       // [head_dim]
  const float* k_norm = nullptr;       // [head_dim]
  const float* head_scale = nullptr;   // [heads]
};

struct Block {
  Attn attn[3];                        // col, feat, nbr
  const float* norm[4] = {};           // col, feat, nbr, ffn   [d]
  Linear w1, w2, w3;                   // SwiGLU
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

// debug_taps=false skips copying x_embed/x_block0 into the Output (bench mode).
Output forward(const Model& m, const Batch& batch, int n_threads = 0,
               bool debug_taps = true);

}  // namespace rt
