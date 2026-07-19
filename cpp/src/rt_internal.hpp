// rt_internal.hpp — internals shared by the CPU / MPS / CUDA backends.
//
// Host-side batch preparation (stable sort, query-group construction, work
// tiling, value/col-name embeddings) is identical for every device and runs
// on the CPU; the 12 transformer blocks + output head then run on the
// selected backend. GPU backends consume the Prepared struct: x is the
// block-0 input, groups/work describe the sparse attention as flat index
// lists so no S x S mask is ever materialized on device either.
#pragma once

#include <vector>

#include "rt.hpp"

namespace rt {
namespace detail {

// Query-groups for masked attention: every query in a group attends to the
// same key list, so scores/output run as tiles instead of per-query streams.
struct Groups {
  std::vector<int> q;            // flattened query idxs (sorted positions)
  std::vector<int> qoff{0};      // per group
  std::vector<int> k;            // flattened key idxs (shared by the group)
  std::vector<int> koff{0};
  int n() const { return (int)qoff.size() - 1; }
  void add(const std::vector<int>& qs, const std::vector<int>& ks) {
    q.insert(q.end(), qs.begin(), qs.end());
    k.insert(k.end(), ks.begin(), ks.end());
    qoff.push_back((int)q.size());
    koff.push_back((int)k.size());
  }
};

constexpr int kQTile = 64;       // queries per attention work item

// One attention work item: query tile [q0,q1) of group g in batch row b.
// logkv = log(clamp_min(bf16(kv_count), 1)) is precomputed here so every
// backend applies the identical query scaling.
struct Work {
  int b, g, q0, q1;
  float logkv;
};

struct Prepared {
  int B = 0, S = 0;
  std::vector<uint8_t> pad;              // [B*S] sorted order
  std::vector<Groups> g_col, g_feat, g_nbr;  // per batch row
  std::vector<Work> work[3];             // col, feat, nbr
  std::vector<float> x;                  // [B*S, d] block-0 input (embeddings)
};

// Round fp32 to bf16 (round-to-nearest-even) and back — mirrors the Python
// side's `.bfloat16()` cast of kv counts.
float bf16_round(float f);

// Sort, group, tile and embed. Fills out.sort_idxs / sorted_is_target /
// x_embed (when debug_taps) and sizes out.yhat_number.
Prepared prepare(const Model& m, const Batch& batch, Output& out,
                 bool debug_taps);

// Blocks + head on each backend. Consumes prep.x as block-0 input, writes
// out.yhat_number (+ x_block0 when debug_taps).
void run_blocks_cpu(const Model& m, Prepared& prep, Output& out, int n_threads,
                    bool debug_taps, bool want_text_head = false);
#ifdef RT_METAL
bool metal_available();
void run_blocks_metal(const Model& m, Prepared& prep, Output& out,
                      bool debug_taps);
#endif
#ifdef RT_CUDA
bool cuda_available();
void run_blocks_cuda(const Model& m, Prepared& prep, Output& out,
                     bool debug_taps);
#endif

}  // namespace detail
}  // namespace rt
