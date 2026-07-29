// bench_synth.hpp — synthetic relational batches shared by the bench tools.
#pragma once

#include <cmath>
#include <random>
#include <vector>

#include "rt.hpp"

namespace rt_bench {

// Synthetic relational batch: entity row + fact rows (FK->entity, FK->item)
// + item rows + task label history — the shape real samplers emit.
inline rt::Batch synth(int B, int S, uint32_t seed) {
  using rt::kDText;
  using rt::kMaxF2p;
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  rt::Batch b;
  b.B = B; b.S = S;
  size_t BS = (size_t)B * S;
  b.node_idxs.resize(BS); b.f2p.assign(BS * kMaxF2p, -1);
  b.col_idxs.resize(BS); b.table_idxs.resize(BS);
  b.is_padding.assign(BS, 0); b.sem_types.resize(BS);
  b.is_target.assign(BS, 0);
  b.number_v.assign(BS, 0.f); b.datetime_v.assign(BS, 0.f);
  b.boolean_v.assign(BS, 0.f);
  b.text_v.assign(BS * kDText, 0.f); b.col_name_v.assign(BS * kDText, 0.f);
  const int n_items = std::max(2, S / 16);
  for (int r = 0; r < B; r++) {
    size_t base = (size_t)r * S;
    int64_t next_node = 0;
    int64_t entity = next_node++;
    std::vector<int64_t> items(n_items);
    for (auto& it : items) it = next_node++;
    int s = 0;
    auto put = [&](int64_t nodeid, int col, int table, int sem, float val,
                   int64_t p0, int64_t p1, bool target = false) {
      size_t i = base + s;
      b.node_idxs[i] = nodeid;
      b.col_idxs[i] = col;
      b.table_idxs[i] = table;
      b.sem_types[i] = sem;
      b.is_target[i] = target;
      if (sem == rt::kNumber) b.number_v[i] = val;
      else if (sem == rt::kDatetime) b.datetime_v[i] = val;
      else if (sem == rt::kText)
        for (int d = 0; d < kDText; d++) b.text_v[i * kDText + d] = nd(rng) * 0.1f;
      b.f2p[i * kMaxF2p] = p0;
      b.f2p[i * kMaxF2p + 1] = p1;
      for (int d = 0; d < kDText; d++)
        b.col_name_v[i * kDText + d] = std::sin(0.1f * (col * 7 + d));  // stable per column
      s++;
    };
    // task row: masked target + timestamp, FK -> entity
    int64_t task = next_node++;
    put(task, 0, 0, rt::kNumber, 0.f, entity, -1, /*target=*/true);
    put(task, 1, 0, rt::kDatetime, 0.5f, entity, -1);
    // entity row
    put(entity, 2, 1, rt::kNumber, nd(rng), -1, -1);
    put(entity, 3, 1, rt::kDatetime, nd(rng) * 0.3f, -1, -1);
    // items
    for (int it = 0; it < n_items && s < S - 1; it++) {
      put(items[it], 4, 2, rt::kNumber, nd(rng), -1, -1);
      put(items[it], 5, 2, rt::kText, 0.f, -1, -1);
    }
    // label history (self labels) + fact rows until full
    int hist = 0;
    while (s < S) {
      if (hist++ % 6 == 0 && s + 1 < S) {
        int64_t t2 = next_node++;
        put(t2, 0, 0, rt::kNumber, nd(rng) > 0 ? 1.41f : -0.71f, entity, -1);
        put(t2, 1, 0, rt::kDatetime, -nd(rng) * 0.5f, entity, -1);
      } else if (s + 2 < S) {
        int64_t fact = next_node++;
        int64_t item = items[rng() % n_items];
        put(fact, 6, 3, rt::kNumber, nd(rng), entity, item);
        put(fact, 7, 3, rt::kDatetime, -std::abs(nd(rng)), entity, item);
        put(fact, 8, 3, rt::kNumber, nd(rng), entity, item);
      } else {
        put(next_node++, 9, 3, rt::kNumber, nd(rng), entity, -1);
      }
    }
  }
  return b;
}

}  // namespace rt_bench
