// bench.cpp — batching correctness + speed/memory benchmarks for rt.cpp.
//
//   ./rt_bench <testdata_dir> <model.safetensors> [--device cpu|mps|cuda]
//
// 1. Batching correctness: the golden B=5 batch executed (a) all at once,
//    (b) one row at a time, (c) with batch rows permuted — per-entity outputs
//    must agree (attention must never leak across batch rows).
// 2. Speed: sweep batch size (replicated golden rows) and context length
//    (synthetic relational batches, S up to 2048).
// 3. Memory: ru_maxrss checkpoints (post-load, post-forward) + analytic
//    weight/activation accounting.
#include "rt.hpp"

#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using rt::Batch;
using rt::kDText;
using rt::kMaxF2p;

namespace {

template <typename T>
std::vector<T> read_bin(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
  size_t bytes = (size_t)f.tellg();
  std::vector<T> v(bytes / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), bytes);
  return v;
}

Batch load_golden(const std::string& dir) {
  Batch b;
  b.node_idxs = read_bin<int64_t>(dir + "/node_idxs.bin");
  b.S = 16;
  b.B = (int)b.node_idxs.size() / b.S;
  b.f2p = read_bin<int64_t>(dir + "/f2p_nbr_idxs.bin");
  b.col_idxs = read_bin<int64_t>(dir + "/col_name_idxs.bin");
  b.table_idxs = read_bin<int64_t>(dir + "/table_name_idxs.bin");
  b.is_padding = read_bin<uint8_t>(dir + "/is_padding.bin");
  b.sem_types = read_bin<int64_t>(dir + "/sem_types.bin");
  b.is_target = read_bin<uint8_t>(dir + "/is_targets.bin");
  b.number_v = read_bin<float>(dir + "/number_values.bin");
  b.datetime_v = read_bin<float>(dir + "/datetime_values.bin");
  b.boolean_v = read_bin<float>(dir + "/boolean_values.bin");
  b.text_v = read_bin<float>(dir + "/text_values.bin");
  b.col_name_v = read_bin<float>(dir + "/col_name_values.bin");
  return b;
}

// Copy selected batch rows (by index, possibly repeated) into a new Batch.
Batch take_rows(const Batch& b, const std::vector<int>& rows) {
  Batch o;
  o.B = (int)rows.size();
  o.S = b.S;
  auto cp = [&](auto& dst, const auto& src, size_t per) {
    dst.resize(rows.size() * b.S * per);
    for (size_t r = 0; r < rows.size(); r++)
      std::memcpy(&dst[r * b.S * per], &src[(size_t)rows[r] * b.S * per],
                  b.S * per * sizeof(dst[0]));
  };
  cp(o.node_idxs, b.node_idxs, 1);
  cp(o.f2p, b.f2p, kMaxF2p);
  cp(o.col_idxs, b.col_idxs, 1);
  cp(o.table_idxs, b.table_idxs, 1);
  cp(o.is_padding, b.is_padding, 1);
  cp(o.sem_types, b.sem_types, 1);
  cp(o.is_target, b.is_target, 1);
  cp(o.number_v, b.number_v, 1);
  cp(o.datetime_v, b.datetime_v, 1);
  cp(o.boolean_v, b.boolean_v, 1);
  cp(o.text_v, b.text_v, kDText);
  cp(o.col_name_v, b.col_name_v, kDText);
  return o;
}

// Synthetic relational batch: entity row + fact rows (FK->entity, FK->item)
// + item rows + task label history — the shape real samplers emit.
Batch synth(int B, int S, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  Batch b;
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

double target_score(const rt::Output& o, int row) {
  for (int s = 0; s < o.S; s++) {
    size_t i = (size_t)row * o.S + s;
    if (o.sorted_is_target[i]) return o.yhat_number[i];
  }
  return NAN;
}

double rss_mb() {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
  return u.ru_maxrss / (1024.0 * 1024.0);   // bytes on macOS
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <testdata> <safetensors> [--device cpu|mps|cuda]\n", argv[0]); return 2; }
  rt::ForwardOpts opts;
  opts.debug_taps = false;
  for (int i = 3; i + 1 < argc; i++) {
    if (std::string(argv[i]) == "--device") {
      std::string d = argv[i + 1];
      opts.device = d == "mps" ? rt::Device::MPS
                    : d == "cuda" ? rt::Device::CUDA
                                  : rt::Device::CPU;
    }
  }
  if (!rt::device_available(opts.device)) {
    fprintf(stderr, "device %s not available\n", rt::device_name(opts.device));
    return 2;
  }
  printf("device: %s\n", rt::device_name(opts.device));
  double rss0 = rss_mb();
  auto t0 = std::chrono::steady_clock::now();
  rt::Model model = rt::Model::load(argv[2]);
  printf("== memory ==\n");
  printf("checkpoint load: %.0f ms, RSS %.0f MB (weights fp32 ~%.0f MB)\n",
         ms_since(t0), rss_mb(),
         [&] { double n = 0; for (auto& [k, t] : model.store) n += t.numel(); return n * 4 / 1e6; }());

  Batch golden = load_golden(argv[1]);

  // ---- 1. batching correctness -------------------------------------------
  printf("\n== batching correctness ==\n");
  rt::Output all = rt::forward(model, golden, opts);
  double max_dev_single = 0, max_dev_perm = 0;
  for (int r = 0; r < golden.B; r++) {                    // one row at a time
    rt::Output one = rt::forward(model, take_rows(golden, {r}), opts);
    max_dev_single = std::max(max_dev_single,
        std::fabs(target_score(all, r) - target_score(one, 0)));
  }
  std::vector<int> perm = {3, 0, 4, 2, 1};                // permuted batch
  rt::Output p = rt::forward(model, take_rows(golden, perm), opts);
  for (int r = 0; r < (int)perm.size(); r++)
    max_dev_perm = std::max(max_dev_perm,
        std::fabs(target_score(all, perm[r]) - target_score(p, r)));
  // batch of duplicates: every copy must produce the identical score
  rt::Output dup = rt::forward(model, take_rows(golden, {1, 1, 1, 1}), opts);
  double max_dev_dup = 0;
  for (int r = 1; r < 4; r++)
    max_dev_dup = std::max(max_dev_dup,
        std::fabs(target_score(dup, 0) - target_score(dup, r)));
  printf("batched vs single-row  max|Δ| = %.3e\n", max_dev_single);
  printf("batched vs permuted    max|Δ| = %.3e\n", max_dev_perm);
  printf("duplicate rows in batch max|Δ| = %.3e\n", max_dev_dup);
  bool ok = max_dev_single < 1e-5 && max_dev_perm < 1e-5 && max_dev_dup < 1e-6;
  printf(ok ? "BATCHING OK\n" : "BATCHING FAIL\n");

  // ---- 2. speed sweeps ----------------------------------------------------
  auto bench = [&](const Batch& b, int iters) {
    rt::forward(model, b, opts);                      // warm
    auto s0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) rt::forward(model, b, opts);
    double ms = ms_since(s0) / iters;
    double tok = (double)b.B * b.S / (ms / 1e3);
    printf("B=%-3d S=%-5d  %8.1f ms/fwd   %9.0f tok/s   %6.1f ms/entity\n",
           b.B, b.S, ms, tok, ms / b.B);
  };
  // Smell test only: one tiny shape to confirm the harness runs. Real
  // relational inference is long-context (S >= 1024) — everything below that
  // is latency-dominated by fixed overhead and isn't representative.
  printf("\n== smell test (tiny shape, correctness of the harness) ==\n");
  {
    std::vector<int> rows(5);
    for (int i = 0; i < 5; i++) rows[i] = i % golden.B;
    bench(take_rows(golden, rows), 20);
  }
  printf("\n== speed: single-entity long context ==\n");
  bench(synth(1, 1024, 2), 5);
  bench(synth(1, 2048, 3), 3);
  bench(synth(1, 4096, 20), 2);
  bench(synth(1, 8192, 21), 2);
  printf("\n== speed: batched long context ==\n");
  bench(synth(8, 1024, 4), 3);
  bench(synth(8, 2048, 12), 2);

  // ---- 3. memory after the big forward ------------------------------------
  printf("\n== memory after S=2048/B=8 forwards ==\n");
  printf("RSS start %.0f MB -> now %.0f MB\n", rss0, rss_mb());
  return ok ? 0 : 1;
}
