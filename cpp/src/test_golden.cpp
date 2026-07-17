// Golden test: run the C++ RT forward on the dumped demo batch and compare
// x_embed / x_block0 / yhat_number against the PyTorch reference (fp32).
//
//   ./rt_test <testdata_dir> <model.safetensors> [--bench N]
#include "rt.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_bin(const std::string& path, size_t expect = 0) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "missing %s\n", path.c_str()); exit(2); }
  size_t bytes = static_cast<size_t>(f.tellg());
  std::vector<T> v(bytes / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), bytes);
  if (expect && v.size() != expect) {
    fprintf(stderr, "%s: expected %zu elems, got %zu\n", path.c_str(), expect, v.size());
    exit(2);
  }
  return v;
}

struct Diff { double max_abs = 0, mean_abs = 0; };

Diff diff(const std::vector<float>& a, const std::vector<float>& b,
          const std::vector<uint8_t>* mask_bs, int per) {
  Diff d;
  size_t n = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (mask_bs && (*mask_bs)[i / per]) continue;   // skip padded tokens
    double e = std::fabs((double)a[i] - (double)b[i]);
    d.max_abs = std::max(d.max_abs, e);
    d.mean_abs += e;
    n++;
  }
  d.mean_abs /= std::max<size_t>(n, 1);
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <testdata> <safetensors> [--bench N]\n", argv[0]); return 2; }
  std::string dir = argv[1], ckpt = argv[2];
  int bench = 0;
  if (argc >= 5 && std::string(argv[3]) == "--bench") bench = atoi(argv[4]);

  // Shapes for the demo batch are fixed by the dump (B=5, S=16).
  rt::Batch b;
  b.node_idxs = read_bin<int64_t>(dir + "/node_idxs.bin");
  b.S = 16;
  b.B = static_cast<int>(b.node_idxs.size()) / b.S;
  size_t BS = b.node_idxs.size();
  b.f2p = read_bin<int64_t>(dir + "/f2p_nbr_idxs.bin", BS * rt::kMaxF2p);
  b.col_idxs = read_bin<int64_t>(dir + "/col_name_idxs.bin", BS);
  b.table_idxs = read_bin<int64_t>(dir + "/table_name_idxs.bin", BS);
  b.is_padding = read_bin<uint8_t>(dir + "/is_padding.bin", BS);
  b.sem_types = read_bin<int64_t>(dir + "/sem_types.bin", BS);
  b.is_target = read_bin<uint8_t>(dir + "/is_targets.bin", BS);
  b.number_v = read_bin<float>(dir + "/number_values.bin", BS);
  b.datetime_v = read_bin<float>(dir + "/datetime_values.bin", BS);
  b.boolean_v = read_bin<float>(dir + "/boolean_values.bin", BS);
  b.text_v = read_bin<float>(dir + "/text_values.bin", BS * rt::kDText);
  b.col_name_v = read_bin<float>(dir + "/col_name_values.bin", BS * rt::kDText);

  auto t0 = std::chrono::steady_clock::now();
  rt::Model model = rt::Model::load(ckpt);
  auto t1 = std::chrono::steady_clock::now();
  printf("loaded checkpoint in %.2fs\n",
         std::chrono::duration<double>(t1 - t0).count());

  rt::Output out = rt::forward(model, b);

  // reference (post-sort order)
  auto ref_sort = read_bin<int64_t>(dir + "/sort_idxs.bin", BS);
  auto ref_embed = read_bin<float>(dir + "/x_embed.bin", BS * rt::kDModel);
  auto ref_block0 = read_bin<float>(dir + "/x_block0.bin", BS * rt::kDModel);
  auto ref_yhat = read_bin<float>(dir + "/yhat_number.bin", BS);
  auto ref_tgt = read_bin<uint8_t>(dir + "/sorted_is_targets.bin", BS);

  int sort_mismatch = 0;
  for (size_t i = 0; i < BS; i++) sort_mismatch += (out.sort_idxs[i] != ref_sort[i]);

  // padded positions (post-sort): recompute from sorted padding
  std::vector<uint8_t> pad_sorted(BS);
  for (int bb = 0; bb < b.B; bb++)
    for (int s = 0; s < b.S; s++)
      pad_sorted[(size_t)bb * b.S + s] =
          b.is_padding[(size_t)bb * b.S + out.sort_idxs[(size_t)bb * b.S + s]];

  Diff d_e = diff(out.x_embed, ref_embed, &pad_sorted, rt::kDModel);
  Diff d_b = diff(out.x_block0, ref_block0, &pad_sorted, rt::kDModel);
  Diff d_y = diff(out.yhat_number, ref_yhat, &pad_sorted, 1);

  printf("sort mismatches : %d\n", sort_mismatch);
  printf("x_embed    max|Δ| %.3e  mean %.3e\n", d_e.max_abs, d_e.mean_abs);
  printf("x_block0   max|Δ| %.3e  mean %.3e\n", d_b.max_abs, d_b.mean_abs);
  printf("yhat       max|Δ| %.3e  mean %.3e\n", d_y.max_abs, d_y.mean_abs);

  printf("target scores (cpp vs torch):\n");
  for (int bb = 0; bb < b.B; bb++) {
    for (int s = 0; s < b.S; s++) {
      size_t i = (size_t)bb * b.S + s;
      if (ref_tgt[i]) printf("  row %d: %+0.5f  vs  %+0.5f\n",
                             bb, out.yhat_number[i], ref_yhat[i]);
    }
  }

  bool ok = sort_mismatch == 0 && d_e.max_abs < 5e-4 && d_b.max_abs < 5e-3 &&
            d_y.max_abs < 5e-3;
  printf(ok ? "GOLDEN TEST PASS\n" : "GOLDEN TEST FAIL\n");

  if (bench > 0) {
    rt::forward(model, b);  // warm
    auto s0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench; i++) rt::forward(model, b);
    auto s1 = std::chrono::steady_clock::now();
    double per = std::chrono::duration<double, std::milli>(s1 - s0).count() / bench;
    printf("bench: %.2f ms / forward (B=%d, S=%d)\n", per, b.B, b.S);
  }
  return ok ? 0 : 1;
}
