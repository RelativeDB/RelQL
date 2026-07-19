#include "rt_train.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int argmax(const float* x, int n) {
  return (int)(std::max_element(x, x + n) - x);
}

}  // namespace

int main() {
  if (!rt::device_available(rt::Device::MPS)) {
    std::puts("SKIP: Metal unavailable");
    return 0;
  }

  // Three-class linearly separable frozen features.
  constexpr int C = 3, N = 18, D = rt::kDModel;
  std::vector<float> x((size_t)N * D, 0.f), y(N);
  for (int i = 0; i < N; i++) {
    int c = i % C;
    y[i] = (float)c;
    x[(size_t)i * D + c] = 2.f;
    x[(size_t)i * D + 8 + c] = 1.f;
  }
  rt::FineTuneHead mc;
  mc.task = rt::FineTuneTask::Multiclass;
  mc.outputs = C;
  mc.weight.assign((size_t)C * D, 0.f);
  mc.bias.assign(C, 0.f);
  rt::FineTuneOptions o;
  o.epochs = 80;
  o.learning_rate = 0.03f;
  o.weight_decay = 0.f;
  auto mr = rt::fit_head_metal(mc, x.data(), y.data(), N, nullptr, 0, o);
  auto logits = mc.predict(x.data(), N);
  int correct = 0;
  for (int i = 0; i < N; i++)
    correct += argmax(&logits[(size_t)i * C], C) == (int)y[i];
  if (!(mr.final_loss < mr.initial_loss * 0.1f) || correct != N) {
    std::fprintf(stderr, "multiclass failed: loss %.6f -> %.6f, %d/%d\n",
                 mr.initial_loss, mr.final_loss, correct, N);
    return 1;
  }

  // Two listwise groups. The relevant candidate is encoded by feature 0.
  constexpr int RN = 8;
  std::vector<float> rx((size_t)RN * D, 0.f), ry(RN, 0.f);
  int32_t offsets[] = {0, 4, 8};
  ry[2] = 1.f; ry[7] = 1.f;
  rx[(size_t)2 * D] = 2.f; rx[(size_t)7 * D] = 2.f;
  for (int i = 0; i < RN; i++) rx[(size_t)i * D + 3] = (float)i / RN;
  rt::FineTuneHead rank;
  rank.task = rt::FineTuneTask::Ranking;
  rank.outputs = 1;
  rank.weight.assign(D, 0.f);
  rank.bias.assign(1, 0.f);
  o.epochs = 80;
  auto rr = rt::fit_head_metal(rank, rx.data(), ry.data(), RN, offsets, 2, o);
  auto scores = rank.predict(rx.data(), RN);
  bool ranked = scores[2] > scores[0] && scores[2] > scores[1] &&
                scores[2] > scores[3] && scores[7] > scores[4] &&
                scores[7] > scores[5] && scores[7] > scores[6];
  if (!(rr.final_loss < rr.initial_loss * 0.2f) || !ranked) {
    std::fprintf(stderr, "ranking failed: loss %.6f -> %.6f\n",
                 rr.initial_loss, rr.final_loss);
    return 1;
  }

  const char* path = "/tmp/rt_train_test.safetensors";
  mc.save(path);
  auto loaded = rt::FineTuneHead::load(path);
  auto logits2 = loaded.predict(x.data(), N);
  if (logits != logits2) {
    std::fputs("checkpoint round-trip failed\n", stderr);
    return 1;
  }
  std::remove(path);
  std::printf("TRAIN TEST PASS multiclass %.6f->%.6f ranking %.6f->%.6f\n",
              mr.initial_loss, mr.final_loss, rr.initial_loss, rr.final_loss);
  return 0;
}
