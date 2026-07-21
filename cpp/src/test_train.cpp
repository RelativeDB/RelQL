#include "rt_train.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

int argmax(const float* x, int n) {
  return (int)(std::max_element(x, x + n) - x);
}

template <typename T>
std::vector<T> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("missing " + path);
  size_t bytes=(size_t)f.tellg();std::vector<T> v(bytes/sizeof(T));f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()),bytes);return v;
}

}  // namespace

int main(int argc, char** argv) {
  // Standardization must not alter an initialized head at epoch zero.
  rt::FineTuneHead init;
  init.task = rt::FineTuneTask::Multiclass;
  init.outputs = 2;
  init.weight.resize((size_t)init.outputs * rt::kDModel);
  init.bias = {0.25f, -0.75f};
  for (size_t i = 0; i < init.weight.size(); i++)
    init.weight[i] = (float)((int)(i % 13) - 6) * 0.01f;
  constexpr int ZN = 3;
  std::vector<float> raw((size_t)ZN * rt::kDModel);
  std::vector<float> mean(rt::kDModel), sd(rt::kDModel), standardized(raw.size());
  for (int d = 0; d < rt::kDModel; d++) {
    mean[d] = (float)((d % 7) - 3) * 0.1f;
    sd[d] = 0.25f + (float)(d % 11) * 0.07f;
    for (int n = 0; n < ZN; n++) {
      raw[(size_t)n * rt::kDModel + d] = mean[d] + sd[d] * (n - 1.0f);
      standardized[(size_t)n * rt::kDModel + d] = n - 1.0f;
    }
  }
  auto expected = init.predict(raw.data(), ZN);
  init.reparameterize_for_standardized_features(mean.data(), sd.data());
  auto actual = init.predict(standardized.data(), ZN);
  for (size_t i = 0; i < expected.size(); i++) {
    if (std::abs(expected[i] - actual[i]) > 2e-5f) {
      std::fprintf(stderr, "zero-shot reparameterization failed at %zu: %.8f != %.8f\n",
                   i, expected[i], actual[i]);
      return 1;
    }
  }

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
  if (argc >= 3) {
    std::string dir=argv[1];rt::Batch batch;batch.S=16;
    batch.node_idxs=read_bin<int64_t>(dir+"/node_idxs.bin");batch.B=(int)batch.node_idxs.size()/batch.S;
    batch.f2p=read_bin<int64_t>(dir+"/f2p_nbr_idxs.bin");batch.col_idxs=read_bin<int64_t>(dir+"/col_name_idxs.bin");
    batch.table_idxs=read_bin<int64_t>(dir+"/table_name_idxs.bin");batch.is_padding=read_bin<uint8_t>(dir+"/is_padding.bin");
    batch.sem_types=read_bin<int64_t>(dir+"/sem_types.bin");batch.is_target=read_bin<uint8_t>(dir+"/is_targets.bin");
    batch.number_v=read_bin<float>(dir+"/number_values.bin");batch.datetime_v=read_bin<float>(dir+"/datetime_values.bin");
    batch.boolean_v=read_bin<float>(dir+"/boolean_values.bin");batch.text_v=read_bin<float>(dir+"/text_values.bin");
    batch.col_name_v=read_bin<float>(dir+"/col_name_values.bin");
    rt::Model full=rt::Model::load(argv[2]);
    const char* changed_keys[]={"enc_dict.col_name.weight","mask_embs.number",
      "blocks.0.attns.col.wq.weight","blocks.11.ffn.w3.weight",
      "norm_out.scale","dec_dict.number.weight"};
    std::vector<std::vector<float>> before;
    for(const char* key:changed_keys)before.push_back(full.store.at(key).data);
    rt::FullFineTuneOptions fo;fo.learning_rate=1e-6f;auto step=rt::fit_model_metal_step(full,batch,fo);
    bool changed=true;
    for(size_t k=0;k<before.size();k++)changed&=before[k]!=full.store.at(changed_keys[k]).data;
    if(!std::isfinite(step.loss)||!std::isfinite(step.grad_norm)||!changed){
      std::fprintf(stderr,"full-model step failed loss=%g grad=%g representative_parameters_changed=%d\n",step.loss,step.grad_norm,changed);return 1;
    }
    const char* fullpath="/tmp/rt_full_train_test.safetensors";full.save(fullpath);
    rt::Model round=rt::Model::load(fullpath);std::remove(fullpath);
    if(round.store.at(changed_keys[2]).data!=full.store.at(changed_keys[2]).data){std::fputs("full checkpoint round-trip failed\n",stderr);return 1;}
    std::printf("FULL MODEL MPS PASS loss %.6f grad %.6f seconds %.3f\n",step.loss,step.grad_norm,step.seconds);
  }
  std::printf("TRAIN TEST PASS multiclass %.6f->%.6f ranking %.6f->%.6f\n",
              mr.initial_loss, mr.final_loss, rr.initial_loss, rr.final_loss);
  return 0;
}
