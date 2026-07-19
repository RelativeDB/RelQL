#include "rt_train.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "rt_math.hpp"

namespace rt {

namespace {

const char* task_name(FineTuneTask task) {
  switch (task) {
    case FineTuneTask::Binary: return "binary";
    case FineTuneTask::Regression: return "regression";
    case FineTuneTask::Multiclass: return "multiclass";
    case FineTuneTask::Ranking: return "ranking";
  }
  return "unknown";
}

void validate_shape(const FineTuneHead& h) {
  if (h.outputs <= 0) throw std::runtime_error("fine-tune head has no outputs");
  if (h.task != FineTuneTask::Multiclass && h.outputs != 1)
    throw std::runtime_error(std::string(task_name(h.task)) +
                             " head must have exactly one output");
  if (h.task == FineTuneTask::Multiclass && h.outputs < 2)
    throw std::runtime_error("multiclass head needs at least two outputs");
  if (h.weight.size() != (size_t)h.outputs * kDModel ||
      h.bias.size() != (size_t)h.outputs)
    throw std::runtime_error("fine-tune head tensor shape mismatch");
}

}  // namespace

FineTuneHead FineTuneHead::from_model(const Model& model, FineTuneTask task,
                                      int n_outputs,
                                      const float* class_embeddings) {
  FineTuneHead h;
  h.task = task;
  h.outputs = task == FineTuneTask::Multiclass ? n_outputs : 1;
  h.weight.assign((size_t)h.outputs * kDModel, 0.f);
  h.bias.assign(h.outputs, 0.f);
  validate_shape(h);

  if (task != FineTuneTask::Multiclass) {
    std::memcpy(h.weight.data(), model.dec_number.w,
                (size_t)kDModel * sizeof(float));
    h.bias[0] = model.dec_number.b[0];
    return h;
  }
  if (!class_embeddings) return h;

  // Project the released predicted-text head into the requested class-label
  // embedding basis.  Since every class vector is L2-normalized, argmax of
  // this linear head is exactly argmax cosine(predicted_text, class) — the
  // norm of predicted_text is a common positive factor across classes.
  for (int c = 0; c < h.outputs; c++) {
    const float* e = class_embeddings + (size_t)c * kDText;
    float ss = 0.f;
    for (int t = 0; t < kDText; t++) ss += e[t] * e[t];
    const float inv = 1.f / std::sqrt(std::max(ss, 1e-20f));
    for (int t = 0; t < kDText; t++) {
      const float a = e[t] * inv;
      h.bias[c] += a * model.dec_text.b[t];
      const float* w = model.dec_text.w + (size_t)t * kDModel;
      float* dst = h.weight.data() + (size_t)c * kDModel;
      for (int d = 0; d < kDModel; d++) dst[d] += a * w[d];
    }
  }
  return h;
}

std::vector<float> FineTuneHead::predict(const float* features, int N) const {
  validate_shape(*this);
  if (!features || N < 0) throw std::runtime_error("invalid prediction features");
  std::vector<float> out((size_t)N * outputs);
  math::gemm_nt(features, weight.data(), out.data(), N, outputs, kDModel,
                kDModel, kDModel, outputs);
  for (int n = 0; n < N; n++)
    for (int c = 0; c < outputs; c++) out[(size_t)n * outputs + c] += bias[c];
  return out;
}

void FineTuneHead::save(const std::string& path) const {
  validate_shape(*this);
  const uint64_t bias_bytes = (uint64_t)bias.size() * sizeof(float);
  const uint64_t info_bytes = 3 * sizeof(float);
  const uint64_t weight_bytes = (uint64_t)weight.size() * sizeof(float);
  const uint64_t off_bias = 0;
  const uint64_t off_info = off_bias + bias_bytes;
  const uint64_t off_weight = off_info + info_bytes;

  std::ostringstream os;
  os << "{\"head.bias\":{\"dtype\":\"F32\",\"shape\":[" << outputs
     << "],\"data_offsets\":[" << off_bias << ',' << off_info
     << "]},\"head.info\":{\"dtype\":\"F32\",\"shape\":[3],"
        "\"data_offsets\":["
     << off_info << ',' << off_weight
     << "]},\"head.weight\":{\"dtype\":\"F32\",\"shape\":["
     << outputs << ',' << kDModel << "],\"data_offsets\":[" << off_weight
     << ',' << off_weight + weight_bytes << "]}}";
  std::string header = os.str();
  while (header.size() % 8) header.push_back(' ');

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot create " + path);
  const uint64_t hlen = header.size();
  f.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
  f.write(header.data(), (std::streamsize)header.size());
  f.write(reinterpret_cast<const char*>(bias.data()),
          (std::streamsize)bias_bytes);
  const float info[3] = {1.f, (float)(int32_t)task, (float)outputs};
  f.write(reinterpret_cast<const char*>(info), sizeof(info));
  f.write(reinterpret_cast<const char*>(weight.data()),
          (std::streamsize)weight_bytes);
  if (!f) throw std::runtime_error("failed writing " + path);
}

FineTuneHead FineTuneHead::load(const std::string& path) {
  auto tensors = load_safetensors(path);
  auto get = [&](const char* name) -> const Tensor& {
    auto it = tensors.find(name);
    if (it == tensors.end())
      throw std::runtime_error(std::string("fine-tune checkpoint missing ") + name);
    if (it->second.qtype != (uint8_t)WType::F32)
      throw std::runtime_error(std::string(name) + " must be F32");
    return it->second;
  };
  const Tensor& info = get("head.info");
  const Tensor& w = get("head.weight");
  const Tensor& b = get("head.bias");
  if (info.data.size() != 3 || std::lround(info.data[0]) != 1)
    throw std::runtime_error("unsupported fine-tune checkpoint version");
  const int ti = (int)std::lround(info.data[1]);
  if (ti < 0 || ti > 3) throw std::runtime_error("bad fine-tune task id");
  FineTuneHead h;
  h.task = (FineTuneTask)ti;
  h.outputs = (int)std::lround(info.data[2]);
  if (w.shape != std::vector<int64_t>{h.outputs, kDModel} ||
      b.shape != std::vector<int64_t>{h.outputs})
    throw std::runtime_error("fine-tune checkpoint tensor shape mismatch");
  h.weight = w.data;
  h.bias = b.data;
  validate_shape(h);
  return h;
}

#ifndef RT_METAL
FineTuneResult fit_head_metal(FineTuneHead&, const float*, const float*, int,
                              const int32_t*, int,
                              const FineTuneOptions&) {
  throw std::runtime_error("Metal fine-tuning backend was not compiled");
}
#endif

}  // namespace rt
