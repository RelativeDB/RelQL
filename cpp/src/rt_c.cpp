#include "rt_c.h"

#include <cstring>
#include <string>

#include "rt.hpp"
#include "rt_train.hpp"

namespace {
void set_err(char* err, size_t errlen, const std::string& msg) {
  if (!err || errlen == 0) return;
  std::strncpy(err, msg.c_str(), errlen - 1);
  err[errlen - 1] = '\0';
}
}  // namespace

struct rt_model {
  rt::Model model;
};

struct rt_finetune_head {
  rt::FineTuneHead head;
};

extern "C" {

rt_model* rt_model_load(const char* path, char* err, size_t errlen) {
  try {
    auto* m = new rt_model{rt::Model::load(path)};
    return m;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return nullptr;
  }
}

void rt_model_free(rt_model* m) { delete m; }

int64_t rt_model_num_params(const rt_model* m) {
  int64_t n = 0;
  for (const auto& [k, t] : m->model.store) n += t.numel();
  return n;
}

int rt_device_available(int32_t device) {
  if (device < 0 || device > 2) return 0;
  return rt::device_available(static_cast<rt::Device>(device)) ? 1 : 0;
}

int rt_forward_device(const rt_model* m, int32_t B, int32_t S,
                      const int64_t* node_idxs, const int64_t* f2p,
                      const int64_t* col_idxs, const int64_t* table_idxs,
                      const uint8_t* is_padding, const int64_t* sem_types,
                      const uint8_t* is_target, const float* number_v,
                      const float* datetime_v, const float* boolean_v,
                      const float* text_v, const float* col_name_v,
                      int32_t n_threads, int32_t device,
                      float* out_target_scores, char* err, size_t errlen) {
  try {
    if (B <= 0 || S <= 0) throw std::runtime_error("B and S must be positive");
    if (device < 0 || device > 2) throw std::runtime_error("bad device id");
    const size_t BS = (size_t)B * S;
    rt::Batch b;
    b.B = B;
    b.S = S;
    b.node_idxs.assign(node_idxs, node_idxs + BS);
    b.f2p.assign(f2p, f2p + BS * rt::kMaxF2p);
    b.col_idxs.assign(col_idxs, col_idxs + BS);
    b.table_idxs.assign(table_idxs, table_idxs + BS);
    b.is_padding.assign(is_padding, is_padding + BS);
    b.sem_types.assign(sem_types, sem_types + BS);
    b.is_target.assign(is_target, is_target + BS);
    b.number_v.assign(number_v, number_v + BS);
    b.datetime_v.assign(datetime_v, datetime_v + BS);
    b.boolean_v.assign(boolean_v, boolean_v + BS);
    b.text_v.assign(text_v, text_v + BS * rt::kDText);
    b.col_name_v.assign(col_name_v, col_name_v + BS * rt::kDText);

    rt::ForwardOpts opts;
    opts.device = static_cast<rt::Device>(device);
    opts.n_threads = n_threads;
    opts.debug_taps = false;
    rt::Output out = rt::forward(m->model, b, opts);
    for (int r = 0; r < B; r++) {
      float acc = 0.f;
      for (int s = 0; s < S; s++) {
        size_t i = (size_t)r * S + s;
        if (out.sorted_is_target[i]) acc += out.yhat_number[i];
      }
      out_target_scores[r] = acc;
    }
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return 1;
  }
}

int rt_forward_ex(const rt_model* m, int32_t B, int32_t S,
                  const int64_t* node_idxs, const int64_t* f2p,
                  const int64_t* col_idxs, const int64_t* table_idxs,
                  const uint8_t* is_padding, const int64_t* sem_types,
                  const uint8_t* is_target, const float* number_v,
                  const float* datetime_v, const float* boolean_v,
                  const float* text_v, const float* col_name_v,
                  int32_t n_threads, float* out_target_scores,
                  float* out_target_text, char* err, size_t errlen) {
  try {
    if (B <= 0 || S <= 0) throw std::runtime_error("B and S must be positive");
    const size_t BS = (size_t)B * S;
    rt::Batch b;
    b.B = B;
    b.S = S;
    b.node_idxs.assign(node_idxs, node_idxs + BS);
    b.f2p.assign(f2p, f2p + BS * rt::kMaxF2p);
    b.col_idxs.assign(col_idxs, col_idxs + BS);
    b.table_idxs.assign(table_idxs, table_idxs + BS);
    b.is_padding.assign(is_padding, is_padding + BS);
    b.sem_types.assign(sem_types, sem_types + BS);
    b.is_target.assign(is_target, is_target + BS);
    b.number_v.assign(number_v, number_v + BS);
    b.datetime_v.assign(datetime_v, datetime_v + BS);
    b.boolean_v.assign(boolean_v, boolean_v + BS);
    b.text_v.assign(text_v, text_v + BS * rt::kDText);
    b.col_name_v.assign(col_name_v, col_name_v + BS * rt::kDText);

    rt::ForwardOpts opts;
    opts.device = rt::Device::CPU;   // CPU only, matching rt_forward
    opts.n_threads = n_threads;
    opts.debug_taps = false;
    opts.want_text_head = (out_target_text != nullptr);
    rt::Output out = rt::forward(m->model, b, opts);

    for (int r = 0; r < B; r++) {
      float acc = 0.f;
      for (int s = 0; s < S; s++) {
        size_t i = (size_t)r * S + s;
        if (out.sorted_is_target[i]) acc += out.yhat_number[i];
      }
      out_target_scores[r] = acc;
    }

    if (out_target_text) {
      // Sum the text head over each row's target positions (mirrors the number
      // head reduction above). Rows with no target stay all-zeros.
      const int DT = rt::kDText;
      std::memset(out_target_text, 0, (size_t)B * DT * sizeof(float));
      for (int r = 0; r < B; r++) {
        float* dst = out_target_text + (size_t)r * DT;
        for (int s = 0; s < S; s++) {
          size_t i = (size_t)r * S + s;
          if (!out.sorted_is_target[i]) continue;
          const float* src = &out.yhat_text[i * (size_t)DT];
          for (int d = 0; d < DT; d++) dst[d] += src[d];
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return 1;
  }
}

int rt_forward(const rt_model* m, int32_t B, int32_t S,
               const int64_t* node_idxs, const int64_t* f2p,
               const int64_t* col_idxs, const int64_t* table_idxs,
               const uint8_t* is_padding, const int64_t* sem_types,
               const uint8_t* is_target, const float* number_v,
               const float* datetime_v, const float* boolean_v,
               const float* text_v, const float* col_name_v,
               int32_t n_threads, float* out_target_scores,
               char* err, size_t errlen) {
  return rt_forward_ex(m, B, S, node_idxs, f2p, col_idxs, table_idxs,
                       is_padding, sem_types, is_target, number_v, datetime_v,
                       boolean_v, text_v, col_name_v, n_threads,
                       out_target_scores, /*out_target_text=*/nullptr, err,
                       errlen);
}

int rt_encode_targets_device(const rt_model* m, int32_t B, int32_t S,
                      const int64_t* node_idxs, const int64_t* f2p,
                      const int64_t* col_idxs, const int64_t* table_idxs,
                      const uint8_t* is_padding, const int64_t* sem_types,
                      const uint8_t* is_target, const float* number_v,
                      const float* datetime_v, const float* boolean_v,
                      const float* text_v, const float* col_name_v,
                      int32_t n_threads, int32_t device,
                      float* out_target_features,
                      char* err, size_t errlen) {
  try {
    if (!m || !out_target_features)
      throw std::runtime_error("model and output features are required");
    if (B <= 0 || S <= 0) throw std::runtime_error("B and S must be positive");
    if (device < 0 || device > 2) throw std::runtime_error("bad device id");
    if (device == RT_DEVICE_CUDA)
      throw std::runtime_error("target feature extraction is not implemented on CUDA");
    const size_t BS = (size_t)B * S;
    rt::Batch b;
    b.B = B; b.S = S;
    b.node_idxs.assign(node_idxs, node_idxs + BS);
    b.f2p.assign(f2p, f2p + BS * rt::kMaxF2p);
    b.col_idxs.assign(col_idxs, col_idxs + BS);
    b.table_idxs.assign(table_idxs, table_idxs + BS);
    b.is_padding.assign(is_padding, is_padding + BS);
    b.sem_types.assign(sem_types, sem_types + BS);
    b.is_target.assign(is_target, is_target + BS);
    b.number_v.assign(number_v, number_v + BS);
    b.datetime_v.assign(datetime_v, datetime_v + BS);
    b.boolean_v.assign(boolean_v, boolean_v + BS);
    b.text_v.assign(text_v, text_v + BS * rt::kDText);
    b.col_name_v.assign(col_name_v, col_name_v + BS * rt::kDText);
    rt::ForwardOpts opts;
    opts.device = static_cast<rt::Device>(device);
    opts.n_threads = n_threads;
    opts.debug_taps = false;
    opts.want_target_features = true;
    rt::Output out = rt::forward(m->model, b, opts);
    if (out.target_features.size() != (size_t)B * rt::kDModel)
      throw std::runtime_error("backend did not return target features");
    std::memcpy(out_target_features, out.target_features.data(),
                out.target_features.size() * sizeof(float));
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return 1;
  }
}

rt_finetune_head* rt_finetune_head_create(const rt_model* m, int32_t task,
                                           int32_t n_outputs,
                                           const float* class_embeddings,
                                           char* err, size_t errlen) {
  try {
    if (!m) throw std::runtime_error("model is required");
    if (task < RT_FINETUNE_BINARY || task > RT_FINETUNE_RANKING)
      throw std::runtime_error("bad fine-tune task id");
    auto h = rt::FineTuneHead::from_model(
        m->model, static_cast<rt::FineTuneTask>(task), n_outputs,
        class_embeddings);
    return new rt_finetune_head{std::move(h)};
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return nullptr;
  }
}

rt_finetune_head* rt_finetune_head_load(const char* path,
                                         char* err, size_t errlen) {
  try {
    if (!path) throw std::runtime_error("checkpoint path is required");
    return new rt_finetune_head{rt::FineTuneHead::load(path)};
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return nullptr;
  }
}

void rt_finetune_head_free(rt_finetune_head* h) { delete h; }

int rt_finetune_head_save(const rt_finetune_head* h, const char* path,
                          char* err, size_t errlen) {
  try {
    if (!h || !path) throw std::runtime_error("head and path are required");
    h->head.save(path);
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return 1;
  }
}

int rt_finetune_head_fit_metal(rt_finetune_head* h, int32_t N,
                               const float* features, const float* labels,
                               const int32_t* group_offsets, int32_t n_groups,
                               int32_t epochs, float learning_rate,
                               float weight_decay,
                               float* out_initial_loss, float* out_final_loss,
                               double* out_seconds,
                               char* err, size_t errlen) {
  try {
    if (!h) throw std::runtime_error("fine-tune head is required");
    rt::FineTuneOptions opts;
    opts.epochs = epochs;
    opts.learning_rate = learning_rate;
    opts.weight_decay = weight_decay;
    rt::FineTuneResult r = rt::fit_head_metal(
        h->head, features, labels, N, group_offsets, n_groups, opts);
    if (out_initial_loss) *out_initial_loss = r.initial_loss;
    if (out_final_loss) *out_final_loss = r.final_loss;
    if (out_seconds) *out_seconds = r.seconds;
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return 1;
  }
}

int rt_finetune_head_predict(const rt_finetune_head* h, int32_t N,
                             const float* features, float* out_logits,
                             char* err, size_t errlen) {
  try {
    if (!h || !out_logits) throw std::runtime_error("head and output are required");
    std::vector<float> y = h->head.predict(features, N);
    std::memcpy(out_logits, y.data(), y.size() * sizeof(float));
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return 1;
  }
}

int32_t rt_finetune_head_outputs(const rt_finetune_head* h) {
  return h ? h->head.outputs : 0;
}

int32_t rt_finetune_head_task(const rt_finetune_head* h) {
  return h ? static_cast<int32_t>(h->head.task) : -1;
}

}  // extern "C"
