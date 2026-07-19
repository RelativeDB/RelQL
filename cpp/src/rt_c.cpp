#include "rt_c.h"

#include <cstring>
#include <string>

#include "rt.hpp"

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

}  // extern "C"
