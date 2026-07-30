#include "minilm_c.h"

#include <cstring>
#include <string>
#include <vector>

#include "minilm.hpp"

namespace {

void set_err(char* err, size_t errlen, const std::string& msg) {
  if (!err || errlen == 0) return;
  size_t n = msg.size();
  if (n > errlen - 1) n = errlen - 1;
  std::memcpy(err, msg.data(), n);
  err[n] = '\0';
}

}  // namespace

struct minilm_t {
  std::unique_ptr<minilm::MiniLM> model;
};

extern "C" {

minilm_t* minilm_load(const char* snapshot_dir, char* err, size_t errlen) {
  std::string e;
  auto m = minilm::MiniLM::load(snapshot_dir ? snapshot_dir : "", &e);
  if (!m) {
    set_err(err, errlen, e);
    return nullptr;
  }
  auto* h = new minilm_t();
  h->model = std::move(m);
  return h;
}

void minilm_free(minilm_t* h) { delete h; }

int minilm_encode(const minilm_t* h, const char* const* texts, int32_t n,
                  int32_t normalize, float* out, char* err, size_t errlen) {
  if (!h || !h->model || (n > 0 && (!texts || !out))) {
    set_err(err, errlen, "minilm_encode: bad arguments");
    return 1;
  }
  std::vector<std::string> in;
  in.reserve(n > 0 ? (size_t)n : 0);
  for (int32_t i = 0; i < n; ++i) in.emplace_back(texts[i] ? texts[i] : "");
  std::string e;
  if (!h->model->encode(in, normalize != 0, out, &e)) {
    set_err(err, errlen, e);
    return 1;
  }
  return 0;
}

int32_t minilm_dim(const minilm_t*) { return minilm::kDim; }

}  // extern "C"
