#include "flat_c.h"

#include <cstring>
#include <exception>
#include <string>

#include "flat.hpp"
#include "json.hpp"

namespace {

void set_str(char* dst, size_t dstlen, const std::string& msg) {
  if (!dst || dstlen == 0) return;
  size_t n = msg.size();
  if (n > dstlen - 1) n = dstlen - 1;
  std::memcpy(dst, msg.data(), n);
  dst[n] = '\0';
}

int fail(char* err, size_t errlen) {
  try {
    throw;
  } catch (const std::exception& e) {
    set_str(err, errlen, e.what());
  } catch (...) {
    set_str(err, errlen, "unknown error");
  }
  return 1;
}

}  // namespace

extern "C" {

int relql_flat_features(const char* spec_json, const char* contexts_json,
                        float* out, int n_contexts, int n_features,
                        char* err, size_t errlen) {
  try {
    if (spec_json == nullptr) throw std::runtime_error("null spec");
    if (contexts_json == nullptr) throw std::runtime_error("null contexts");
    if (out == nullptr) throw std::runtime_error("null output matrix");
    relql::JsonValue spec = relql::json_parse(spec_json);
    if (static_cast<int>(relql::flat_spec_size(spec)) != n_features)
      throw std::runtime_error(
          "feature count mismatch: spec has " +
          std::to_string(relql::flat_spec_size(spec)) +
          ", caller allocated " + std::to_string(n_features));
    relql::JsonValue contexts = relql::json_parse(contexts_json);
    if (contexts.kind != relql::JsonValue::Kind::Arr)
      throw std::runtime_error("contexts must be a JSON array");
    if (static_cast<int>(contexts.arr.size()) != n_contexts)
      throw std::runtime_error(
          "context count mismatch: JSON has " +
          std::to_string(contexts.arr.size()) + ", caller declared " +
          std::to_string(n_contexts));
    for (int i = 0; i < n_contexts; ++i)
      relql::flat_features(spec, contexts.arr[i],
                           out + static_cast<size_t>(i) * n_features);
    return 0;
  } catch (...) {
    return fail(err, errlen);
  }
}

}  // extern "C"
