#include "flat_c.h"

#include <cstring>
#include <exception>
#include <string>

#include "analyze.hpp"
#include "flat.hpp"
#include "json.hpp"
#include "relql.hpp"
#include "schema.hpp"

namespace {

void set_str(char* dst, size_t dstlen, const std::string& msg) {
  if (!dst || dstlen == 0) return;
  size_t n = msg.size();
  if (n > dstlen - 1) n = dstlen - 1;
  std::memcpy(dst, msg.data(), n);
  dst[n] = '\0';
}

relql::FlatSpec analyzed_spec(const char* query, const char* schema_json,
                              const char* params_json) {
  if (query == nullptr) throw relql::RelqlError("null query");
  if (schema_json == nullptr) throw relql::SchemaError("null schema");
  relql::Schema schema = relql::schema_from_json(schema_json);
  relql::ParsedQuery q = relql::parse(query);
  q = relql::validate(q, schema);
  q = relql::bind_params(q, params_json ? params_json : "");
  return relql::derive_flat_spec(q, schema);
}

int classify(char* err, size_t errlen) {
  try {
    throw;
  } catch (const relql::RelqlError& e) {
    set_str(err, errlen, std::string("syntax: ") + e.what());
  } catch (const relql::ValidationError& e) {
    set_str(err, errlen, std::string("invalid: ") + e.what());
  } catch (const relql::SchemaError& e) {
    set_str(err, errlen, std::string("schema: ") + e.what());
  } catch (const std::exception& e) {
    set_str(err, errlen, e.what());
  } catch (...) {
    set_str(err, errlen, "unknown error");
  }
  return 1;
}

}  // namespace

extern "C" {

int relql_flat_analyze(const char* query, const char* schema_json,
                       const char* params_json, char* out, size_t outlen,
                       char* err, size_t errlen) {
  try {
    std::string json =
        relql::flat_spec_to_json(analyzed_spec(query, schema_json, params_json));
    if (out == nullptr || outlen == 0 || json.size() + 1 > outlen) {
      set_str(out, outlen, json);  // truncated copy
      set_str(err, errlen, "output buffer too small for flat spec JSON");
      return 2;
    }
    set_str(out, outlen, json);
    return 0;
  } catch (...) {
    return classify(err, errlen);
  }
}

int relql_flat_features(const char* query, const char* schema_json,
                        const char* params_json, const char* contexts_json,
                        float* out, int n_contexts, int n_features,
                        char* err, size_t errlen) {
  try {
    if (contexts_json == nullptr) throw relql::JsonError("null contexts");
    if (out == nullptr) throw relql::JsonError("null output matrix");
    relql::FlatSpec spec = analyzed_spec(query, schema_json, params_json);
    if (!spec.eligible)
      throw relql::ValidationError(
          "query is not flat-eligible: " + spec.reason);
    if (static_cast<int>(spec.features.size()) != n_features)
      throw relql::ValidationError(
          "feature count mismatch: spec has " +
          std::to_string(spec.features.size()) + ", caller allocated " +
          std::to_string(n_features));
    relql::JsonValue contexts = relql::json_parse(contexts_json);
    if (contexts.kind != relql::JsonValue::Kind::Arr)
      throw relql::JsonError("contexts must be a JSON array");
    if (static_cast<int>(contexts.arr.size()) != n_contexts)
      throw relql::ValidationError(
          "context count mismatch: JSON has " +
          std::to_string(contexts.arr.size()) + ", caller declared " +
          std::to_string(n_contexts));
    for (int i = 0; i < n_contexts; ++i)
      relql::flat_features(spec, contexts.arr[i],
                           out + static_cast<size_t>(i) * n_features);
    return 0;
  } catch (...) {
    return classify(err, errlen);
  }
}

}  // extern "C"
