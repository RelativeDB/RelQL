#include "relql_c.h"

#include <cstring>
#include <exception>
#include <string>

#include "analyze.hpp"
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
}  // namespace

extern "C" {

int relql_parse(const char* query, char* out, size_t outlen, char* err,
              size_t errlen) {
  try {
    if (query == nullptr) throw relql::RelqlError("null query");
    std::string json = relql::parse_to_json(std::string(query));
    if (out == nullptr || outlen == 0) {
      set_str(err, errlen, "output buffer too small");
      return 2;
    }
    if (json.size() + 1 > outlen) {
      set_str(out, outlen, json);  // truncated copy
      set_str(err, errlen, "output buffer too small for JSON AST");
      return 2;
    }
    set_str(out, outlen, json);
    return 0;
  } catch (const std::exception& e) {
    set_str(err, errlen, e.what());
    return 1;
  } catch (...) {
    set_str(err, errlen, "unknown error");
    return 1;
  }
}

int relql_analyze(const char* query, const char* schema_json,
                  const char* params_json, char* out, size_t outlen,
                  char* err, size_t errlen) {
  try {
    if (query == nullptr) throw relql::RelqlError("null query");
    if (schema_json == nullptr) throw relql::SchemaError("null schema");
    std::string json = relql::analyze_to_json(
        std::string(query), std::string(schema_json),
        params_json ? std::string(params_json) : std::string());
    if (out == nullptr || outlen == 0) {
      set_str(err, errlen, "output buffer too small");
      return 2;
    }
    if (json.size() + 1 > outlen) {
      set_str(out, outlen, json);  // truncated copy
      set_str(err, errlen, "output buffer too small for JSON AST");
      return 2;
    }
    set_str(out, outlen, json);
    return 0;
  } catch (const relql::RelqlError& e) {
    // Prefixed so a binding can raise the exception type its users expect
    // without the C layer needing to know what those are.
    set_str(err, errlen, std::string("syntax: ") + e.what());
    return 1;
  } catch (const relql::ValidationError& e) {
    set_str(err, errlen, std::string("invalid: ") + e.what());
    return 1;
  } catch (const relql::SchemaError& e) {
    set_str(err, errlen, std::string("schema: ") + e.what());
    return 1;
  } catch (const std::exception& e) {
    set_str(err, errlen, e.what());
    return 1;
  } catch (...) {
    set_str(err, errlen, "unknown error");
    return 1;
  }
}

}  // extern "C"
