/* json.hpp — the minimal JSON reader the query front end needs.
 *
 * Covers exactly the document shapes this layer consumes: the schema
 * (Schema.to_json_dict) and a bind-parameter map. Deliberately small rather
 * than general — the alternative was a third-party dependency in a library
 * that has none.
 */
#ifndef RELATIVEDB_JSON_HPP
#define RELATIVEDB_JSON_HPP

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace relql {

class JsonError : public std::runtime_error {
 public:
  explicit JsonError(const std::string& m) : std::runtime_error(m) {}
};

struct JsonValue {
  enum class Kind { Null, Bool, Num, Str, Arr, Obj } kind = Kind::Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<JsonValue> arr;
  std::map<std::string, JsonValue> obj;

  bool is_null() const { return kind == Kind::Null; }
  const JsonValue* find(const std::string& key) const;
  std::string str_or(const std::string& key) const;
};

JsonValue json_parse(const std::string& text);

}  // namespace relql

#endif  // RELATIVEDB_JSON_HPP
