/* schema.cpp — schema model plus the minimal JSON reader it needs.
 *
 * The reader covers exactly the document shape Schema.to_json_dict emits:
 * objects, arrays, strings, numbers, booleans and null. It is deliberately
 * small rather than general — the alternative was a third-party dependency in
 * a library that has none, for one document per query.
 */
#include "schema.hpp"

#include <cctype>
#include <cstdlib>

namespace relql {

const ColumnDef* TableDef::column(const std::string& col) const {
  for (const ColumnDef& c : columns)
    if (c.name == col) return &c;
  return nullptr;
}

const TableDef* Schema::table(const std::string& name) const {
  for (const TableDef& t : tables)
    if (t.name == name) return &t;
  return nullptr;
}

std::vector<const LinkDef*> Schema::links_from(const std::string& table) const {
  std::vector<const LinkDef*> out;
  for (const LinkDef& l : links)
    if (l.from_table == table) out.push_back(&l);
  return out;
}

const char* value_type_name(ValueType t) {
  switch (t) {
    case ValueType::NUMBER: return "number";
    case ValueType::TEXT: return "text";
    case ValueType::BOOLEAN: return "boolean";
    case ValueType::DATETIME: return "datetime";
    case ValueType::CATEGORICAL: return "categorical";
    default: return "unknown";
  }
}

namespace {

ValueType value_type_from(const std::string& s) {
  if (s == "number") return ValueType::NUMBER;
  if (s == "text") return ValueType::TEXT;
  if (s == "boolean") return ValueType::BOOLEAN;
  if (s == "datetime") return ValueType::DATETIME;
  if (s == "categorical") return ValueType::CATEGORICAL;
  return ValueType::UNKNOWN;
}

// ---------------------------------------------------------------------------
// minimal JSON
// ---------------------------------------------------------------------------

struct JsonValue {
  enum class Kind { Null, Bool, Num, Str, Arr, Obj } kind = Kind::Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<JsonValue> arr;
  std::map<std::string, JsonValue> obj;

  bool is_null() const { return kind == Kind::Null; }
  const JsonValue* find(const std::string& key) const {
    if (kind != Kind::Obj) return nullptr;
    auto it = obj.find(key);
    return it == obj.end() ? nullptr : &it->second;
  }
  // A string field, or "" when absent/null — schema fields like time_column
  // are legitimately null rather than missing.
  std::string str_or(const std::string& key) const {
    const JsonValue* v = find(key);
    if (!v || v->kind != Kind::Str) return "";
    return v->str;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& s) : s_(s) {}

  JsonValue parse() {
    skip();
    JsonValue v = value();
    skip();
    if (i_ != s_.size()) fail("trailing characters after JSON value");
    return v;
  }

 private:
  const std::string& s_;
  size_t i_ = 0;

  [[noreturn]] void fail(const std::string& why) const {
    throw SchemaError("schema JSON at offset " + std::to_string(i_) + ": " +
                      why);
  }
  void skip() {
    while (i_ < s_.size() && std::isspace((unsigned char)s_[i_])) ++i_;
  }
  bool accept(char c) {
    skip();
    if (i_ < s_.size() && s_[i_] == c) { ++i_; return true; }
    return false;
  }
  void expect(char c) {
    if (!accept(c)) fail(std::string("expected '") + c + "'");
  }

  JsonValue value() {
    skip();
    if (i_ >= s_.size()) fail("unexpected end of input");
    char c = s_[i_];
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') { JsonValue v; v.kind = JsonValue::Kind::Str; v.str = string(); return v; }
    if (c == 't' || c == 'f') return boolean();
    if (c == 'n') return null();
    return number();
  }

  JsonValue object() {
    JsonValue v;
    v.kind = JsonValue::Kind::Obj;
    expect('{');
    if (accept('}')) return v;
    for (;;) {
      skip();
      std::string key = string();
      expect(':');
      v.obj[key] = value();
      if (accept(',')) continue;
      expect('}');
      return v;
    }
  }

  JsonValue array() {
    JsonValue v;
    v.kind = JsonValue::Kind::Arr;
    expect('[');
    if (accept(']')) return v;
    for (;;) {
      v.arr.push_back(value());
      if (accept(',')) continue;
      expect(']');
      return v;
    }
  }

  JsonValue boolean() {
    JsonValue v;
    v.kind = JsonValue::Kind::Bool;
    if (s_.compare(i_, 4, "true") == 0) { v.b = true; i_ += 4; return v; }
    if (s_.compare(i_, 5, "false") == 0) { v.b = false; i_ += 5; return v; }
    fail("expected true/false");
  }

  JsonValue null() {
    if (s_.compare(i_, 4, "null") != 0) fail("expected null");
    i_ += 4;
    return JsonValue();
  }

  JsonValue number() {
    size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
    while (i_ < s_.size() &&
           (std::isdigit((unsigned char)s_[i_]) || s_[i_] == '.' ||
            s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '-' || s_[i_] == '+'))
      ++i_;
    if (start == i_) fail("expected a value");
    JsonValue v;
    v.kind = JsonValue::Kind::Num;
    v.num = std::strtod(s_.substr(start, i_ - start).c_str(), nullptr);
    return v;
  }

  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size()) {
      char c = s_[i_++];
      if (c == '"') return out;
      if (c != '\\') { out.push_back(c); continue; }
      if (i_ >= s_.size()) fail("unterminated escape");
      char e = s_[i_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (i_ + 4 > s_.size()) fail("truncated \\u escape");
          unsigned cp = (unsigned)std::strtoul(s_.substr(i_, 4).c_str(),
                                               nullptr, 16);
          i_ += 4;
          // Schema identifiers are ASCII in practice; encode the BMP code
          // point as UTF-8 so a non-ASCII table name still round-trips.
          if (cp < 0x80) {
            out.push_back((char)cp);
          } else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
          } else {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
          }
          break;
        }
        default: fail("unknown escape");
      }
    }
    fail("unterminated string");
  }
};

}  // namespace

Schema schema_from_json(const std::string& json) {
  JsonValue root = JsonParser(json).parse();
  if (root.kind != JsonValue::Kind::Obj)
    throw SchemaError("schema JSON must be an object");

  Schema schema;
  const JsonValue* tables = root.find("tables");
  if (tables && tables->kind == JsonValue::Kind::Arr) {
    for (const JsonValue& t : tables->arr) {
      TableDef def;
      def.name = t.str_or("name");
      if (def.name.empty()) throw SchemaError("a table has no name");
      def.primary_key = t.str_or("primary_key");
      def.time_column = t.str_or("time_column");
      const JsonValue* cols = t.find("columns");
      if (cols && cols->kind == JsonValue::Kind::Arr) {
        for (const JsonValue& c : cols->arr) {
          ColumnDef col;
          col.name = c.str_or("name");
          col.type = value_type_from(c.str_or("type"));
          def.columns.push_back(col);
        }
      }
      schema.tables.push_back(def);
    }
  }

  const JsonValue* links = root.find("links");
  if (links && links->kind == JsonValue::Kind::Arr) {
    for (const JsonValue& l : links->arr) {
      LinkDef def;
      def.from_table = l.str_or("from_table");
      def.fk_column = l.str_or("fk_column");
      def.to_table = l.str_or("to_table");
      schema.links.push_back(def);
    }
  }
  return schema;
}

}  // namespace relql
