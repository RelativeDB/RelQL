/* json.cpp — the minimal JSON reader (see json.hpp). */
#include "json.hpp"

#include <cctype>
#include <cstdlib>

namespace relql {

const JsonValue* JsonValue::find(const std::string& key) const {
  if (kind != Kind::Obj) return nullptr;
  auto it = obj.find(key);
  return it == obj.end() ? nullptr : &it->second;
}

// A string field, or "" when absent or null — fields like time_column are
// legitimately null rather than missing.
std::string JsonValue::str_or(const std::string& key) const {
  const JsonValue* v = find(key);
  if (!v || v->kind != Kind::Str) return "";
  return v->str;
}

namespace {

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
    throw JsonError("JSON at offset " + std::to_string(i_) + ": " +
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

JsonValue json_parse(const std::string& text) {
  return JsonParser(text).parse();
}

}  // namespace relql
