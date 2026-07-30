#include "flat.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace relql {
namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

double unit_seconds(const std::string& u) {
  if (u == "seconds") return 1.0;
  if (u == "minutes") return 60.0;
  if (u == "hours") return 3600.0;
  if (u == "days") return 86400.0;
  if (u == "weeks") return 7 * 86400.0;
  // Calendar frames use the same 30/365-day approximation as the Python
  // window arithmetic (relql.ast.TimeUnit.delta).
  if (u == "months") return 30 * 86400.0;
  if (u == "years") return 365 * 86400.0;
  return 86400.0;
}

double bound_of(const JsonValue* v) {
  if (!v) return kNan;
  if (v->kind == JsonValue::Kind::Num) return v->num;
  if (v->kind == JsonValue::Kind::Str) {
    if (v->str == "inf") return std::numeric_limits<double>::infinity();
    if (v->str == "-inf") return -std::numeric_limits<double>::infinity();
  }
  return kNan;
}

// Stable across processes and runs: the categorical encoding must agree
// between fit time and score time without carrying a vocabulary.
double hash_feature(const std::string& s) {
  std::uint32_t h = 2166136261u;
  for (unsigned char c : s) { h ^= c; h *= 16777619u; }
  return static_cast<double>(h);
}

// "YYYY-MM-DD[THH:MM[:SS]...]" -> epoch seconds (UTC). NaN when not a date.
double parse_iso_seconds(const std::string& s) {
  if (s.size() < 10 || s[4] != '-' || s[7] != '-') return kNan;
  auto num = [&](size_t pos, size_t len) -> long {
    long v = 0;
    for (size_t i = pos; i < pos + len; ++i) {
      if (s[i] < '0' || s[i] > '9') return -1;
      v = v * 10 + (s[i] - '0');
    }
    return v;
  };
  long y = num(0, 4), m = num(5, 2), d = num(8, 2);
  if (y < 0 || m < 1 || m > 12 || d < 1 || d > 31) return kNan;
  // Howard Hinnant's days-from-civil.
  y -= m <= 2;
  const long era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  double secs = (era * 146097LL + static_cast<long long>(doe) - 719468) * 86400.0;
  if (s.size() >= 16 && s[10] == 'T' && s[13] == ':') {
    long hh = num(11, 2), mi = num(14, 2);
    long ss = (s.size() >= 19 && s[16] == ':') ? num(17, 2) : 0;
    if (hh >= 0 && mi >= 0 && ss >= 0)
      secs += hh * 3600.0 + mi * 60.0 + ss;
  }
  return secs;
}

// A cell as a number, honoring the encoder contract (datetimes arrive as
// epoch seconds, but tolerate ISO strings). NaN when not numeric.
double cell_number(const JsonValue& v) {
  switch (v.kind) {
    case JsonValue::Kind::Num: return v.num;
    case JsonValue::Kind::Bool: return v.b ? 1.0 : 0.0;
    case JsonValue::Kind::Str: return parse_iso_seconds(v.str);
    default: return kNan;
  }
}

struct Ref {
  const JsonValue* row = nullptr;
  bool has_ts = false;
  double ts = 0.0;
};

// A row's value for `col`, falling back to its foreign keys — FK columns are
// edges, not cells, but they are legal aggregation subjects.
const JsonValue* cell_of(const JsonValue& row, const std::string& col) {
  if (const JsonValue* cells = row.find("cells"))
    if (const JsonValue* v = cells->find(col))
      if (!v->is_null()) return v;
  if (const JsonValue* parents = row.find("parents"))
    if (const JsonValue* v = parents->find(col))
      if (!v->is_null()) return v;
  return nullptr;
}

// ---------------------------------------------------------------------------
// inline WHERE inside an aggregation, mirroring evaluate.eval_row_predicate.
// The filter tree arrives as spec JSON (cond/logic/not, literal RHS); a
// comparison this evaluator cannot decide excludes the row.
// ---------------------------------------------------------------------------

bool like_match(const std::string& s, const std::string& pat, size_t si,
                size_t pi) {
  while (pi < pat.size()) {
    if (pat[pi] == '%') {
      for (size_t k = si; k <= s.size(); ++k)
        if (like_match(s, pat, k, pi + 1)) return true;
      return false;
    }
    if (si >= s.size()) return false;
    if (pat[pi] != '_' &&
        std::tolower(static_cast<unsigned char>(pat[pi])) !=
            std::tolower(static_cast<unsigned char>(s[si])))
      return false;
    ++si; ++pi;
  }
  return si == s.size();
}

// The literal side of a filter comparison, as spec JSON: JSON-native scalars,
// {"date": "<iso>"} for dates, arrays for IN lists.
bool is_date_lit(const JsonValue& lit) {
  return lit.kind == JsonValue::Kind::Obj && lit.find("date");
}

bool lit_matches(const JsonValue& cell, const JsonValue& lit) {
  if (is_date_lit(lit))
    return cell_number(cell) == parse_iso_seconds(lit.find("date")->str);
  switch (lit.kind) {
    case JsonValue::Kind::Num:
      return cell.kind != JsonValue::Kind::Str && cell_number(cell) == lit.num;
    case JsonValue::Kind::Bool:
      return (cell.kind == JsonValue::Kind::Bool && cell.b == lit.b) ||
             (cell.kind == JsonValue::Kind::Num &&
              cell.num == (lit.b ? 1.0 : 0.0));
    case JsonValue::Kind::Str:
      return cell.kind == JsonValue::Kind::Str && cell.str == lit.str;
    case JsonValue::Kind::Null:
      return cell.is_null();
    default:
      return false;  // lists handled by IN
  }
}

double lit_number(const JsonValue& lit) {
  if (is_date_lit(lit)) return parse_iso_seconds(lit.find("date")->str);
  switch (lit.kind) {
    case JsonValue::Kind::Num: return lit.num;
    case JsonValue::Kind::Bool: return lit.b ? 1.0 : 0.0;
    default: return kNan;
  }
}

bool row_condition(const JsonValue& cond, const JsonValue& row) {
  const JsonValue* col = cond.find("column");
  const JsonValue* opv = cond.find("op");
  if (!col || col->kind != JsonValue::Kind::Str || !opv ||
      opv->kind != JsonValue::Kind::Str)
    return false;
  const std::string& op = opv->str;
  const JsonValue* cell = cell_of(row, col->str);
  if (op == "IS_NULL") return cell == nullptr;
  if (op == "IS_NOT_NULL") return cell != nullptr;
  if (cell == nullptr) return false;
  const JsonValue* r = cond.find("right");
  static const JsonValue kNull;
  if (!r) r = &kNull;
  if (op == "EQ") return lit_matches(*cell, *r);
  if (op == "NEQ") return !lit_matches(*cell, *r);
  if (op == "IN" || op == "NOT_IN") {
    bool found = false;
    if (r->kind == JsonValue::Kind::Arr)
      for (const JsonValue& item : r->arr)
        if (lit_matches(*cell, item)) { found = true; break; }
    return op == "IN" ? found : !found;
  }
  if (op == "GT" || op == "LT" || op == "GE" || op == "LE") {
    double l = cell_number(*cell), rv = lit_number(*r);
    if (std::isnan(l) || std::isnan(rv)) return false;
    if (op == "GT") return l > rv;
    if (op == "LT") return l < rv;
    if (op == "GE") return l >= rv;
    return l <= rv;
  }
  if (op == "STARTS_WITH" || op == "ENDS_WITH" || op == "CONTAINS" ||
      op == "NOT_CONTAINS" || op == "LIKE" || op == "NOT_LIKE") {
    if (cell->kind != JsonValue::Kind::Str || r->kind != JsonValue::Kind::Str)
      return false;
    const std::string& s = cell->str;
    const std::string& p = r->str;
    if (op == "STARTS_WITH")
      return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
    if (op == "ENDS_WITH")
      return s.size() >= p.size() &&
             s.compare(s.size() - p.size(), p.size(), p) == 0;
    if (op == "CONTAINS") return s.find(p) != std::string::npos;
    if (op == "NOT_CONTAINS") return s.find(p) == std::string::npos;
    if (op == "LIKE") return like_match(s, p, 0, 0);
    return !like_match(s, p, 0, 0);
  }
  return false;
}

bool row_predicate(const JsonValue& e, const JsonValue& row) {
  const JsonValue* kind = e.find("kind");
  if (!kind || kind->kind != JsonValue::Kind::Str) return false;
  if (kind->str == "logic") {
    const JsonValue* l = e.find("left");
    const JsonValue* r = e.find("right");
    const JsonValue* op = e.find("op");
    bool lb = l && row_predicate(*l, row);
    bool rb = r && row_predicate(*r, row);
    return (op && op->kind == JsonValue::Kind::Str && op->str == "AND")
               ? (lb && rb) : (lb || rb);
  }
  if (kind->str == "not") {
    const JsonValue* inner = e.find("expr");
    return inner && !row_predicate(*inner, row);
  }
  if (kind->str == "cond") return row_condition(e, row);
  return false;
}

// ---------------------------------------------------------------------------
// aggregation over context rows, mirroring evaluate.eval_value
// ---------------------------------------------------------------------------

using RowsByTable = std::map<std::string, std::vector<Ref>>;

// Distinctness key: kind-tagged so 1.0 and "1" stay distinct.
std::string distinct_key(const JsonValue& v) {
  switch (v.kind) {
    case JsonValue::Kind::Num: return "n:" + std::to_string(v.num);
    case JsonValue::Kind::Str: return "s:" + v.str;
    case JsonValue::Kind::Bool: return v.b ? "b:1" : "b:0";
    default: return "?";
  }
}

double eval_agg(const JsonValue& agg, const RowsByTable& by_table,
                bool has_anchor, double anchor) {
  static const std::vector<Ref> kEmpty;
  const JsonValue* tablev = agg.find("table");
  const JsonValue* funcv = agg.find("func");
  const JsonValue* columnv = agg.find("column");
  if (!tablev || !funcv || !columnv) return kNan;
  const std::string& func = funcv->str;
  const std::string& column = columnv->str;
  auto it = by_table.find(tablev->str);
  const std::vector<Ref>& all = (it == by_table.end()) ? kEmpty : it->second;

  std::vector<Ref> rows;
  const JsonValue* window = agg.find("window");
  if (window && window->kind == JsonValue::Kind::Obj) {
    // (anchor+start, anchor+end], start excluded, end included; undated rows
    // never enter a windowed frame.
    if (!has_anchor) return kNan;
    const JsonValue* unitv = window->find("unit");
    double us = unit_seconds(unitv && unitv->kind == JsonValue::Kind::Str
                                 ? unitv->str : "days");
    double lo = bound_of(window->find("start")) * us;  // may be -inf
    double hi = bound_of(window->find("end")) * us;    // may be +inf
    for (const Ref& r : all) {
      if (!r.has_ts) continue;
      if (std::isfinite(lo) && !(r.ts > anchor + lo)) continue;
      if (std::isfinite(hi) && !(r.ts <= anchor + hi)) continue;
      rows.push_back(r);
    }
  } else {
    rows = all;
  }
  std::stable_sort(rows.begin(), rows.end(), [](const Ref& a, const Ref& b) {
    if (a.has_ts != b.has_ts) return !a.has_ts;  // undated first
    return a.ts < b.ts;
  });
  if (const JsonValue* filter = agg.find("filter")) {
    if (filter->kind == JsonValue::Kind::Obj) {
      std::vector<Ref> kept;
      for (const Ref& r : rows)
        if (row_predicate(*filter, *r.row)) kept.push_back(r);
      rows.swap(kept);
    }
  }

  if (func == "EXISTS") return rows.empty() ? 0.0 : 1.0;
  const bool star = column == "*";
  if (func == "COUNT") {
    if (star) return static_cast<double>(rows.size());
    std::size_t n = 0;
    for (const Ref& r : rows)
      if (cell_of(*r.row, column)) ++n;
    return static_cast<double>(n);
  }

  std::vector<const JsonValue*> values;
  for (const Ref& r : rows) {
    const JsonValue* v = star ? nullptr : cell_of(*r.row, column);
    if (star || v) values.push_back(v);
  }
  if (func == "COUNT_DISTINCT") {
    std::set<std::string> seen;
    for (const JsonValue* v : values)
      if (v) seen.insert(distinct_key(*v));
    return static_cast<double>(seen.size());
  }
  auto scalar = [](const JsonValue* v) -> double {
    if (!v) return kNan;
    if (v->kind == JsonValue::Kind::Str && std::isnan(parse_iso_seconds(v->str)))
      return hash_feature(v->str);  // categorical FIRST/LAST stays usable
    return cell_number(*v);
  };
  if (func == "FIRST")
    return values.empty() ? kNan : scalar(values.front());
  if (func == "LAST")
    return values.empty() ? kNan : scalar(values.back());

  std::vector<double> nums;
  for (const JsonValue* v : values) {
    if (!v) return kNan;  // SUM(t.*) has no numeric meaning
    double d = cell_number(*v);
    if (std::isnan(d)) return kNan;  // whole column of the wrong type
    nums.push_back(d);
  }
  if (func == "SUM") {
    double s = 0;
    for (double d : nums) s += d;
    return s;
  }
  if (nums.empty()) return kNan;
  if (func == "AVG") {
    double s = 0;
    for (double d : nums) s += d;
    return s / static_cast<double>(nums.size());
  }
  if (func == "MIN") return *std::min_element(nums.begin(), nums.end());
  if (func == "MAX") return *std::max_element(nums.begin(), nums.end());
  return kNan;  // ARRAY_AGG / LIST_DISTINCT are never flat features
}

const JsonValue& spec_features(const JsonValue& spec) {
  const JsonValue* feats = spec.find("features");
  if (!feats || feats->kind != JsonValue::Kind::Arr)
    throw std::runtime_error("flat spec has no 'features' array");
  return *feats;
}

}  // namespace

std::size_t flat_spec_size(const JsonValue& spec) {
  return spec_features(spec).arr.size();
}

void flat_features(const JsonValue& spec, const JsonValue& context,
                   float* out) {
  const JsonValue* entity_tablev = spec.find("entity_table");
  const std::string entity_table =
      entity_tablev && entity_tablev->kind == JsonValue::Kind::Str
          ? entity_tablev->str : "";

  bool has_anchor = false;
  double anchor = 0.0;
  if (const JsonValue* a = context.find("anchor")) {
    if (a->kind == JsonValue::Kind::Num) { has_anchor = true; anchor = a->num; }
  }
  const JsonValue* entity_id = context.find("entity_id");

  RowsByTable by_table;
  const JsonValue* entity_row = nullptr;
  if (const JsonValue* rows = context.find("rows")) {
    for (const JsonValue& r : rows->arr) {
      const JsonValue* table = r.find("table");
      if (!table || table->kind != JsonValue::Kind::Str) continue;
      Ref ref;
      ref.row = &r;
      if (const JsonValue* ts = r.find("ts")) {
        if (ts->kind == JsonValue::Kind::Num) { ref.has_ts = true; ref.ts = ts->num; }
      }
      by_table[table->str].push_back(ref);
      if (!entity_row && table->str == entity_table && entity_id) {
        const JsonValue* id = r.find("id");
        if (id && id->kind == entity_id->kind &&
            ((id->kind == JsonValue::Kind::Str && id->str == entity_id->str) ||
             (id->kind == JsonValue::Kind::Num && id->num == entity_id->num)))
          entity_row = &r;
      }
    }
  }

  const JsonValue& feats = spec_features(spec);
  for (std::size_t i = 0; i < feats.arr.size(); ++i) {
    const JsonValue& f = feats.arr[i];
    const JsonValue* kindv = f.find("kind");
    const std::string kind =
        kindv && kindv->kind == JsonValue::Kind::Str ? kindv->str : "";
    double v = kNan;
    if (kind == "entity_column") {
      const JsonValue* colv = f.find("column");
      const JsonValue* cell =
          (entity_row && colv && colv->kind == JsonValue::Kind::Str)
              ? cell_of(*entity_row, colv->str) : nullptr;
      if (cell) {
        const JsonValue* ctv = f.find("col_type");
        const std::string col_type =
            ctv && ctv->kind == JsonValue::Kind::Str ? ctv->str : "";
        if (col_type == "datetime") {
          double when = cell_number(*cell);
          if (has_anchor && !std::isnan(when))
            v = (anchor - when) / 86400.0;
        } else if (cell->kind == JsonValue::Kind::Str &&
                   std::isnan(parse_iso_seconds(cell->str))) {
          // categorical value in a scalar slot: the stable hash keeps it
          // usable without carrying a vocabulary
          v = hash_feature(cell->str);
        } else {
          v = cell_number(*cell);
        }
      }
    } else if (kind == "aggregate") {
      if (const JsonValue* agg = f.find("agg"))
        v = eval_agg(*agg, by_table, has_anchor, anchor);
    } else if (kind == "days_since_last") {
      const JsonValue* tablev = f.find("table");
      if (has_anchor && tablev && tablev->kind == JsonValue::Kind::Str) {
        auto it = by_table.find(tablev->str);
        if (it != by_table.end()) {
          double latest = -std::numeric_limits<double>::infinity();
          for (const Ref& r : it->second)
            if (r.has_ts && r.ts <= anchor && r.ts > latest) latest = r.ts;
          if (std::isfinite(latest)) v = (anchor - latest) / 86400.0;
        }
      }
    }
    out[i] = static_cast<float>(v);
  }
}

}  // namespace relql
