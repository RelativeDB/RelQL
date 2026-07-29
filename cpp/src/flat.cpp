#include "flat.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "analyze.hpp"
#include "plan.hpp"

namespace relql {
namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

double unit_seconds(TimeUnit u) {
  switch (u) {
    case TimeUnit::SECONDS: return 1.0;
    case TimeUnit::MINUTES: return 60.0;
    case TimeUnit::HOURS: return 3600.0;
    case TimeUnit::DAYS: return 86400.0;
    case TimeUnit::WEEKS: return 7 * 86400.0;
    // Calendar frames use the same 30/365-day approximation as the Python
    // window arithmetic (relql.ast.TimeUnit.delta).
    case TimeUnit::MONTHS: return 30 * 86400.0;
    case TimeUnit::YEARS: return 365 * 86400.0;
  }
  return 86400.0;
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
// A comparison this evaluator cannot decide excludes the row.
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

bool lit_matches(const JsonValue& cell, const Lit& lit) {
  switch (lit.kind) {
    case LitKind::Int:
      return cell.kind != JsonValue::Kind::Str &&
             cell_number(cell) == static_cast<double>(lit.ival);
    case LitKind::Float: {
      double d = lit.dval;
      return cell.kind != JsonValue::Kind::Str && cell_number(cell) == d;
    }
    case LitKind::Bool:
      return (cell.kind == JsonValue::Kind::Bool && cell.b == lit.bval) ||
             (cell.kind == JsonValue::Kind::Num &&
              cell.num == (lit.bval ? 1.0 : 0.0));
    case LitKind::Str:
      return cell.kind == JsonValue::Kind::Str && cell.str == lit.sval;
    case LitKind::Date:
      return cell_number(cell) == parse_iso_seconds(lit.sval);
    case LitKind::Null:
      return cell.is_null();
    case LitKind::List:
      return false;  // handled by IN
  }
  return false;
}

double lit_number(const Lit& lit) {
  switch (lit.kind) {
    case LitKind::Int: return static_cast<double>(lit.ival);
    case LitKind::Float: return lit.dval;
    case LitKind::Bool: return lit.bval ? 1.0 : 0.0;
    case LitKind::Date: return parse_iso_seconds(lit.sval);
    default: return kNan;
  }
}

bool row_condition(const Expr& cond, const JsonValue& row) {
  if (cond.kind != ExprKind::Cond || !cond.left ||
      cond.left->kind != ExprKind::Col)
    return false;
  const JsonValue* cell = cell_of(row, cond.left->column);
  if (cond.op == Operator::IS_NULL) return cell == nullptr;
  if (cond.op == Operator::IS_NOT_NULL) return cell != nullptr;
  if (cell == nullptr) return false;
  const Lit& r = cond.right;
  switch (cond.op) {
    case Operator::EQ: return lit_matches(*cell, r);
    case Operator::NEQ: return !lit_matches(*cell, r);
    case Operator::IN:
    case Operator::NOT_IN: {
      bool found = false;
      for (const Lit& item : r.items)
        if (lit_matches(*cell, item)) { found = true; break; }
      return cond.op == Operator::IN ? found : !found;
    }
    case Operator::GT: case Operator::LT:
    case Operator::GE: case Operator::LE: {
      double l = cell_number(*cell), rv = lit_number(r);
      if (std::isnan(l) || std::isnan(rv)) return false;
      if (cond.op == Operator::GT) return l > rv;
      if (cond.op == Operator::LT) return l < rv;
      if (cond.op == Operator::GE) return l >= rv;
      return l <= rv;
    }
    case Operator::STARTS_WITH: case Operator::ENDS_WITH:
    case Operator::CONTAINS: case Operator::NOT_CONTAINS:
    case Operator::LIKE: case Operator::NOT_LIKE: {
      if (cell->kind != JsonValue::Kind::Str || r.kind != LitKind::Str)
        return false;
      const std::string& s = cell->str;
      const std::string& p = r.sval;
      switch (cond.op) {
        case Operator::STARTS_WITH:
          return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
        case Operator::ENDS_WITH:
          return s.size() >= p.size() &&
                 s.compare(s.size() - p.size(), p.size(), p) == 0;
        case Operator::CONTAINS: return s.find(p) != std::string::npos;
        case Operator::NOT_CONTAINS: return s.find(p) == std::string::npos;
        case Operator::LIKE: return like_match(s, p, 0, 0);
        default: return !like_match(s, p, 0, 0);
      }
    }
    default: return false;
  }
}

bool row_predicate(const Expr& e, const JsonValue& row) {
  switch (e.kind) {
    case ExprKind::Logic: {
      bool l = e.rleft && row_predicate(*e.rleft, row);
      bool r = e.rright && row_predicate(*e.rright, row);
      return e.bop == BoolOp::AND ? (l && r) : (l || r);
    }
    case ExprKind::Not:
      return e.inner && !row_predicate(*e.inner, row);
    case ExprKind::Cond:
      return row_condition(e, row);
    default:
      return false;
  }
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

double eval_agg(const Expr& agg, const RowsByTable& by_table, bool has_anchor,
                double anchor) {
  static const std::vector<Ref> kEmpty;
  auto it = by_table.find(agg.table);
  const std::vector<Ref>& all = (it == by_table.end()) ? kEmpty : it->second;

  std::vector<Ref> rows;
  if (agg.has_window) {
    // (anchor+start, anchor+end], start excluded, end included; undated rows
    // never enter a windowed frame.
    if (!has_anchor) return kNan;
    double us = unit_seconds(agg.window.unit);
    double lo = agg.window.start * us;  // may be -inf
    double hi = agg.window.end * us;    // may be +inf
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
  if (agg.filter) {
    std::vector<Ref> kept;
    for (const Ref& r : rows)
      if (row_predicate(*agg.filter, *r.row)) kept.push_back(r);
    rows.swap(kept);
  }

  if (agg.func == AggFunc::EXISTS) return rows.empty() ? 0.0 : 1.0;
  const bool star = agg.column == "*";
  if (agg.func == AggFunc::COUNT) {
    if (star) return static_cast<double>(rows.size());
    std::size_t n = 0;
    for (const Ref& r : rows)
      if (cell_of(*r.row, agg.column)) ++n;
    return static_cast<double>(n);
  }

  std::vector<const JsonValue*> values;
  for (const Ref& r : rows) {
    const JsonValue* v = star ? nullptr : cell_of(*r.row, agg.column);
    if (star || v) values.push_back(v);
  }
  if (agg.func == AggFunc::COUNT_DISTINCT) {
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
  if (agg.func == AggFunc::FIRST)
    return values.empty() ? kNan : scalar(values.front());
  if (agg.func == AggFunc::LAST)
    return values.empty() ? kNan : scalar(values.back());

  std::vector<double> nums;
  for (const JsonValue* v : values) {
    if (!v) return kNan;  // SUM(t.*) has no numeric meaning
    double d = cell_number(*v);
    if (std::isnan(d)) return kNan;  // whole column of the wrong type
    nums.push_back(d);
  }
  if (agg.func == AggFunc::SUM) {
    double s = 0;
    for (double d : nums) s += d;
    return s;
  }
  if (nums.empty()) return kNan;
  if (agg.func == AggFunc::AVG) {
    double s = 0;
    for (double d : nums) s += d;
    return s / static_cast<double>(nums.size());
  }
  if (agg.func == AggFunc::MIN) return *std::min_element(nums.begin(), nums.end());
  if (agg.func == AggFunc::MAX) return *std::max_element(nums.begin(), nums.end());
  return kNan;  // ARRAY_AGG / LIST_DISTINCT are never flat features
}

// ---------------------------------------------------------------------------
// spec derivation
// ---------------------------------------------------------------------------

void collect_aggs(const Expr* e, std::vector<const Expr*>& out) {
  if (!e) return;
  if (e->kind == ExprKind::Agg) out.push_back(e);
  collect_aggs(e->filter.get(), out);
  collect_aggs(e->left.get(), out);
  collect_aggs(e->right_expr.get(), out);
  collect_aggs(e->rleft.get(), out);
  collect_aggs(e->rright.get(), out);
  collect_aggs(e->inner.get(), out);
  collect_aggs(e->a_left.get(), out);
  collect_aggs(e->a_right.get(), out);
  for (const ExprPtr& a : e->args) collect_aggs(a.get(), out);
  for (const ExprPtr& c : e->when_conds) collect_aggs(c.get(), out);
  for (const ExprPtr& t : e->when_thens) collect_aggs(t.get(), out);
  collect_aggs(e->case_else.get(), out);
}

ExprPtr make_agg(AggFunc func, const std::string& table,
                 const std::string& column, double start_days,
                 double end_days) {
  auto e = std::make_shared<Expr>();
  e->kind = ExprKind::Agg;
  e->func = func;
  e->table = table;
  e->column = column;
  if (std::isfinite(start_days)) {
    // Unbounded frames stay windowless: assembly already bounds the past,
    // and a windowed frame would drop static (undated) rows entirely.
    e->has_window = true;
    e->window.start = start_days;
    e->window.end = end_days;
    e->window.unit = TimeUnit::DAYS;
  }
  return e;
}

const double kPastDays[] = {7, 30, 90};

// Tables whose rows can appear in an entity-scoped context: within two link
// hops of the entity table, direction-blind (children, parents, siblings).
std::set<std::string> reachable_tables(const Schema& schema,
                                       const std::string& entity_table) {
  std::set<std::string> seen{entity_table};
  std::vector<std::string> frontier{entity_table};
  for (int hop = 0; hop < 2; ++hop) {
    std::vector<std::string> next;
    for (const std::string& t : frontier) {
      for (const LinkDef& l : schema.links) {
        std::string other;
        if (l.from_table == t) other = l.to_table;
        else if (l.to_table == t) other = l.from_table;
        else continue;
        if (seen.insert(other).second) next.push_back(other);
      }
    }
    frontier.swap(next);
  }
  seen.erase(entity_table);
  return seen;
}

void add_feature(FlatSpec& spec, std::set<std::string>& names, FlatFeature f) {
  if (names.insert(f.name).second) spec.features.push_back(std::move(f));
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (static_cast<unsigned char>(c) < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else out += c;
  }
  return out;
}

const char* task_name(TaskType t) {
  switch (t) {
    case TaskType::REGRESSION: return "regression";
    case TaskType::BINARY_CLASSIFICATION: return "binary_classification";
    case TaskType::MULTICLASS_CLASSIFICATION: return "multiclass_classification";
    case TaskType::MULTILABEL_RANKING: return "multilabel_ranking";
    case TaskType::FORECASTING: return "forecasting";
  }
  return "?";
}

}  // namespace

FlatSpec derive_flat_spec(const ParsedQuery& q, const Schema& schema) {
  FlatSpec spec;
  spec.entity_table = q.entity_table;
  spec.task = task_type(q, schema);
  auto decline = [&spec](const std::string& why) -> FlatSpec& {
    spec.eligible = false;
    spec.reason = why;
    spec.features.clear();
    return spec;
  };

  if (spec.task != TaskType::REGRESSION &&
      spec.task != TaskType::BINARY_CLASSIFICATION)
    return decline("flat features cover scalar regression and binary "
                   "classification; " + std::string(task_name(spec.task)) +
                   " needs the sequence model");
  if (q.rank != RankKind::NONE || q.has_top_k)
    return decline("RANK/CLASSIFY targets need the sequence model");
  if (q.has_num_forecasts && q.num_forecasts > 1)
    return decline("multi-horizon forecasting needs the sequence model");
  if (q.assuming)
    return decline("ASSUMING is a counterfactual on the context; a fitted "
                   "tree model cannot honor it");
  if (!q.ablations.empty())
    return decline("ABLATE changes the context a sequence model reads; flat "
                   "features have no equivalent");
  if (q.ret.present && q.ret.kind != ReturnKind::EXPECTED_VALUE &&
      q.ret.kind != ReturnKind::PROBABILITY)
    return decline("only RETURN EXPECTED VALUE / PROBABILITY map onto a "
                   "scalar tree prediction");

  std::vector<const Expr*> target_aggs;
  collect_aggs(q.target.get(), target_aggs);
  for (const Expr* a : target_aggs) {
    if (a->func == AggFunc::ARRAY_AGG || a->func == AggFunc::LIST_DISTINCT)
      return decline("list-valued aggregations in the target need the "
                     "sequence model");
    if (a->has_window && a->window.horizons > 1)
      return decline("multi-horizon windows need the sequence model");
  }

  spec.eligible = true;
  std::set<std::string> names;
  const TableDef* entity = schema.table(q.entity_table);

  // 1. The entity row's own scalar columns.
  if (entity) {
    for (const ColumnDef& c : entity->columns) {
      if (c.name == entity->primary_key) continue;
      if (c.type == ValueType::TEXT || c.type == ValueType::UNKNOWN) continue;
      FlatFeature f;
      f.kind = FlatFeature::Kind::EntityColumn;
      f.column = c.name;
      f.col_type = c.type;
      f.name = "entity." + c.name +
               (c.type == ValueType::DATETIME ? "_age_days" : "");
      add_feature(spec, names, std::move(f));
    }
  }

  // 2. The target mirrored into the recent past — the autoregressive signal.
  for (const Expr* a : target_aggs) {
    bool finite = a->has_window && std::isfinite(a->window.start) &&
                  std::isfinite(a->window.end);
    if (finite) {
      double w = a->window.end - a->window.start;
      if (w <= 0) w = 1;
      for (int i = 1; i <= 3; ++i) {
        auto clone = std::make_shared<Expr>(*a);
        clone->window.start = a->window.start - i * w;
        clone->window.end = a->window.end - i * w;
        FlatFeature f;
        f.kind = FlatFeature::Kind::Aggregate;
        f.agg = clone;
        f.name = "hist" + std::to_string(i) + ":" + expr_to_string(*clone);
        add_feature(spec, names, std::move(f));
      }
    } else {
      for (double d : kPastDays) {
        auto clone = std::make_shared<Expr>(*a);
        clone->has_window = true;
        clone->window = Window{};
        clone->window.start = -d;
        clone->window.end = 0;
        clone->window.unit = TimeUnit::DAYS;
        FlatFeature f;
        f.kind = FlatFeature::Kind::Aggregate;
        f.agg = clone;
        f.name = "hist:" + expr_to_string(*clone);
        add_feature(spec, names, std::move(f));
      }
    }
  }

  // 3. Whatever the WHERE clause already computes over the past.
  std::vector<const Expr*> where_aggs;
  collect_aggs(q.where.get(), where_aggs);
  for (const Expr* a : where_aggs) {
    if (a->func == AggFunc::ARRAY_AGG || a->func == AggFunc::LIST_DISTINCT)
      continue;
    if (a->has_window && a->window.horizons > 1) continue;
    auto clone = std::make_shared<Expr>(*a);
    FlatFeature f;
    f.kind = FlatFeature::Kind::Aggregate;
    f.agg = clone;
    f.name = "where:" + expr_to_string(*clone);
    add_feature(spec, names, std::move(f));
  }

  // 4. The standard per-table recipe over every linked table.
  std::set<std::string> nearby = reachable_tables(schema, q.entity_table);
  for (const TableDef& t : schema.tables) {
    if (!nearby.count(t.name)) continue;
    for (double d : kPastDays) {
      FlatFeature f;
      f.kind = FlatFeature::Kind::Aggregate;
      f.agg = make_agg(AggFunc::COUNT, t.name, "*", -d, 0);
      f.name = t.name + ".count_" + std::to_string(static_cast<int>(d)) + "d";
      add_feature(spec, names, std::move(f));
    }
    {
      FlatFeature f;
      f.kind = FlatFeature::Kind::Aggregate;
      f.agg = make_agg(AggFunc::COUNT, t.name, "*",
                       -std::numeric_limits<double>::infinity(), 0);
      f.name = t.name + ".count_all";
      add_feature(spec, names, std::move(f));
    }
    if (!t.time_column.empty()) {
      FlatFeature f;
      f.kind = FlatFeature::Kind::DaysSinceLast;
      f.table = t.name;
      f.name = t.name + ".recency_days";
      add_feature(spec, names, std::move(f));
    }
    for (const ColumnDef& c : t.columns) {
      if (c.type != ValueType::NUMBER || c.name == t.primary_key) continue;
      struct { AggFunc func; const char* tag; } recipes[] = {
          {AggFunc::SUM, "sum"}, {AggFunc::AVG, "avg"}, {AggFunc::MAX, "max"}};
      for (const auto& r : recipes) {
        FlatFeature f30;
        f30.kind = FlatFeature::Kind::Aggregate;
        f30.agg = make_agg(r.func, t.name, c.name, -30, 0);
        f30.name = t.name + "." + c.name + "_" + r.tag + "_30d";
        add_feature(spec, names, std::move(f30));
        FlatFeature fall;
        fall.kind = FlatFeature::Kind::Aggregate;
        fall.agg = make_agg(r.func, t.name, c.name,
                            -std::numeric_limits<double>::infinity(), 0);
        fall.name = t.name + "." + c.name + "_" + r.tag + "_all";
        add_feature(spec, names, std::move(fall));
      }
    }
  }
  return spec;
}

std::string flat_spec_to_json(const FlatSpec& spec) {
  std::string out = "{\"eligible\":";
  out += spec.eligible ? "true" : "false";
  out += ",\"reason\":\"" + json_escape(spec.reason) + "\"";
  out += ",\"task_type\":\"";
  out += task_name(spec.task);
  out += "\",\"entity_table\":\"" + json_escape(spec.entity_table) + "\"";
  out += ",\"features\":[";
  for (std::size_t i = 0; i < spec.features.size(); ++i) {
    if (i) out += ",";
    out += "\"" + json_escape(spec.features[i].name) + "\"";
  }
  out += "]}";
  return out;
}

void flat_features(const FlatSpec& spec, const JsonValue& context,
                   float* out) {
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
      if (!entity_row && table->str == spec.entity_table && entity_id) {
        const JsonValue* id = r.find("id");
        if (id && id->kind == entity_id->kind &&
            ((id->kind == JsonValue::Kind::Str && id->str == entity_id->str) ||
             (id->kind == JsonValue::Kind::Num && id->num == entity_id->num)))
          entity_row = &r;
      }
    }
  }

  for (std::size_t i = 0; i < spec.features.size(); ++i) {
    const FlatFeature& f = spec.features[i];
    double v = kNan;
    switch (f.kind) {
      case FlatFeature::Kind::EntityColumn: {
        const JsonValue* cell =
            entity_row ? cell_of(*entity_row, f.column) : nullptr;
        if (!cell) break;
        if (f.col_type == ValueType::DATETIME) {
          double when = cell_number(*cell);
          if (has_anchor && !std::isnan(when))
            v = (anchor - when) / 86400.0;
        } else if (f.col_type == ValueType::CATEGORICAL &&
                   cell->kind == JsonValue::Kind::Str) {
          v = hash_feature(cell->str);
        } else {
          v = cell_number(*cell);
        }
        break;
      }
      case FlatFeature::Kind::Aggregate:
        v = eval_agg(*f.agg, by_table, has_anchor, anchor);
        break;
      case FlatFeature::Kind::DaysSinceLast: {
        if (!has_anchor) break;
        auto it = by_table.find(f.table);
        if (it == by_table.end()) break;
        double latest = -std::numeric_limits<double>::infinity();
        for (const Ref& r : it->second)
          if (r.has_ts && r.ts <= anchor && r.ts > latest) latest = r.ts;
        if (std::isfinite(latest)) v = (anchor - latest) / 86400.0;
        break;
      }
    }
    out[i] = static_cast<float>(v);
  }
}

}  // namespace relql
