/* plan.cpp — logical planning.
 *
 * Ported from python/src/relativedb/plan.py, which every frontend would
 * otherwise reimplement. The rendered strings are user-facing (they are what
 * EXPLAIN prints), so the formats match the Python originals exactly.
 */
#include "plan.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <set>

#include "analyze.hpp"

namespace relql {

namespace {

// JSON-safe numeric: finite values pass through, infinities stringify.
std::string num_str(double v) {
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  if (v == (long long)v) return std::to_string((long long)v);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%g", v);
  return buf;
}

const char* unit_name(TimeUnit u) {
  switch (u) {
    case TimeUnit::SECONDS: return "seconds";
    case TimeUnit::MINUTES: return "minutes";
    case TimeUnit::HOURS: return "hours";
    case TimeUnit::DAYS: return "days";
    case TimeUnit::WEEKS: return "weeks";
    case TimeUnit::MONTHS: return "months";
    case TimeUnit::YEARS: return "years";
  }
  return "days";
}

const char* agg_name(AggFunc f) {
  switch (f) {
    case AggFunc::SUM: return "SUM";
    case AggFunc::AVG: return "AVG";
    case AggFunc::MIN: return "MIN";
    case AggFunc::MAX: return "MAX";
    case AggFunc::COUNT: return "COUNT";
    case AggFunc::COUNT_DISTINCT: return "COUNT_DISTINCT";
    case AggFunc::LIST_DISTINCT: return "LIST_DISTINCT";
    case AggFunc::ARRAY_AGG: return "ARRAY_AGG";
    case AggFunc::FIRST: return "FIRST";
    case AggFunc::LAST: return "LAST";
    case AggFunc::EXISTS: return "EXISTS";
  }
  return "COUNT";
}

const char* op_name(Operator o) {
  switch (o) {
    case Operator::GT: return ">";
    case Operator::LT: return "<";
    case Operator::EQ: return "=";
    case Operator::NEQ: return "!=";
    case Operator::GE: return ">=";
    case Operator::LE: return "<=";
    case Operator::STARTS_WITH: return "STARTS WITH";
    case Operator::ENDS_WITH: return "ENDS WITH";
    case Operator::CONTAINS: return "CONTAINS";
    case Operator::NOT_CONTAINS: return "NOT CONTAINS";
    case Operator::LIKE: return "LIKE";
    case Operator::NOT_LIKE: return "NOT LIKE";
    case Operator::IN: return "IN";
    case Operator::NOT_IN: return "NOT IN";
    case Operator::IS_NULL: return "IS NULL";
    case Operator::IS_NOT_NULL: return "IS NOT NULL";
  }
  return "=";
}

// Matches Python's repr() for the literal kinds a query can carry: strings
// quoted, numbers bare, lists parenthesized.
std::string lit_str(const Lit& l) {
  switch (l.kind) {
    case LitKind::Str:
    case LitKind::Date:
      return "'" + l.sval + "'";
    case LitKind::Int:
      return std::to_string(l.ival);
    case LitKind::Float:
      return l.sval.empty() ? num_str(l.dval) : l.sval;
    case LitKind::Bool:
      return l.bval ? "True" : "False";
    case LitKind::Null:
      return "None";
    case LitKind::List: {
      std::string s = "(";
      for (size_t i = 0; i < l.items.size(); ++i) {
        if (i) s += ", ";
        s += lit_str(l.items[i]);
      }
      return s + ")";
    }
  }
  return "None";
}

// The bare value of a literal. lit_str renders for display ('C1'); a pinned
// cohort is a list of ids the caller looks entities up by, so it must not
// carry the quotes.
std::string lit_value(const Lit& l) {
  switch (l.kind) {
    case LitKind::Str:
    case LitKind::Date:
      return l.sval;
    case LitKind::Int:
      return std::to_string(l.ival);
    case LitKind::Float:
      return l.sval.empty() ? num_str(l.dval) : l.sval;
    case LitKind::Bool:
      return l.bval ? "True" : "False";
    default:
      return "";
  }
}

std::string window_str(const Window& w) {
  std::string s = "OVER (" + num_str(w.start) + ", " + num_str(w.end) + "] " +
                  unit_name(w.unit);
  if (w.horizons > 1) s += " HORIZONS " + std::to_string(w.horizons);
  return s;
}

void collect_windows(const Expr* e, const Schema& schema,
                     const std::string& role, std::vector<PlanWindow>& out) {
  if (!e) return;
  if (e->kind == ExprKind::Agg && e->has_window) {
    PlanWindow w;
    w.table = e->table;
    const TableDef* t = schema.table(e->table);
    w.time_column = t ? t->time_column : "";
    w.start = e->window.start;
    w.end = e->window.end;
    w.unit = unit_name(e->window.unit);
    w.horizons = e->window.horizons;
    w.has_step = e->window.has_step;
    w.step = e->window.step;
    w.role = role;
    out.push_back(w);
  }
  collect_windows(e->filter.get(), schema, role, out);
  collect_windows(e->left.get(), schema, role, out);
  collect_windows(e->right_expr.get(), schema, role, out);
  collect_windows(e->rleft.get(), schema, role, out);
  collect_windows(e->rright.get(), schema, role, out);
  collect_windows(e->inner.get(), schema, role, out);
  collect_windows(e->a_left.get(), schema, role, out);
  collect_windows(e->a_right.get(), schema, role, out);
  for (const ExprPtr& a : e->args) collect_windows(a.get(), schema, role, out);
  for (const ExprPtr& c : e->when_conds)
    collect_windows(c.get(), schema, role, out);
  for (const ExprPtr& t : e->when_thens)
    collect_windows(t.get(), schema, role, out);
  collect_windows(e->case_else.get(), schema, role, out);
}

// The cohort a WHERE pins the primary key to. Only conjunctive top-level
// predicates count: `pk = v` and `pk IN (...)` joined by AND. Under an OR (or
// a NOT) the clause no longer restricts the cohort on its own, so there is
// nothing safe to push down. Several ANDed pk predicates intersect.
bool pinned_ids(const Expr* where, const std::string& table,
                const std::string& column, std::vector<std::string>& out) {
  if (!where) return false;
  if (where->kind == ExprKind::Logic && where->bop == BoolOp::AND) {
    std::vector<std::string> l, r;
    bool has_l = pinned_ids(where->rleft.get(), table, column, l);
    bool has_r = pinned_ids(where->rright.get(), table, column, r);
    if (!has_l && !has_r) return false;
    if (!has_l) { out = r; return true; }
    if (!has_r) { out = l; return true; }
    std::set<std::string> keep(r.begin(), r.end());
    for (const std::string& v : l)
      if (keep.count(v)) out.push_back(v);   // intersect, order-stable
    return true;
  }
  if (where->kind != ExprKind::Cond) return false;
  const Expr* lhs = where->left.get();
  if (!lhs || lhs->kind != ExprKind::Col || lhs->table != table ||
      lhs->column != column)
    return false;
  if (where->right_expr) return false;       // unbound param / expression
  if (where->op == Operator::EQ) {
    out.push_back(lit_value(where->right));
    return true;
  }
  if (where->op == Operator::IN) {
    for (const Lit& item : where->right.items) out.push_back(lit_value(item));
    return true;
  }
  return false;
}

// The assignments and count bounds an ASSUMING clause states. Two shapes
// have one concrete answer the engine can build: `column = literal`, and a
// COUNT/EXISTS bound (`COUNT(t.*) OVER (...) >= k`, EXISTS, NOT EXISTS) —
// realized by adding or removing the entity's own rows. Anything else
// describes a set of possible worlds, so it is reported rather than applied.
bool assuming_count_bound(const Expr* e, std::string& out) {
  auto agg_ok = [](const Expr* a) {
    return a && a->kind == ExprKind::Agg &&
           (a->func == AggFunc::COUNT || a->func == AggFunc::EXISTS);
  };
  if (e->kind == ExprKind::Cond && agg_ok(e->left.get()) && !e->right_expr &&
      (e->right.kind == LitKind::Int || e->right.kind == LitKind::Float) &&
      (e->op == Operator::GE || e->op == Operator::GT ||
       e->op == Operator::EQ || e->op == Operator::LE ||
       e->op == Operator::LT)) {
    out = expr_to_string(*e);
    return true;
  }
  if (agg_ok(e) && e->func == AggFunc::EXISTS) {
    out = expr_to_string(*e);
    return true;
  }
  if (e->kind == ExprKind::Not && agg_ok(e->inner.get()) &&
      e->inner->func == AggFunc::EXISTS) {
    out = expr_to_string(*e);
    return true;
  }
  return false;
}

bool assuming_assignments(const Expr* e, std::string& out) {
  if (!e) return false;
  if (e->kind == ExprKind::Logic && e->bop == BoolOp::AND) {
    std::string l, r;
    if (!assuming_assignments(e->rleft.get(), l)) return false;
    if (!assuming_assignments(e->rright.get(), r)) return false;
    out = l + ", " + r;
    return true;
  }
  if (e->kind == ExprKind::Cond && e->op == Operator::EQ && e->left &&
      e->left->kind == ExprKind::Col && e->left->column != "*" &&
      !e->right_expr && e->right.kind != LitKind::List) {
    out = e->left->table + "." + e->left->column + " := " + lit_str(e->right);
    return true;
  }
  return assuming_count_bound(e, out);
}

const char* default_output(TaskType t) {
  switch (t) {
    case TaskType::REGRESSION: return "value";
    case TaskType::BINARY_CLASSIFICATION: return "probability";
    case TaskType::MULTICLASS_CLASSIFICATION: return "class";
    case TaskType::MULTILABEL_RANKING: return "ranked";
    case TaskType::FORECASTING: return "value-per-horizon";
  }
  return "value";
}

std::string lower(const std::string& s) {
  std::string out = s;
  for (char& c : out) c = (char)std::tolower((unsigned char)c);
  return out;
}

const char* return_kind_name(ReturnKind k) {
  switch (k) {
    case ReturnKind::EXPECTED_VALUE: return "expected_value";
    case ReturnKind::PROBABILITY: return "probability";
    case ReturnKind::CLASS: return "class";
    case ReturnKind::DISTRIBUTION: return "distribution";
    case ReturnKind::MULTILABEL: return "multilabel";
    case ReturnKind::MULTICLASS: return "multiclass";
  }
  return "expected_value";
}

const char* task_name(TaskType t) {
  switch (t) {
    case TaskType::REGRESSION: return "regression";
    case TaskType::BINARY_CLASSIFICATION: return "binary_classification";
    case TaskType::MULTICLASS_CLASSIFICATION:
      return "multiclass_classification";
    case TaskType::MULTILABEL_RANKING: return "multilabel_ranking";
    case TaskType::FORECASTING: return "forecasting";
  }
  return "regression";
}

// A finite bound is a JSON number; an infinite one has no JSON form, so it
// travels as the string "inf"/"-inf" the bindings already understand.
std::string num_json(double v) {
  if (std::isinf(v)) return v > 0 ? "\"inf\"" : "\"-inf\"";
  return num_str(v);
}

void json_escape(std::string& out, const std::string& s) {
  out += '"';
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  out += '"';
}

}  // namespace

std::string expr_to_string(const Expr& e) {
  switch (e.kind) {
    case ExprKind::Col:
      return e.table + "." + e.column;
    case ExprKind::Lit:
      return lit_str(e.lit);
    case ExprKind::Param:
      return ":" + e.param_name;
    case ExprKind::Agg: {
      std::string s = std::string(agg_name(e.func)) + "(" + e.table + "." +
                      e.column + ")";
      if (e.has_window) s += " " + window_str(e.window);
      return s;
    }
    case ExprKind::Cond: {
      std::string rhs = e.right_expr ? expr_to_string(*e.right_expr)
                                     : lit_str(e.right);
      return expr_to_string(*e.left) + " " + op_name(e.op) + " " + rhs;
    }
    case ExprKind::Logic:
      return "(" + expr_to_string(*e.rleft) + " " +
             (e.bop == BoolOp::AND ? "AND" : "OR") + " " +
             expr_to_string(*e.rright) + ")";
    case ExprKind::Not:
      return "NOT (" + expr_to_string(*e.inner) + ")";
    case ExprKind::Arith:
      return "(" + expr_to_string(*e.a_left) + " " + std::string(1, e.arith_op) +
             " " + expr_to_string(*e.a_right) + ")";
    case ExprKind::Func: {
      std::string s = e.func_name + "(";
      for (size_t i = 0; i < e.args.size(); ++i) {
        if (i) s += ", ";
        s += expr_to_string(*e.args[i]);
      }
      return s + ")";
    }
    case ExprKind::Case: {
      std::string s = "CASE";
      for (size_t i = 0; i < e.when_conds.size(); ++i)
        s += " WHEN " + expr_to_string(*e.when_conds[i]) + " THEN " +
             expr_to_string(*e.when_thens[i]);
      if (e.case_else) s += " ELSE " + expr_to_string(*e.case_else);
      return s + " END";
    }
  }
  return "";
}

bool pure_pin(const Expr* where, const std::string& entity_table,
              const std::string& entity_column) {
  if (!where) return false;
  if (where->kind == ExprKind::Logic && where->bop == BoolOp::AND)
    return pure_pin(where->rleft.get(), entity_table, entity_column) &&
           pure_pin(where->rright.get(), entity_table, entity_column);
  std::vector<std::string> ids;
  return pinned_ids(where, entity_table, entity_column, ids);
}

LogicalPlan build_logical_plan(const ParsedQuery& q, const Schema& schema) {
  LogicalPlan p;
  p.target = expr_to_string(*q.target);
  p.task = task_type(q, schema);
  p.entity_table = q.entity_table;
  p.entity_pk = q.entity_column;

  std::vector<std::string> ids;
  if (pinned_ids(q.where.get(), q.entity_table, q.entity_column, ids)) {
    p.selector_all = false;
    p.selector = ids;
  }

  p.output = q.ret.present ? return_kind_name(q.ret.kind)
                           : default_output(p.task);
  p.output = lower(p.output);

  collect_windows(q.target.get(), schema, "target", p.windows);
  collect_windows(q.where.get(), schema, "where", p.windows);
  collect_windows(q.assuming.get(), schema, "assuming", p.windows);

  p.where_present = q.where != nullptr;
  p.assuming_present = q.assuming != nullptr;
  if (q.assuming) {
    std::string rendered;
    if (assuming_assignments(q.assuming.get(), rendered)) {
      p.has_assuming_plan = true;
      p.assuming = rendered;
    } else {
      // EXPLAIN must describe any query that parses, so an inapplicable
      // clause is reported rather than raised.
      p.warnings.push_back(
          "ASSUMING '" + expr_to_string(*q.assuming) +
          "' cannot be applied: a counterfactual must state one concrete "
          "world - `column = literal`, or a count bound like `COUNT(t.*) "
          "OVER (...) >= k` / EXISTS(t.*), joined by AND. IN, OR, and "
          "inequalities on plain columns describe a set of possible worlds, "
          "so the engine cannot build the context they imply.");
    }
  }

  if (!q.as_of.present || q.as_of.kind == AnchorKind::NOW) {
    p.as_of_source = "execution-anchor";
  } else if (q.as_of.kind == AnchorKind::DATE) {
    p.as_of_source = "query-date";
  } else {
    p.as_of_source = "query-param";
    p.as_of_param = q.as_of.value;
  }

  for (const Ablation& a : q.ablations) p.ablations.push_back(a.name);
  return p;
}

std::string plan_to_json(const LogicalPlan& p) {
  std::string o = "{";
  o += "\"target\":";
  json_escape(o, p.target);
  o += ",\"task_type\":";
  json_escape(o, task_name(p.task));
  o += ",\"entity_table\":";
  json_escape(o, p.entity_table);
  o += ",\"entity_pk\":";
  json_escape(o, p.entity_pk);
  o += ",\"selector_all\":";
  o += p.selector_all ? "true" : "false";
  o += ",\"selector\":[";
  for (size_t i = 0; i < p.selector.size(); ++i) {
    if (i) o += ",";
    json_escape(o, p.selector[i]);
  }
  o += "],\"output\":";
  json_escape(o, p.output);
  o += ",\"windows\":[";
  for (size_t i = 0; i < p.windows.size(); ++i) {
    const PlanWindow& w = p.windows[i];
    if (i) o += ",";
    o += "{\"table\":";
    json_escape(o, w.table);
    o += ",\"time_column\":";
    if (w.time_column.empty()) o += "null"; else json_escape(o, w.time_column);
    o += ",\"start\":";
    o += num_json(w.start);
    o += ",\"end\":";
    o += num_json(w.end);
    o += ",\"unit\":";
    json_escape(o, w.unit);
    o += ",\"horizons\":" + std::to_string(w.horizons);
    o += ",\"step\":";
    if (w.has_step) o += num_json(w.step); else o += "null";
    o += ",\"role\":";
    json_escape(o, w.role);
    o += "}";
  }
  o += "],\"where_present\":";
  o += p.where_present ? "true" : "false";
  o += ",\"assuming_present\":";
  o += p.assuming_present ? "true" : "false";
  o += ",\"assuming\":";
  if (p.has_assuming_plan) json_escape(o, p.assuming); else o += "null";
  o += ",\"as_of_source\":";
  json_escape(o, p.as_of_source);
  o += ",\"as_of_param\":";
  if (p.as_of_param.empty()) o += "null"; else json_escape(o, p.as_of_param);
  o += ",\"ablations\":[";
  for (size_t i = 0; i < p.ablations.size(); ++i) {
    if (i) o += ",";
    json_escape(o, p.ablations[i]);
  }
  o += "],\"warnings\":[";
  for (size_t i = 0; i < p.warnings.size(); ++i) {
    if (i) o += ",";
    json_escape(o, p.warnings[i]);
  }
  o += "]}";
  return o;
}

}  // namespace relql
