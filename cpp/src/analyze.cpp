/* analyze.cpp — schema-aware validation and task-type inference.
 *
 * Ported from python/src/relativedb/relql/parser.py::validate and
 * ParsedQuery.task_type, which every binding would otherwise reimplement. The
 * messages are kept close to the originals: they are the user-facing contract
 * for a rejected query, and the Python tests assert on their wording.
 */
#include "analyze.hpp"

#include <set>

namespace relql {

namespace {

std::string col_str(const Expr& e) { return e.table + "." + e.column; }

// The primary key is a legal reference even though tables do not list it among
// their columns; pinning a cohort (`WHERE t.pk IN :ids`) depends on it.
void check_column(const Expr& col, const Schema& schema, bool allow_star,
                  bool allow_pk, bool allow_fk) {
  const TableDef* table = schema.table(col.table);
  if (!table)
    throw ValidationError("unknown table '" + col.table + "'");
  if (col.column == "*") {
    if (!allow_star)
      throw ValidationError(col_str(col) + ": '*' not allowed here");
    return;
  }
  if (table->column(col.column)) return;
  if (allow_pk && !table->primary_key.empty() &&
      col.column == table->primary_key)
    return;
  if (allow_fk) {
    for (const LinkDef* l : schema.links_from(col.table))
      if (l->fk_column == col.column) return;
  }
  throw ValidationError("unknown column '" + col.column + "' on table '" +
                        col.table + "'");
}

void walk_columns(const Expr* e, const Schema& schema) {
  if (!e) return;
  switch (e->kind) {
    case ExprKind::Agg: {
      // FK columns are legal aggregation targets for set/count aggregations;
      // FIRST/LAST exclude them per the docs.
      bool fk_ok = e->func == AggFunc::LIST_DISTINCT ||
                   e->func == AggFunc::ARRAY_AGG ||
                   e->func == AggFunc::COUNT ||
                   e->func == AggFunc::COUNT_DISTINCT;
      Expr col;
      col.kind = ExprKind::Col;
      col.table = e->table;
      col.column = e->column;
      check_column(col, schema, /*allow_star=*/true, /*allow_pk=*/false, fk_ok);
      // Only a frame the query actually wrote needs a time column to cut rows
      // against. The implied default expresses no temporal intent, so a table
      // without a time column stays legal there.
      if (e->has_window && !e->window.implied) {
        const TableDef* t = schema.table(e->table);
        if (t && t->time_column.empty())
          throw ValidationError("windowed aggregation over '" + e->table +
                                "', which has no time_column");
      }
      walk_columns(e->filter.get(), schema);
      return;
    }
    case ExprKind::Col:
      check_column(*e, schema, /*allow_star=*/false, /*allow_pk=*/true,
                   /*allow_fk=*/false);
      return;
    case ExprKind::Cond:
      walk_columns(e->left.get(), schema);
      walk_columns(e->right_expr.get(), schema);
      return;
    case ExprKind::Logic:
      walk_columns(e->rleft.get(), schema);
      walk_columns(e->rright.get(), schema);
      return;
    case ExprKind::Not:
      walk_columns(e->inner.get(), schema);
      return;
    case ExprKind::Arith:
      walk_columns(e->a_left.get(), schema);
      walk_columns(e->a_right.get(), schema);
      return;
    case ExprKind::Func:
      for (const ExprPtr& a : e->args) walk_columns(a.get(), schema);
      return;
    case ExprKind::Case:
      for (const ExprPtr& c : e->when_conds) walk_columns(c.get(), schema);
      for (const ExprPtr& t : e->when_thens) walk_columns(t.get(), schema);
      walk_columns(e->case_else.get(), schema);
      return;
    case ExprKind::Lit:
    case ExprKind::Param:
      return;
  }
}

void collect_aggregations(const Expr* e, std::vector<const Expr*>& out) {
  if (!e) return;
  if (e->kind == ExprKind::Agg) out.push_back(e);
  collect_aggregations(e->filter.get(), out);
  collect_aggregations(e->left.get(), out);
  collect_aggregations(e->right_expr.get(), out);
  collect_aggregations(e->rleft.get(), out);
  collect_aggregations(e->rright.get(), out);
  collect_aggregations(e->inner.get(), out);
  collect_aggregations(e->a_left.get(), out);
  collect_aggregations(e->a_right.get(), out);
  for (const ExprPtr& a : e->args) collect_aggregations(a.get(), out);
  for (const ExprPtr& c : e->when_conds) collect_aggregations(c.get(), out);
  for (const ExprPtr& t : e->when_thens) collect_aggregations(t.get(), out);
  collect_aggregations(e->case_else.get(), out);
}

// A bare column's task depends on its declared type; without a schema the
// caller's default stands.
TaskType static_or_categorical(const std::string& table,
                               const std::string& column, const Schema& schema,
                               TaskType fallback) {
  const TableDef* t = schema.table(table);
  const ColumnDef* c = t ? t->column(column) : nullptr;
  if (!c) return fallback;
  switch (c->type) {
    case ValueType::NUMBER: return TaskType::REGRESSION;
    case ValueType::BOOLEAN: return TaskType::BINARY_CLASSIFICATION;
    case ValueType::DATETIME: return TaskType::REGRESSION;
    default: return TaskType::MULTICLASS_CLASSIFICATION;
  }
}

// RETURN kind -> task types it is compatible with (contract section 1).
const std::map<std::string, std::set<TaskType>>& return_compatibility() {
  static const std::map<std::string, std::set<TaskType>> table = {
      {"EXPECTED_VALUE",
       {TaskType::REGRESSION, TaskType::FORECASTING,
        TaskType::BINARY_CLASSIFICATION}},
      {"PROBABILITY", {TaskType::BINARY_CLASSIFICATION}},
      {"CLASS",
       {TaskType::BINARY_CLASSIFICATION,
        TaskType::MULTICLASS_CLASSIFICATION}},
      {"DISTRIBUTION",
       {TaskType::BINARY_CLASSIFICATION,
        TaskType::MULTICLASS_CLASSIFICATION}},
      {"MULTILABEL", {TaskType::MULTILABEL_RANKING}},
      {"MULTICLASS", {TaskType::MULTICLASS_CLASSIFICATION}},
  };
  return table;
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

const char* return_kind_name(ReturnKind k) {
  switch (k) {
    case ReturnKind::EXPECTED_VALUE: return "EXPECTED_VALUE";
    case ReturnKind::PROBABILITY: return "PROBABILITY";
    case ReturnKind::CLASS: return "CLASS";
    case ReturnKind::DISTRIBUTION: return "DISTRIBUTION";
    case ReturnKind::MULTILABEL: return "MULTILABEL";
    case ReturnKind::MULTICLASS: return "MULTICLASS";
    default: return "";
  }
}

void validate_return(const ReturnSpec& ret, TaskType task) {
  const std::string kind = return_kind_name(ret.kind);
  auto it = return_compatibility().find(kind);
  if (it == return_compatibility().end())
    throw ValidationError("unknown RETURN kind '" + kind + "'");
  if (it->second.count(task)) return;
  std::string allowed;
  std::set<std::string> names;
  for (TaskType t : it->second) names.insert(task_name(t));
  for (const std::string& n : names) {
    if (!allowed.empty()) allowed += ", ";
    allowed += n;
  }
  throw ValidationError("RETURN " + kind + " is not compatible with inferred "
                        "task '" + std::string(task_name(task)) +
                        "' (allowed tasks: " + allowed + ")");
}

}  // namespace

TaskType task_type(const ParsedQuery& q, const Schema& schema) {
  if (q.has_num_forecasts) return TaskType::FORECASTING;
  if (q.rank == RankKind::RANK) return TaskType::MULTILABEL_RANKING;
  if (q.rank == RankKind::CLASSIFY) return TaskType::MULTICLASS_CLASSIFICATION;
  const Expr& t = *q.target;
  if (t.kind == ExprKind::Cond || t.kind == ExprKind::Logic ||
      t.kind == ExprKind::Not)
    return TaskType::BINARY_CLASSIFICATION;
  if (t.kind == ExprKind::Lit)
    return t.lit.kind == LitKind::Bool ? TaskType::BINARY_CLASSIFICATION
                                       : TaskType::REGRESSION;
  if (t.kind == ExprKind::Arith || t.kind == ExprKind::Func ||
      t.kind == ExprKind::Case)
    return TaskType::REGRESSION;
  if (t.kind == ExprKind::Agg) {
    if (t.func == AggFunc::EXISTS) return TaskType::BINARY_CLASSIFICATION;
    if (t.func == AggFunc::LIST_DISTINCT || t.func == AggFunc::ARRAY_AGG)
      return TaskType::MULTILABEL_RANKING;
    if (t.func == AggFunc::FIRST || t.func == AggFunc::LAST)
      return static_or_categorical(t.table, t.column, schema,
                                   TaskType::MULTICLASS_CLASSIFICATION);
    return TaskType::REGRESSION;
  }
  // bare column
  return static_or_categorical(t.table, t.column, schema,
                               TaskType::MULTICLASS_CLASSIFICATION);
}

ParsedQuery validate(const ParsedQuery& q, const Schema& schema) {
  const TableDef* table = schema.table(q.entity_table);
  if (!table) {
    const char* origin =
        q.has_from ? "named by FROM" : "inferred from the PREDICT target";
    throw ValidationError("unknown entity table '" + q.entity_table + "' (" +
                          origin + ")");
  }
  if (table->primary_key.empty())
    throw ValidationError("table '" + q.entity_table +
                          "' declares no primary key, so it cannot be a "
                          "population");
  if (!q.entity_column.empty() && table->primary_key != q.entity_column)
    throw ValidationError(q.entity_table + "." + q.entity_column + ": '" +
                          q.entity_column + "' is not the primary key of '" +
                          q.entity_table + "' (expected '" +
                          table->primary_key + "')");

  ParsedQuery bound = q;
  bound.entity_column = table->primary_key;   // bind_entity_key

  walk_columns(bound.target.get(), schema);
  std::vector<const Expr*> target_aggs;
  collect_aggregations(bound.target.get(), target_aggs);
  for (const Expr* agg : target_aggs) {
    if (agg->has_window && agg->window.start < 0)
      throw ValidationError(
          "target window (" + std::to_string((long long)agg->window.start) +
          ", " + std::to_string((long long)agg->window.end) +
          "] must be future-facing (start >= 0)");
  }

  const std::pair<const char*, const Expr*> clauses[] = {
      {"WHERE", bound.where.get()}, {"ASSUMING", bound.assuming.get()}};
  for (const auto& [name, clause] : clauses) {
    if (!clause) continue;
    walk_columns(clause, schema);
    std::vector<const Expr*> aggs;
    collect_aggregations(clause, aggs);
    for (const Expr* agg : aggs) {
      if (agg->has_window && agg->window.horizons > 1)
        throw ValidationError(
            "HORIZONS > 1 is only allowed on the PREDICT target, not in " +
            std::string(name));
    }
  }

  TaskType task = task_type(bound, schema);
  if (bound.ret.present) validate_return(bound.ret, task);
  return bound;
}

std::string analyze_to_json(const std::string& query,
                            const std::string& schema_json) {
  Schema schema = schema_from_json(schema_json);
  ParsedQuery bound = validate(parse(query), schema);
  std::string out = "{\"query\":";
  out += to_json(bound);
  out += ",\"task_type\":\"";
  out += task_name(task_type(bound, schema));
  out += "\"}";
  return out;
}

}  // namespace relql
