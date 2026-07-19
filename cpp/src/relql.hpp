/* relql.hpp — dependency-light C++20 RelQL lexer + recursive-descent parser.
 *
 * Parses a RelQL predictive query into an AST and serializes it to a JSON string
 * in the schema the language bindings deserialize. No third-party dependencies.
 *
 * Grammar v2 (see RelQL_EVOLUTION.md): temporal frames use a trailing
 * `OVER (window_spec)` / `OVER window_name` clause and `WINDOW name AS (...)`
 * declarations; the old positional `AGG(col, start, end[, unit])` form is gone.
 * Adds EXISTS, multi-horizon windows (HORIZONS/STEP), richer value expressions
 * (arithmetic, CASE, COALESCE/NULLIF/ABS/LOG/EXP/LEAST/GREATEST, TRUE/FALSE),
 * and query-level AS OF / RETURN / ABLATE / EXPLAIN clauses.
 */
#ifndef RELATIVEDB_RelQL_HPP
#define RELATIVEDB_RelQL_HPP

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace relql {

// ---------------------------------------------------------------------------
// Enums (names/values mirror the binding ASTs)
// ---------------------------------------------------------------------------
enum class AggFunc {
  SUM, AVG, MIN, MAX, COUNT, COUNT_DISTINCT, LIST_DISTINCT, ARRAY_AGG, FIRST,
  LAST, EXISTS
};

enum class TimeUnit { SECONDS, MINUTES, HOURS, DAYS, WEEKS, MONTHS, YEARS };

enum class Operator {
  GT, LT, EQ, NEQ, GE, LE,
  STARTS_WITH, ENDS_WITH, CONTAINS, NOT_CONTAINS, LIKE, NOT_LIKE,
  IN, NOT_IN, IS_NULL, IS_NOT_NULL
};

enum class BoolOp { AND, OR };

enum class RankKind { NONE, CLASSIFY, RANK };

enum class TaskType {
  REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION,
  MULTILABEL_RANKING, FORECASTING
};

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------
enum class LitKind { Int, Float, Str, Bool, Null, Date, List };

struct Lit {
  LitKind kind = LitKind::Null;
  long long ival = 0;
  double dval = 0.0;
  std::string sval;   // Str body, Date text, or raw numeric text for Float
  bool bval = false;
  std::vector<Lit> items;  // List
};

// ---------------------------------------------------------------------------
// Window / Expr
// ---------------------------------------------------------------------------
// A normalized temporal frame `(start, end]` in `unit`, with an optional
// multi-horizon projection. `start`/`end` are offsets from the query anchor
// NOW (may be +/- infinity). `horizons` >= 1; when > 1 the query forecasts.
// `step` (present only when explicit) is the horizon stride in `unit`; when
// absent the effective step defaults to the frame width `end - start`.
//
// An aggregation with no `OVER` clause gets an implied unbounded frame whose
// direction follows the clause it sits in: future `(0, +inf]` in PREDICT and
// ASSUMING, past `(-inf, 0]` in WHERE. `implied` records that, so EXPLAIN can
// distinguish "unbounded by default" from an explicit UNBOUNDED frame.
//
// `top_k` is the frame's `RANK TOP k` directive: keep only the k most likely
// values from this frame. It lives on the frame, not the query, so different
// aggregations in one target may rank independently.
struct Window {
  double start = 0.0;   // may be +/- infinity
  double end = 0.0;
  TimeUnit unit = TimeUnit::DAYS;
  long long horizons = 1;
  bool has_step = false;
  double step = 0.0;
  bool implied = false;
  bool has_top_k = false;
  long long top_k = 0;
};

enum class ExprKind { Agg, Col, Cond, Logic, Not, Arith, Func, Case, Lit, Param };

struct Expr {
  ExprKind kind = ExprKind::Col;

  // Col
  std::string table;
  std::string column;

  // Agg
  AggFunc func = AggFunc::COUNT;
  std::shared_ptr<Expr> filter;   // inline WHERE inside the agg (nullable)
  bool has_window = false;
  Window window;
  std::string window_ref;         // OVER <name> reference, resolved post-parse

  // Cond
  std::shared_ptr<Expr> left;     // value expr
  Operator op = Operator::EQ;
  bool has_right = false;         // false => right is null (IS NULL etc.)
  Lit right;
  std::shared_ptr<Expr> right_expr;  // non-literal comparison RHS (nullable)

  // Logic
  BoolOp bop = BoolOp::AND;
  std::shared_ptr<Expr> rleft;    // logic left
  std::shared_ptr<Expr> rright;   // logic right

  // Not
  std::shared_ptr<Expr> inner;

  // Arith (op in {+,-,*,/})
  char arith_op = '+';
  std::shared_ptr<Expr> a_left;
  std::shared_ptr<Expr> a_right;

  // Func (COALESCE/NULLIF/ABS/LOG/EXP/LEAST/GREATEST)
  std::string func_name;
  std::vector<std::shared_ptr<Expr>> args;

  // Case
  std::vector<std::shared_ptr<Expr>> when_conds;
  std::vector<std::shared_ptr<Expr>> when_thens;
  std::shared_ptr<Expr> case_else;   // nullable

  // Lit (literal in value position)
  Lit lit;

  // Param — a `:name` bind parameter standing in for a literal or, for
  // `IN :name`, for the whole list. Bound to a value at execution time.
  std::string param_name;
};

using ExprPtr = std::shared_ptr<Expr>;

// ---------------------------------------------------------------------------
// Query-level clauses
// ---------------------------------------------------------------------------
enum class ExplainMode { NONE, PLAN, CONTEXT, ANALYZE, ABLATION };
enum class ExplainFormat { TEXT, JSON };

struct Explain {
  bool present = false;
  ExplainMode mode = ExplainMode::PLAN;
  ExplainFormat format = ExplainFormat::TEXT;
};

enum class AnchorKind { PARAM, DATE, NOW };

struct AsOf {
  bool present = false;
  AnchorKind kind = AnchorKind::NOW;
  std::string value;  // param name (without ':') or date text
};

struct Ablation {
  std::string kind = "table";
  std::string name;
};

enum class ReturnKind {
  EXPECTED_VALUE, PROBABILITY, CLASS, DISTRIBUTION, MULTILABEL, MULTICLASS
};

struct ReturnSpec {
  bool present = false;
  ReturnKind kind = ReturnKind::EXPECTED_VALUE;
};

struct ParsedQuery {
  Explain explain;
  ExprPtr target;
  // The population. `entity_table` comes from `FROM <table>`, or is inferred
  // from the target's own table when there is no FROM clause. `entity_column`
  // is the primary key; the parser has no schema, so it stays empty and the
  // binding resolves it from the table's declared key during validation.
  std::string entity_table;
  std::string entity_column;
  bool has_from = false;          // false => population inferred from target
  ExprPtr where;                  // nullable
  ExprPtr assuming;               // nullable
  AsOf as_of;
  std::vector<Ablation> ablations;
  ReturnSpec ret;
  std::map<std::string, Window> windows;  // declared WINDOW templates
  // Derived from the target: CLASSIFY sets rank directly; RANK TOP k is a
  // frame directive (`Window::top_k`) lifted here so task-type inference and
  // the engine keep a single query-level view.
  RankKind rank = RankKind::NONE;
  bool has_top_k = false;
  long long top_k = 0;
  bool has_num_forecasts = false;
  long long num_forecasts = 0;    // derived from target window horizons
};

// Thrown on any lex/parse (syntax) error.
class RelqlError : public std::runtime_error {
 public:
  explicit RelqlError(const std::string& msg) : std::runtime_error(msg) {}
};

// Parse a RelQL query. Throws RelqlError on syntax error.
ParsedQuery parse(const std::string& query);

// Schema-less task-type inference.
TaskType task_type(const ParsedQuery& q);

// Serialize a parsed query to the JSON AST schema.
std::string to_json(const ParsedQuery& q);

// Convenience: parse + serialize. Throws RelqlError on syntax error.
std::string parse_to_json(const std::string& query);

}  // namespace relql

#endif  // RELATIVEDB_RelQL_HPP
