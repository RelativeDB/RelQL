/* pql.hpp — dependency-light C++20 PQL lexer + recursive-descent parser.
 *
 * Faithful to python/src/relativedb/pql/parser.py and java Pql.g4. Parses a
 * PQL predictive query into an AST and serializes it to a JSON string in the
 * schema the language bindings deserialize. No third-party dependencies.
 */
#ifndef RELATIVEDB_PQL_HPP
#define RELATIVEDB_PQL_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pql {

// ---------------------------------------------------------------------------
// Enums (names/values mirror ast.py)
// ---------------------------------------------------------------------------
enum class AggFunc {
  SUM, AVG, MIN, MAX, COUNT, COUNT_DISTINCT, LIST_DISTINCT, FIRST, LAST
};

enum class TimeUnit { SECONDS, MINUTES, HOURS, DAYS, WEEKS, MONTHS };

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
struct Window {
  double start = 0.0;   // may be +/- infinity
  double end = 0.0;
  TimeUnit unit = TimeUnit::DAYS;
};

enum class ExprKind { Agg, Col, Cond, Logic, Not };

struct Expr {
  ExprKind kind;

  // Col
  std::string table;
  std::string column;

  // Agg
  AggFunc func = AggFunc::COUNT;
  std::shared_ptr<Expr> filter;   // inline WHERE inside the agg (nullable)
  bool has_window = false;
  Window window;

  // Cond
  std::shared_ptr<Expr> left;     // value expr
  Operator op = Operator::EQ;
  bool has_right = false;         // false => right is null (IS NULL etc.)
  Lit right;

  // Logic
  BoolOp bop = BoolOp::AND;
  std::shared_ptr<Expr> rleft;    // logic left
  std::shared_ptr<Expr> rright;   // logic right

  // Not
  std::shared_ptr<Expr> inner;
};

using ExprPtr = std::shared_ptr<Expr>;

struct ParsedQuery {
  ExprPtr target;
  std::string entity_table;
  std::string entity_column;
  std::vector<Lit> entity_ids;
  ExprPtr where;                  // nullable
  ExprPtr assuming;               // nullable
  RankKind rank = RankKind::NONE;
  bool has_top_k = false;
  long long top_k = 0;
  bool has_num_forecasts = false;
  long long num_forecasts = 0;
};

// Thrown on any lex/parse (syntax) error.
class PqlError : public std::runtime_error {
 public:
  explicit PqlError(const std::string& msg) : std::runtime_error(msg) {}
};

// Parse a PQL query. Throws PqlError on syntax error.
ParsedQuery parse(const std::string& query);

// Schema-less task-type inference (matches ParsedQuery.task_type(None)).
TaskType task_type(const ParsedQuery& q);

// Serialize a parsed query to the JSON AST schema.
std::string to_json(const ParsedQuery& q);

// Convenience: parse + serialize. Throws PqlError on syntax error.
std::string parse_to_json(const std::string& query);

}  // namespace pql

#endif  // RELATIVEDB_PQL_HPP
