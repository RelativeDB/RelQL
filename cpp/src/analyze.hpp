/* analyze.hpp — schema-aware validation and task-type inference.
 *
 * The parser answers "is this syntactically RelQL". This answers "is it
 * meaningful against this schema, and what task is it" — the semantic pass
 * that previously lived in each binding (python/.../relql/parser.py::validate).
 * Both questions now have one implementation, shared by every frontend.
 */
#ifndef RELATIVEDB_ANALYZE_HPP
#define RELATIVEDB_ANALYZE_HPP

#include <stdexcept>
#include <string>

#include "relql.hpp"
#include "schema.hpp"

namespace relql {

// A query that parses but cannot mean anything against this schema.
class ValidationError : public std::runtime_error {
 public:
  explicit ValidationError(const std::string& msg)
      : std::runtime_error(msg) {}
};

// Task-type inference with a schema in hand. Falls back to the schema-less
// rules (task_type) wherever the schema does not disambiguate; a bare column
// is the case that needs it, since the answer depends on the column's type.
TaskType task_type(const ParsedQuery& q, const Schema& schema);

// Validate `q` against `schema` and bind the population's primary key.
// Throws ValidationError with a message naming the offending clause.
// Returns the bound query; callers must use it rather than their input.
ParsedQuery validate(const ParsedQuery& q, const Schema& schema);

// Substitute every `:name` bind parameter with its value from `params_json`
// (a JSON object). A parameter in comparison-RHS position collapses onto the
// literal slot, so downstream code never has to know parameters existed.
// Throws ValidationError naming any parameter the map does not supply.
ParsedQuery bind_params(const ParsedQuery& q, const std::string& params_json);

// parse + validate + bind + plan, serialized as the JSON bindings consume:
//   {"query": <bound AST>, "task_type": "...", "plan": {...}}
// `params_json` may be empty or "{}" when the query has no parameters. The
// plan is built from the BOUND query: a cohort pinned through `IN :ids` is
// only visible once the parameter is substituted.
// Throws RelqlError (syntax) or ValidationError (semantics).
std::string analyze_to_json(const std::string& query,
                            const std::string& schema_json,
                            const std::string& params_json = "");

}  // namespace relql

#endif  // RELATIVEDB_ANALYZE_HPP
