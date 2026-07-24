/* plan.hpp — the logical query plan, shared by every binding.
 *
 * What a query means and how it selects its population is a property of the
 * query and the schema, not of the language calling in: the target's rendered
 * form, the cohort a WHERE pins, the output form, the temporal frames, where
 * the anchor came from, and what an ASSUMING clause asserts. That half of
 * planning lives here so a frontend inherits it rather than reimplementing it.
 *
 * The *physical* half — which execution strategy runs, which sampler, how work
 * is batched — stays in the binding, because it names that binding's own
 * objects and threading. A frontend adds those fields to what this produces.
 */
#ifndef RELATIVEDB_PLAN_HPP
#define RELATIVEDB_PLAN_HPP

#include <string>
#include <vector>

#include "relql.hpp"
#include "schema.hpp"

namespace relql {

struct PlanWindow {
  std::string table;
  std::string time_column;   // empty when the table declares none
  double start = 0;
  double end = 0;
  std::string unit;
  long long horizons = 1;
  bool has_step = false;
  double step = 0;
  std::string role;          // target | where | assuming
};

struct LogicalPlan {
  std::string target;              // normalized rendering of the target
  TaskType task = TaskType::REGRESSION;
  std::string entity_table;
  std::string entity_pk;
  // The cohort a primary-key pin names. `selector_all` means the WHERE does
  // not pin one and the population is the whole table.
  bool selector_all = true;
  std::vector<std::string> selector;
  std::string output;              // probability | value | class | ...
  std::vector<PlanWindow> windows;
  bool where_present = false;
  bool assuming_present = false;
  bool has_assuming_plan = false;
  std::string assuming;            // "t.c := v, ..." when applicable
  std::string as_of_source;        // execution-anchor | query-date | query-param
  std::string as_of_param;         // set only for query-param
  std::vector<std::string> ablations;
  std::vector<std::string> warnings;
};

// Render one expression the way EXPLAIN shows it.
std::string expr_to_string(const Expr& e);

// True when every WHERE leaf is a primary-key pin, i.e. the clause selects the
// cohort and nothing else.
bool pure_pin(const Expr* where, const std::string& entity_table,
              const std::string& entity_column);

// Build the logical plan for a validated query.
LogicalPlan build_logical_plan(const ParsedQuery& q, const Schema& schema);

// Serialize for the C ABI.
std::string plan_to_json(const LogicalPlan& p);

}  // namespace relql

#endif  // RELATIVEDB_PLAN_HPP
