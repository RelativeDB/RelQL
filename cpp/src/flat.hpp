/* flat.hpp — can a RelQL query run as flat features, and what are they?
 *
 * Gradient-boosted trees consume one fixed-width numeric vector per entity.
 * That representation cannot see graph structure, but for scalar targets it
 * is a strong technique — so the decision of WHICH queries qualify, the
 * derivation of the feature columns, and the evaluation of those columns
 * over an assembled context all live here, single-sourced for every binding.
 * A binding (e.g. the Python XGBoost backend) only moves matrices.
 *
 * Eligibility is deliberately narrow: scalar regression / binary targets,
 * one horizon, no RANK/CLASSIFY, no ASSUMING (a fitted tree cannot honor a
 * counterfactual), no ABLATE. Everything else stays with the sequence model.
 *
 * The feature columns are the classic tabular recipe:
 *   - the entity row's own scalar columns (dates become age-at-anchor,
 *     categoricals a stable hash),
 *   - the target aggregation mirrored into recent PAST windows (the
 *     autoregressive signal),
 *   - every windowed aggregation the WHERE clause already computes,
 *   - per linked table: COUNT over standard past windows, recency, and
 *     SUM/AVG/MAX of each numeric column.
 *
 * Aggregation semantics mirror relativedb.evaluate: window (anchor+start,
 * anchor+end], undated rows excluded from windowed frames, inline filters on
 * the aggregated table's own columns, NULL in -> NaN out.
 */
#ifndef RELATIVEDB_FLAT_HPP
#define RELATIVEDB_FLAT_HPP

#include <string>
#include <vector>

#include "json.hpp"
#include "relql.hpp"
#include "schema.hpp"

namespace relql {

struct FlatFeature {
  enum class Kind { EntityColumn, Aggregate, DaysSinceLast };
  Kind kind = Kind::Aggregate;
  std::string name;

  // EntityColumn
  std::string column;
  ValueType col_type = ValueType::UNKNOWN;

  // Aggregate: a (possibly synthesized) Agg expr evaluated over the context.
  ExprPtr agg;

  // DaysSinceLast
  std::string table;
};

struct FlatSpec {
  bool eligible = false;
  std::string reason;                 // filled when ineligible
  TaskType task = TaskType::REGRESSION;
  std::string entity_table;
  std::vector<FlatFeature> features;  // empty when ineligible
};

// Derive the flat plan from a BOUND query (validate + bind_params first).
// Never throws for ineligibility — that is an answer, not an error.
FlatSpec derive_flat_spec(const ParsedQuery& bound, const Schema& schema);

// {"eligible":…,"reason":…,"task_type":…,"entity_table":…,"features":[names]}
std::string flat_spec_to_json(const FlatSpec& spec);

// One assembled context, decoded from the binding's JSON:
//   {"entity_id":…, "anchor": <epoch seconds|null>,
//    "rows":[{"table":…,"id":…,"ts":<epoch seconds|null>,
//             "cells":{…},"parents":{…}}]}
// Rows must be the FOCAL rows (the entity's own subgraph): the evaluator
// aggregates whatever it is given, and peer rows would count into features.
// Writes spec.features.size() floats; anything unevaluable becomes NaN.
void flat_features(const FlatSpec& spec, const JsonValue& context, float* out);

}  // namespace relql

#endif  // RELATIVEDB_FLAT_HPP
