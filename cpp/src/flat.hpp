/* flat.hpp — evaluate flat feature columns over assembled contexts.
 *
 * Gradient-boosted trees consume one fixed-width numeric vector per entity.
 * WHICH queries qualify and WHICH feature columns a schema yields is decided
 * in Python (relativedb.flat) next to the parser and planner — no RelQL text
 * ever reaches this layer. What stays native is the numeric evaluation: the
 * binding sends the feature-spec JSON Python derived plus the encoded
 * contexts, and this module fills a dense row-major float matrix.
 *
 * Aggregation semantics mirror relativedb.evaluate: window (anchor+start,
 * anchor+end], undated rows excluded from windowed frames, inline filters on
 * the aggregated table's own columns, NULL in -> NaN out.
 */
#ifndef RELATIVEDB_FLAT_HPP
#define RELATIVEDB_FLAT_HPP

#include <cstddef>
#include <string>

#include "json.hpp"

namespace relql {

/* Feature spec (relativedb.flat.flat_spec_to_json):
 *   {"entity_table": str, "task_type": str, "features": [
 *      {"kind":"entity_column","name":…,"column":…,"col_type":…} |
 *      {"kind":"aggregate","name":…,"agg":{"func":…,"table":…,"column":…,
 *         "filter"?: <cond tree>, "window"?: {"start":…,"end":…,"unit":…}}} |
 *      {"kind":"days_since_last","name":…,"table":…}]}
 * Frame bounds are numbers or "inf"/"-inf"; filter trees use the same
 * cond/logic/not shapes the RelQL AST serializes to, with literal RHS only.
 */

// Number of features in a parsed spec; throws JsonError-style on bad shape.
std::size_t flat_spec_size(const JsonValue& spec);

/* Evaluate the spec's feature columns over one assembled context:
 *   {"entity_id":…, "anchor": <epoch seconds|null>,
 *    "rows":[{"table":…,"id":…,"ts":<epoch seconds|null>,
 *             "cells":{…},"parents":{…}}]}
 * Rows must be the entity's FOCAL rows (the entity's own subgraph): the
 * evaluator aggregates whatever it is given, and peer rows would count into
 * features. Writes flat_spec_size(spec) floats; anything unevaluable becomes
 * NaN (a tree model's native missing value). */
void flat_features(const JsonValue& spec, const JsonValue& context,
                   float* out);

}  // namespace relql

#endif  // RELATIVEDB_FLAT_HPP
