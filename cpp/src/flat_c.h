/* flat_c.h — C ABI for the flat-feature planner/evaluator (flat.hpp).
 *
 * The full pipeline a tree-model binding needs:
 *   relql_flat_analyze  — can this query run as flat features, and what are
 *                         the feature columns? (parse+validate+bind inside)
 *   relql_flat_features — evaluate those columns over assembled contexts
 *                         into a dense row-major float matrix.
 *
 * Return codes follow relql_c.h: 0 ok, 1 error (err filled, prefixed
 * "syntax: "/"invalid: "/"schema: " where applicable), 2 output buffer too
 * small (analyze only; grow and retry).
 */
#ifndef RELATIVEDB_FLAT_C_H
#define RELATIVEDB_FLAT_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Analyze `query` against `schema_json` (Schema.to_json_dict) with optional
 * `params_json` bindings. Writes JSON to `out`:
 *   {"eligible":bool,"reason":str,"task_type":str,"entity_table":str,
 *    "features":[name,...]}
 * Ineligibility is an ANSWER (rc 0, eligible:false with a reason), not an
 * error; rc 1 is reserved for queries that do not parse/validate at all. */
int relql_flat_analyze(const char* query, const char* schema_json,
                       const char* params_json, char* out, size_t outlen,
                       char* err, size_t errlen);

/* Evaluate the query's feature columns over `contexts_json`:
 *   [{"entity_id":…, "anchor":<epoch seconds|null>,
 *     "rows":[{"table":…,"id":…,"ts":<epoch seconds|null>,
 *              "cells":{…},"parents":{…}}]}, …]
 * Rows must be the entity's FOCAL rows; datetimes arrive as epoch seconds.
 * Fills `out` row-major [n_contexts x n_features]; unevaluable cells are NaN
 * (a tree model's native missing value). `n_features` must match what
 * relql_flat_analyze reported, and the query must be eligible. */
int relql_flat_features(const char* query, const char* schema_json,
                        const char* params_json, const char* contexts_json,
                        float* out, int n_contexts, int n_features,
                        char* err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* RELATIVEDB_FLAT_C_H */
