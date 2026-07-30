/* flat_c.h — C ABI for the flat-feature evaluator (flat.hpp).
 *
 * Eligibility and feature derivation happen in Python (relativedb.flat) —
 * no RelQL text reaches this layer. The binding sends the feature-spec JSON
 * Python derived plus the encoded contexts; this evaluates the columns into
 * a dense row-major float matrix.
 *
 * Return codes: 0 ok, 1 error (err filled).
 */
#ifndef RELATIVEDB_FLAT_C_H
#define RELATIVEDB_FLAT_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluate the spec's feature columns over `contexts_json`:
 *   spec_json — relativedb.flat.flat_spec_to_json(analysis)
 *   contexts_json — [{"entity_id":…, "anchor":<epoch seconds|null>,
 *     "rows":[{"table":…,"id":…,"ts":<epoch seconds|null>,
 *              "cells":{…},"parents":{…}}]}, …]
 * Rows must be the entity's FOCAL rows; datetimes arrive as epoch seconds.
 * Fills `out` row-major [n_contexts x n_features]; unevaluable cells are NaN
 * (a tree model's native missing value). `n_features` must match the spec's
 * feature count. */
int relql_flat_features(const char* spec_json, const char* contexts_json,
                        float* out, int n_contexts, int n_features,
                        char* err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* RELATIVEDB_FLAT_C_H */
