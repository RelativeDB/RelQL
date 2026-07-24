/* relql_c.h — C ABI for the shared relativedb RelQL parser.
 *
 * One parser for all three language bindings (Python/Java/Rust). Wraps the
 * C++20 recursive-descent parser in relql.cpp. Follows the rt_c.h convention:
 * opaque errors via (char* err, size_t errlen), nonzero return on failure.
 */
#ifndef RelQL_C_H
#define RelQL_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Parse a RelQL query. On success writes a JSON AST (see relql.hpp / the schema
 * doc) into out (NUL-terminated, capped at outlen) and returns 0. On a syntax
 * error, writes a human-readable message into err (capped at errlen) and
 * returns nonzero. If the JSON does not fit in outlen it is truncated and a
 * nonzero value is returned. */
int relql_parse(const char* query, char* out, size_t outlen, char* err,
              size_t errlen);

/* Parse AND validate against a schema, then infer the task type — the semantic
 * pass each binding used to reimplement. `schema_json` is the document
 * relativedb.schema.Schema.to_json_dict emits. On success writes
 *   {"query": <bound AST>, "task_type": "..."}
 * into out and returns 0. The AST is *bound*: the population's primary key is
 * resolved, so callers must use it rather than their own parse.
 *
 * Returns 1 with a message in err for a syntax error, an invalid query, or a
 * malformed schema; 2 when out is too small. Distinguishing the three is the
 * binding's job — it knows which exception type its users expect — so the
 * message is prefixed "syntax: ", "invalid: " or "schema: ". */
int relql_analyze(const char* query, const char* schema_json, char* out,
                  size_t outlen, char* err, size_t errlen);

#ifdef __cplusplus
}
#endif
#endif /* RelQL_C_H */
