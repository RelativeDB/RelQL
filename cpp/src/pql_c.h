/* pql_c.h — C ABI for the shared relativedb PQL parser.
 *
 * One parser for all three language bindings (Python/Java/Rust). Wraps the
 * C++20 recursive-descent parser in pql.cpp. Follows the rt_c.h convention:
 * opaque errors via (char* err, size_t errlen), nonzero return on failure.
 */
#ifndef PQL_C_H
#define PQL_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Parse a PQL query. On success writes a JSON AST (see pql.hpp / the schema
 * doc) into out (NUL-terminated, capped at outlen) and returns 0. On a syntax
 * error, writes a human-readable message into err (capped at errlen) and
 * returns nonzero. If the JSON does not fit in outlen it is truncated and a
 * nonzero value is returned. */
int pql_parse(const char* query, char* out, size_t outlen, char* err,
              size_t errlen);

#ifdef __cplusplus
}
#endif
#endif /* PQL_C_H */
