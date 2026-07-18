/* csc_c.h — C ABI for the shared relativedb CSC adjacency index.
 *
 * One implementation of the time-bounded "latest <= anchor" children query
 * for all three language bindings (Python/Java/Rust). Wraps the C++20
 * CscAdjacency in csc.cpp. Follows the rt_c.h / pql_c.h convention: opaque
 * handle, (char* err, size_t errlen), nonzero return on failure.
 *
 * The index owns only the array algorithm — the CSC/CSR adjacency and the
 * binary-searched children query. Callers keep their own id<->dense mapping
 * and row storage; ids here are dense integers, ts is epoch seconds (use
 * -inf, i.e. -HUGE_VAL, for static rows so they sort first).
 */
#ifndef CSC_C_H
#define CSC_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct csc_index csc_index;

/* Build from edge arrays (edge order arbitrary; the index sorts internally by
 * (parent, ts ascending)). edge_parent/edge_child/edge_ts each have length
 * n_edges; parent/child are dense ids, ts is epoch seconds (-inf for static).
 * Edges whose parent is outside [0, n_parents) are dropped. Returns NULL on
 * error (message in err, capped at errlen). */
csc_index* csc_build(int64_t n_parents, int64_t n_edges,
                     const int64_t* edge_parent, const int64_t* edge_child,
                     const double* edge_ts, char* err, size_t errlen);

void csc_free(csc_index*);

/* Latest <= anchor_ts children of parent_dense, newest-first. Writes up to
 * `limit` ids into out_child (caller-allocated, length >= max(limit,0)); sets
 * *out_n to how many were written. limit <= 0 or an out-of-range parent yields
 * *out_n = 0. Returns 0 on success, nonzero on error (message in err). */
int csc_children(const csc_index*, int64_t parent_dense, double anchor_ts,
                 int32_t limit, int64_t* out_child, int32_t* out_n,
                 char* err, size_t errlen);

#ifdef __cplusplus
}
#endif
#endif /* CSC_C_H */
