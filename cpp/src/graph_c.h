/* graph_c.h — C ABI for array-backed context assembly.
 *
 * Build the graph once from flat arrays, then assemble many contexts from it.
 * Each assemble returns the ORDERED node ids the context emits, so the caller
 * materializes only those rows rather than the whole database.
 *
 * Node numbering is the caller's: these ids index the arrays it passed in, and
 * they seed the walk and BFS streams, so the caller's ordering decides the
 * sampling. Follows the rt_c.h convention: opaque handle, nonzero on failure.
 */
#ifndef RELATIVEDB_GRAPH_C_H
#define RELATIVEDB_GRAPH_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rel_graph rel_graph;

/* node_ts is epoch seconds with NaN for a timeless row; node_cells is the per
 * node cell cost the budget counts; node_table indexes the caller's table
 * order; node_is_task marks derived task rows. Edges are (parent, child) node
 * id pairs. Returns NULL on failure with a message in err. */
rel_graph* rel_graph_build(int64_t n_nodes, const double* node_ts,
                           const int32_t* node_cells, const int32_t* node_table,
                           const uint8_t* node_is_task, int64_t n_edges,
                           const int64_t* edge_parent, const int64_t* edge_child,
                           char* err, size_t errlen);

void rel_graph_free(rel_graph*);

/* Assemble one context around `target`. `eligible` (length n_nodes, 1 = the
 * walk may land here) turns on the peer-ranking walk; pass NULL for the
 * target's neighbourhood alone. `cutoff_ts` bounds which rows are visible.
 * Writes emitted ids into out_nodes and a 1/0 focal flag into out_focal, both
 * caller-allocated with room for max_nodes, and sets *out_count. Returns 0 on
 * success. */
int rel_graph_assemble(const rel_graph*, int64_t target, double cutoff_ts,
                       const uint8_t* eligible, int32_t max_context_cells,
                       int32_t local_context_cells, int32_t bfs_width,
                       int32_t num_walks, int32_t walk_length, uint64_t seed,
                       uint8_t prefer_latest, int64_t fallback_base,
                       int64_t fallback_n, int64_t* out_nodes,
                       uint8_t* out_focal, int32_t max_nodes,
                       int32_t* out_count, char* err, size_t errlen);

#ifdef __cplusplus
}
#endif
#endif /* RELATIVEDB_GRAPH_C_H */
