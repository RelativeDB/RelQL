#include "graph_c.h"

#include <cstring>
#include <exception>
#include <string>

#include "graph.hpp"

namespace {
void set_err(char* dst, size_t dstlen, const std::string& msg) {
  if (!dst || dstlen == 0) return;
  size_t n = msg.size();
  if (n > dstlen - 1) n = dstlen - 1;
  std::memcpy(dst, msg.data(), n);
  dst[n] = '\0';
}
}  // namespace

struct rel_graph {
  relgraph::Graph g;
};

extern "C" {

rel_graph* rel_graph_build(int64_t n_nodes, const double* node_ts,
                           const int32_t* node_cells, const int32_t* node_table,
                           const uint8_t* node_is_task, int64_t n_edges,
                           const int64_t* edge_parent, const int64_t* edge_child,
                           char* err, size_t errlen) {
  try {
    if (n_nodes <= 0) throw std::runtime_error("n_nodes must be positive");
    if (!node_ts || !node_cells || !node_table || !node_is_task)
      throw std::runtime_error("null node array");
    if (n_edges < 0) throw std::runtime_error("n_edges cannot be negative");
    if (n_edges && (!edge_parent || !edge_child))
      throw std::runtime_error("null edge array");
    for (int64_t e = 0; e < n_edges; ++e) {
      if (edge_parent[e] < 0 || edge_parent[e] >= n_nodes ||
          edge_child[e] < 0 || edge_child[e] >= n_nodes)
        throw std::runtime_error("edge endpoint out of range");
    }
    return new rel_graph{relgraph::Graph(n_nodes, node_ts, node_cells,
                                         node_table, node_is_task, n_edges,
                                         edge_parent, edge_child)};
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return nullptr;
  } catch (...) {
    set_err(err, errlen, "unknown error");
    return nullptr;
  }
}

void rel_graph_free(rel_graph* g) { delete g; }

int rel_graph_assemble(const rel_graph* handle, int64_t target, double cutoff_ts,
                       const uint8_t* eligible, int32_t max_context_cells,
                       int32_t local_context_cells, int32_t bfs_width,
                       int32_t num_walks, int32_t walk_length, uint64_t seed,
                       uint8_t prefer_latest, int64_t fallback_base,
                       int64_t fallback_n, int64_t* out_nodes,
                       uint8_t* out_focal, int32_t max_nodes,
                       int32_t* out_count, char* err, size_t errlen) {
  try {
    if (!handle) throw std::runtime_error("null graph");
    if (!out_nodes || !out_focal || !out_count)
      throw std::runtime_error("null output buffer");
    if (max_nodes <= 0) throw std::runtime_error("max_nodes must be positive");
    relgraph::Policy p;
    p.max_context_cells = max_context_cells;
    p.local_context_cells = local_context_cells;
    p.bfs_width = bfs_width;
    p.num_walks = num_walks;
    p.walk_length = walk_length;
    p.seed = seed;
    p.prefer_latest = prefer_latest != 0;
    const int32_t n = handle->g.assemble(target, cutoff_ts, eligible, p,
                                         fallback_base, fallback_n, out_nodes,
                                         out_focal, max_nodes);
    if (n < 0) throw std::runtime_error("target node out of range");
    *out_count = n;
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return 1;
  } catch (...) {
    set_err(err, errlen, "unknown error");
    return 1;
  }
}

}  // extern "C"
