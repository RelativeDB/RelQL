/* graph.hpp — array-backed context assembly.
 *
 * The engine's only real demand on storage is CONTEXT POPULATION: hand back
 * the few thousand rows that belong in each prediction's context. Doing that
 * from a binding means materializing every database row as a language object
 * up front, which caps out in the low millions and made rel-event take minutes
 * per task where an array-backed path took seconds.
 *
 * This holds the graph as flat arrays -- CSR adjacency, timestamps and cell
 * counts as vectors -- and returns the ORDERED node ids a context emits. The
 * binding materializes only those, so cost scales with the context (thousands)
 * rather than the database (millions).
 *
 * Node numbering is supplied by the caller, not invented here. That is the
 * whole difference from the columnar path this replaces: that one numbered
 * nodes its own way, and since node indices seed the walk/BFS RNG it produced
 * different contexts from the same data and seed. A caller that numbers nodes
 * in reference order gets reference-order sampling.
 */
#ifndef RELATIVEDB_GRAPH_HPP
#define RELATIVEDB_GRAPH_HPP

#include <cstdint>
#include <vector>

namespace relgraph {

// Context assembly geometry. Mirrors relativedb.engine.ContextPolicy.
struct Policy {
  std::int32_t max_context_cells = 2048;
  std::int32_t local_context_cells = 256;
  std::int32_t bfs_width = 32;
  std::int32_t num_walks = 10000;
  std::int32_t walk_length = 20;
  std::uint64_t seed = 0;
  bool prefer_latest = true;
};

// The graph, as arrays. Built once per snapshot and queried many times.
class Graph {
 public:
  // `node_ts` is epoch seconds, NaN for a static (timeless) row. `node_cells`
  // is the per-node cell cost the budget counts. `node_table` indexes the
  // caller's table ordering. Edges are (parent, child) node id pairs; both
  // directions are derived here.
  // `node_is_task` marks derived task rows (1) from database rows (0): a
  // seed keeps every task row of its own table but samples database children
  // under the fanout cap, so the two cannot be told apart by table index
  // alone.
  Graph(std::int64_t n_nodes, const double* node_ts,
        const std::int32_t* node_cells, const std::int32_t* node_table,
        const std::uint8_t* node_is_task, std::int64_t n_edges,
        const std::int64_t* edge_parent, const std::int64_t* edge_child);

  std::int64_t n_nodes() const { return n_nodes_; }
  std::int64_t n_edges() const { return (std::int64_t)p2f_.size(); }

  // Copy out the adjacency this class built and ordered. Bindings need it for
  // their own retriever surfaces, and exporting it is what keeps the ordering
  // single-sourced: a binding that rebuilt the same CSR would be a second
  // implementation of the rule that decides which rows a context sees.
  // `out_offsets` has room for n_nodes+1, `out_values` for n_edges.
  void adjacency(bool children, std::int64_t* out_offsets,
                 std::int64_t* out_values) const;

  // Assemble one context. `target` is the focal node; `eligible` marks nodes
  // the walk may land on (length n_nodes, 1 = eligible); `cutoff_ts` bounds
  // which rows are visible. Writes emitted node ids in order into `out_nodes`
  // (caller-allocated, length >= max_nodes) and a 1/0 focal flag per emitted
  // node into `out_focal`. Returns how many were written, or -1 if the target
  // is out of range.
  // `fallback_base`/`fallback_n` name the task table's contiguous node range.
  // When the target and its walk tier leave the context short of the budget,
  // rows are sampled from that range to pad it -- the stage the reference
  // traversal ends with. Pass fallback_n = 0 to skip it.
  std::int32_t assemble(std::int64_t target, double cutoff_ts,
                        const std::uint8_t* eligible, const Policy& policy,
                        std::int64_t fallback_base, std::int64_t fallback_n,
                        std::int64_t* out_nodes, std::uint8_t* out_focal,
                        std::int32_t max_nodes) const;

 private:
  std::int64_t n_nodes_;
  std::vector<double> ts_;
  std::vector<std::int32_t> cells_;
  std::vector<std::int32_t> table_;
  std::vector<std::uint8_t> is_task_;
  // parents-of-node (f2p) and children-of-node (p2f), CSR.
  std::vector<std::int64_t> f2p_off_, f2p_;
  std::vector<std::int64_t> p2f_off_, p2f_;
};

}  // namespace relgraph

#endif  // RELATIVEDB_GRAPH_HPP
