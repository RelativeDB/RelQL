/* csc.hpp — dependency-light C++20 CSC adjacency for the time-bounded
 * "latest <= anchor" children query.
 *
 * Faithful to python/src/relativedb/csc.py: edges are sorted by
 * (parent_dense, ts ascending); a colptr of length n_parents+1 gives each
 * parent's contiguous child bucket; children(parent, anchor, limit) returns
 * the last `limit` children whose ts <= anchor, reversed to newest-first.
 * Static rows use ts = -inf so they sort first and are admitted under every
 * bound. Only the array algorithm lives here — id<->dense mapping and row
 * storage stay in the language bindings. No third-party dependencies.
 */
#ifndef RELATIVEDB_CSC_HPP
#define RELATIVEDB_CSC_HPP

#include <cstdint>
#include <vector>

namespace csc {

// CSC/CSR adjacency over dense parent/child ids. `colptr` has length
// n_parents+1; child[colptr[p] : colptr[p+1]] are that parent's child dense
// ids sorted by ts ascending, with ts[] the matching timestamps.
class CscAdjacency {
 public:
  // Build from edge arrays (order arbitrary; sorted internally by
  // (parent, ts asc)). Edges with parent outside [0, n_parents) are dropped,
  // mirroring the reference which only keeps resolvable parents.
  CscAdjacency(std::int64_t n_parents, std::int64_t n_edges,
               const std::int64_t* edge_parent, const std::int64_t* edge_child,
               const double* edge_ts);

  // Latest <= anchor_ts children of parent_dense, newest-first. Writes up to
  // `limit` dense child ids into out_child (caller-allocated, length >=
  // max(limit,0)); returns how many were written. limit <= 0 or an
  // out-of-range parent yields 0.
  std::int32_t children(std::int64_t parent_dense, double anchor_ts,
                        std::int32_t limit, std::int64_t* out_child) const;

  std::int64_t n_parents() const { return n_parents_; }
  std::int64_t n_edges() const { return static_cast<std::int64_t>(child_.size()); }

 private:
  std::int64_t n_parents_;
  std::vector<std::int64_t> colptr_;  // length n_parents_ + 1
  std::vector<std::int64_t> child_;   // length n_edges (kept)
  std::vector<double> ts_;            // length n_edges (kept), ts asc per bucket
};

}  // namespace csc

#endif  // RELATIVEDB_CSC_HPP
