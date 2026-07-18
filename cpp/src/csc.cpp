#include "csc.hpp"

#include <algorithm>
#include <numeric>

namespace csc {

CscAdjacency::CscAdjacency(std::int64_t n_parents, std::int64_t n_edges,
                           const std::int64_t* edge_parent,
                           const std::int64_t* edge_child,
                           const double* edge_ts)
    : n_parents_(n_parents < 0 ? 0 : n_parents) {
  // Keep only edges whose parent resolves to a dense id in range; the
  // reference drops dangling FKs the same way.
  std::vector<std::int64_t> order;
  order.reserve(n_edges > 0 ? static_cast<std::size_t>(n_edges) : 0);
  for (std::int64_t i = 0; i < n_edges; ++i) {
    std::int64_t p = edge_parent[i];
    if (p >= 0 && p < n_parents_) order.push_back(i);
  }
  // Stable sort by (parent, ts asc). std::stable_sort keeps original edge
  // order among equal (parent, ts) — matches numpy lexsort's stability, so
  // ties break identically to the Python reference.
  std::stable_sort(order.begin(), order.end(),
                   [&](std::int64_t a, std::int64_t b) {
                     if (edge_parent[a] != edge_parent[b])
                       return edge_parent[a] < edge_parent[b];
                     return edge_ts[a] < edge_ts[b];
                   });

  const std::size_t m = order.size();
  child_.resize(m);
  ts_.resize(m);
  colptr_.assign(static_cast<std::size_t>(n_parents_) + 1, 0);
  for (std::size_t k = 0; k < m; ++k) {
    std::int64_t e = order[k];
    child_[k] = edge_child[e];
    ts_[k] = edge_ts[e];
    ++colptr_[static_cast<std::size_t>(edge_parent[e]) + 1];
  }
  // Cumulative counts -> colptr.
  for (std::size_t p = 0; p < static_cast<std::size_t>(n_parents_); ++p)
    colptr_[p + 1] += colptr_[p];
}

std::int32_t CscAdjacency::children(std::int64_t parent_dense, double anchor_ts,
                                    std::int32_t limit,
                                    std::int64_t* out_child) const {
  if (limit <= 0) return 0;
  if (parent_dense < 0 || parent_dense >= n_parents_) return 0;
  std::int64_t s = colptr_[static_cast<std::size_t>(parent_dense)];
  std::int64_t e = colptr_[static_cast<std::size_t>(parent_dense) + 1];
  // ts within [s,e) is ascending: hi = first index > anchor (searchsorted
  // side="right"). Admitted children are [s, s+cnt).
  auto begin = ts_.begin() + s;
  auto end = ts_.begin() + e;
  std::int64_t cnt =
      std::upper_bound(begin, end, anchor_ts) - begin;  // # with ts <= anchor
  // Take the last `limit` of those, reversed to newest-first.
  std::int64_t take = cnt < static_cast<std::int64_t>(limit)
                          ? cnt
                          : static_cast<std::int64_t>(limit);
  for (std::int64_t k = 0; k < take; ++k)
    out_child[k] = child_[s + cnt - 1 - k];  // last `take` admitted, newest-first
  return static_cast<std::int32_t>(take);
}

}  // namespace csc
