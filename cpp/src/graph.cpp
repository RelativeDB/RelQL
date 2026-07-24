/* graph.cpp — array-backed context assembly.
 *
 * The BFS emission is ported from the traversal that used to run in the
 * bindings. Its observable output depends on the exact order the RNG is drawn
 * in -- one extra or missing draw shifts every subsequent choice -- so the
 * structure here follows the original statement for statement rather than
 * being rewritten into something tidier.
 */
#include "graph.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "stdrng.hpp"

namespace relgraph {

namespace {

constexpr std::uint64_t kU64 = ~0ULL;

// rand::seq::index::sample for the u32-sized cases contexts use. Only the
// in-place (partial Fisher-Yates) and Floyd branches are reachable at these
// sizes; the rejection path is for samples far larger than a context.
std::vector<std::int32_t> rand_sample(relrng::StdRng091& rng,
                                      std::int32_t length,
                                      std::int32_t amount) {
  std::vector<std::int32_t> out;
  if (amount <= 0 || amount > length) return out;
  bool use_inplace;
  const int j = length >= 500000 ? 1 : 0;
  if (amount < 163) {
    const double a[2] = {10.0, 70.0 / 9.0};
    const double b[2] = {1.6, 8.0 / 45.0};
    use_inplace = amount > 11 &&
                  (double)length < (a[j] + b[j] * amount) * amount;
  } else {
    const double c[2] = {270.0, 330.0 / 9.0};
    use_inplace = (double)length < c[j] * amount;
  }
  if (use_inplace) {
    std::vector<std::int32_t> idx(length);
    for (std::int32_t i = 0; i < length; ++i) idx[i] = i;
    for (std::int32_t i = 0; i < amount; ++i) {
      // range(length, i): a draw over [i, length)
      const std::int32_t k =
          i + (std::int32_t)rng.range((std::uint32_t)(length - i));
      std::swap(idx[i], idx[k]);
    }
    out.assign(idx.begin(), idx.begin() + amount);
    return out;
  }
  // Floyd, in rand's variant -- NOT the textbook form. On a collision it
  // overwrites the EXISTING occurrence with j and still appends t, where the
  // textbook version substitutes j for t. The two draw the same numbers and
  // return different sequences, which is a silent divergence in every context
  // the fallback stage pads.
  out.reserve(amount);
  for (std::int32_t j = length - amount; j < length; ++j) {
    const std::int32_t t = (std::int32_t)rng.range((std::uint32_t)(j + 1));
    auto it = std::find(out.begin(), out.end(), t);
    if (it != out.end()) *it = j;
    out.push_back(t);
  }
  return out;
}

}  // namespace

Graph::Graph(std::int64_t n_nodes, const double* node_ts,
             const std::int32_t* node_cells, const std::int32_t* node_table,
             const std::uint8_t* node_is_task, std::int64_t n_edges,
             const std::int64_t* edge_parent, const std::int64_t* edge_child)
    : n_nodes_(n_nodes),
      ts_(node_ts, node_ts + n_nodes),
      cells_(node_cells, node_cells + n_nodes),
      table_(node_table, node_table + n_nodes),
      is_task_(node_is_task, node_is_task + n_nodes) {
  // CSR both ways. Children of one parent are ordered by (has_ts, ts, node),
  // matching the stable order the row path produces.
  std::vector<std::int64_t> fcount(n_nodes + 1, 0), pcount(n_nodes + 1, 0);
  for (std::int64_t e = 0; e < n_edges; ++e) {
    ++fcount[edge_child[e]];
    ++pcount[edge_parent[e]];
  }
  f2p_off_.assign(n_nodes + 1, 0);
  p2f_off_.assign(n_nodes + 1, 0);
  for (std::int64_t i = 0; i < n_nodes; ++i) {
    f2p_off_[i + 1] = f2p_off_[i] + fcount[i];
    p2f_off_[i + 1] = p2f_off_[i] + pcount[i];
  }
  f2p_.resize(n_edges);
  p2f_.resize(n_edges);
  std::vector<std::int64_t> fat(f2p_off_.begin(), f2p_off_.end() - 1);
  std::vector<std::int64_t> pat(p2f_off_.begin(), p2f_off_.end() - 1);
  for (std::int64_t e = 0; e < n_edges; ++e) {
    f2p_[fat[edge_child[e]]++] = edge_parent[e];
    p2f_[pat[edge_parent[e]]++] = edge_child[e];
  }
  for (std::int64_t c = 0; c < n_nodes; ++c)
    std::sort(f2p_.begin() + f2p_off_[c], f2p_.begin() + f2p_off_[c + 1]);
  for (std::int64_t p = 0; p < n_nodes; ++p) {
    auto lo = p2f_.begin() + p2f_off_[p], hi = p2f_.begin() + p2f_off_[p + 1];
    std::stable_sort(lo, hi, [&](std::int64_t a, std::int64_t b) {
      const bool ha = !std::isnan(ts_[a]), hb = !std::isnan(ts_[b]);
      if (ha != hb) return hb;            // timeless first
      const double ta = ha ? ts_[a] : -std::numeric_limits<double>::infinity();
      const double tb = hb ? ts_[b] : -std::numeric_limits<double>::infinity();
      if (ta != tb) return ta < tb;
      return a < b;
    });
  }
}

std::int32_t Graph::assemble(std::int64_t target, double cutoff_ts,
                             const std::uint8_t* eligible,
                             const Policy& policy, std::int64_t fallback_base,
                             std::int64_t fallback_n, std::int64_t* out_nodes,
                             std::uint8_t* out_focal,
                             std::int32_t max_nodes) const {
  if (target < 0 || target >= n_nodes_) return -1;

  const std::uint64_t context_seed = relrng::StdRng091(policy.seed).u64();
  const std::uint64_t step_seed = relrng::StdRng091(context_seed).u64();
  relrng::StdRng091 bfs_rng(
      (step_seed + (std::uint64_t)target + 0xB0B0B0B0B0B0B0B0ULL) & kU64);

  std::unordered_map<std::int64_t, std::int32_t> visited_depth;
  std::unordered_set<std::int64_t> emitted;
  std::vector<std::int64_t> ordered;
  std::unordered_set<std::int64_t> focal;
  std::int32_t cells = 1;
  bool full = false;

  // Children of `node` admitted by the seed's cut-off.
  auto kids_of = [&](std::int64_t node, double seed_cut) {
    std::vector<std::int64_t> out;
    for (std::int64_t i = p2f_off_[node]; i < p2f_off_[node + 1]; ++i) {
      const std::int64_t c = p2f_[i];
      const double t = ts_[c];
      if (std::isnan(t) || t <= seed_cut) out.push_back(c);
    }
    return out;
  };

  auto extend = [&](std::int64_t seed_node, bool is_focal) {
    std::int32_t local_cells = 0;
    std::vector<std::pair<std::int32_t, std::int64_t>> f2p_stack;
    std::vector<std::vector<std::int64_t>> levels{{seed_node}};
    double seed_cut = ts_[seed_node];
    if (std::isnan(seed_cut)) seed_cut = std::numeric_limits<double>::infinity();
    const std::int32_t seed_table = table_[seed_node];
    for (;;) {
      std::int32_t depth;
      std::int64_t node;
      if (!f2p_stack.empty()) {
        depth = f2p_stack.back().first;
        node = f2p_stack.back().second;
        f2p_stack.pop_back();
      } else {
        depth = -1;
        for (std::size_t i = 0; i < levels.size(); ++i)
          if (!levels[i].empty()) { depth = (std::int32_t)i; break; }
        if (depth < 0) return;
        auto& level = levels[depth];
        const std::uint32_t sel = bfs_rng.range((std::uint32_t)level.size());
        std::swap(level[sel], level.back());
        node = level.back();
        level.pop_back();
      }
      auto prev = visited_depth.find(node);
      if (prev != visited_depth.end() && prev->second <= depth) continue;
      const std::int32_t cost = cells_[node];
      local_cells += cost;
      if (local_cells >= policy.local_context_cells) return;
      visited_depth[node] = depth;
      if (!emitted.count(node)) {
        std::int32_t emitted_cost = cost;
        if (node == target) emitted_cost = std::max(0, emitted_cost - 1);
        if (cells >= policy.max_context_cells) { full = true; return; }
        emitted.insert(node);
        ordered.push_back(node);
        cells += emitted_cost;
        if (cells >= policy.max_context_cells) full = true;
        if (is_focal) focal.insert(node);
      }
      for (std::int64_t i = f2p_off_[node]; i < f2p_off_[node + 1]; ++i)
        f2p_stack.push_back({depth + 1, f2p_[i]});
      std::vector<std::int64_t> valid = kids_of(node, seed_cut);
      // Task rows of the seed's own table are always kept; database children
      // are subject to the fanout cap.
      std::vector<std::int64_t> task_kids, db_kids;
      for (std::int64_t k : valid) {
        if (is_task_[k]) {
          if (table_[k] == seed_table) task_kids.push_back(k);
        } else {
          db_kids.push_back(k);
        }
      }
      if ((std::int32_t)db_kids.size() > policy.bfs_width) {
        auto sel = rand_sample(bfs_rng, (std::int32_t)db_kids.size(),
                               policy.bfs_width);
        std::vector<std::int64_t> kept;
        kept.reserve(sel.size());
        for (std::int32_t i : sel) kept.push_back(db_kids[i]);
        db_kids.swap(kept);
      }
      while ((std::int32_t)levels.size() <= depth + 1) levels.push_back({});
      auto& next = levels[depth + 1];
      next.insert(next.end(), task_kids.begin(), task_kids.end());
      next.insert(next.end(), db_kids.begin(), db_kids.end());
    }
  };

  // ---- tier 1: rank peers by a random walk from the target ---------------
  // The walk is over the bidirectional neighbourhood with children cut at the
  // anchor; parents are always admitted. Ranking is by visit count, then
  // recency when prefer_latest, with a per-node RNG draw breaking ties -- the
  // same three keys, in the same order, as the traversal this replaces.
  std::vector<std::int64_t> tier1;
  std::unordered_set<std::int64_t> visits;
  if (eligible) {
    std::vector<std::int32_t> off(n_nodes_ + 1, 0);
    std::vector<std::int32_t> flat;
    flat.reserve(f2p_.size() + p2f_.size());
    for (std::int64_t n = 0; n < n_nodes_; ++n) {
      off[n] = (std::int32_t)flat.size();
      for (std::int64_t i = f2p_off_[n]; i < f2p_off_[n + 1]; ++i)
        flat.push_back((std::int32_t)f2p_[i]);           // parents: no bound
      for (std::int64_t i = p2f_off_[n]; i < p2f_off_[n + 1]; ++i) {
        const std::int64_t c = p2f_[i];
        if (std::isnan(ts_[c]) || ts_[c] <= cutoff_ts)
          flat.push_back((std::int32_t)c);
      }
    }
    off[n_nodes_] = (std::int32_t)flat.size();

    std::vector<std::uint8_t> walk_eligible(eligible, eligible + n_nodes_);
    walk_eligible[target] = 0;
    std::vector<std::uint32_t> counts(n_nodes_, 0);
    relrng::StdRng091 walk_rng(
        (step_seed + (std::uint64_t)target + 0xD0D0D0D0D0D0D0D0ULL) & kU64);
    for (std::int32_t w = 0; w < policy.num_walks; ++w) {
      std::int64_t current = target;
      for (std::int32_t step = 0; step < policy.walk_length; ++step) {
        if (walk_eligible[current]) ++counts[current];
        const std::int32_t begin = off[current], end = off[current + 1];
        if (begin == end) break;
        current = flat[begin + walk_rng.range((std::uint32_t)(end - begin))];
      }
    }
    for (std::int64_t n = 0; n < n_nodes_; ++n)
      if (counts[n]) { tier1.push_back(n); visits.insert(n); }
    std::vector<std::uint64_t> tie(tier1.size());
    for (std::size_t i = 0; i < tier1.size(); ++i)
      tie[i] = relrng::StdRng091(
          (step_seed + (std::uint64_t)tier1[i]) & kU64).u64();
    std::vector<std::size_t> idx(tier1.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    const bool latest = policy.prefer_latest;
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
      if (latest) {
        // Recency is the PRIMARY key when prefer_latest: numpy's lexsort
        // treats its last argument as primary, so ordering counts first was
        // a different ranking, not a cosmetic difference.
        const double ta = std::isnan(ts_[tier1[a]])
            ? -std::numeric_limits<double>::infinity() : ts_[tier1[a]];
        const double tb = std::isnan(ts_[tier1[b]])
            ? -std::numeric_limits<double>::infinity() : ts_[tier1[b]];
        if (ta != tb) return ta > tb;
      }
      if (counts[tier1[a]] != counts[tier1[b]])
        return counts[tier1[a]] > counts[tier1[b]];
      return tie[a] < tie[b];
    });
    std::vector<std::int64_t> ranked;
    ranked.reserve(idx.size());
    for (std::size_t i : idx) ranked.push_back(tier1[i]);
    tier1.swap(ranked);
  }

  extend(target, true);
  for (std::int64_t node : tier1) {
    if (full) break;
    extend(node, false);
  }

  // ---- fallback: pad a short context from the task table ------------------
  if (!full && fallback_n > 0) {
    relrng::StdRng091 fallback_rng(
        (step_seed + (std::uint64_t)target + 0xA5A5A5A5A5A5A5A5ULL) & kU64);
    const std::int32_t amount = (std::int32_t)std::min<std::int64_t>(
        std::max(policy.max_context_cells - cells, 0), fallback_n);
    auto sel = rand_sample(fallback_rng, (std::int32_t)fallback_n, amount);
    for (std::int32_t pos : sel) {
      if (full) break;
      const std::int64_t node = fallback_base + pos;
      if (node == target || visits.count(node)) continue;
      if (!(std::isnan(ts_[node]) || ts_[node] <= cutoff_ts)) continue;
      if (eligible && !eligible[node]) continue;
      extend(node, false);
    }
  }

  const std::int32_t n =
      (std::int32_t)std::min<std::size_t>(ordered.size(), (std::size_t)max_nodes);
  for (std::int32_t i = 0; i < n; ++i) {
    out_nodes[i] = ordered[i];
    out_focal[i] = focal.count(ordered[i]) ? 1 : 0;
  }
  return n;
}

}  // namespace relgraph
