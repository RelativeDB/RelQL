/* test_csc.cpp — framework-free conformance test for the CSC adjacency.
 *
 * Builds randomized graphs with a fixed-seed LCG (no time-based seeding) and
 * checks csc::CscAdjacency::children against a brute-force reference for many
 * (parent, anchor, limit) triples. The reference filters a parent's edge
 * bucket by ts <= anchor, sorts by ts ascending, then takes the last `limit`
 * reversed to newest-first — the exact semantics of python/.../csc.py.
 * Prints "PASS: N/N" and exits nonzero on any mismatch.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "csc.hpp"

namespace {

// Deterministic 64-bit linear-congruential generator (Numerical Recipes
// constants). Fixed seed -> reproducible graphs, no time-based seeding.
struct Lcg {
  std::uint64_t s;
  explicit Lcg(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s;
  }
  std::int64_t range(std::int64_t lo, std::int64_t hi) {  // [lo, hi]
    if (hi <= lo) return lo;
    std::uint64_t span = static_cast<std::uint64_t>(hi - lo + 1);
    return lo + static_cast<std::int64_t>(next() % span);
  }
};

// Brute-force reference over the raw edge arrays for one parent.
std::vector<std::int64_t> ref_children(
    std::int64_t parent, double anchor, std::int32_t limit,
    const std::vector<std::int64_t>& ep, const std::vector<std::int64_t>& ec,
    const std::vector<double>& et) {
  if (limit <= 0) return {};
  // Collect this parent's edges preserving original order (stable), so ties in
  // ts break by insertion order — same as stable_sort in the index.
  std::vector<std::size_t> bucket;
  for (std::size_t i = 0; i < ep.size(); ++i)
    if (ep[i] == parent) bucket.push_back(i);
  // Stable sort by ts ascending.
  std::stable_sort(bucket.begin(), bucket.end(),
                   [&](std::size_t a, std::size_t b) { return et[a] < et[b]; });
  // Keep ts <= anchor.
  std::vector<std::size_t> admitted;
  for (std::size_t i : bucket)
    if (et[i] <= anchor) admitted.push_back(i);
  // Last `limit`, reversed to newest-first.
  std::vector<std::int64_t> out;
  std::size_t n = admitted.size();
  std::size_t take =
      n < static_cast<std::size_t>(limit) ? n : static_cast<std::size_t>(limit);
  for (std::size_t k = 0; k < take; ++k)
    out.push_back(ec[admitted[n - 1 - k]]);
  return out;
}

bool run_case(std::uint64_t seed, std::int64_t n_parents, std::int64_t n_edges,
              int n_queries, std::int64_t& checks, std::int64_t& fails) {
  Lcg rng(seed);
  std::vector<std::int64_t> ep, ec;
  std::vector<double> et;
  ep.reserve(n_edges);
  ec.reserve(n_edges);
  et.reserve(n_edges);
  // Distinct timestamps mostly, but force some ties and some static (-inf).
  for (std::int64_t i = 0; i < n_edges; ++i) {
    ep.push_back(rng.range(0, n_parents - 1));
    ec.push_back(rng.range(0, 100000));
    std::int64_t roll = rng.range(0, 9);
    if (roll == 0) {
      et.push_back(-std::numeric_limits<double>::infinity());  // static row
    } else if (roll <= 3) {
      et.push_back(static_cast<double>(rng.range(0, 5)));  // heavy ties
    } else {
      et.push_back(static_cast<double>(rng.range(0, 1000)));
    }
  }

  csc::CscAdjacency adj(n_parents, n_edges, ep.data(), ec.data(), et.data());

  std::vector<std::int64_t> out(64);
  for (int q = 0; q < n_queries; ++q) {
    // Include out-of-range parents (-2, n_parents) and various limits/anchors.
    std::int64_t parent = rng.range(-2, n_parents);
    std::int32_t limit =
        static_cast<std::int32_t>(rng.range(-1, 8));  // includes 0 and >bucket
    // anchor spans before-all (-inf), exact tie values, and after-all (+inf).
    double anchor;
    std::int64_t aroll = rng.range(0, 12);
    if (aroll == 0)
      anchor = -std::numeric_limits<double>::infinity();
    else if (aroll == 1)
      anchor = std::numeric_limits<double>::infinity();
    else if (aroll <= 5)
      anchor = static_cast<double>(rng.range(0, 5));  // land on tie values
    else
      anchor = static_cast<double>(rng.range(-5, 1005));

    if (limit > static_cast<std::int32_t>(out.size()))
      out.resize(static_cast<std::size_t>(limit));

    std::int32_t n = adj.children(parent, anchor, limit, out.data());
    std::vector<std::int64_t> got(out.begin(), out.begin() + (n < 0 ? 0 : n));
    std::vector<std::int64_t> want = ref_children(parent, anchor, limit, ep, ec, et);

    ++checks;
    if (got != want) {
      ++fails;
      if (fails <= 10) {
        std::printf(
            "MISMATCH seed=%llu parent=%lld anchor=%g limit=%d: got[%zu] want[%zu]\n",
            (unsigned long long)seed, (long long)parent, anchor, limit,
            got.size(), want.size());
      }
    }
  }
  return fails == 0;
}

}  // namespace

int main() {
  std::int64_t checks = 0, fails = 0;

  // Edge cases: parent with no edges is covered by out-of-range/empty buckets;
  // an empty graph exercises the n_edges==0 path.
  {
    csc::CscAdjacency empty(5, 0, nullptr, nullptr, nullptr);
    std::int64_t out[4];
    std::int32_t n = empty.children(2, 1e9, 4, out);
    ++checks;
    if (n != 0) { ++fails; std::printf("MISMATCH empty-graph n=%d\n", n); }
  }
  // limit 0 explicitly.
  {
    std::vector<std::int64_t> ep{0, 0}, ec{7, 8};
    std::vector<double> et{1.0, 2.0};
    csc::CscAdjacency a(1, 2, ep.data(), ec.data(), et.data());
    std::int64_t out[4];
    std::int32_t n = a.children(0, 100.0, 0, out);
    ++checks;
    if (n != 0) { ++fails; std::printf("MISMATCH limit0 n=%d\n", n); }
  }

  // Randomized batteries with distinct fixed seeds and shapes.
  struct Cfg { std::uint64_t seed; std::int64_t np; std::int64_t ne; int q; };
  const Cfg cfgs[] = {
      {0x1234ULL, 1, 20, 500},
      {0xABCDULL, 8, 200, 4000},
      {0xF00DULL, 50, 2000, 8000},
      {0xBEEFULL, 200, 50, 4000},   // sparse: many parents with no edges
      {0x5A5AULL, 4, 5000, 6000},   // dense: heavy ties per parent
  };
  for (const Cfg& c : cfgs) run_case(c.seed, c.np, c.ne, c.q, checks, fails);

  std::printf("PASS: %lld/%lld\n", (long long)(checks - fails), (long long)checks);
  return fails == 0 ? 0 : 1;
}
