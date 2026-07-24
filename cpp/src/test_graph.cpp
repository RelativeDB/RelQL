/* test_graph.cpp — array-backed context assembly.
 *
 * This exists so a binding can hand back the few thousand rows a context needs
 * without materializing the whole database first. What it must guarantee is
 * pinned here: the same graph and seed always give the same ordered context,
 * the budgets actually bind, and nothing past the cut-off is ever emitted.
 */
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "graph.hpp"

namespace {

int checks = 0, fails = 0;

void ok(bool cond, const std::string& what) {
  ++checks;
  if (!cond) {
    ++fails;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// A small star-of-stars: 4 "customers" (timeless), 40 "orders" hanging off
// them with spread timestamps, and 5 timeless "products" each order points at.
struct Fixture {
  std::int64_t n_nodes = 0;
  std::vector<double> ts;
  std::vector<std::int32_t> cells, table;
  std::vector<std::uint8_t> is_task;
  std::vector<std::int64_t> ep, ec;

  Fixture() {
    const int n_cust = 4, n_prod = 5, n_ord = 40;
    n_nodes = n_cust + n_ord + n_prod;
    ts.assign(n_nodes, kNaN);
    cells.assign(n_nodes, 2);
    table.assign(n_nodes, 0);
    is_task.assign(n_nodes, 0);
    for (int i = 0; i < n_ord; ++i) {
      const std::int64_t node = n_cust + i;
      ts[node] = 100.0 + i;          // distinct, increasing
      table[node] = 1;
    }
    for (int i = 0; i < n_prod; ++i) table[n_cust + n_ord + i] = 2;
    for (int i = 0; i < n_ord; ++i) {
      const std::int64_t order = n_cust + i;
      ep.push_back(i % n_cust);                          // customer -> order
      ec.push_back(order);
      ep.push_back(n_cust + n_ord + (i % n_prod));       // product  -> order
      ec.push_back(order);
    }
  }

  relgraph::Graph build() const {
    return relgraph::Graph(n_nodes, ts.data(), cells.data(), table.data(),
                           is_task.data(), (std::int64_t)ep.size(), ep.data(),
                           ec.data());
  }
};

relgraph::Policy policy() {
  relgraph::Policy p;
  p.max_context_cells = 64;
  p.local_context_cells = 32;
  p.bfs_width = 3;
  p.seed = 0;
  return p;
}

std::vector<std::int64_t> assemble(const relgraph::Graph& g, std::int64_t target,
                                   double cutoff, const relgraph::Policy& p,
                                   std::vector<std::uint8_t>* focal_out = nullptr) {
  std::vector<std::int64_t> nodes(4096);
  std::vector<std::uint8_t> focal(4096);
  const std::int32_t n =
      g.assemble(target, cutoff, nullptr, p, nodes.data(), focal.data(), 4096);
  if (n < 0) return {};
  nodes.resize(n);
  if (focal_out) { focal.resize(n); *focal_out = focal; }
  return nodes;
}

void test_determinism() {
  Fixture f;
  auto g = f.build();
  auto a = assemble(g, 0, 1e18, policy());
  auto b = assemble(g, 0, 1e18, policy());
  ok(!a.empty(), "a context is produced");
  ok(a == b, "same graph and seed give the same ordered context");
  // A different seed must actually change the sampling, or the seed is inert.
  auto p2 = policy();
  p2.seed = 7;
  ok(assemble(g, 0, 1e18, p2) != a, "a different seed changes the context");
}

void test_target_comes_first_and_is_focal() {
  Fixture f;
  auto g = f.build();
  std::vector<std::uint8_t> focal;
  auto nodes = assemble(g, 0, 1e18, policy(), &focal);
  ok(!nodes.empty() && nodes[0] == 0, "the target is emitted first");
  ok(!focal.empty() && focal[0] == 1, "the target is focal");
}

void test_budget_binds() {
  Fixture f;
  auto g = f.build();
  auto tight = policy();
  tight.max_context_cells = 8;
  auto loose = policy();
  loose.max_context_cells = 512;
  loose.local_context_cells = 512;
  const auto small = assemble(g, 0, 1e18, tight);
  const auto big = assemble(g, 0, 1e18, loose);
  ok(small.size() < big.size(), "a tighter cell budget emits fewer rows");
  // 2 cells per node, budget 8 -> a handful, never the whole graph.
  ok((std::int64_t)small.size() < f.n_nodes, "the tight budget really bound");
}

void test_no_duplicates() {
  Fixture f;
  auto g = f.build();
  auto loose = policy();
  loose.max_context_cells = 4096;
  loose.local_context_cells = 4096;
  const auto nodes = assemble(g, 0, 1e18, loose);
  const std::set<std::int64_t> unique(nodes.begin(), nodes.end());
  ok(unique.size() == nodes.size(), "no node is emitted twice");
}

void test_cutoff_excludes_the_future() {
  Fixture f;
  auto g = f.build();
  auto loose = policy();
  loose.max_context_cells = 4096;
  loose.local_context_cells = 4096;
  // The target is a timeless customer, so its own cut-off is +inf; bound the
  // walk instead by asking for a target whose time cuts the orders.
  const std::int64_t mid_order = 4 + 20;     // ts = 120
  const auto nodes = assemble(g, mid_order, 1e18, loose);
  for (std::int64_t n : nodes) {
    if (std::isnan(f.ts[n])) continue;
    ok(f.ts[n] <= f.ts[mid_order],
       "no row later than the seed's cut-off is emitted");
    if (fails) return;                        // one report is enough
  }
  ++checks;                                   // count the loop as one check
}

void test_fanout_cap_is_applied() {
  Fixture f;
  auto g = f.build();
  // Customer 0 has 10 orders; a width of 3 must not let all 10 through in one
  // hop. Give a budget big enough that truncation is the only limiter.
  auto p = policy();
  p.bfs_width = 3;
  p.max_context_cells = 4096;
  p.local_context_cells = 4096;
  const auto nodes = assemble(g, 0, 1e18, p);
  std::int64_t orders = 0;
  for (std::int64_t n : nodes) if (f.table[n] == 1) ++orders;
  ok(orders > 0, "some orders are reached");
  ok(orders < 40, "the fanout cap keeps the whole table out of one context");
}

void test_out_of_range_target() {
  Fixture f;
  auto g = f.build();
  std::vector<std::int64_t> nodes(16);
  std::vector<std::uint8_t> focal(16);
  auto p = policy();
  ok(g.assemble(-1, 1e18, nullptr, p, nodes.data(), focal.data(), 16) == -1,
     "a negative target is rejected");
  ok(g.assemble(f.n_nodes, 1e18, nullptr, p, nodes.data(), focal.data(), 16) == -1,
     "a target past the end is rejected");
}

void test_output_is_capped_not_overrun() {
  Fixture f;
  auto g = f.build();
  auto loose = policy();
  loose.max_context_cells = 4096;
  loose.local_context_cells = 4096;
  std::vector<std::int64_t> nodes(5, -1);
  std::vector<std::uint8_t> focal(5, 0);
  const std::int32_t n =
      g.assemble(0, 1e18, nullptr, loose, nodes.data(), focal.data(), 5);
  ok(n >= 0 && n <= 5, "the emitted count respects the caller's buffer");
}

}  // namespace

int main() {
  test_determinism();
  test_target_comes_first_and_is_focal();
  test_budget_binds();
  test_no_duplicates();
  test_cutoff_excludes_the_future();
  test_fanout_cap_is_applied();
  test_out_of_range_target();
  test_output_is_capped_not_overrun();
  std::printf("PASS: %d/%d\n", checks - fails, checks);
  return fails == 0 ? 0 : 1;
}
