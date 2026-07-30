/* test_flat.cpp — flat-feature evaluation over a spec JSON.
 *
 * Eligibility and feature derivation moved to Python (relativedb.flat);
 * what is pinned here is the native evaluator: the exact numbers a known
 * context produces for a hand-written spec, through both the C++ API and
 * the C ABI. Framework-free, no fixtures on disk, no network.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "flat.hpp"
#include "flat_c.h"
#include "json.hpp"

namespace {

int checks = 0, fails = 0;

void ok(bool cond, const std::string& what) {
  ++checks;
  if (!cond) {
    ++fails;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

void feq(float got, double want, const std::string& what) {
  ok(std::abs(got - want) < 1e-6, what + " (got " + std::to_string(got) +
                                      ", want " + std::to_string(want) + ")");
}

// The spec relativedb.flat.flat_spec_to_json produces for a churn-style
// schema: entity columns, an autoregressive window, a filtered count, a
// windowless (unbounded) sum, a categorical LAST, and a recency feature.
const char* kSpec = R"({
  "entity_table": "customers",
  "task_type": "regression",
  "features": [
    {"kind":"entity_column","name":"entity.age","column":"age","col_type":"number"},
    {"kind":"entity_column","name":"entity.signup_age_days","column":"signup","col_type":"datetime"},
    {"kind":"aggregate","name":"hist1","agg":{"func":"SUM","table":"orders","column":"qty",
       "window":{"start":-30,"end":0,"unit":"days"}}},
    {"kind":"aggregate","name":"big_orders","agg":{"func":"COUNT","table":"orders","column":"*",
       "filter":{"kind":"cond","column":"qty","op":"GT","right":1},
       "window":{"start":"-inf","end":0,"unit":"days"}}},
    {"kind":"aggregate","name":"sum_all","agg":{"func":"SUM","table":"orders","column":"qty"}},
    {"kind":"aggregate","name":"last_status","agg":{"func":"LAST","table":"status","column":"state"}},
    {"kind":"days_since_last","name":"orders.recency_days","table":"orders"}
  ]})";

// Anchor at day 100 (epoch seconds 8,640,000). Orders at day 80 (qty 2),
// day 95 (qty 1), day 99 (qty 3); one undated status row.
const char* kContext = R"({
  "entity_id": "C7",
  "anchor": 8640000,
  "rows": [
    {"table":"customers","id":"C7","ts":null,
     "cells":{"age": 41, "signup": 864000}},
    {"table":"orders","id":"O1","ts":6912000,"cells":{"qty":2},"parents":{"cid":"C7"}},
    {"table":"orders","id":"O2","ts":8208000,"cells":{"qty":1},"parents":{"cid":"C7"}},
    {"table":"orders","id":"O3","ts":8553600,"cells":{"qty":3},"parents":{"cid":"C7"}},
    {"table":"status","id":"S1","ts":null,"cells":{"state":"active"}}
  ]})";

double hash_active() {  // FNV-1a of "active", as the evaluator computes it
  std::uint32_t h = 2166136261u;
  for (unsigned char c : std::string("active")) { h ^= c; h *= 16777619u; }
  return static_cast<double>(h);
}

}  // namespace

int main() {
  relql::JsonValue spec = relql::json_parse(kSpec);
  relql::JsonValue ctx = relql::json_parse(kContext);
  ok(relql::flat_spec_size(spec) == 7, "spec has 7 features");

  float out[7];
  relql::flat_features(spec, ctx, out);
  feq(out[0], 41.0, "entity.age");
  // signup at day 10, anchor at day 100 -> 90 days old
  feq(out[1], 90.0, "entity.signup_age_days");
  // (anchor-30d, anchor] = (day 70, day 100]: all three orders qualify
  feq(out[2], 6.0, "hist window sum");
  // qty > 1 across all time: days 80 and 99
  feq(out[3], 2.0, "filtered count");
  // windowless sum sees every order, dated or not
  feq(out[4], 6.0, "unbounded sum");
  // categorical LAST hashes the string, stable across runs
  feq(out[5], static_cast<float>(hash_active()), "categorical LAST hash");
  // latest order at day 99 -> 1 day before the anchor
  feq(out[6], 1.0, "recency days");

  // Windowed aggregation with no anchor is unevaluable -> NaN.
  {
    relql::JsonValue noanchor = relql::json_parse(
        R"({"entity_id":"C7","rows":[]})");
    float v[7];
    relql::flat_features(spec, noanchor, v);
    ok(std::isnan(v[2]), "windowed agg without anchor is NaN");
    feq(v[4], 0.0, "unbounded SUM over no rows is 0");
    ok(std::isnan(v[6]), "recency without anchor is NaN");
  }

  // The C ABI: same numbers, matrix layout, and its error surface.
  {
    std::string contexts = std::string("[") + kContext + "," + kContext + "]";
    float m[14];
    char err[256] = {0};
    int rc = relql_flat_features(kSpec, contexts.c_str(), m, 2, 7, err,
                                 sizeof(err));
    ok(rc == 0, std::string("flat_c evaluates: ") + err);
    feq(m[0], 41.0f, "flat_c row 0 age");
    feq(m[7 + 2], 6.0, "flat_c row 1 hist sum");

    rc = relql_flat_features(kSpec, contexts.c_str(), m, 2, 6, err,
                             sizeof(err));
    ok(rc == 1, "feature count mismatch is an error");
    rc = relql_flat_features("{not json", contexts.c_str(), m, 2, 7, err,
                             sizeof(err));
    ok(rc == 1, "bad spec JSON is an error");
  }

  std::printf("%s: %d checks, %d failures\n", fails ? "FAIL" : "PASS", checks,
              fails);
  return fails ? 1 : 0;
}
