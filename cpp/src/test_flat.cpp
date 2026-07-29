/* test_flat.cpp — flat-feature eligibility, spec derivation and evaluation.
 *
 * The XGBoost binding delegates all of its decisions to this layer, so its
 * behaviour is pinned here: which queries qualify, which feature columns a
 * schema yields, and the exact numbers the evaluator produces for a known
 * context. Framework-free, no fixtures on disk, no network.
 */
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "analyze.hpp"
#include "flat.hpp"
#include "flat_c.h"
#include "relql.hpp"
#include "schema.hpp"

namespace {

int checks = 0, fails = 0;

void ok(bool cond, const std::string& what) {
  ++checks;
  if (!cond) {
    ++fails;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

const char* kSchema = R"({
  "tables": [
    {"name": "customers",
     "columns": [{"name": "age", "type": "number"},
                 {"name": "tier", "type": "categorical"},
                 {"name": "signup_date", "type": "datetime"}],
     "primary_key": "customer_id", "time_column": null},
    {"name": "products",
     "columns": [{"name": "price", "type": "number"},
                 {"name": "name", "type": "text"}],
     "primary_key": "product_id", "time_column": null},
    {"name": "orders",
     "columns": [{"name": "qty", "type": "number"},
                 {"name": "order_date", "type": "datetime"}],
     "primary_key": "order_id", "time_column": "order_date"}
  ],
  "links": [
    {"from_table": "orders", "fk_column": "customer_id", "to_table": "customers"},
    {"from_table": "orders", "fk_column": "product_id", "to_table": "products"}
  ]
})";

relql::FlatSpec spec_for(const std::string& q) {
  relql::Schema schema = relql::schema_from_json(kSchema);
  relql::ParsedQuery pq = relql::validate(relql::parse(q), schema);
  return relql::derive_flat_spec(pq, schema);
}

int feature_index(const relql::FlatSpec& spec, const std::string& prefix) {
  for (std::size_t i = 0; i < spec.features.size(); ++i)
    if (spec.features[i].name.rfind(prefix, 0) == 0)
      return static_cast<int>(i);
  return -1;
}

// anchor and row timestamps in whole days for readable arithmetic.
double day(double n) { return n * 86400.0; }

}  // namespace

int main() {
  using relql::FlatSpec;

  // -- eligibility ----------------------------------------------------------
  {
    FlatSpec s = spec_for(
        "PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) FROM customers");
    ok(s.eligible, "windowed COUNT regression is flat-eligible");
    ok(s.task == relql::TaskType::REGRESSION, "task is regression");
    ok(!s.features.empty(), "eligible query derives features");
  }
  ok(!spec_for("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FROM "
               "customers ASSUMING COUNT(orders.*) OVER (7 DAYS FOLLOWING) > 2")
          .eligible,
     "ASSUMING declines");
  ok(!spec_for("PREDICT LIST_DISTINCT(orders.product_id) OVER "
               "(30 DAYS FOLLOWING) CLASSIFY FROM customers")
          .eligible,
     "CLASSIFY declines");
  ok(!spec_for("PREDICT ARRAY_AGG(orders.product_id) OVER "
               "(30 DAYS FOLLOWING RANK TOP 5) FROM customers")
          .eligible,
     "RANK TOP declines");
  ok(!spec_for("PREDICT SUM(orders.qty) OVER (1 DAY FOLLOWING HORIZONS 28) "
               "FROM customers")
          .eligible,
     "multi-horizon forecasting declines");
  ok(spec_for("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
              "FROM customers RETURN PROBABILITY")
         .task == relql::TaskType::BINARY_CLASSIFICATION,
     "binary churn shape is flat-eligible");

  // -- spec shape -----------------------------------------------------------
  FlatSpec spec = spec_for(
      "PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) FROM customers "
      "WHERE SUM(orders.qty) OVER (60 DAYS PRECEDING) > 0");
  ok(feature_index(spec, "entity.age") >= 0, "entity numeric column present");
  ok(feature_index(spec, "entity.tier") >= 0, "entity categorical present");
  ok(feature_index(spec, "entity.signup_date_age_days") >= 0,
     "entity datetime becomes age");
  ok(feature_index(spec, "hist1:") >= 0 && feature_index(spec, "hist3:") >= 0,
     "target history features present");
  ok(feature_index(spec, "where:") >= 0, "WHERE aggregate becomes a feature");
  ok(feature_index(spec, "orders.count_7d") >= 0 &&
         feature_index(spec, "orders.count_all") >= 0,
     "per-table counts present");
  ok(feature_index(spec, "orders.recency_days") >= 0, "recency present");
  ok(feature_index(spec, "orders.qty_sum_30d") >= 0,
     "numeric column aggregate present");
  ok(feature_index(spec, "products.count_all") >= 0,
     "two-hop table (via orders) reaches the recipe");

  // -- evaluation over a known context -------------------------------------
  // anchor = day 20000; orders 3, 20 and 100 days old with qty 1, 2, 5;
  // signup 200 days before the anchor.
  char ctx[4096];
  std::snprintf(ctx, sizeof(ctx), R"([{
    "entity_id": "C1", "anchor": %.1f,
    "rows": [
      {"table": "customers", "id": "C1",
       "cells": {"age": 34, "tier": "gold", "signup_date": %.1f}},
      {"table": "orders", "id": "O1", "ts": %.1f,
       "cells": {"qty": 1}, "parents": {"customer_id": "C1", "product_id": "P1"}},
      {"table": "orders", "id": "O2", "ts": %.1f,
       "cells": {"qty": 2}, "parents": {"customer_id": "C1", "product_id": "P1"}},
      {"table": "orders", "id": "O3", "ts": %.1f,
       "cells": {"qty": 5}, "parents": {"customer_id": "C1", "product_id": "P2"}},
      {"table": "products", "id": "P1", "cells": {"price": 25.0}},
      {"table": "products", "id": "P2", "cells": {"price": 90.0}}
    ]}])",
                day(20000), day(19800), day(19997), day(19980), day(19900));

  const char* query =
      "PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) FROM customers "
      "WHERE SUM(orders.qty) OVER (60 DAYS PRECEDING) > 0";
  int n = static_cast<int>(spec.features.size());
  std::vector<float> feats(n, 0.0f);
  char err[512] = {0};
  int rc = relql_flat_features(query, kSchema, "{}", ctx, feats.data(), 1, n,
                               err, sizeof(err));
  ok(rc == 0, std::string("relql_flat_features rc=0 (") + err + ")");

  auto at = [&](const std::string& prefix) {
    int i = feature_index(spec, prefix);
    return i < 0 ? -12345.0f : feats[i];
  };
  ok(at("entity.age") == 34.0f, "entity.age == 34");
  ok(at("entity.signup_date_age_days") == 200.0f, "signup age == 200 days");
  ok(at("entity.tier") > 0.0f, "categorical hashes to a stable positive code");
  ok(at("orders.count_7d") == 1.0f, "orders.count_7d == 1");
  ok(at("orders.count_30d") == 2.0f, "orders.count_30d == 2");
  ok(at("orders.count_all") == 3.0f, "orders.count_all == 3");
  ok(at("orders.recency_days") == 3.0f, "recency == 3 days");
  ok(at("orders.qty_sum_30d") == 3.0f, "qty sum over 30d == 1+2");
  ok(at("orders.qty_sum_all") == 8.0f, "qty sum unbounded == 1+2+5");
  ok(at("orders.qty_max_all") == 5.0f, "qty max unbounded == 5");
  ok(at("products.count_all") == 2.0f, "static products counted unwindowed");
  ok(at("hist1:") == 2.0f, "target mirrored into (-30,0] == 2");
  ok(at("hist2:") == 0.0f, "target mirrored into (-60,-30] == 0");
  ok(at("where:") == 3.0f, "WHERE SUM(qty) over 60d == 1+2");

  // inline filter inside the mirrored target
  {
    const char* fq =
        "PREDICT COUNT(orders.* WHERE orders.qty > 1) OVER "
        "(30 DAYS FOLLOWING) FROM customers";
    FlatSpec fs = spec_for(fq);
    int fn = static_cast<int>(fs.features.size());
    std::vector<float> ff(fn, 0.0f);
    rc = relql_flat_features(fq, kSchema, "{}", ctx, ff.data(), 1, fn, err,
                             sizeof(err));
    ok(rc == 0, std::string("filtered features rc=0 (") + err + ")");
    int i = feature_index(fs, "hist1:");
    ok(i >= 0 && ff[i] == 1.0f, "inline filter keeps only qty>1 in hist1");
  }

  // -- the C ABI analyze entry ---------------------------------------------
  {
    char out[1 << 15] = {0};
    rc = relql_flat_analyze(
        "PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) FROM customers",
        kSchema, nullptr, out, sizeof(out), err, sizeof(err));
    ok(rc == 0, std::string("relql_flat_analyze rc=0 (") + err + ")");
    ok(std::string(out).find("\"eligible\":true") != std::string::npos,
       "analyze JSON reports eligible");
    rc = relql_flat_analyze(
        "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FROM customers "
        "ASSUMING COUNT(orders.*) OVER (7 DAYS FOLLOWING) > 2",
        kSchema, nullptr, out, sizeof(out), err, sizeof(err));
    ok(rc == 0 &&
           std::string(out).find("\"eligible\":false") != std::string::npos,
       "ineligible is an answer, not an error");
    rc = relql_flat_analyze("PREDICT bogus syntax", kSchema, nullptr, out,
                            sizeof(out), err, sizeof(err));
    ok(rc == 1, "a query that does not parse is an error");
  }

  // NaN, not zero, for a missing entity row: trees treat NaN as missing.
  {
    const char* empty_ctx =
        R"([{"entity_id": "C9", "anchor": 1728000000.0, "rows": []}])";
    std::vector<float> ef(n, 0.0f);
    rc = relql_flat_features(query, kSchema, "{}", empty_ctx, ef.data(), 1, n,
                             err, sizeof(err));
    ok(rc == 0, "empty context evaluates");
    int i = feature_index(spec, "entity.age");
    ok(i >= 0 && std::isnan(ef[i]), "missing entity cell is NaN");
    int c = feature_index(spec, "orders.count_all");
    ok(c >= 0 && ef[c] == 0.0f, "count over no rows is 0, not NaN");
  }

  std::printf("%d checks, %d failures\n", checks, fails);
  return fails == 0 ? 0 : 1;
}
