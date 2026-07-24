/* test_analyze.cpp — the schema-aware semantic pass.
 *
 * Validation and task-type inference moved here from the bindings, so this is
 * where their behaviour is pinned: every frontend now inherits whatever this
 * asserts. Framework-free, no fixtures on disk, no network.
 */
#include <cstdio>
#include <string>

#include "analyze.hpp"
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

// The churn schema the bindings' own tests use.
const char* kSchema = R"({
  "tables": [
    {"name": "customers",
     "columns": [{"name": "age", "type": "number"},
                 {"name": "tier", "type": "text"},
                 {"name": "active", "type": "boolean"},
                 {"name": "signup_date", "type": "datetime"}],
     "primary_key": "customer_id", "time_column": null},
    {"name": "products",
     "columns": [{"name": "price", "type": "number"},
                 {"name": "name", "type": "text"}],
     "primary_key": "product_id", "time_column": null},
    {"name": "orders",
     "columns": [{"name": "qty", "type": "number"},
                 {"name": "order_date", "type": "datetime"}],
     "primary_key": "order_id", "time_column": "order_date"},
    {"name": "notes",
     "columns": [{"name": "body", "type": "text"}],
     "primary_key": "note_id", "time_column": null}
  ],
  "links": [
    {"from_table": "orders", "fk_column": "customer_id", "to_table": "customers"},
    {"from_table": "orders", "fk_column": "product_id", "to_table": "products"}
  ]
})";

relql::Schema schema() { return relql::schema_from_json(kSchema); }

// Validate, reporting which exception (if any) came out.
enum class Outcome { Ok, Syntax, Invalid };

Outcome run(const std::string& q, std::string* msg = nullptr) {
  try {
    relql::validate(relql::parse(q), schema());
    return Outcome::Ok;
  } catch (const relql::ValidationError& e) {
    if (msg) *msg = e.what();
    return Outcome::Invalid;
  } catch (const relql::RelqlError& e) {
    if (msg) *msg = e.what();
    return Outcome::Syntax;
  }
}

relql::TaskType task_of(const std::string& q) {
  relql::Schema s = schema();
  return relql::task_type(relql::validate(relql::parse(q), s), s);
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

void test_schema_parsing() {
  relql::Schema s = schema();
  ok(s.tables.size() == 4, "schema has 4 tables");
  ok(s.table("customers") != nullptr, "customers resolves");
  ok(s.table("nope") == nullptr, "unknown table resolves to null");
  ok(s.table("customers")->primary_key == "customer_id", "primary key read");
  ok(s.table("orders")->time_column == "order_date", "time column read");
  // time_column: null must read as "no time column", not the string "null"
  ok(s.table("customers")->time_column.empty(), "null time_column is empty");
  ok(s.table("customers")->column("age")->type == relql::ValueType::NUMBER,
     "column type read");
  ok(s.links_from("orders").size() == 2, "two links from orders");
  ok(s.links_from("customers").empty(), "no links from customers");

  bool threw = false;
  try {
    relql::schema_from_json("{\"tables\": [");
  } catch (const relql::SchemaError&) {
    threw = true;
  }
  ok(threw, "malformed schema JSON raises SchemaError");
}

void test_task_type_inference() {
  using T = relql::TaskType;
  ok(task_of("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) FROM customers") ==
         T::REGRESSION, "COUNT -> regression");
  ok(task_of("PREDICT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers") ==
         T::BINARY_CLASSIFICATION, "EXISTS -> binary");
  ok(task_of("PREDICT SUM(orders.qty) OVER (90 DAYS FOLLOWING) > 3 FROM customers") ==
         T::BINARY_CLASSIFICATION, "condition -> binary");
  // The schema-aware cases: a bare column's task follows its declared type,
  // which is the whole reason this pass needs a schema.
  ok(task_of("PREDICT customers.age FROM customers") == T::REGRESSION,
     "bare NUMBER column -> regression");
  ok(task_of("PREDICT customers.tier FROM customers") ==
         T::MULTICLASS_CLASSIFICATION, "bare TEXT column -> multiclass");
  ok(task_of("PREDICT customers.active FROM customers") ==
         T::BINARY_CLASSIFICATION, "bare BOOLEAN column -> binary");
  ok(task_of("PREDICT customers.signup_date FROM customers") == T::REGRESSION,
     "bare DATETIME column -> regression");
  // Same rule under FIRST/LAST.
  ok(task_of("PREDICT LAST(customers.tier) FROM customers") ==
         T::MULTICLASS_CLASSIFICATION, "LAST(TEXT) -> multiclass");
  ok(task_of("PREDICT LAST(customers.age) FROM customers") == T::REGRESSION,
     "LAST(NUMBER) -> regression");
}

void test_entity_binding() {
  // The parser leaves entity_column empty; validation resolves it.
  relql::Schema s = schema();
  relql::ParsedQuery bound = relql::validate(
      relql::parse("PREDICT customers.age FROM customers"), s);
  ok(bound.entity_column == "customer_id", "primary key bound onto the query");

  std::string msg;
  ok(run("PREDICT customers.age FROM nosuch", &msg) == Outcome::Invalid &&
         contains(msg, "unknown entity table"),
     "unknown entity table rejected");
  ok(contains(msg, "named by FROM"), "message names the clause of origin");
}

void test_column_checking() {
  std::string msg;
  ok(run("PREDICT customers.nope FROM customers", &msg) == Outcome::Invalid &&
         contains(msg, "unknown column"),
     "unknown column rejected");
  ok(run("PREDICT COUNT(nosuch.*) OVER (90 DAYS FOLLOWING) FROM customers", &msg) ==
         Outcome::Invalid && contains(msg, "unknown table"),
     "unknown table inside an aggregation rejected");

  // The primary key is referenceable even though tables do not list it among
  // their columns -- pinning a cohort depends on it.
  ok(run("PREDICT customers.age FROM customers "
         "WHERE customers.customer_id = 'C1'") == Outcome::Ok,
     "primary key is a legal column reference");
  // FK columns are legal targets for set/count aggregations only.
  ok(run("PREDICT COUNT_DISTINCT(orders.product_id) OVER (90 DAYS FOLLOWING) "
         "FROM customers") == Outcome::Ok,
     "FK legal under COUNT_DISTINCT");
  ok(run("PREDICT LAST(orders.product_id) OVER (90 DAYS FOLLOWING) FROM customers") ==
         Outcome::Invalid, "FK rejected under LAST");
}

void test_window_rules() {
  std::string msg;
  ok(run("PREDICT COUNT(orders.*) OVER (30 DAYS PRECEDING) FROM customers", &msg) ==
         Outcome::Invalid && contains(msg, "future-facing"),
     "past-facing target window rejected");
  // A table with no time column cannot carry an explicit frame, but the
  // implied default expresses no temporal intent and stays legal.
  ok(run("PREDICT COUNT(notes.*) OVER (90 DAYS FOLLOWING) FROM customers", &msg) ==
         Outcome::Invalid && contains(msg, "no time_column"),
     "explicit window over a time-less table rejected");
  ok(run("PREDICT COUNT(notes.*) FROM customers") == Outcome::Ok,
     "implied window over a time-less table allowed");
  ok(run("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING HORIZONS 3) FROM customers "
         "WHERE COUNT(orders.*) OVER (30 DAYS PRECEDING HORIZONS 2) > 1", &msg) ==
         Outcome::Invalid && contains(msg, "HORIZONS"),
     "HORIZONS outside the target rejected");
}

void test_return_compatibility() {
  std::string msg;
  ok(run("PREDICT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers "
         "RETURN PROBABILITY") == Outcome::Ok,
     "PROBABILITY on a binary task");
  ok(run("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) FROM customers "
         "RETURN PROBABILITY", &msg) == Outcome::Invalid &&
         contains(msg, "not compatible"),
     "PROBABILITY on a regression task rejected");
  ok(contains(msg, "allowed tasks:"), "message lists the allowed tasks");
}

void test_analyze_json() {
  std::string json = relql::analyze_to_json(
      "PREDICT customers.age FROM customers", kSchema);
  ok(contains(json, "\"task_type\":\"regression\""),
     "analyze_to_json carries the task type");
  ok(contains(json, "\"query\":"), "analyze_to_json carries the bound AST");
  ok(contains(json, "customer_id"), "bound AST includes the resolved key");
}

}  // namespace

int main() {
  test_schema_parsing();
  test_task_type_inference();
  test_entity_binding();
  test_column_checking();
  test_window_rules();
  test_return_compatibility();
  test_analyze_json();
  std::printf("PASS: %d/%d\n", checks - fails, checks);
  return fails == 0 ? 0 : 1;
}
