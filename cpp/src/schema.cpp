/* schema.cpp — schema model plus the minimal JSON reader it needs.
 *
 * The reader covers exactly the document shape Schema.to_json_dict emits:
 * objects, arrays, strings, numbers, booleans and null. It is deliberately
 * small rather than general — the alternative was a third-party dependency in
 * a library that has none, for one document per query.
 */
#include "schema.hpp"

#include "json.hpp"

namespace relql {

const ColumnDef* TableDef::column(const std::string& col) const {
  for (const ColumnDef& c : columns)
    if (c.name == col) return &c;
  return nullptr;
}

const TableDef* Schema::table(const std::string& name) const {
  for (const TableDef& t : tables)
    if (t.name == name) return &t;
  return nullptr;
}

std::vector<const LinkDef*> Schema::links_from(const std::string& table) const {
  std::vector<const LinkDef*> out;
  for (const LinkDef& l : links)
    if (l.from_table == table) out.push_back(&l);
  return out;
}

const char* value_type_name(ValueType t) {
  switch (t) {
    case ValueType::NUMBER: return "number";
    case ValueType::TEXT: return "text";
    case ValueType::BOOLEAN: return "boolean";
    case ValueType::DATETIME: return "datetime";
    case ValueType::CATEGORICAL: return "categorical";
    default: return "unknown";
  }
}

namespace {

ValueType value_type_from(const std::string& s) {
  if (s == "number") return ValueType::NUMBER;
  if (s == "text") return ValueType::TEXT;
  if (s == "boolean") return ValueType::BOOLEAN;
  if (s == "datetime") return ValueType::DATETIME;
  if (s == "categorical") return ValueType::CATEGORICAL;
  return ValueType::UNKNOWN;
}

}  // namespace

Schema schema_from_json(const std::string& json) {
  JsonValue root;
  try {
    root = json_parse(json);
  } catch (const JsonError& e) {
    throw SchemaError(e.what());
  }
  if (root.kind != JsonValue::Kind::Obj)
    throw SchemaError("schema JSON must be an object");

  Schema schema;
  const JsonValue* tables = root.find("tables");
  if (tables && tables->kind == JsonValue::Kind::Arr) {
    for (const JsonValue& t : tables->arr) {
      TableDef def;
      def.name = t.str_or("name");
      if (def.name.empty()) throw SchemaError("a table has no name");
      def.primary_key = t.str_or("primary_key");
      def.time_column = t.str_or("time_column");
      const JsonValue* cols = t.find("columns");
      if (cols && cols->kind == JsonValue::Kind::Arr) {
        for (const JsonValue& c : cols->arr) {
          ColumnDef col;
          col.name = c.str_or("name");
          col.type = value_type_from(c.str_or("type"));
          def.columns.push_back(col);
        }
      }
      schema.tables.push_back(def);
    }
  }

  const JsonValue* links = root.find("links");
  if (links && links->kind == JsonValue::Kind::Arr) {
    for (const JsonValue& l : links->arr) {
      LinkDef def;
      def.from_table = l.str_or("from_table");
      def.fk_column = l.str_or("fk_column");
      def.to_table = l.str_or("to_table");
      schema.links.push_back(def);
    }
  }
  return schema;
}

}  // namespace relql
