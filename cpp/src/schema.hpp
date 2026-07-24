/* schema.hpp — the relational schema, shared by every binding.
 *
 * Validation and schema-aware task-type inference used to live in each
 * binding's own language (python/src/relativedb/relql/parser.py). Every new
 * frontend had to reimplement them, which is the same divergence risk that the
 * grammar and the CSC adjacency are single-sourced in C++ to avoid.
 *
 * The schema arrives as the JSON that Schema.to_json_dict already emits, so a
 * binding hands over what it has rather than marshalling a parallel C struct.
 * No third-party dependencies: the JSON reader below is the small subset this
 * document shape needs.
 */
#ifndef RELATIVEDB_SCHEMA_HPP
#define RELATIVEDB_SCHEMA_HPP

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace relql {

// Mirrors relativedb.schema.ValueType.
enum class ValueType { NUMBER, TEXT, BOOLEAN, DATETIME, CATEGORICAL, UNKNOWN };

struct ColumnDef {
  std::string name;
  ValueType type = ValueType::UNKNOWN;
};

struct LinkDef {
  std::string from_table;
  std::string fk_column;
  std::string to_table;
};

struct TableDef {
  std::string name;
  std::vector<ColumnDef> columns;
  std::string primary_key;   // empty when the table declares none
  std::string time_column;   // empty when the table is static

  const ColumnDef* column(const std::string& col) const;
};

class Schema {
 public:
  std::vector<TableDef> tables;
  std::vector<LinkDef> links;

  const TableDef* table(const std::string& name) const;
  std::vector<const LinkDef*> links_from(const std::string& table) const;
};

// Thrown when the schema document itself is malformed.
class SchemaError : public std::runtime_error {
 public:
  explicit SchemaError(const std::string& msg) : std::runtime_error(msg) {}
};

// Parse the JSON produced by relativedb.schema.Schema.to_json_dict.
Schema schema_from_json(const std::string& json);

const char* value_type_name(ValueType t);

}  // namespace relql

#endif  // RELATIVEDB_SCHEMA_HPP
