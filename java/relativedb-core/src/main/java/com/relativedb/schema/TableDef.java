package com.relativedb.schema;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** A table declaration: typed columns plus optional primary-key and time column. */
public final class TableDef {
    private final String name;
    private final Map<String, ColumnDef> columns;
    private final String primaryKey;   // nullable
    private final String timeColumn;   // nullable

    private TableDef(String name, Map<String, ColumnDef> columns, String primaryKey, String timeColumn) {
        this.name = name;
        this.columns = columns;
        this.primaryKey = primaryKey;
        this.timeColumn = timeColumn;
    }

    public static Builder newTable(String name) { return new BuilderImpl(name); }

    public interface Builder {
        /** Canonical form — ColumnDef is the extension point for future per-column metadata. */
        Builder column(ColumnDef column);
        /** Convenience overload; equivalent to {@code column(ColumnDef.of(name, type))}. */
        Builder column(String name, ValueType type);
        /** Primary key: identity only — never surfaced as a cell (F17). */
        Builder primaryKey(String column);
        /** Row timestamp: drives temporal filtering (F24) and windows. */
        Builder timeColumn(String column);
        TableDef build();
    }

    public String name() { return name; }
    public List<ColumnDef> columns() { return List.copyOf(columns.values()); }
    public Optional<ColumnDef> column(String name) { return Optional.ofNullable(columns.get(name)); }
    public Optional<String> primaryKey() { return Optional.ofNullable(primaryKey); }
    public Optional<String> timeColumn() { return Optional.ofNullable(timeColumn); }

    @Override public String toString() { return "TableDef[" + name + "]"; }

    private static final class BuilderImpl implements Builder {
        private final String name;
        private final Map<String, ColumnDef> columns = new LinkedHashMap<>();
        private String primaryKey;
        private String timeColumn;

        BuilderImpl(String name) {
            this.name = Objects.requireNonNull(name, "table name");
            if (name.isBlank()) throw new IllegalArgumentException("table name must not be blank");
        }

        @Override public Builder column(ColumnDef column) {
            Objects.requireNonNull(column, "column");
            if (columns.putIfAbsent(column.name(), column) != null) {
                throw new SchemaException("duplicate column '" + column.name() + "' in table '" + name + "'");
            }
            return this;
        }
        @Override public Builder column(String name, ValueType type) {
            return column(ColumnDef.of(name, type));
        }
        @Override public Builder primaryKey(String column) { this.primaryKey = column; return this; }
        @Override public Builder timeColumn(String column) { this.timeColumn = column; return this; }
        @Override public TableDef build() {
            // Time column must be a declared DATETIME column, or (like the PK)
            // an identity/metadata column not surfaced as a cell. If declared,
            // it must be DATETIME.
            ColumnDef tc = timeColumn == null ? null : columns.get(timeColumn);
            if (tc != null && tc.type() != ValueType.DATETIME) {
                throw new SchemaException("time column '" + timeColumn + "' of table '" + name
                        + "' must be DATETIME, was " + tc.type());
            }
            return new TableDef(name, new LinkedHashMap<>(columns), primaryKey, timeColumn);
        }
    }
}
