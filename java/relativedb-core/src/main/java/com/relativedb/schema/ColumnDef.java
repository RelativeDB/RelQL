package com.relativedb.schema;

import java.util.Objects;

/**
 * A typed column. The extension point for future per-column metadata
 * (timestamp format, encoder hints, vocab).
 */
public final class ColumnDef {
    private final String name;
    private final ValueType type;

    private ColumnDef(String name, ValueType type) {
        this.name = Objects.requireNonNull(name, "name");
        this.type = Objects.requireNonNull(type, "type");
        if (name.isBlank()) throw new IllegalArgumentException("column name must not be blank");
    }

    public static ColumnDef of(String name, ValueType type) {
        return new ColumnDef(name, type);
    }

    public String name() { return name; }
    public ValueType type() { return type; }

    @Override public boolean equals(Object o) {
        return o instanceof ColumnDef c && c.name.equals(name) && c.type == type;
    }
    @Override public int hashCode() { return Objects.hash(name, type); }
    @Override public String toString() { return name + ":" + type; }
}
