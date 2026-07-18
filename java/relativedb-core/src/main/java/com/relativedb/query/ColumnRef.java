package com.relativedb.query;

/** {@code table.column} — column may be {@code "*"}. */
public record ColumnRef(String table, String column) implements TargetExpr {
    public boolean isWildcard() { return "*".equals(column); }
    @Override public String toString() { return table + "." + column; }
}
