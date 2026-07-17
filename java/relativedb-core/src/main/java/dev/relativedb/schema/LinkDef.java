package dev.relativedb.schema;

import java.util.Objects;

/** A foreign-key link: {@code fromTable.fkColumn -> toTable.primaryKey}. */
public final class LinkDef {
    private final String fromTable;
    private final String fkColumn;
    private final String toTable;

    private LinkDef(String fromTable, String fkColumn, String toTable) {
        this.fromTable = Objects.requireNonNull(fromTable, "fromTable");
        this.fkColumn = Objects.requireNonNull(fkColumn, "fkColumn");
        this.toTable = Objects.requireNonNull(toTable, "toTable");
    }

    public static LinkDef link(String fromTable, String fkColumn, String toTable) {
        return new LinkDef(fromTable, fkColumn, toTable);
    }

    public String fromTable() { return fromTable; }
    public String fkColumn() { return fkColumn; }
    public String toTable() { return toTable; }

    @Override public boolean equals(Object o) {
        return o instanceof LinkDef l && l.fromTable.equals(fromTable)
                && l.fkColumn.equals(fkColumn) && l.toTable.equals(toTable);
    }
    @Override public int hashCode() { return Objects.hash(fromTable, fkColumn, toTable); }
    @Override public String toString() { return fromTable + "." + fkColumn + " -> " + toTable; }
}
