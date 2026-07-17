package dev.relativedb.schema;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The declared relational graph: tables + FK links. Only <em>shape</em> lives
 * here — no URLs, no credentials. (The design doc's {@code RqlSchema}.)
 */
public final class RelativeDbSchema {
    private final Map<String, TableDef> tables;
    private final List<LinkDef> links;

    private RelativeDbSchema(Map<String, TableDef> tables, List<LinkDef> links) {
        this.tables = tables;
        this.links = links;
    }

    public static Builder newSchema() { return new BuilderImpl(); }

    public interface Builder {
        Builder table(TableDef table);
        Builder link(LinkDef link);
        /** Validates: PKs exist, links resolve, FK columns exist, etc. */
        RelativeDbSchema build();
    }

    public Optional<TableDef> table(String name) { return Optional.ofNullable(tables.get(name)); }
    public List<TableDef> tables() { return List.copyOf(tables.values()); }
    public List<LinkDef> links() { return links; }

    /** Links whose {@code fromTable} is {@code table} — the F→P (parent) edges. */
    public List<LinkDef> linksFrom(String table) {
        List<LinkDef> out = new ArrayList<>();
        for (LinkDef l : links) if (l.fromTable().equals(table)) out.add(l);
        return out;
    }

    /** Links whose {@code toTable} is {@code table} — the P→F (children) edges. */
    public List<LinkDef> linksTo(String table) {
        List<LinkDef> out = new ArrayList<>();
        for (LinkDef l : links) if (l.toTable().equals(table)) out.add(l);
        return out;
    }

    private static final class BuilderImpl implements Builder {
        private final Map<String, TableDef> tables = new LinkedHashMap<>();
        private final List<LinkDef> links = new ArrayList<>();

        @Override public Builder table(TableDef table) {
            if (tables.putIfAbsent(table.name(), table) != null) {
                throw new SchemaException("duplicate table '" + table.name() + "'");
            }
            return this;
        }
        @Override public Builder link(LinkDef link) { links.add(link); return this; }

        @Override public RelativeDbSchema build() {
            for (LinkDef l : links) {
                TableDef from = tables.get(l.fromTable());
                if (from == null) throw new SchemaException("link " + l + ": unknown table '" + l.fromTable() + "'");
                TableDef to = tables.get(l.toTable());
                if (to == null) throw new SchemaException("link " + l + ": unknown table '" + l.toTable() + "'");
                if (to.primaryKey().isEmpty()) {
                    throw new SchemaException("link " + l + ": target table '" + l.toTable() + "' has no primary key");
                }
                // The FK column is an edge, never a feature cell (F17) — it must
                // NOT be declared as a typed feature column of the child table.
                if (from.column(l.fkColumn()).isPresent()) {
                    throw new SchemaException("link " + l + ": fk column '" + l.fkColumn()
                            + "' must not also be declared as a feature column of '" + l.fromTable()
                            + "' (PK/FK values are edges, not data — F17)");
                }
            }
            for (TableDef t : tables.values()) {
                // PK must not be a declared feature column either (identity only, F17).
                if (t.primaryKey().isPresent() && t.column(t.primaryKey().get()).isPresent()) {
                    throw new SchemaException("table '" + t.name() + "': primary key '"
                            + t.primaryKey().get() + "' must not also be a feature column (F17)");
                }
            }
            return new RelativeDbSchema(new LinkedHashMap<>(tables), List.copyOf(links));
        }
    }
}
