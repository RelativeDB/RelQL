package dev.relativedb.retrieve;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * One row's typed feature cells. IDs and FK values are NOT cells (F17) — links
 * are reported separately (as {@link #parents()}) so the engine can traverse
 * without ever tokenizing identifiers.
 */
public final class Row {
    private final String table;
    private final EntityId id;
    private final Instant timestamp;               // nullable
    private final Map<String, Object> cells;       // Double | String | Instant | Boolean
    private final Map<String, EntityId> parents;   // fkColumn -> parent id

    private Row(String table, EntityId id, Instant timestamp,
                Map<String, Object> cells, Map<String, EntityId> parents) {
        this.table = table;
        this.id = id;
        this.timestamp = timestamp;
        this.cells = cells;
        this.parents = parents;
    }

    public static Builder newRow(String table, EntityId id) { return new BuilderImpl(table, id); }

    public interface Builder {
        Builder cell(String column, double number);
        Builder cell(String column, String text);
        Builder cell(String column, Instant datetime);
        Builder cell(String column, boolean bool);
        /** Missing/null: simply omit the cell — nulls emit no token. */
        Builder timestamp(Instant rowTime);                  // required if table has timeColumn
        Builder parent(String fkColumn, EntityId parentId);  // F→P edge
        Row build();
    }

    public String table() { return table; }
    public EntityId id() { return id; }
    public Optional<Instant> timestamp() { return Optional.ofNullable(timestamp); }
    public Map<String, Object> cells() { return cells; }
    public Map<String, EntityId> parents() { return parents; }

    /** Number of feature cells (the unit the engine's context budget counts). */
    public int cellCount() { return cells.size(); }

    @Override public String toString() {
        return table + "[" + id + "]" + (timestamp != null ? "@" + timestamp : "");
    }

    private static final class BuilderImpl implements Builder {
        private final String table;
        private final EntityId id;
        private Instant timestamp;
        private final Map<String, Object> cells = new LinkedHashMap<>();
        private final Map<String, EntityId> parents = new LinkedHashMap<>();

        BuilderImpl(String table, EntityId id) {
            this.table = Objects.requireNonNull(table, "table");
            this.id = Objects.requireNonNull(id, "id");
        }
        private Builder put(String column, Object value) {
            cells.put(Objects.requireNonNull(column, "column"), value);
            return this;
        }
        @Override public Builder cell(String column, double number) { return put(column, number); }
        @Override public Builder cell(String column, String text) {
            return text == null ? this : put(column, text);
        }
        @Override public Builder cell(String column, Instant datetime) {
            return datetime == null ? this : put(column, datetime);
        }
        @Override public Builder cell(String column, boolean bool) { return put(column, bool); }
        @Override public Builder timestamp(Instant rowTime) { this.timestamp = rowTime; return this; }
        @Override public Builder parent(String fkColumn, EntityId parentId) {
            parents.put(Objects.requireNonNull(fkColumn), Objects.requireNonNull(parentId));
            return this;
        }
        @Override public Row build() {
            return new Row(table, id, timestamp,
                    new LinkedHashMap<>(cells), new LinkedHashMap<>(parents));
        }
    }
}
