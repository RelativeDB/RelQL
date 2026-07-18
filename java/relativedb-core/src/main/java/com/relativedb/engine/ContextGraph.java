package com.relativedb.engine;

import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.Row;

import java.util.List;

/** The assembled per-entity context: seed row first, then traversal order. */
public record ContextGraph(String seedTable, EntityId seedId, List<Row> rows, int totalCells) {

    public boolean contains(String table, EntityId id) {
        return rows.stream().anyMatch(r -> r.table().equals(table) && r.id().equals(id));
    }
}
