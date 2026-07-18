package com.relativedb.engine;

import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.Row;
import com.relativedb.retrieve.TemporalBound;
import com.relativedb.schema.LinkDef;

import java.util.List;
import java.util.Optional;

/**
 * Internal data-access abstraction the hop loop runs against — implemented by
 * {@link RetrieverContextSource} (pull-per-hop) and {@link CscIndex} (in-memory).
 * Implementations must never return a row newer than {@code bound}; the
 * assembler re-checks defensively anyway.
 */
interface ContextSource {

    List<Row> byIds(String table, List<EntityId> ids, TemporalBound bound);

    /** Children of {@code parentId} along {@code link}, newest-first, at most {@code limit}. */
    List<Row> children(LinkDef link, EntityId parentId, TemporalBound bound, int limit);

    /** Similar-entity ids, if a cohort source exists for {@code table}. */
    Optional<List<EntityId>> cohort(String table, EntityId anchor, TemporalBound bound, int limit);
}
