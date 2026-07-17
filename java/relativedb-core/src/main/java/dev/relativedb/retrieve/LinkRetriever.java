package dev.relativedb.retrieve;

import dev.relativedb.schema.LinkDef;
import java.util.List;
import java.util.concurrent.CompletionStage;

/**
 * Traversal along one P→F link: the children of a parent row, newest-first,
 * capped at {@code limit} (the engine supplies its BFS width bound, F23).
 * MUST NOT return rows newer than {@code bound} — and the engine re-checks
 * defensively either way.
 */
@FunctionalInterface
public interface LinkRetriever {
    CompletionStage<List<Row>> fetchChildren(LinkDef link, EntityId parentId,
                                             TemporalBound bound, int limit);
}
