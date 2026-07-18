package com.relativedb.retrieve;

import java.util.List;
import java.util.concurrent.CompletionStage;

/**
 * OPTIONAL: similar/other entities of the same table for in-context examples
 * (RT-J Tier 1/2). Without one, context is target-entity-local only.
 */
@FunctionalInterface
public interface CohortRetriever {
    CompletionStage<List<EntityId>> fetchCohort(String table, EntityId anchor,
                                                TemporalBound bound, int limit);
}
