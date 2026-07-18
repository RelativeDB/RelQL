package com.relativedb.retrieve;

import java.util.List;
import java.util.concurrent.CompletionStage;

/** Batched point lookup: rows of one table by id. GraphQL's DataFetcher analog. */
@FunctionalInterface
public interface EntityRetriever {
    CompletionStage<List<Row>> fetchByIds(String table, List<EntityId> ids, TemporalBound bound);
}
