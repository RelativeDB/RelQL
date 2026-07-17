package dev.relativedb.engine;

import dev.relativedb.retrieve.CohortRetriever;
import dev.relativedb.retrieve.EntityId;
import dev.relativedb.retrieve.EntityRetriever;
import dev.relativedb.retrieve.LinkRetriever;
import dev.relativedb.retrieve.RetrieverWiring;
import dev.relativedb.retrieve.Row;
import dev.relativedb.retrieve.TemporalBound;
import dev.relativedb.schema.LinkDef;

import java.util.List;
import java.util.Optional;

/** RETRIEVER mode: every expansion is a call into user-supplied retrievers. */
final class RetrieverContextSource implements ContextSource {
    private final RetrieverWiring wiring;

    RetrieverContextSource(RetrieverWiring wiring) { this.wiring = wiring; }

    @Override public List<Row> byIds(String table, List<EntityId> ids, TemporalBound bound) {
        EntityRetriever r = wiring.entityRetriever(table).orElseThrow(() ->
                new IllegalStateException("no EntityRetriever wired for table '" + table + "'"));
        return r.fetchByIds(table, ids, bound).toCompletableFuture().join();
    }

    @Override public List<Row> children(LinkDef link, EntityId parentId, TemporalBound bound, int limit) {
        LinkRetriever r = wiring.linkRetriever(link.fromTable()).orElseThrow(() ->
                new IllegalStateException("no LinkRetriever wired for table '" + link.fromTable()
                        + "' (and no defaultLinks)"));
        return r.fetchChildren(link, parentId, bound, limit).toCompletableFuture().join();
    }

    @Override public Optional<List<EntityId>> cohort(String table, EntityId anchor,
                                                     TemporalBound bound, int limit) {
        Optional<CohortRetriever> r = wiring.cohortRetriever(table);
        return r.map(cr -> cr.fetchCohort(table, anchor, bound, limit).toCompletableFuture().join());
    }
}
