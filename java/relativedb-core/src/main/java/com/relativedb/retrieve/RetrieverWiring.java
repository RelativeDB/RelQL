package com.relativedb.retrieve;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Wiring: schema element → implementation. GraphQL RuntimeWiring analog. */
public final class RetrieverWiring {
    private final Map<String, EntityRetriever> entities;
    private final Map<String, LinkRetriever> links;
    private final LinkRetriever defaultLinks;      // nullable
    private final Map<String, CohortRetriever> cohorts;
    private final Map<String, TableScanner> scanners;
    private final StatsProvider stats;             // nullable

    private RetrieverWiring(BuilderImpl b) {
        this.entities = Map.copyOf(b.entities);
        this.links = Map.copyOf(b.links);
        this.defaultLinks = b.defaultLinks;
        this.cohorts = Map.copyOf(b.cohorts);
        this.scanners = Map.copyOf(b.scanners);
        this.stats = b.stats;
    }

    public static Builder newWiring() { return new BuilderImpl(); }

    public interface Builder {
        Builder entities(String table, EntityRetriever retriever);
        Builder links(String fromTable, LinkRetriever retriever);   // per-table or...
        Builder defaultLinks(LinkRetriever retriever);              // ...one for all
        Builder cohort(String table, CohortRetriever retriever);    // optional
        Builder scanner(String table, TableScanner scanner);        // optional (CSC mode)
        Builder stats(StatsProvider stats);
        RetrieverWiring build();
    }

    public Optional<EntityRetriever> entityRetriever(String table) {
        return Optional.ofNullable(entities.get(table));
    }

    /** Link retriever for children living in {@code fromTable}, falling back to the default. */
    public Optional<LinkRetriever> linkRetriever(String fromTable) {
        LinkRetriever r = links.get(fromTable);
        return Optional.ofNullable(r != null ? r : defaultLinks);
    }

    public Optional<CohortRetriever> cohortRetriever(String table) {
        return Optional.ofNullable(cohorts.get(table));
    }

    public Optional<TableScanner> scanner(String table) {
        return Optional.ofNullable(scanners.get(table));
    }

    public Optional<StatsProvider> stats() { return Optional.ofNullable(stats); }

    private static final class BuilderImpl implements Builder {
        final Map<String, EntityRetriever> entities = new HashMap<>();
        final Map<String, LinkRetriever> links = new HashMap<>();
        LinkRetriever defaultLinks;
        final Map<String, CohortRetriever> cohorts = new HashMap<>();
        final Map<String, TableScanner> scanners = new HashMap<>();
        StatsProvider stats;

        @Override public Builder entities(String table, EntityRetriever retriever) {
            entities.put(Objects.requireNonNull(table), Objects.requireNonNull(retriever));
            return this;
        }
        @Override public Builder links(String fromTable, LinkRetriever retriever) {
            links.put(Objects.requireNonNull(fromTable), Objects.requireNonNull(retriever));
            return this;
        }
        @Override public Builder defaultLinks(LinkRetriever retriever) {
            this.defaultLinks = Objects.requireNonNull(retriever);
            return this;
        }
        @Override public Builder cohort(String table, CohortRetriever retriever) {
            cohorts.put(Objects.requireNonNull(table), Objects.requireNonNull(retriever));
            return this;
        }
        @Override public Builder scanner(String table, TableScanner scanner) {
            scanners.put(Objects.requireNonNull(table), Objects.requireNonNull(scanner));
            return this;
        }
        @Override public Builder stats(StatsProvider stats) { this.stats = stats; return this; }
        @Override public RetrieverWiring build() { return new RetrieverWiring(this); }
    }
}
