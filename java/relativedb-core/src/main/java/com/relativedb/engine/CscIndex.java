package com.relativedb.engine;

import com.relativedb.csc.NativeCsc;
import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.RetrieverWiring;
import com.relativedb.retrieve.Row;
import com.relativedb.retrieve.TableScanner;
import com.relativedb.retrieve.TemporalBound;
import com.relativedb.schema.RelativeDbSchema;
import com.relativedb.schema.LinkDef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Flow;

/**
 * CSC-mode materialized graph: each table's {@link TableScanner} is drained
 * once and rows get dense ids. Per-link adjacency — the build and the
 * time-bounded "latest {@code <=} anchor" children query — is delegated to the
 * shared C++ implementation ({@code csc_*} in {@code librt_c}) via
 * {@link NativeCsc}, the single source of truth across the Python/Java/Rust
 * bindings. This class keeps only the Java-side bookkeeping: row storage, the
 * id↔dense mapping, and the seed/cohort lookups. Dense child ids returned by
 * the native index point back into this index's own row lists.
 *
 * <p>{@code librt_c} is a hard dependency (the same native library the RT-J
 * model and PQL parser require): {@link #build} fails fast if it is absent.
 */
public final class CscIndex implements ContextSource {

    /** One scanned table: dense-id row store. */
    private static final class TableStore {
        final List<Row> rows = new ArrayList<>();
        final Map<EntityId, Integer> denseId = new HashMap<>();
    }

    private final RelativeDbSchema schema;
    private final RetrieverWiring wiring;
    private final Map<String, TableStore> tables = new LinkedHashMap<>();
    private final Map<LinkDef, NativeCsc> adjacency = new LinkedHashMap<>();
    private final Map<LinkDef, Integer> edgeCounts = new LinkedHashMap<>();

    private CscIndex(RelativeDbSchema schema, RetrieverWiring wiring) {
        this.schema = schema;
        this.wiring = wiring;
    }

    /** Drains every wired TableScanner and builds the native CSC adjacency. */
    public static CscIndex build(RelativeDbSchema schema, RetrieverWiring wiring, TemporalBound bound) {
        CscIndex index = new CscIndex(schema, wiring);
        index.load(bound);
        return index;
    }

    /** Rebuild from the scanners (the CSC snapshot is otherwise frozen). */
    public CscIndex refresh(TemporalBound bound) {
        return build(schema, wiring, bound);
    }

    private void load(TemporalBound bound) {
        for (var table : schema.tables()) {
            Optional<TableScanner> scanner = wiring.scanner(table.name());
            if (scanner.isEmpty()) continue;
            TableStore store = new TableStore();
            for (Row row : drain(scanner.get().scan(table.name(), bound))) {
                // Defensive temporal re-check even at load time (F24).
                if (!row.timestamp().map(bound::admits).orElse(true)) continue;
                Integer prev = store.denseId.putIfAbsent(row.id(), store.rows.size());
                if (prev == null) store.rows.add(row);
            }
            tables.put(table.name(), store);
        }
        for (LinkDef link : schema.links()) {
            TableStore children = tables.get(link.fromTable());
            TableStore parents = tables.get(link.toTable());
            if (children == null || parents == null) continue;
            adjacency.put(link, buildAdjacency(link, children, parents));
        }
    }

    /**
     * Extract this link's edges {@code (parentDense, childDense, ts)} and hand
     * them to the native index; the native side sorts and buckets them.
     * {@code ts} is the child row time in epoch seconds, {@code -inf} for static
     * rows (so they sort first and are admitted under every temporal bound).
     */
    private NativeCsc buildAdjacency(LinkDef link, TableStore children, TableStore parents) {
        int nChildren = children.rows.size();
        long[] ep = new long[nChildren];
        long[] ec = new long[nChildren];
        double[] et = new double[nChildren];
        int n = 0;
        for (int c = 0; c < nChildren; c++) {
            Row child = children.rows.get(c);
            EntityId pid = child.parents().get(link.fkColumn());
            if (pid == null) continue;
            Integer p = parents.denseId.get(pid);
            if (p == null) continue;   // dangling FK: edge dropped, row still scannable
            ep[n] = p;
            ec[n] = c;
            et[n] = epochSeconds(child);
            n++;
        }
        edgeCounts.put(link, n);
        NativeCsc index = new NativeCsc(parents.rows.size(),
                Arrays.copyOf(ep, n),
                Arrays.copyOf(ec, n),
                Arrays.copyOf(et, n));
        return index;
    }

    /** Row time as epoch seconds; static rows sort first ({@code -inf}). */
    private static double epochSeconds(Row row) {
        return row.timestamp()
                .map(t -> t.getEpochSecond() + t.getNano() / 1_000_000_000.0)
                .orElse(Double.NEGATIVE_INFINITY);
    }

    // ------------------------------------------------------------------
    //  ContextSource
    // ------------------------------------------------------------------

    @Override public List<Row> byIds(String table, List<EntityId> ids, TemporalBound bound) {
        TableStore store = requireTable(table);
        List<Row> out = new ArrayList<>(ids.size());
        for (EntityId id : ids) {
            Integer dense = store.denseId.get(id);
            if (dense == null) continue;
            Row row = store.rows.get(dense);
            if (row.timestamp().map(bound::admits).orElse(true)) out.add(row);
        }
        return out;
    }

    @Override public List<Row> children(LinkDef link, EntityId parentId, TemporalBound bound, int limit) {
        NativeCsc adj = adjacency.get(link);
        TableStore parents = requireTable(link.toTable());
        TableStore childStore = requireTable(link.fromTable());
        if (adj == null) return List.of();
        Integer p = parents.denseId.get(parentId);
        if (p == null) return List.of();

        double anchor = bound.asOf()
                .map(t -> t.getEpochSecond() + t.getNano() / 1_000_000_000.0)
                .orElse(Double.POSITIVE_INFINITY);
        long[] denseChildren = adj.children(p, anchor, limit);
        List<Row> out = new ArrayList<>(denseChildren.length);
        for (long ci : denseChildren) {
            out.add(childStore.rows.get((int) ci));
        }
        return out;
    }

    @Override public Optional<List<EntityId>> cohort(String table, EntityId anchor,
                                                     TemporalBound bound, int limit) {
        var wired = wiring.cohortRetriever(table);
        if (wired.isPresent()) {
            return Optional.of(wired.get().fetchCohort(table, anchor, bound, limit)
                    .toCompletableFuture().join());
        }
        // Same-table sampling is an array scan in CSC mode.
        TableStore store = tables.get(table);
        if (store == null) return Optional.empty();
        List<EntityId> out = new ArrayList<>();
        for (Row row : store.rows) {
            if (out.size() >= limit) break;
            if (!row.id().equals(anchor)) out.add(row.id());
        }
        return Optional.of(out);
    }

    /** Number of edges materialized for {@code link} (test/inspection hook). */
    public int edgeCount(LinkDef link) {
        return edgeCounts.getOrDefault(link, 0);
    }

    /** Number of rows materialized for {@code table}. */
    public int rowCount(String table) {
        TableStore s = tables.get(table);
        return s == null ? 0 : s.rows.size();
    }

    private TableStore requireTable(String table) {
        TableStore s = tables.get(table);
        if (s == null) {
            throw new IllegalStateException("CSC mode: no TableScanner wired for table '"
                    + table + "'");
        }
        return s;
    }

    private static List<Row> drain(Flow.Publisher<Row> publisher) {
        List<Row> rows = new ArrayList<>();
        CompletableFuture<Void> done = new CompletableFuture<>();
        publisher.subscribe(new Flow.Subscriber<>() {
            @Override public void onSubscribe(Flow.Subscription s) { s.request(Long.MAX_VALUE); }
            @Override public void onNext(Row row) { rows.add(row); }
            @Override public void onError(Throwable t) { done.completeExceptionally(t); }
            @Override public void onComplete() { done.complete(null); }
        });
        done.join();
        return rows;
    }
}
