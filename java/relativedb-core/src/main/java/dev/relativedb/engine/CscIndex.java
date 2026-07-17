package dev.relativedb.engine;

import dev.relativedb.retrieve.EntityId;
import dev.relativedb.retrieve.RetrieverWiring;
import dev.relativedb.retrieve.Row;
import dev.relativedb.retrieve.TableScanner;
import dev.relativedb.retrieve.TemporalBound;
import dev.relativedb.schema.RelativeDbSchema;
import dev.relativedb.schema.LinkDef;

import java.time.Instant;
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
 * once, rows get dense ids, and per-link adjacency is stored as
 * {@code colptr}/{@code row} arrays with neighbor lists sorted by timestamp.
 * "Latest w children with time ≤ anchor" is then a
 * binary search + tail slice per node — no per-hop I/O.
 */
public final class CscIndex implements ContextSource {

    /** One scanned table: dense-id row store. */
    private static final class TableStore {
        final List<Row> rows = new ArrayList<>();
        final Map<EntityId, Integer> denseId = new HashMap<>();
    }

    /** One link's CSC adjacency (children of each parent, time-sorted ascending). */
    static final class Adjacency {
        int[] colptr;     // length nParents + 1
        int[] childRow;   // dense child row ids, grouped by parent, sorted by time asc
        long[] childTime; // epoch millis aligned with childRow (MIN_VALUE = no timestamp)
    }

    private final RelativeDbSchema schema;
    private final RetrieverWiring wiring;
    private final Map<String, TableStore> tables = new LinkedHashMap<>();
    private final Map<LinkDef, Adjacency> adjacency = new LinkedHashMap<>();

    private CscIndex(RelativeDbSchema schema, RetrieverWiring wiring) {
        this.schema = schema;
        this.wiring = wiring;
    }

    /** Drains every wired TableScanner and builds the CSC arrays. */
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

    private static Adjacency buildAdjacency(LinkDef link, TableStore children, TableStore parents) {
        int nParents = parents.rows.size();
        int[] counts = new int[nParents];
        for (Row child : children.rows) {
            EntityId pid = child.parents().get(link.fkColumn());
            if (pid == null) continue;
            Integer p = parents.denseId.get(pid);
            if (p != null) counts[p]++;
        }
        Adjacency adj = new Adjacency();
        adj.colptr = new int[nParents + 1];
        for (int i = 0; i < nParents; i++) adj.colptr[i + 1] = adj.colptr[i] + counts[i];
        int nEdges = adj.colptr[nParents];
        adj.childRow = new int[nEdges];
        adj.childTime = new long[nEdges];

        int[] cursor = Arrays.copyOf(adj.colptr, nParents);
        for (int c = 0; c < children.rows.size(); c++) {
            Row child = children.rows.get(c);
            EntityId pid = child.parents().get(link.fkColumn());
            if (pid == null) continue;
            Integer p = parents.denseId.get(pid);
            if (p == null) continue;
            int pos = cursor[p]++;
            adj.childRow[pos] = c;
            adj.childTime[pos] = child.timestamp().map(Instant::toEpochMilli).orElse(Long.MIN_VALUE);
        }
        // Sort each parent's neighbor list by timestamp ascending.
        for (int p = 0; p < nParents; p++) {
            sortByTime(adj, adj.colptr[p], adj.colptr[p + 1]);
        }
        return adj;
    }

    /** Insertion sort on the (childTime, childRow) parallel arrays in [from, to). */
    private static void sortByTime(Adjacency adj, int from, int to) {
        for (int i = from + 1; i < to; i++) {
            long t = adj.childTime[i];
            int r = adj.childRow[i];
            int j = i - 1;
            while (j >= from && adj.childTime[j] > t) {
                adj.childTime[j + 1] = adj.childTime[j];
                adj.childRow[j + 1] = adj.childRow[j];
                j--;
            }
            adj.childTime[j + 1] = t;
            adj.childRow[j + 1] = r;
        }
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
        Adjacency adj = adjacency.get(link);
        TableStore parents = requireTable(link.toTable());
        TableStore childStore = requireTable(link.fromTable());
        if (adj == null) return List.of();
        Integer p = parents.denseId.get(parentId);
        if (p == null) return List.of();

        int lo = adj.colptr[p];
        int hi = adj.colptr[p + 1];
        long anchor = bound.asOf().map(Instant::toEpochMilli).orElse(Long.MAX_VALUE);
        // Binary search: first index in [lo, hi) with time > anchor.
        int cut = upperBound(adj.childTime, lo, hi, anchor);
        // Tail slice = the newest admissible children; emit newest-first.
        int n = Math.min(limit, cut - lo);
        List<Row> out = new ArrayList<>(n);
        for (int i = cut - 1; i >= cut - n; i--) {
            out.add(childStore.rows.get(adj.childRow[i]));
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
        Adjacency adj = adjacency.get(link);
        return adj == null ? 0 : adj.childRow.length;
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

    private static int upperBound(long[] a, int lo, int hi, long key) {
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (a[mid] <= key) lo = mid + 1; else hi = mid;
        }
        return lo;
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
