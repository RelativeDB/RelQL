package com.relativedb;

import com.relativedb.engine.RelativeDbEngine;
import com.relativedb.engine.ContextGraph;
import com.relativedb.engine.ContextPolicy;
import com.relativedb.engine.CscIndex;
import com.relativedb.engine.SamplerMode;
import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.RetrieverWiring;
import com.relativedb.retrieve.Row;
import com.relativedb.retrieve.TemporalBound;
import com.relativedb.schema.LinkDef;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;

import static com.relativedb.TestData.*;
import static org.junit.jupiter.api.Assertions.*;

/** CSC-mode: index construction from TableScanner streams + temporal queries. */
class CscIndexTest {

    private static final LinkDef LINK = SCHEMA.links().get(0); // orders.customer_id -> customers

    private Store store;
    private RetrieverWiring wiring;

    @BeforeEach
    void setUp() {
        store = new Store();
        store.customers.add(customer(1, 34));
        store.customers.add(customer(2, 55));
        // Deliberately out of time order — the index must sort neighbor lists.
        store.orders.add(order(102, 1, 7, "2026-06-01T00:00:00Z"));
        store.orders.add(order(100, 1, 2, "2026-01-01T00:00:00Z"));
        store.orders.add(order(103, 1, 9, "2026-08-01T00:00:00Z"));
        store.orders.add(order(101, 1, 5, "2026-03-01T00:00:00Z"));
        store.orders.add(order(200, 2, 1, "2026-02-01T00:00:00Z"));
        wiring = RetrieverWiring.newWiring()
                .scanner("customers", store::scan)
                .scanner("orders", store::scan)
                .build();
    }

    @Test
    void buildsDenseRowsAndEdges() {
        CscIndex index = CscIndex.build(SCHEMA, wiring, TemporalBound.unbounded());
        assertEquals(2, index.rowCount("customers"));
        assertEquals(5, index.rowCount("orders"));
        assertEquals(5, index.edgeCount(LINK));
    }

    @Test
    void neighborListsAreTimeSortedNewestFirst() {
        CscIndex index = CscIndex.build(SCHEMA, wiring, TemporalBound.unbounded());
        List<Row> children = index.children(LINK, EntityId.of(1L), TemporalBound.unbounded(), 10);
        assertEquals(List.of(103L, 102L, 101L, 100L),
                children.stream().map(r -> r.id().raw()).toList());
    }

    @Test
    void boundIsBinarySearchedNotScanned() {
        CscIndex index = CscIndex.build(SCHEMA, wiring, TemporalBound.unbounded());
        TemporalBound bound = TemporalBound.atOrBefore(Instant.parse("2026-05-01T00:00:00Z"));
        List<Row> children = index.children(LINK, EntityId.of(1L), bound, 10);
        assertEquals(List.of(101L, 100L), children.stream().map(r -> r.id().raw()).toList());
    }

    @Test
    void limitTakesNewestAdmissible() {
        CscIndex index = CscIndex.build(SCHEMA, wiring, TemporalBound.unbounded());
        TemporalBound bound = TemporalBound.atOrBefore(Instant.parse("2026-07-01T00:00:00Z"));
        List<Row> children = index.children(LINK, EntityId.of(1L), bound, 2);
        assertEquals(List.of(102L, 101L), children.stream().map(r -> r.id().raw()).toList());
    }

    @Test
    void loadTimeBoundFiltersRows() {
        CscIndex index = CscIndex.build(SCHEMA, wiring,
                TemporalBound.atOrBefore(Instant.parse("2026-04-01T00:00:00Z")));
        assertEquals(3, index.rowCount("orders"));  // 100, 101, 200
        assertEquals(0, index.children(LINK, EntityId.of(1L), TemporalBound.unbounded(), 10)
                .stream().filter(r -> (Long) r.id().raw() >= 102).count());
    }

    @Test
    void cohortIsAnArrayScanInCscMode() {
        CscIndex index = CscIndex.build(SCHEMA, wiring, TemporalBound.unbounded());
        List<EntityId> cohort = index.cohort("customers", EntityId.of(1L),
                TemporalBound.unbounded(), 10).orElseThrow();
        assertEquals(List.of(EntityId.of(2L)), cohort);
    }

    @Test
    void cscEngineAssemblesContextWithoutRetrievers() {
        RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring)
                .samplerMode(SamplerMode.CSC)
                .contextPolicy(ContextPolicy.newPolicy().fanouts(2, 2).build())
                .build();
        TemporalBound bound = TemporalBound.atOrBefore(Instant.parse("2026-07-01T00:00:00Z"));

        ContextGraph ctx = engine.assembleContext("customers", EntityId.of(1L), bound);

        assertTrue(ctx.contains("orders", EntityId.of(102L)));
        assertTrue(ctx.contains("orders", EntityId.of(101L)));
        assertFalse(ctx.contains("orders", EntityId.of(103L)), "future row must not enter context");
        assertFalse(ctx.contains("orders", EntityId.of(100L)), "fanout 2 keeps only the newest two");
    }
}
