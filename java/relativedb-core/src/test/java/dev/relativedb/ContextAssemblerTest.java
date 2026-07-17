package dev.relativedb;

import dev.relativedb.engine.RelativeDbEngine;
import dev.relativedb.engine.ContextGraph;
import dev.relativedb.engine.ContextPolicy;
import dev.relativedb.retrieve.EntityId;
import dev.relativedb.retrieve.RetrieverWiring;
import dev.relativedb.retrieve.Row;
import dev.relativedb.retrieve.TemporalBound;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static dev.relativedb.TestData.*;
import static org.junit.jupiter.api.Assertions.*;

/** RETRIEVER-mode hop-loop tests: temporal bound, newest-first, fanouts. */
class ContextAssemblerTest {

    private Store store;

    @BeforeEach
    void setUp() {
        store = new Store();
        store.customers.add(customer(1, 34));
        store.customers.add(customer(2, 55));
        store.orders.add(order(100, 1, 2, "2026-01-01T00:00:00Z"));
        store.orders.add(order(101, 1, 5, "2026-03-01T00:00:00Z"));
        store.orders.add(order(102, 1, 7, "2026-06-01T00:00:00Z"));
        store.orders.add(order(103, 1, 9, "2026-08-01T00:00:00Z"));  // future vs anchor below
        store.orders.add(order(200, 2, 1, "2026-02-01T00:00:00Z"));
    }

    private RelativeDbEngine engine(ContextPolicy policy, RetrieverWiring wiring) {
        return RelativeDbEngine.newEngine(SCHEMA, wiring).contextPolicy(policy).build();
    }

    private RetrieverWiring honestWiring() {
        return RetrieverWiring.newWiring()
                .entities("customers", store::byIds)
                .entities("orders", store::byIds)
                .defaultLinks(store::children)
                .build();
    }

    @Test
    void temporalBoundExcludesNewerRows() {
        RelativeDbEngine engine = engine(ContextPolicy.defaults(), honestWiring());
        TemporalBound bound = TemporalBound.atOrBefore(Instant.parse("2026-07-01T00:00:00Z"));

        ContextGraph ctx = engine.assembleContext("customers", EntityId.of(1L), bound);

        assertTrue(ctx.contains("orders", EntityId.of(102L)));
        assertFalse(ctx.contains("orders", EntityId.of(103L)),
                "a row newer than the bound must never enter context");
    }

    @Test
    void buggyRetrieverCannotLeakTheFuture() {
        // A retriever that IGNORES the bound and returns everything, including
        // the future row 103 — the engine's defensive re-check must drop it.
        RetrieverWiring buggy = RetrieverWiring.newWiring()
                .entities("customers", (t, ids, b) -> store.byIds(t, ids, TemporalBound.unbounded()))
                .entities("orders", (t, ids, b) -> store.byIds(t, ids, TemporalBound.unbounded()))
                .defaultLinks((link, parent, b, limit) ->
                        store.children(link, parent, TemporalBound.unbounded(), limit))
                .build();
        RelativeDbEngine engine = engine(ContextPolicy.defaults(), buggy);
        TemporalBound bound = TemporalBound.atOrBefore(Instant.parse("2026-07-01T00:00:00Z"));

        ContextGraph ctx = engine.assembleContext("customers", EntityId.of(1L), bound);

        assertTrue(ctx.contains("orders", EntityId.of(102L)));
        assertFalse(ctx.contains("orders", EntityId.of(103L)),
                "defense in depth (F24): future rows from a buggy retriever must be dropped");
    }

    @Test
    void newestFirstChildSelectionUnderFanout() {
        // Fanout of 2: of orders 100/101/102 (past of the bound), the two newest
        // (102, 101) must win.
        RelativeDbEngine engine = engine(
                ContextPolicy.newPolicy().fanouts(2).build(), honestWiring());
        TemporalBound bound = TemporalBound.atOrBefore(Instant.parse("2026-07-01T00:00:00Z"));

        ContextGraph ctx = engine.assembleContext("customers", EntityId.of(1L), bound);

        assertTrue(ctx.contains("orders", EntityId.of(102L)));
        assertTrue(ctx.contains("orders", EntityId.of(101L)));
        assertFalse(ctx.contains("orders", EntityId.of(100L)), "oldest order must be cut by the fanout");
    }

    @Test
    void bfsWidthActsAsUniformFanout() {
        RelativeDbEngine engine = engine(
                ContextPolicy.newPolicy().bfsWidth(1).build(), honestWiring());
        ContextGraph ctx = engine.assembleContext("customers", EntityId.of(1L),
                TemporalBound.unbounded());
        long orderCount = ctx.rows().stream().filter(r -> r.table().equals("orders")).count();
        assertEquals(1, orderCount);
        assertTrue(ctx.contains("orders", EntityId.of(103L)), "the single slot goes to the newest");
    }

    @Test
    void parentsAreAlwaysFollowed() {
        // Seed on an ORDER: its customer parent must be pulled in.
        RelativeDbEngine engine = engine(ContextPolicy.defaults(), honestWiring());
        ContextGraph ctx = engine.assembleContext("orders", EntityId.of(100L),
                TemporalBound.unbounded());
        assertTrue(ctx.contains("customers", EntityId.of(1L)));
    }

    @Test
    void cellBudgetCapsContext() {
        // Every row has exactly 1 cell; budget of 2 = seed + one more row.
        RelativeDbEngine engine = engine(
                ContextPolicy.newPolicy().maxContextCells(2).build(), honestWiring());
        ContextGraph ctx = engine.assembleContext("customers", EntityId.of(1L),
                TemporalBound.unbounded());
        assertTrue(ctx.totalCells() <= 2, "totalCells=" + ctx.totalCells());
    }

    @Test
    void cohortRowsJoinContextWhenConfigured() {
        RetrieverWiring wiring = RetrieverWiring.newWiring()
                .entities("customers", store::byIds)
                .entities("orders", store::byIds)
                .defaultLinks(store::children)
                .cohort("customers", (table, anchor, bound, limit) ->
                        CompletableFuture.completedFuture(List.of(EntityId.of(2L))))
                .build();
        RelativeDbEngine engine = engine(
                ContextPolicy.newPolicy().cohortSize(4).build(), wiring);
        ContextGraph ctx = engine.assembleContext("customers", EntityId.of(1L),
                TemporalBound.unbounded());
        assertTrue(ctx.contains("customers", EntityId.of(2L)));
    }

    @Test
    void seedRowComesFirst() {
        RelativeDbEngine engine = engine(ContextPolicy.defaults(), honestWiring());
        ContextGraph ctx = engine.assembleContext("customers", EntityId.of(1L),
                TemporalBound.unbounded());
        Row first = ctx.rows().get(0);
        assertEquals("customers", first.table());
        assertEquals(EntityId.of(1L), first.id());
    }
}
