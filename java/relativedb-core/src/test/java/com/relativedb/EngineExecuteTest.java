package com.relativedb;

import com.relativedb.engine.RelativeDbEngine;
import com.relativedb.engine.ExecutionInput;
import com.relativedb.engine.PredictionResult;
import com.relativedb.model.TokenBatch;
import com.relativedb.query.TaskType;
import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.RetrieverWiring;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.concurrent.CompletionException;

import static com.relativedb.TestData.*;
import static org.junit.jupiter.api.Assertions.*;

/** End-to-end execute() with the shared deterministic stub backend. */
class EngineExecuteTest {

    private Store store;
    private RetrieverWiring wiring;
    private final StubBackend fakeBackend = new StubBackend();

    @BeforeEach
    void setUp() {
        store = new Store();
        store.customers.add(customer(1, 34));
        store.orders.add(order(100, 1, 2, "2026-01-01T00:00:00Z"));
        store.orders.add(order(103, 1, 9, "2026-08-01T00:00:00Z"));  // after anchor
        wiring = RetrieverWiring.newWiring()
                .entities("customers", store::byIds)
                .entities("orders", store::byIds)
                .defaultLinks(store::children)
                .build();
    }

    @Test
    void quickstartChurnQueryExecutes() {
        RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring)
                .modelBackend(fakeBackend)
                .build();

        PredictionResult churn = engine.execute(ExecutionInput.newInput()
                .query("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id")
                .anchorTime(Instant.parse("2026-07-01T00:00:00Z"))
                .entityIds(List.of(1L))
                .build()).toCompletableFuture().join();

        assertEquals(TaskType.BINARY_CLASSIFICATION, churn.taskType());
        assertEquals(1, churn.predictions().size());
        PredictionResult.EntityPrediction p = churn.predictions().get(0);
        assertEquals(EntityId.of(1L), p.id());
        assertEquals(0.83, p.probability().orElseThrow(), 1e-9);

        // The token batch fed to the model must respect the anchor: order 103
        // (2026-08-01) is newer than the anchor and must not be tokenized.
        TokenBatch batch = fakeBackend.lastBatch.get();
        assertNotNull(batch);
        assertTrue(batch.tokens().stream().noneMatch(t -> t.table().equals("orders")
                        && t.normalizedValue() == 9.0),
                "future order's qty must not reach the model");
    }

    @Test
    void queryEntitySelectorProvidesIds() {
        RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring)
                .modelBackend(fakeBackend)
                .build();
        PredictionResult r = engine.execute(ExecutionInput.newInput()
                .query("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id")
                .entityIds(List.of(1L))
                .build()).toCompletableFuture().join();
        assertEquals(TaskType.REGRESSION, r.taskType());
        assertEquals(EntityId.of(1L), r.predictions().get(0).id());
        assertEquals(12.5, r.predictions().get(0).value().orElseThrow(), 1e-9);
    }

    @Test
    void forEachWithoutIdsFailsClearlyInRetrieverMode() {
        RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring)
                .modelBackend(fakeBackend)
                .build();
        CompletionException e = assertThrows(CompletionException.class, () ->
                engine.execute(ExecutionInput.newInput()
                        .query("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id")
                        .build()).toCompletableFuture().join());
        assertTrue(e.getCause().getMessage().contains("entityIds"));
    }

    @Test
    void missingBackendFailsWithRoutingHint() {
        RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring).build();
        CompletionException e = assertThrows(CompletionException.class, () ->
                engine.execute(ExecutionInput.newInput()
                        .query("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id")
                        .entityIds(List.of(1L))
                        .build()).toCompletableFuture().join());
        assertTrue(e.getCause() instanceof IllegalStateException);
        assertTrue(e.getCause().getMessage().contains("hf://stanford-star/rt-j/regression"),
                "error should name the checkpoint the routing selected");
    }
}
