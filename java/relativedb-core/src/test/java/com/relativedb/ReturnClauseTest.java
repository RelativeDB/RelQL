package com.relativedb;

import com.relativedb.engine.ExecutionInput;
import com.relativedb.engine.PredictionResult;
import com.relativedb.engine.PredictionResult.EntityPrediction;
import com.relativedb.engine.RelativeDbEngine;
import com.relativedb.retrieve.RetrieverWiring;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletionException;

import static com.relativedb.TestData.*;
import static org.junit.jupiter.api.Assertions.*;

/** End-to-end execution of the {@code RETURN <output>} clause. */
class ReturnClauseTest {

    private static final Instant ANCHOR = Instant.parse("2026-04-01T00:00:00Z");

    private RetrieverWiring wiring;

    /** The shared deterministic stub: binary p=0.83, regression value=12.5. */
    private final StubBackend fakeBackend = new StubBackend();

    @BeforeEach
    void setUp() {
        Store store = new Store();
        store.customers.add(customer(1, 34));
        store.orders.add(order(200, 1, 5, "2026-01-15T00:00:00Z"));
        store.orders.add(order(203, 1, 50, "2026-03-15T00:00:00Z"));
        wiring = RetrieverWiring.newWiring()
                .entities("customers", store::byIds)
                .entities("orders", store::byIds)
                .defaultLinks(store::children)
                .build();
    }

    private EntityPrediction runOne(String pql) {
        RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring)
                .modelBackend(fakeBackend)
                .build();
        PredictionResult r = engine.execute(ExecutionInput.newInput()
                .query(pql)
                .anchorTime(ANCHOR)
                .entityIds(List.of(1L))
                .build()).toCompletableFuture().join();
        assertEquals(1, r.predictions().size());
        return r.predictions().get(0);
    }

    @Test
    void returnClassSetsHardLabel() {
        EntityPrediction p = runOne(
                "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                + "FOR EACH customers.customer_id RETURN CLASS");
        assertEquals("true", p.predictedClass().orElseThrow());   // p=0.83 >= 0.5
        assertTrue(p.probability().isEmpty(), "CLASS returns a label, not the score");
    }

    @Test
    void returnDistributionGivesTwoEntryClassDistribution() {
        EntityPrediction p = runOne(
                "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                + "FOR EACH customers.customer_id RETURN DISTRIBUTION");
        Map<String, Double> dist = p.classProbs();
        assertEquals(2, dist.size());
        assertEquals(0.83, dist.get("true"), 1e-9);
        assertEquals(0.17, dist.get("false"), 1e-9);
    }

    @Test
    void returnExpectedValueOnBinaryIsProbability() {
        EntityPrediction p = runOne(
                "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                + "FOR EACH customers.customer_id RETURN EXPECTED VALUE");
        assertEquals(0.83, p.value().orElseThrow(), 1e-9);
    }

    @Test
    void returnQuantilesIsUnsupportedWithoutAQuantileHead() {
        CompletionException e = assertThrows(CompletionException.class, () -> runOne(
                "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                + "FOR EACH customers.customer_id RETURN QUANTILES (0.1, 0.5, 0.9)"));
        assertTrue(e.getCause() instanceof UnsupportedOperationException);
        assertTrue(e.getCause().getMessage().contains("quantile/distribution head"),
                "error should explain the checkpoint lacks a quantile/distribution head");
    }

    @Test
    void returnIntervalIsUnsupportedWithoutADistributionHead() {
        CompletionException e = assertThrows(CompletionException.class, () -> runOne(
                "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                + "FOR EACH customers.customer_id RETURN INTERVAL 80%"));
        assertTrue(e.getCause() instanceof UnsupportedOperationException);
        assertTrue(e.getCause().getMessage().contains("quantile/distribution head"),
                "error should explain the checkpoint lacks a quantile/distribution head");
    }

    @Test
    void defaultOutputUnchangedWhenNoReturnClause() {
        EntityPrediction p = runOne(
                "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id");
        assertEquals(12.5, p.value().orElseThrow(), 1e-9);
        assertTrue(p.predictedClass().isEmpty());
        assertTrue(p.quantiles().isEmpty());
        assertTrue(p.interval().isEmpty());
    }
}
