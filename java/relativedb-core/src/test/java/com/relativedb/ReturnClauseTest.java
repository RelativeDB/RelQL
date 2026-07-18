package com.relativedb;

import com.relativedb.engine.ExecutionInput;
import com.relativedb.engine.PredictionResult;
import com.relativedb.engine.PredictionResult.EntityPrediction;
import com.relativedb.engine.PredictionResult.Interval;
import com.relativedb.engine.RelativeDbEngine;
import com.relativedb.model.ModelBackend;
import com.relativedb.model.ModelCapabilities;
import com.relativedb.model.ModelOutput;
import com.relativedb.model.TokenBatch;
import com.relativedb.query.TaskType;
import com.relativedb.retrieve.RetrieverWiring;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

import static com.relativedb.TestData.*;
import static org.junit.jupiter.api.Assertions.*;

/** End-to-end execution of the {@code RETURN <output>} clause. */
class ReturnClauseTest {

    private static final Instant ANCHOR = Instant.parse("2026-04-01T00:00:00Z");

    private RetrieverWiring wiring;

    /** Scripted backend: fixed binary p and regression value. */
    private final ModelBackend fakeBackend = new ModelBackend() {
        @Override public ModelCapabilities capabilities() { return ModelCapabilities.all(8192); }
        @Override public CompletionStage<ModelOutput> score(TokenBatch batch, TaskType taskType) {
            return CompletableFuture.completedFuture(
                    taskType == TaskType.BINARY_CLASSIFICATION
                            ? ModelOutput.binary(0.83) : ModelOutput.regression(12.5));
        }
    };

    @BeforeEach
    void setUp() {
        Store store = new Store();
        store.customers.add(customer(1, 34));
        // Three consecutive trailing 30-day windows before the anchor with
        // increasing summed qty: 10, 30, 50 (oldest -> newest).
        store.orders.add(order(200, 1, 5, "2026-01-15T00:00:00Z"));   // window k=3
        store.orders.add(order(201, 1, 5, "2026-01-20T00:00:00Z"));   // window k=3  -> 10
        store.orders.add(order(202, 1, 30, "2026-02-15T00:00:00Z"));  // window k=2  -> 30
        store.orders.add(order(203, 1, 50, "2026-03-15T00:00:00Z"));  // window k=1  -> 50
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
                + "FOR customers.customer_id = 1 RETURN CLASS");
        assertEquals("true", p.predictedClass().orElseThrow());   // p=0.83 >= 0.5
        assertTrue(p.probability().isEmpty(), "CLASS returns a label, not the score");
    }

    @Test
    void returnDistributionGivesTwoEntryClassDistribution() {
        EntityPrediction p = runOne(
                "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                + "FOR customers.customer_id = 1 RETURN DISTRIBUTION");
        Map<String, Double> dist = p.classProbs();
        assertEquals(2, dist.size());
        assertEquals(0.83, dist.get("true"), 1e-9);
        assertEquals(0.17, dist.get("false"), 1e-9);
    }

    @Test
    void returnExpectedValueOnBinaryIsProbability() {
        EntityPrediction p = runOne(
                "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
                + "FOR customers.customer_id = 1 RETURN EXPECTED VALUE");
        assertEquals(0.83, p.value().orElseThrow(), 1e-9);
    }

    @Test
    void returnQuantilesGivesMonotonicEntries() {
        EntityPrediction p = runOne(
                "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                + "FOR customers.customer_id = 1 RETURN QUANTILES (0.1, 0.5, 0.9)");
        Map<Double, Double> q = p.quantiles();
        assertEquals(3, q.size());
        double q10 = q.get(0.1), q50 = q.get(0.5), q90 = q.get(0.9);
        // Sample [10, 30, 50] -> linear-interpolation quantiles 14, 30, 46.
        assertEquals(14.0, q10, 1e-9);
        assertEquals(30.0, q50, 1e-9);
        assertEquals(46.0, q90, 1e-9);
        assertTrue(q10 <= q50 && q50 <= q90, "quantiles must be monotonic non-decreasing");
        // The point estimate still comes from the model backend.
        assertEquals(12.5, p.value().orElseThrow(), 1e-9);
    }

    @Test
    void returnIntervalGivesOrderedBounds() {
        EntityPrediction p = runOne(
                "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                + "FOR customers.customer_id = 1 RETURN INTERVAL 80%");
        Interval iv = p.interval().orElseThrow();
        assertTrue(iv.lower() <= iv.upper(), "interval lower <= upper");
        // 80% central interval -> quantiles 0.1 and 0.9 of [10, 30, 50] = 14, 46.
        assertEquals(14.0, iv.lower(), 1e-9);
        assertEquals(46.0, iv.upper(), 1e-9);
    }

    @Test
    void defaultOutputUnchangedWhenNoReturnClause() {
        EntityPrediction p = runOne(
                "PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR customers.customer_id = 1");
        assertEquals(12.5, p.value().orElseThrow(), 1e-9);
        assertTrue(p.predictedClass().isEmpty());
        assertTrue(p.quantiles().isEmpty());
        assertTrue(p.interval().isEmpty());
    }
}
