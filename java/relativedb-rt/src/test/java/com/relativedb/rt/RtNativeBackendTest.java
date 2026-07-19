package com.relativedb.rt;

import com.relativedb.model.ModelConfig;
import com.relativedb.model.ModelOutput;
import com.relativedb.model.TokenBatch;
import com.relativedb.query.TaskType;
import com.relativedb.schema.ValueType;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Plumbing test: a hand-built TokenBatch through RtNativeBackend against the
 * real classification checkpoint, with a PrecomputedEncoder of random-but-fixed
 * vectors. Checks finite output and batch determinism.
 */
class RtNativeBackendTest {

    private static TokenBatch sampleBatch() {
        return TokenBatch.newBatch()
            .numeric(1, List.of(), "customers", "age", ValueType.NUMBER, 0.42, false)
            .numeric(1, List.of(), "customers", "signup_date", ValueType.DATETIME, -0.1, false)
            .text(1, List.of(), "customers", "plan", "premium", false)
            .numeric(2, List.of(1), "orders", "qty", ValueType.NUMBER, 1.3, false)
            .numeric(3, List.of(1), "orders", "qty", ValueType.NUMBER, -0.7, false)
            .numeric(1, List.of(), "customers", "churned", ValueType.BOOLEAN, 0.0, true)
            .build();
    }

    private static PrecomputedEncoder fixedEncoder() {
        Map<String, float[]> table = new HashMap<>();
        for (String key : new String[] { "age", "signup_date", "plan", "qty", "churned", "premium",
                // col_name_v schema phrases ("<column> of <table>").
                "age of customers", "signup_date of customers", "plan of customers",
                "qty of orders", "churned of customers" }) {
            table.put(key, fixedVector(key));
        }
        return new PrecomputedEncoder(table);
    }

    /** Random-but-fixed unit-scale vector, deterministic per key. */
    private static float[] fixedVector(String key) {
        Random rnd = new Random(key.hashCode() * 2654435761L);
        float[] v = new float[TextEncoder.DIMENSION];
        for (int i = 0; i < v.length; i++) v[i] = (float) (rnd.nextGaussian() * 0.1);
        return v;
    }

    @Test
    void scoresHandBuiltBatchAndIsDeterministicAcrossBatchRows() {
        assumeTrue(RtNative.isAvailable(), "librt_c not available");
        assumeTrue(GoldenData.classificationCheckpointPresent(),
            "classification checkpoint not in local HF cache");

        TokenBatch batch = sampleBatch();
        try (RtNativeBackend backend = new RtNativeBackend(ModelConfig.defaults(), fixedEncoder())) {
            ModelOutput out = backend.score(batch, TaskType.BINARY_CLASSIFICATION)
                .toCompletableFuture().join();
            assertTrue(Double.isFinite(out.probability()), "probability must be finite");
            assertTrue(out.probability() > 0.0 && out.probability() < 1.0,
                "sigmoid(logit) must be a probability, got " + out.probability());

            // Two identical batch rows in one forward pass score identically.
            float[] scores = backend.scoreAll(List.of(batch, batch),
                TaskType.BINARY_CLASSIFICATION);
            assertEquals(2, scores.length);
            assertTrue(Float.isFinite(scores[0]));
            assertEquals(scores[0], scores[1], 0.0f,
                "identical rows in one batch must yield identical logits");

            // And the single-batch path agrees with the batched path.
            assertEquals(RtNativeBackend.sigmoid(scores[0]), out.probability(), 1e-12);
        }
    }
}
