package com.relativedb.rt;

import com.relativedb.model.ModelConfig;
import com.relativedb.model.ModelOutput;
import com.relativedb.model.TokenBatch;
import com.relativedb.query.TaskType;
import com.relativedb.schema.ValueType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Native multiclass ({@code rt_forward_ex} text head → L2-norm → cosine →
 * softmax) and ranking (per-candidate sigmoid → top-k) through
 * {@link RtNativeBackend} against the real classification checkpoint, with a
 * PrecomputedEncoder of random-but-fixed vectors. Checks the class distribution
 * is a proper, deterministic simplex and the ranking is ordered and complete.
 */
class RtMulticlassRankingTest {

    private static final List<String> CLASSES = List.of("premium", "basic", "gold", "enterprise");

    private static TokenBatch entityBatch(double age, double qtyA, double qtyB) {
        return TokenBatch.newBatch()
            .numeric(1, List.of(), "customers", "age", ValueType.NUMBER, age, false)
            .text(1, List.of(), "customers", "plan", "premium", false)
            .numeric(2, List.of(1), "orders", "qty", ValueType.NUMBER, qtyA, false)
            .numeric(3, List.of(1), "orders", "qty", ValueType.NUMBER, qtyB, false)
            // The masked label target cell (masked as TEXT by the multiclass path).
            .numeric(1, List.of(), "customers", "churned", ValueType.BOOLEAN, 0.0, true)
            .build();
    }

    private static PrecomputedEncoder fixedEncoder() {
        Map<String, float[]> table = new HashMap<>();
        List<String> keys = new ArrayList<>(List.of("age", "plan", "qty", "churned", "premium",
            // col_name_v schema phrases ("<column> of <table>").
            "age of customers", "plan of customers", "qty of orders", "churned of customers"));
        keys.addAll(CLASSES);
        for (String key : keys) table.put(key, fixedVector(key));
        return new PrecomputedEncoder(table);
    }

    private static float[] fixedVector(String key) {
        Random rnd = new Random(key.hashCode() * 2654435761L);
        float[] v = new float[TextEncoder.DIMENSION];
        for (int i = 0; i < v.length; i++) v[i] = (float) (rnd.nextGaussian() * 0.1);
        return v;
    }

    @Test
    void multiclassProducesADeterministicProperDistribution() {
        assumeTrue(RtNative.isAvailable(), "librt_c not available");
        assumeTrue(GoldenData.classificationCheckpointPresent(),
            "classification checkpoint not in local HF cache");

        try (RtNativeBackend backend = new RtNativeBackend(ModelConfig.defaults(), fixedEncoder())) {
            ModelOutput out = backend.classifyMulticlass(
                entityBatch(0.42, 1.3, -0.7), CLASSES, TaskType.MULTICLASS_CLASSIFICATION);

            Map<String, Double> probs = out.classProbs();
            assertEquals(CLASSES.size(), probs.size());
            assertEquals(CLASSES, new ArrayList<>(probs.keySet()),
                "distribution must be keyed in the given class order");
            double sum = 0.0;
            for (double pk : probs.values()) {
                assertTrue(pk > 0.0 && pk < 1.0 && Double.isFinite(pk),
                    "each class probability must be a finite simplex weight, got " + pk);
                sum += pk;
            }
            assertEquals(1.0, sum, 1e-9, "softmax must sum to 1");

            // Determinism.
            ModelOutput again = backend.classifyMulticlass(
                entityBatch(0.42, 1.3, -0.7), CLASSES, TaskType.MULTICLASS_CLASSIFICATION);
            assertEquals(probs, again.classProbs());
        }
    }

    @Test
    void rankingScoresEveryCandidateInDescendingOrder() {
        assumeTrue(RtNative.isAvailable(), "librt_c not available");
        assumeTrue(GoldenData.classificationCheckpointPresent(),
            "classification checkpoint not in local HF cache");

        try (RtNativeBackend backend = new RtNativeBackend(ModelConfig.defaults(), fixedEncoder())) {
            List<TokenBatch> candidates = List.of(
                entityBatch(0.42, 1.3, -0.7),
                entityBatch(0.42, -1.1, 2.0),
                entityBatch(0.42, 0.0, 0.0));
            List<String> ids = List.of("a10", "a20", "a30");

            ModelOutput out = backend.rankCandidates(candidates, ids, TaskType.MULTILABEL_RANKING);
            Map<String, Double> ranked = out.rankedScores();

            assertEquals(3, ranked.size(), "every candidate must be scored");
            assertTrue(ranked.keySet().containsAll(ids));
            double prev = Double.POSITIVE_INFINITY;
            for (double pk : ranked.values()) {
                assertTrue(pk > 0.0 && pk < 1.0 && Double.isFinite(pk),
                    "sigmoid score must be a probability, got " + pk);
                assertTrue(pk <= prev, "ranked scores must be non-increasing");
                prev = pk;
            }

            // Determinism.
            assertEquals(ranked, backend.rankCandidates(candidates, ids,
                TaskType.MULTILABEL_RANKING).rankedScores());
        }
    }
}
