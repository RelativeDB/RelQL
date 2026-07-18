package com.relativedb.rt;

import com.relativedb.model.ModelConfig;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Golden regression gate: feed the PRE-sort golden batch from cpp/testdata
 * straight into the JNA binding and match the PyTorch-verified target scores
 * for BOTH checkpoints. Skips (via assumptions) only when the dylib,
 * checkpoints, or testdata are absent.
 */
class RtGoldenForwardTest {

    private static final int B = 5;
    private static final int S = 16;
    private static final double TOL = 2e-3;

    private static final double[] EXPECTED_CLASSIFICATION =
        { -0.18470, -0.33108, +0.43363, -0.14449, +0.46848 };
    private static final double[] EXPECTED_REGRESSION =
        { -0.27052, -0.41538, +0.39998, -0.30649, +0.26804 };

    @Test
    void classificationCheckpointMatchesGolden() {
        runGolden(ModelConfig.DEFAULT_CLASSIFICATION_MODEL_URI,
            GoldenData.classificationCheckpointPresent(), EXPECTED_CLASSIFICATION);
    }

    @Test
    void regressionCheckpointMatchesGolden() {
        runGolden(ModelConfig.DEFAULT_REGRESSION_MODEL_URI,
            GoldenData.regressionCheckpointPresent(), EXPECTED_REGRESSION);
    }

    private void runGolden(String modelUri, boolean checkpointPresent, double[] expected) {
        assumeTrue(RtNative.isAvailable(), "librt_c not available");
        Path data = GoldenData.testdataDir();
        assumeTrue(data != null, "cpp/testdata not found");
        assumeTrue(checkpointPresent, "checkpoint not in local HF cache: " + modelUri);

        int n = B * S;
        long[] nodeIdxs = GoldenData.longs(data, "node_idxs.bin", n);
        long[] f2p = GoldenData.longs(data, "f2p_nbr_idxs.bin", n * 5);
        long[] colIdxs = GoldenData.longs(data, "col_name_idxs.bin", n);
        long[] tableIdxs = GoldenData.longs(data, "table_name_idxs.bin", n);
        byte[] isPadding = GoldenData.bytes(data, "is_padding.bin", n);
        long[] semTypes = GoldenData.longs(data, "sem_types.bin", n);
        byte[] isTarget = GoldenData.bytes(data, "is_targets.bin", n);
        float[] numberV = GoldenData.floats(data, "number_values.bin", n);
        float[] datetimeV = GoldenData.floats(data, "datetime_values.bin", n);
        float[] booleanV = GoldenData.floats(data, "boolean_values.bin", n);
        float[] textV = GoldenData.floats(data, "text_values.bin", n * 384);
        float[] colNameV = GoldenData.floats(data, "col_name_values.bin", n * 384);

        try (RtModel model = RtModel.load(CheckpointResolver.resolve(modelUri))) {
            assertTrue(model.numParams() > 0, "model reports no parameters");
            float[] scores = model.forward(B, S, nodeIdxs, f2p, colIdxs, tableIdxs,
                isPadding, semTypes, isTarget, numberV, datetimeV, booleanV,
                textV, colNameV, 0);
            assertEquals(B, scores.length);
            for (int i = 0; i < B; i++) {
                assertEquals(expected[i], scores[i], TOL,
                    "score[" + i + "] for " + modelUri);
            }
        }
    }
}
