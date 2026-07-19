package com.relativedb.rt;

import com.sun.jna.Library;
import com.sun.jna.Pointer;

/**
 * Raw JNA mapping of the {@code rt_c.h} C ABI (the golden-verified C++ RT-J
 * inference engine, {@code librt_c}).
 *
 * <p>All arrays are caller-owned, densely packed, little-endian, and — per the
 * ABI contract — the RAW PRE-SORT token arrays: the engine sorts and builds its
 * sparse attention masks internally.
 *
 * <p>Array lengths for a forward pass over B batch rows of S tokens:
 * <ul>
 *   <li>{@code B*S}: nodeIdxs, colIdxs, tableIdxs, semTypes (int64);
 *       isPadding, isTarget (uint8); numberV, datetimeV, booleanV (float32)</li>
 *   <li>{@code B*S*5}: f2p (int64, -1 = no parent)</li>
 *   <li>{@code B*S*384}: textV, colNameV (float32, MiniLM-L12-v2 embeddings)</li>
 * </ul>
 */
public interface RtC extends Library {

    /** Returns NULL on failure; message written into {@code err}. */
    Pointer rt_model_load(String safetensorsPath, byte[] err, long errlen);

    void rt_model_free(Pointer model);

    long rt_model_num_params(Pointer model);

    /**
     * Returns 0 on success, nonzero on error (message in {@code err}).
     * {@code outTargetScores} has length B; {@code nThreads <= 0} selects
     * hardware concurrency.
     */
    int rt_forward(Pointer model, int b, int s,
                   long[] nodeIdxs, long[] f2p,
                   long[] colIdxs, long[] tableIdxs,
                   byte[] isPadding, long[] semTypes,
                   byte[] isTarget, float[] numberV,
                   float[] datetimeV, float[] booleanV,
                   float[] textV, float[] colNameV,
                   int nThreads, float[] outTargetScores,
                   byte[] err, long errlen);

    /**
     * Extended forward: identical to {@link #rt_forward} plus one trailing
     * nullable output {@code outTargetText} of length {@code B*384} — the TEXT
     * decoder head (approximate MiniLM-L12-v2 embedding) summed over each row's
     * target cell(s), using the SAME target-cell selection as the number head.
     * NOT L2-normalized (the caller normalizes before matching). When
     * {@code outTargetText} is {@code null} this is byte-identical to
     * {@link #rt_forward}. {@code outTargetScores} (length B) is the same
     * number-head output {@code rt_forward} returns. Returns 0 on success.
     */
    int rt_forward_ex(Pointer model, int b, int s,
                      long[] nodeIdxs, long[] f2p,
                      long[] colIdxs, long[] tableIdxs,
                      byte[] isPadding, long[] semTypes,
                      byte[] isTarget, float[] numberV,
                      float[] datetimeV, float[] booleanV,
                      float[] textV, float[] colNameV,
                      int nThreads, float[] outTargetScores,
                      float[] outTargetText, byte[] err, long errlen);
}
