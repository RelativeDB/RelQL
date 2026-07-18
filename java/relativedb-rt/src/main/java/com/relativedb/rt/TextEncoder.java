package com.relativedb.rt;

/**
 * SPI producing the 384-dim text embeddings the RT checkpoints were trained
 * with (KB F13/F14: all-MiniLM-L12-v2, shared by both variants). The backend
 * uses it for BOTH embedding channels: cell text values ({@code text_v}) and
 * column names ({@code col_name_v}).
 *
 * <p>A real MiniLM encoder (ONNX / sentence-transformers bridge) is a separate
 * concern and deliberately NOT part of this module — implementations here are
 * expected to be thin (precomputed lookup tables, remote embedding services,
 * ...). See {@link PrecomputedEncoder} for the lookup-table implementation.
 */
public interface TextEncoder {

    /** MiniLM-L12-v2 embedding width; the RT input contract is fixed to this. */
    int DIMENSION = 384;

    /** Returns the 384-dim embedding of {@code text} (never null, length 384). */
    float[] encode(String text);
}
