package com.relativedb.rt;

import java.util.HashMap;
import java.util.Map;

/**
 * {@link TextEncoder} backed by a precomputed {@code string -> float[384]}
 * table. Suitable when the set of texts (column names, categorical values) is
 * known ahead of time and embedded offline with all-MiniLM-L12-v2.
 * Unknown strings fail fast with the missing key in the message.
 */
public final class PrecomputedEncoder implements TextEncoder {

    private final Map<String, float[]> table;

    public PrecomputedEncoder(Map<String, float[]> table) {
        Map<String, float[]> copy = new HashMap<>();
        table.forEach((k, v) -> {
            if (v == null || v.length != DIMENSION) {
                throw new IllegalArgumentException(
                    "Embedding for '" + k + "' must have length " + DIMENSION
                    + " (got " + (v == null ? "null" : v.length) + ")");
            }
            copy.put(k, v.clone());
        });
        this.table = copy;
    }

    @Override
    public float[] encode(String text) {
        float[] v = table.get(text);
        if (v == null) {
            throw new IllegalArgumentException(
                "PrecomputedEncoder has no embedding for '" + text + "' ("
                + table.size() + " entries). Add it to the table or use a real "
                + "MiniLM encoder implementation of TextEncoder.");
        }
        return v;
    }
}
