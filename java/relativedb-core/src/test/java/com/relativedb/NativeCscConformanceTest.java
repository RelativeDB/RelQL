package com.relativedb;

import com.relativedb.csc.NativeCsc;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Cross-language conformance: the shared C++ CSC adjacency ({@code csc_build} /
 * {@code csc_children} in {@code librt_c}) must agree with a brute-force
 * reference of the "latest {@code <= anchor} children" semantics, node for node,
 * across randomized graphs. Mirrors {@code python/tests/test_native_csc.py}.
 *
 * <p>Skips cleanly (JUnit {@code Assume}) when {@code librt_c} is not built.
 */
class NativeCscConformanceTest {

    private static final double NEG_INF = Double.NEGATIVE_INFINITY;

    @BeforeAll
    static void requireNative() {
        Assumptions.assumeTrue(NativeCsc.available(),
                "librt_c not built (run cmake in cpp/); native CSC unavailable");
    }

    /** Brute-force reference matching relativedb.csc.CscIndex.children. */
    private static long[] refChildren(long parent, double anchor, int limit,
                                      long[] ep, long[] ec, double[] et) {
        if (limit <= 0) return new long[0];
        // Indices in the parent's bucket, stable-sorted by ts ascending.
        List<Integer> bucket = new ArrayList<>();
        for (int i = 0; i < ep.length; i++) {
            if (ep[i] == parent) bucket.add(i);
        }
        bucket.sort((a, b) -> Double.compare(et[a], et[b]));  // stable: ties keep order
        List<Integer> admitted = new ArrayList<>();
        for (int i : bucket) {
            if (et[i] <= anchor) admitted.add(i);
        }
        int take = Math.min(limit, admitted.size());
        long[] out = new long[take];
        // Last `take`, reversed to newest-first.
        for (int k = 0; k < take; k++) {
            int idx = admitted.get(admitted.size() - 1 - k);
            out[k] = ec[idx];
        }
        return out;
    }

    private static Object[] randomGraph(Random rng, int nParents, int nEdges) {
        long[] ep = new long[nEdges];
        long[] ec = new long[nEdges];
        double[] et = new double[nEdges];
        for (int i = 0; i < nEdges; i++) {
            ep[i] = rng.nextInt(nParents);
            ec[i] = rng.nextInt(100_000);
            int roll = rng.nextInt(10);
            if (roll == 0) {
                et[i] = NEG_INF;                    // static row
            } else if (roll <= 3) {
                et[i] = rng.nextInt(6);             // heavy ties
            } else {
                et[i] = rng.nextInt(1001);
            }
        }
        return new Object[] { ep, ec, et };
    }

    static Stream<Arguments> graphs() {
        return Stream.of(
                Arguments.of(1, 1, 20),
                Arguments.of(2, 8, 200),
                Arguments.of(3, 50, 2000),
                Arguments.of(4, 200, 50),      // sparse: many parents, no edges
                Arguments.of(5, 4, 3000));     // dense: heavy ties per parent
    }

    @ParameterizedTest(name = "seed={0} parents={1} edges={2}")
    @MethodSource("graphs")
    @DisplayName("native CSC children matches brute-force reference")
    void matchesReference(int seed, int nParents, int nEdges) {
        Random rng = new Random(seed);
        Object[] g = randomGraph(rng, nParents, nEdges);
        long[] ep = (long[]) g[0];
        long[] ec = (long[]) g[1];
        double[] et = (double[]) g[2];

        double[] anchors = { NEG_INF, Double.POSITIVE_INFINITY,
                -2, -1, 0, 1, 2, 3, 4, 5, 6, 7 };

        try (NativeCsc idx = new NativeCsc(nParents, ep, ec, et)) {
            for (int t = 0; t < 2000; t++) {
                long parent = rng.nextInt(nParents + 3) - 2;   // includes out-of-range
                int limit = rng.nextInt(10) - 1;               // includes 0 and > bucket
                double anchor = (t % 3 == 0)
                        ? rng.nextInt(1011) - 5
                        : anchors[rng.nextInt(anchors.length)];
                long[] got = idx.children(parent, anchor, limit);
                long[] want = refChildren(parent, anchor, limit, ep, ec, et);
                assertArrayEquals(want, got, () -> "mismatch: parent=" + parent
                        + " anchor=" + anchor + " limit=" + limit);
            }
        }
    }

    @Test
    @DisplayName("CSC edge cases")
    void edgeCases() {
        try (NativeCsc empty = new NativeCsc(5, new long[0], new long[0], new double[0])) {
            assertArrayEquals(new long[0], empty.children(2, 1e9, 4));
        }
        try (NativeCsc idx = new NativeCsc(1, new long[] {0, 0},
                new long[] {7, 8}, new double[] {1.0, 2.0})) {
            assertArrayEquals(new long[0], idx.children(0, 100.0, 0));
            assertArrayEquals(new long[0], idx.children(0, 100.0, -3));
        }
        try (NativeCsc idx2 = new NativeCsc(1, new long[] {0, 0},
                new long[] {9, 10}, new double[] {NEG_INF, 5.0})) {
            assertArrayEquals(new long[] {9}, idx2.children(0, -100.0, 4)); // only static row
            assertArrayEquals(new long[] {10, 9}, idx2.children(0, 100.0, 4)); // newest-first
        }
    }
}
