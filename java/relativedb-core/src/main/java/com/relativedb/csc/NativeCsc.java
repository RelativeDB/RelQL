package com.relativedb.csc;

import com.relativedb.nat.RtCLib;
import com.relativedb.nat.RtCNative;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;

import java.nio.charset.StandardCharsets;

/**
 * Wraps a native CSC adjacency index ({@code csc_build} / {@code csc_children}
 * / {@code csc_free} in {@code librt_c}). Build once from parallel edge arrays,
 * then answer many time-bounded "latest {@code <= anchor}" children queries.
 * The native handle is released on {@link #close()} and, defensively, on GC.
 *
 * <p>Java analogue of {@code python/src/relativedb/csc_native.py}. Ids are dense
 * integers; {@code ts} is epoch seconds ({@code -inf} for static rows so they
 * sort first).
 */
public final class NativeCsc implements AutoCloseable {

    private static final int ERR = 1024;

    private final RtCLib lib;
    private Pointer handle;

    /**
     * Build an index over {@code nParents} parents from the given edges.
     *
     * @throws IllegalStateException if {@code librt_c} is unavailable
     * @throws IllegalArgumentException if the edge arrays differ in length
     * @throws RuntimeException if the native build fails
     */
    public NativeCsc(long nParents, long[] edgeParent, long[] edgeChild, double[] edgeTs) {
        RtCLib l = RtCNative.get();
        if (l == null) {
            throw new IllegalStateException(
                    "The CSC index requires the native runtime 'librt_c', which could not be "
                    + "loaded. Build it (cd cpp && cmake --build build) and point the "
                    + "'relativedb.rt.lib' system property or RELATIVEDB_RT_LIB env var at the "
                    + "library file. Underlying cause: " + RtCNative.failure());
        }
        this.lib = l;
        int nEdges = edgeParent.length;
        if (edgeChild.length != nEdges || edgeTs.length != nEdges) {
            throw new IllegalArgumentException("edge arrays must have equal length");
        }
        byte[] err = new byte[ERR];
        Pointer h = lib.csc_build(nParents, nEdges,
                nEdges == 0 ? null : edgeParent,
                nEdges == 0 ? null : edgeChild,
                nEdges == 0 ? null : edgeTs,
                err, err.length);
        if (h == null) {
            throw new RuntimeException(orDefault(cString(err), "csc_build failed"));
        }
        this.handle = h;
    }

    /** True if the native CSC index can be built. */
    public static boolean available() {
        return RtCNative.isAvailable();
    }

    /**
     * Up to {@code limit} dense child ids with {@code ts <= anchorTs}, newest
     * first. {@code limit <= 0} returns an empty array.
     */
    public long[] children(long parentDense, double anchorTs, int limit) {
        if (handle == null) throw new IllegalStateException("csc_index already freed");
        if (limit <= 0) return new long[0];
        long[] out = new long[limit];
        IntByReference n = new IntByReference(0);
        byte[] err = new byte[ERR];
        int rc = lib.csc_children(handle, parentDense, anchorTs, limit, out, n, err, err.length);
        if (rc != 0) {
            throw new RuntimeException(orDefault(cString(err), "csc_children failed"));
        }
        int count = n.getValue();
        long[] result = new long[count];
        System.arraycopy(out, 0, result, 0, count);
        return result;
    }

    @Override
    public synchronized void close() {
        if (handle != null) {
            lib.csc_free(handle);
            handle = null;
        }
    }

    @Override
    @SuppressWarnings({"deprecation", "removal"})
    protected void finalize() {
        close();
    }

    private static String cString(byte[] buf) {
        int n = 0;
        while (n < buf.length && buf[n] != 0) n++;
        return new String(buf, 0, n, StandardCharsets.UTF_8);
    }

    private static String orDefault(String s, String fallback) {
        return (s == null || s.isEmpty()) ? fallback : s;
    }
}
