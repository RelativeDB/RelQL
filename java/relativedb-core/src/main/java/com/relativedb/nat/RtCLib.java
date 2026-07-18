package com.relativedb.nat;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;

/**
 * Raw JNA mapping of the shared C++ runtime ({@code librt_c}) entry points that
 * relativedb-core delegates to: the RelQL parser ({@code pql_parse}) and the CSC
 * adjacency index ({@code csc_build} / {@code csc_children} / {@code csc_free}).
 *
 * <p>These are the same symbols the Python bindings load via ctypes
 * ({@code python/src/relativedb/pql/native.py}, {@code .../csc_native.py}); the
 * single C++ implementation is the source of truth for all language bindings.
 *
 * <p>The ABI follows the {@code err(char*, size_t)} + nonzero-on-failure
 * convention. {@code size_t} is mapped to a Java {@code long} (correct on the
 * 64-bit targets we support). Byte arrays are caller-owned buffers JNA copies
 * across the boundary; {@code long[]}/{@code double[]} map to packed native
 * {@code int64_t*}/{@code double*}.
 */
public interface RtCLib extends Library {

    /**
     * Parse {@code query} into a JSON AST written into {@code out} (NUL
     * terminated, capped at {@code outlen}). Returns 0 on success; nonzero on a
     * syntax error, with a human-readable message written into {@code err}.
     */
    int pql_parse(String query, byte[] out, long outlen, byte[] err, long errlen);

    /**
     * Build a CSC index from parallel edge arrays. Returns NULL on error
     * (message in {@code err}). Arrays may be {@code null} when {@code n_edges}
     * is 0.
     */
    Pointer csc_build(long n_parents, long n_edges,
                      long[] edge_parent, long[] edge_child, double[] edge_ts,
                      byte[] err, long errlen);

    void csc_free(Pointer index);

    /**
     * Latest-{@code <= anchor_ts} children of {@code parent_dense}, newest
     * first, up to {@code limit}. Writes ids into {@code out_child} and the
     * count into {@code out_n}. Returns 0 on success, nonzero on error.
     */
    int csc_children(Pointer index, long parent_dense, double anchor_ts,
                     int limit, long[] out_child, IntByReference out_n,
                     byte[] err, long errlen);
}
