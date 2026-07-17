package dev.relativedb.rt;

import com.sun.jna.Pointer;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/**
 * A loaded RT-J checkpoint (one {@code rt_model*}). Thread-safe: the native
 * model may be shared across threads and {@link #forward} is reentrant.
 * Free with {@link #close()}.
 */
public final class RtModel implements AutoCloseable {

    private static final int ERR_LEN = 4096;

    private final RtC lib;
    private volatile Pointer handle;
    private final Path source;

    private RtModel(RtC lib, Pointer handle, Path source) {
        this.lib = lib;
        this.handle = handle;
        this.source = source;
    }

    /** Loads a safetensors checkpoint (bf16 or f32). */
    public static RtModel load(Path safetensorsPath) {
        RtC lib = RtNative.get();
        byte[] err = new byte[ERR_LEN];
        Pointer p = lib.rt_model_load(safetensorsPath.toString(), err, err.length);
        if (p == null) {
            throw new RtException("rt_model_load failed for " + safetensorsPath + ": " + cstr(err));
        }
        return new RtModel(lib, p, safetensorsPath);
    }

    public long numParams() {
        return lib.rt_model_num_params(alive());
    }

    public Path source() { return source; }

    /**
     * Runs the forward pass over the RAW PRE-SORT arrays (see {@link RtC} for
     * layouts) and returns the per-batch-row target scores (length B):
     * logits for the classification checkpoint, normalized values for the
     * regression checkpoint. {@code nThreads <= 0} = hardware concurrency.
     */
    public float[] forward(int b, int s,
                           long[] nodeIdxs, long[] f2p,
                           long[] colIdxs, long[] tableIdxs,
                           byte[] isPadding, long[] semTypes,
                           byte[] isTarget, float[] numberV,
                           float[] datetimeV, float[] booleanV,
                           float[] textV, float[] colNameV,
                           int nThreads) {
        check("node_idxs", nodeIdxs.length, b * s);
        check("f2p", f2p.length, b * s * 5);
        check("col_idxs", colIdxs.length, b * s);
        check("table_idxs", tableIdxs.length, b * s);
        check("is_padding", isPadding.length, b * s);
        check("sem_types", semTypes.length, b * s);
        check("is_target", isTarget.length, b * s);
        check("number_v", numberV.length, b * s);
        check("datetime_v", datetimeV.length, b * s);
        check("boolean_v", booleanV.length, b * s);
        check("text_v", textV.length, b * s * 384);
        check("col_name_v", colNameV.length, b * s * 384);
        float[] out = new float[b];
        byte[] err = new byte[ERR_LEN];
        int rc = lib.rt_forward(alive(), b, s, nodeIdxs, f2p, colIdxs, tableIdxs,
            isPadding, semTypes, isTarget, numberV, datetimeV, booleanV,
            textV, colNameV, nThreads, out, err, err.length);
        if (rc != 0) {
            throw new RtException("rt_forward failed (rc=" + rc + "): " + cstr(err));
        }
        return out;
    }

    @Override
    public synchronized void close() {
        Pointer p = handle;
        if (p != null) {
            handle = null;
            lib.rt_model_free(p);
        }
    }

    private Pointer alive() {
        Pointer p = handle;
        if (p == null) throw new RtException("RtModel already closed: " + source);
        return p;
    }

    private static void check(String name, int actual, int expected) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                name + " has length " + actual + ", expected " + expected);
        }
    }

    private static String cstr(byte[] buf) {
        int n = 0;
        while (n < buf.length && buf[n] != 0) n++;
        return new String(buf, 0, n, StandardCharsets.UTF_8);
    }
}
