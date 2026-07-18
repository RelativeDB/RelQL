package com.relativedb.nat;

import com.sun.jna.Native;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Lazy loader for the shared C++ runtime ({@code librt_c}). Nothing native is
 * touched until {@link #get()} is first called; {@link #get()} returns
 * {@code null} when the library cannot be located, and callers that require it
 * (PQL parsing, and the RT-J model) raise a clear error. {@code librt_c} is a
 * hard dependency, mirroring the Python and Rust bindings.
 *
 * <p>Discovery order:
 * <ol>
 *   <li>system property {@code relativedb.rt.lib}</li>
 *   <li>environment variable {@code RELATIVEDB_RT_LIB}</li>
 *   <li>{@code cpp/build/librt_c.{dylib,so,dll}} found by walking up from the
 *       working directory (so it resolves from the repo root or the {@code java/}
 *       tree)</li>
 *   <li>the plain library name {@code rt_c} on {@code jna.library.path}</li>
 * </ol>
 */
public final class RtCNative {

    static final String LIB_PROP = "relativedb.rt.lib";
    static final String LIB_ENV = "RELATIVEDB_RT_LIB";

    private static final String[] LIB_NAMES =
            {"librt_c.dylib", "librt_c.so", "librt_c.dll", "rt_c.dll"};

    private static volatile boolean tried;
    private static volatile RtCLib instance;
    private static volatile String failure;

    private RtCNative() { }

    /** The loaded binding, or {@code null} if {@code librt_c} is unavailable. */
    public static RtCLib get() {
        if (tried) return instance;
        synchronized (RtCNative.class) {
            if (!tried) {
                try {
                    instance = load();
                } catch (Throwable t) {
                    failure = describe(t);
                    instance = null;
                }
                tried = true;
            }
        }
        return instance;
    }

    /** True if the native runtime loaded (loads it as a side effect). */
    public static boolean isAvailable() {
        return get() != null;
    }

    /** A message describing why the library did not load, or {@code null}. */
    public static String failure() {
        get();
        return failure;
    }

    private static RtCLib load() {
        String explicit = System.getProperty(LIB_PROP, System.getenv(LIB_ENV));
        if (explicit != null && !explicit.isBlank()) {
            return Native.load(explicit, RtCLib.class);
        }
        for (File candidate : upwardCandidates()) {
            if (candidate.isFile()) {
                return Native.load(candidate.getAbsolutePath(), RtCLib.class);
            }
        }
        return Native.load("rt_c", RtCLib.class);
    }

    static List<File> upwardCandidates() {
        List<File> out = new ArrayList<>();
        File dir = new File(System.getProperty("user.dir", ".")).getAbsoluteFile();
        while (dir != null) {
            File build = new File(dir, "cpp" + File.separator + "build");
            for (String name : LIB_NAMES) {
                out.add(new File(build, name));
            }
            dir = dir.getParentFile();
        }
        return out;
    }

    private static String describe(Throwable t) {
        String msg = t.getMessage();
        return "librt_c unavailable: " + (msg != null ? msg : t.toString());
    }
}
