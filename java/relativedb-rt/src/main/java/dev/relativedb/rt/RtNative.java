package dev.relativedb.rt;

import com.sun.jna.Native;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Lazy loader for {@code librt_c} (the C++ RT-J inference engine). Nothing
 * native is touched until the first {@link #get()} — depending on this module
 * never breaks a JVM without the library.
 *
 * <p>Search order:
 * <ol>
 *   <li>system property {@code relativedb.rt.lib} — path to the library file</li>
 *   <li>environment variable {@code RELATIVEDB_RT_LIB}</li>
 *   <li>relative lookup of {@code cpp/build/librt_c.dylib} (checked against the
 *       working directory at {@code ../cpp}, {@code ../../cpp} and {@code ./cpp}
 *       so it works both from the repo root and from the {@code java/} tree)</li>
 *   <li>normal loader lookup of {@code rt_c} ({@code java.library.path} /
 *       {@code jna.library.path} apply)</li>
 * </ol>
 */
public final class RtNative {

    static final String LIB_PROP = "relativedb.rt.lib";
    static final String LIB_ENV = "RELATIVEDB_RT_LIB";

    private static volatile RtC instance;

    private RtNative() { }

    /** True if the native engine can be loaded (loads it as a side effect). */
    public static boolean isAvailable() {
        try {
            get();
            return true;
        } catch (RtException e) {
            return false;
        }
    }

    /** The loaded binding, or an {@link RtException} describing how to provide it. */
    public static RtC get() {
        RtC lib = instance;
        if (lib != null) return lib;
        synchronized (RtNative.class) {
            if (instance == null) {
                try {
                    instance = load();
                } catch (Throwable t) {
                    throw new RtException(
                        "The native RT engine 'librt_c' is not available on this system. "
                        + "relativedb-rt is an OPTIONAL module: the engine in relativedb-core works "
                        + "without it. To use it, build the C++ engine (cd cpp && cmake -B build "
                        + "&& cmake --build build → produces librt_c.dylib / librt_c.so) and "
                        + "either run from the repo root (the loader finds cpp/build/ "
                        + "automatically), or point the system property '" + LIB_PROP + "' "
                        + "(or env " + LIB_ENV + ") at the library file. Underlying error: "
                        + t.getMessage(), t);
                }
            }
            return instance;
        }
    }

    private static RtC load() {
        String explicit = System.getProperty(LIB_PROP, System.getenv(LIB_ENV));
        if (explicit != null && !explicit.isBlank()) {
            return Native.load(explicit, RtC.class);
        }
        for (Path candidate : relativeCandidates()) {
            if (Files.isRegularFile(candidate)) {
                return Native.load(candidate.toAbsolutePath().toString(), RtC.class);
            }
        }
        return Native.load("rt_c", RtC.class);
    }

    static List<Path> relativeCandidates() {
        Path cwd = Path.of(System.getProperty("user.dir", "."));
        List<Path> out = new ArrayList<>();
        for (String prefix : new String[] { "..", "../..", "." }) {
            for (String name : new String[] { "librt_c.dylib", "librt_c.so" }) {
                out.add(cwd.resolve(prefix).resolve("cpp/build").resolve(name).normalize());
            }
        }
        return out;
    }
}
