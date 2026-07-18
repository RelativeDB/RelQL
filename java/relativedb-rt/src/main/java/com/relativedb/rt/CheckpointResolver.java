package com.relativedb.rt;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

/**
 * Resolves a {@code ModelConfig} checkpoint URI to a local safetensors file.
 *
 * <p>Supported forms:
 * <ul>
 *   <li>{@code file:///abs/path/model.safetensors} (or a {@code file://} directory —
 *       {@code model.safetensors} is appended)</li>
 *   <li>a plain filesystem path (file or directory, same rule)</li>
 *   <li>{@code hf://<org>/<repo>[/<subdir>]} — resolved against the LOCAL
 *       Hugging Face hub cache only ({@code models--<org>--<repo>/snapshots/<hash>/
 *       [<subdir>/]model.safetensors}). This is deliberately NOT an HF client:
 *       nothing is downloaded. Pre-populate the cache (e.g. {@code huggingface-cli
 *       download stanford-star/rt-j}) or use a file path/override instead.</li>
 * </ul>
 *
 * <p>The cache root is, in priority order: system property
 * {@code relativedb.rt.hf.cache}, env {@code RELATIVEDB_RT_HF_CACHE}, env
 * {@code HF_HUB_CACHE}, env {@code HF_HOME} (+{@code /hub}), then
 * {@code ~/.cache/huggingface/hub}. When several snapshots contain the
 * checkpoint, the lexicographically last one wins (deterministic).
 *
 * <p>System property {@code relativedb.rt.quantized} or env
 * {@code RELATIVEDB_RT_QUANTIZED} selects a quantized checkpoint variant
 * (produced by {@code cpp/rt_quantize}): {@code 1}/{@code true}/{@code q8}
 * pick a {@code model.q8.safetensors} sibling, {@code q4} / {@code f16} the
 * corresponding file, wherever a directory is resolved. Off by default so
 * fp32 golden parity is untouched; explicit file paths are always used as
 * given.
 */
public final class CheckpointResolver {

    static final String CACHE_PROP = "relativedb.rt.hf.cache";
    static final String CACHE_ENV = "RELATIVEDB_RT_HF_CACHE";
    static final String QUANT_PROP = "relativedb.rt.quantized";
    static final String QUANT_ENV = "RELATIVEDB_RT_QUANTIZED";

    private CheckpointResolver() { }

    /** {@code q8}, {@code q4}, {@code f16}, or null (fp32). */
    static String quantizedVariant() {
        String v = System.getProperty(QUANT_PROP, System.getenv(QUANT_ENV));
        if (v == null) return null;
        v = v.toLowerCase();
        if (v.equals("1") || v.equals("true") || v.equals("q8")) return "q8";
        if (v.equals("q4") || v.equals("f16")) return v;
        return null;
    }

    /** dir -> dir/model.&lt;variant&gt;.safetensors when opted in and present,
     *  else dir/model.safetensors. */
    private static Path pickModel(Path dir) {
        String v = quantizedVariant();
        if (v != null) {
            Path q = dir.resolve("model." + v + ".safetensors");
            if (Files.isRegularFile(q)) return q;
        }
        return dir.resolve("model.safetensors");
    }

    /** Resolves to an existing safetensors file or throws {@link RtException}. */
    public static Path resolve(String uri) {
        Path p = tryResolve(uri);
        if (p == null || !Files.isRegularFile(p)) {
            throw new RtException(
                "Cannot resolve model checkpoint '" + uri + "' to a local safetensors file"
                + (p != null ? " (looked at " + p + ")" : "")
                + ". Use a file:// URI or plain path to a model.safetensors, or for hf:// "
                + "URIs pre-populate the local Hugging Face cache (override the cache root "
                + "with -D" + CACHE_PROP + " or env " + CACHE_ENV + ").");
        }
        return p;
    }

    private static Path tryResolve(String uri) {
        if (uri.startsWith("hf://")) {
            return resolveHf(uri.substring("hf://".length()));
        }
        String raw = uri.startsWith("file://") ? uri.substring("file://".length()) : uri;
        Path p = Path.of(raw);
        if (Files.isDirectory(p)) p = pickModel(p);
        return p;
    }

    /** {@code <org>/<repo>[/<subdir...>]} against the local HF hub cache. */
    private static Path resolveHf(String spec) {
        String[] parts = spec.split("/", 3);
        if (parts.length < 2) return null;
        String repoDir = "models--" + parts[0] + "--" + parts[1];
        String subdir = parts.length == 3 ? parts[2] : "";
        Path snapshots = cacheRoot().resolve(repoDir).resolve("snapshots");
        if (!Files.isDirectory(snapshots)) return null;
        try (Stream<Path> snaps = Files.list(snapshots)) {
            return snaps.filter(Files::isDirectory)
                .sorted(Comparator.comparing(Path::getFileName).reversed())
                .map(s -> pickModel(subdir.isEmpty() ? s : s.resolve(subdir)))
                .filter(Files::isRegularFile)
                .findFirst().orElse(null);
        } catch (IOException e) {
            return null;
        }
    }

    static Path cacheRoot() {
        String prop = System.getProperty(CACHE_PROP, System.getenv(CACHE_ENV));
        if (prop != null && !prop.isBlank()) return Path.of(prop);
        String hubCache = System.getenv("HF_HUB_CACHE");
        if (hubCache != null && !hubCache.isBlank()) return Path.of(hubCache);
        String hfHome = System.getenv("HF_HOME");
        if (hfHome != null && !hfHome.isBlank()) return Path.of(hfHome).resolve("hub");
        return Path.of(System.getProperty("user.home"), ".cache", "huggingface", "hub");
    }
}
