package com.relativedb.rt;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/** Test helper: locate and read the little-endian golden .bin arrays from cpp/testdata. */
final class GoldenData {

    private GoldenData() { }

    /** cpp/testdata directory, searched relative to the working directory; null if absent. */
    static Path testdataDir() {
        Path cwd = Path.of(System.getProperty("user.dir", "."));
        for (String prefix : new String[] { ".", "..", "../..", "../../.." }) {
            Path candidate = cwd.resolve(prefix).resolve("cpp/testdata").normalize();
            if (Files.isRegularFile(candidate.resolve("manifest.json"))) return candidate;
        }
        return null;
    }

    static boolean classificationCheckpointPresent() {
        try {
            CheckpointResolver.resolve(com.relativedb.model.ModelConfig.DEFAULT_CLASSIFICATION_MODEL_URI);
            return true;
        } catch (RtException e) {
            return false;
        }
    }

    static boolean regressionCheckpointPresent() {
        try {
            CheckpointResolver.resolve(com.relativedb.model.ModelConfig.DEFAULT_REGRESSION_MODEL_URI);
            return true;
        } catch (RtException e) {
            return false;
        }
    }

    private static ByteBuffer buf(Path dir, String name) {
        try {
            byte[] bytes = Files.readAllBytes(dir.resolve(name));
            return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static long[] longs(Path dir, String name, int expected) {
        ByteBuffer b = buf(dir, name);
        long[] out = new long[b.remaining() / 8];
        b.asLongBuffer().get(out);
        checkLen(name, out.length, expected);
        return out;
    }

    static float[] floats(Path dir, String name, int expected) {
        ByteBuffer b = buf(dir, name);
        float[] out = new float[b.remaining() / 4];
        b.asFloatBuffer().get(out);
        checkLen(name, out.length, expected);
        return out;
    }

    static byte[] bytes(Path dir, String name, int expected) {
        ByteBuffer b = buf(dir, name);
        byte[] out = new byte[b.remaining()];
        b.get(out);
        checkLen(name, out.length, expected);
        return out;
    }

    private static void checkLen(String name, int actual, int expected) {
        if (actual != expected) {
            throw new IllegalStateException(name + ": expected " + expected
                + " elements, got " + actual);
        }
    }
}
