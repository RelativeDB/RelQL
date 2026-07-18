package com.relativedb.query;

/** An {@code EXPLAIN [mode] [FORMAT fmt]} prefix carried on the parsed query. */
public record Explain(Mode mode, Format format) {

    public enum Mode { PLAN, CONTEXT, ANALYZE, ABLATION }

    public enum Format { TEXT, JSON }
}
