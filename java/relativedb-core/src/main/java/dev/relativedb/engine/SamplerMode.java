package dev.relativedb.engine;

/** Traversal strategy. */
public enum SamplerMode {
    /** Pull-per-hop through Entity/Link retrievers (default). */
    RETRIEVER,
    /** Materialized in-memory CSC index built from TableScanners. */
    CSC
}
