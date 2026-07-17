package dev.relativedb.query;

/** Inline aggregation filter: {@code COUNT(t.* WHERE <condition>, ...)}. */
public record Filter(TargetExpr condition) { }
