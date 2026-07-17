package dev.relativedb.query;

/** Boolean negation: {@code NOT inner}. */
public record Not(TargetExpr inner) implements TargetExpr { }
