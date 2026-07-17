package dev.relativedb.query;

/**
 * A leaf comparison: {@code left <op> right}. For IS NULL / IS NOT NULL the
 * right side is {@link Literal#NULL}; for IN / NOT IN it is a LIST literal.
 */
public record Condition(TargetExpr left, Operator op, Literal right) implements TargetExpr { }
