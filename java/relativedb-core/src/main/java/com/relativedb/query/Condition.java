package com.relativedb.query;

import java.util.Optional;

/**
 * A leaf comparison: {@code left <op> right}. For IS NULL / IS NOT NULL the
 * right side is {@link Literal#NULL}; for IN / NOT IN it is a LIST literal.
 *
 * <p>{@code rightExpr} carries a column/expression right-hand side (e.g.
 * {@code a.x > b.y}); when it is present {@code right} is {@link Literal#NULL}.
 * Literal-valued word operators (IN / LIKE / CONTAINS / ...) keep {@code right}.
 */
public record Condition(TargetExpr left, Operator op, Literal right,
                        Optional<TargetExpr> rightExpr) implements TargetExpr {

    /** Convenience for a literal-valued comparison (no expression RHS). */
    public Condition(TargetExpr left, Operator op, Literal right) {
        this(left, op, right, Optional.empty());
    }
}
