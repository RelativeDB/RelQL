package com.relativedb.query;

import java.util.List;

/**
 * A {@code CASE WHEN cond THEN then ... [ELSE elseExpr] END} expression.
 * {@code elseExpr} is {@code null} when the CASE has no ELSE branch.
 */
public record Case(List<When> whens, TargetExpr elseExpr) implements TargetExpr {

    /** One {@code WHEN cond THEN then} arm of a {@link Case}. */
    public record When(TargetExpr cond, TargetExpr then) { }
}
