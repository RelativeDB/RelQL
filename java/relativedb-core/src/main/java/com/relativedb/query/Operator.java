package com.relativedb.query;

/** Comparison / membership / null-test operators. */
public enum Operator {
    GT, LT, EQ, NEQ, GE, LE,
    STARTS_WITH, ENDS_WITH, CONTAINS, NOT_CONTAINS, LIKE, NOT_LIKE,
    IN, NOT_IN,
    IS_NULL, IS_NOT_NULL;

    /** Operators that only apply to text values. */
    public boolean isTextOp() {
        return this == STARTS_WITH || this == ENDS_WITH || this == CONTAINS
                || this == NOT_CONTAINS || this == LIKE || this == NOT_LIKE;
    }

    /** Ordering comparisons requiring an orderable (number/datetime) type. */
    public boolean isOrdering() {
        return this == GT || this == LT || this == GE || this == LE;
    }
}
