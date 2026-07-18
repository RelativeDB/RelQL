package com.relativedb.query;

public enum AggFunc {
    SUM, AVG, MIN, MAX, COUNT, COUNT_DISTINCT, FIRST, LAST, LIST_DISTINCT, EXISTS;

    /** Functions whose operand may be {@code table.*}. */
    public boolean allowsWildcard() { return this == COUNT || this == EXISTS; }

    /** Functions requiring a NUMBER operand. */
    public boolean requiresNumeric() {
        return this == SUM || this == AVG;
    }
}
