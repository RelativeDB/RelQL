package dev.relativedb.query;

/**
 * A node of the PREDICT / WHERE / ASSUMING expression tree.
 * (The design doc's sealed interface, extended with {@link Not} — the grammar
 * has a NOT operator but the doc's LogicalOp is strictly binary.)
 */
public sealed interface TargetExpr permits Aggregation, ColumnRef, Condition, LogicalOp, Not { }
