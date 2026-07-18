package com.relativedb.query;

/** Binary boolean combination: {@code left AND|OR right}. */
public record LogicalOp(TargetExpr left, BoolOp op, TargetExpr right) implements TargetExpr { }
