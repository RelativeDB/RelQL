package com.relativedb.query;

/** Binary arithmetic on value expressions: {@code left <op> right} where op is {@code + - * /}. */
public record Arith(char op, TargetExpr left, TargetExpr right) implements TargetExpr { }
