package com.relativedb.query;

/** A literal appearing in value position (number, TRUE/FALSE, string, date). */
public record LitExpr(Literal value) implements TargetExpr { }
