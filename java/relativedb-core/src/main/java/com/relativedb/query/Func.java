package com.relativedb.query;

import java.util.List;

/** A scalar function call in value position: {@code COALESCE|NULLIF|ABS|LOG|EXP|LEAST|GREATEST(args...)}. */
public record Func(String name, List<TargetExpr> args) implements TargetExpr { }
