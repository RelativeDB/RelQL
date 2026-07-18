package com.relativedb.query;

import java.time.LocalDateTime;
import java.util.List;

/**
 * A RelQL literal. {@code value} is a {@link String}, {@link Double},
 * {@link LocalDateTime} (DATE), {@code List<Literal>} (LIST), or {@code null}.
 */
public record Literal(Kind kind, Object value) {

    public enum Kind { STRING, NUMBER, BOOLEAN, DATE, NULL, LIST }

    public static final Literal NULL = new Literal(Kind.NULL, null);
    public static final Literal TRUE = new Literal(Kind.BOOLEAN, Boolean.TRUE);
    public static final Literal FALSE = new Literal(Kind.BOOLEAN, Boolean.FALSE);

    public static Literal string(String s) { return new Literal(Kind.STRING, s); }
    public static Literal number(double d) { return new Literal(Kind.NUMBER, d); }
    public static Literal bool(boolean b) { return b ? TRUE : FALSE; }
    public static Literal date(LocalDateTime d) { return new Literal(Kind.DATE, d); }
    public static Literal list(List<Literal> items) { return new Literal(Kind.LIST, List.copyOf(items)); }

    @SuppressWarnings("unchecked")
    public List<Literal> items() {
        if (kind != Kind.LIST) throw new IllegalStateException("not a LIST literal: " + this);
        return (List<Literal>) value;
    }

    @Override public String toString() {
        return switch (kind) {
            case STRING -> "'" + value + "'";
            case NULL -> "NULL";
            default -> String.valueOf(value);
        };
    }
}
