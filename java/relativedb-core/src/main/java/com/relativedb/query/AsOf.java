package com.relativedb.query;

/**
 * An {@code AS OF <anchor>} clause. {@code value} is the parameter name (PARAM),
 * the date text (DATE), or {@code null} (NOW).
 */
public record AsOf(Kind kind, String value) {

    public enum Kind { PARAM, DATE, NOW }
}
