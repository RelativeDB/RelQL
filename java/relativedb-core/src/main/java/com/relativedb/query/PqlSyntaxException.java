package com.relativedb.query;

/** Syntax error with source location. */
public class PqlSyntaxException extends RuntimeException {
    private final int line;
    private final int charPositionInLine;

    public PqlSyntaxException(String message, int line, int charPositionInLine) {
        super("line " + line + ":" + charPositionInLine + " " + message);
        this.line = line;
        this.charPositionInLine = charPositionInLine;
    }

    public int line() { return line; }
    public int charPositionInLine() { return charPositionInLine; }
}
