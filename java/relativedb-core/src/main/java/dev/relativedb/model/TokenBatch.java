package dev.relativedb.model;

import dev.relativedb.schema.ValueType;

import java.util.ArrayList;
import java.util.List;

/**
 * Mirrors the RT input contract: per-token → rowId, parentRowIds, table,
 * column, valueType, normalizedValue (or raw text for the frozen encoder),
 * isTarget. Column/table names carried as strings; the backend owns embedding
 * them (frozen text encoder, F13/F14).
 */
public final class TokenBatch {

    /** One cell = one token. Exactly one of {@code normalizedValue}/{@code text} is meaningful. */
    public record Token(
        int rowId,
        List<Integer> parentRowIds,
        String table,
        String column,
        ValueType valueType,
        double normalizedValue,   // NaN when the value is text
        String text,              // null when the value is numeric-encoded
        boolean isTarget
    ) { }

    private final List<Token> tokens;

    private TokenBatch(List<Token> tokens) { this.tokens = List.copyOf(tokens); }

    public static Builder newBatch() { return new Builder(); }

    public List<Token> tokens() { return tokens; }
    public int size() { return tokens.size(); }

    public static final class Builder {
        private final List<Token> tokens = new ArrayList<>();

        public Builder token(Token t) { tokens.add(t); return this; }

        public Builder numeric(int rowId, List<Integer> parents, String table, String column,
                               ValueType type, double normalizedValue, boolean isTarget) {
            return token(new Token(rowId, parents, table, column, type, normalizedValue, null, isTarget));
        }

        public Builder text(int rowId, List<Integer> parents, String table, String column,
                            String text, boolean isTarget) {
            return token(new Token(rowId, parents, table, column, ValueType.TEXT, Double.NaN, text, isTarget));
        }

        public TokenBatch build() { return new TokenBatch(tokens); }
    }
}
