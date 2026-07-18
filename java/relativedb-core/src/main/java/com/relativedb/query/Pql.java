package com.relativedb.query;

import com.relativedb.schema.RelativeDbSchema;

/** Entry point: RelQL parsing and schema-bound validation. */
public final class Pql {
    private Pql() { }

    /**
     * Parse only — no schema needed.
     *
     * <p>Parsing is done by the shared C++ parser ({@code pql_parse} in
     * {@code librt_c}) — the single implementation used across the
     * Python/Java/Rust bindings. There is no in-process fallback parser: if the
     * native library cannot be loaded this throws (see {@link NativePqlParser}),
     * pointing at the missing {@code librt_c}. A malformed query throws
     * {@link PqlSyntaxException}.
     */
    public static ParsedQuery parse(String query) {
        return NativePqlParser.parse(query);
    }

    /**
     * Parse + bind against a schema: tables/columns exist, types line up,
     * window signs, LIST_DISTINCT ⇒ CLASSIFY|RANK, no static/temporal mixing —
     * the semantic rules the grammar deliberately leaves out.
     */
    public static ValidatedQuery validate(String query, RelativeDbSchema schema) {
        return validate(parse(query), schema);
    }

    /** Bind an already-parsed query against a schema. */
    public static ValidatedQuery validate(ParsedQuery query, RelativeDbSchema schema) {
        TaskType taskType = new SemanticValidator(schema).validate(query);
        return new ValidatedQuery(query, taskType);
    }
}
