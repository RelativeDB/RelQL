package dev.relativedb.query;

import dev.relativedb.query.antlr.PqlLexer;
import dev.relativedb.query.antlr.PqlParser;
import dev.relativedb.schema.RelativeDbSchema;
import org.antlr.v4.runtime.BaseErrorListener;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.Recognizer;

/** Entry point: PQL parsing and schema-bound validation. */
public final class Pql {
    private Pql() { }

    /** Parse only — no schema needed. Throws {@link PqlSyntaxException} with location. */
    public static ParsedQuery parse(String query) {
        PqlLexer lexer = new PqlLexer(CharStreams.fromString(query));
        lexer.removeErrorListeners();
        lexer.addErrorListener(THROWING);
        PqlParser parser = new PqlParser(new CommonTokenStream(lexer));
        parser.removeErrorListeners();
        parser.addErrorListener(THROWING);
        PqlParser.QueryContext tree = parser.query();
        return new AstBuilder().build(tree);
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

    private static final BaseErrorListener THROWING = new BaseErrorListener() {
        @Override
        public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
                                int line, int charPositionInLine, String msg,
                                RecognitionException e) {
            throw new PqlSyntaxException(msg, line, charPositionInLine);
        }
    };
}
