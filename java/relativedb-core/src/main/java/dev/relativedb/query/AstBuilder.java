package dev.relativedb.query;

import dev.relativedb.query.antlr.PqlParser;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

/** Maps the ANTLR parse tree onto the typed AST. */
final class AstBuilder {

    ParsedQuery build(PqlParser.QueryContext ctx) {
        PqlParser.TargetContext targetCtx = ctx.target();
        TargetExpr target = expr(targetCtx.expr());

        OptionalInt topK = OptionalInt.empty();
        Optional<ProblemType> problemType = Optional.empty();
        PqlParser.RankClauseContext rank = targetCtx.rankClause();
        if (rank != null) {
            if (rank.CLASSIFY() != null) {
                problemType = Optional.of(ProblemType.CLASSIFY);
            } else {
                problemType = Optional.of(ProblemType.RANK);
                topK = OptionalInt.of(Integer.parseInt(rank.INT().getText()));
            }
        }

        OptionalInt numForecasts = ctx.forecastClause() == null
                ? OptionalInt.empty()
                : OptionalInt.of(Integer.parseInt(ctx.forecastClause().INT().getText()));

        PqlParser.ForEachClauseContext fe = ctx.forEachClause();
        ColumnRef entityKey = columnRef(fe.columnRef());
        List<Literal> entityIds = new ArrayList<>();
        if (fe.entitySelector() != null) {
            PqlParser.EntitySelectorContext sel = fe.entitySelector();
            if (sel.literal() != null) {
                entityIds.add(literal(sel.literal()));
            } else {
                for (PqlParser.LiteralContext lc : sel.listLiteral().literal()) {
                    entityIds.add(literal(lc));
                }
            }
        }

        Optional<TargetExpr> where = ctx.whereClause() == null
                ? Optional.empty() : Optional.of(expr(ctx.whereClause().expr()));
        Optional<TargetExpr> assuming = ctx.assumingClause() == null
                ? Optional.empty() : Optional.of(expr(ctx.assumingClause().expr()));

        return new ParsedQuery(target, entityKey, List.copyOf(entityIds),
                where, assuming, topK, problemType, numForecasts);
    }

    private TargetExpr expr(PqlParser.ExprContext ctx) {
        if (ctx instanceof PqlParser.ParenExprContext p) return expr(p.expr());
        if (ctx instanceof PqlParser.NotExprContext n) return new Not(expr(n.expr()));
        if (ctx instanceof PqlParser.AndExprContext a) {
            return new LogicalOp(expr(a.expr(0)), BoolOp.AND, expr(a.expr(1)));
        }
        if (ctx instanceof PqlParser.OrExprContext o) {
            return new LogicalOp(expr(o.expr(0)), BoolOp.OR, expr(o.expr(1)));
        }
        return predicate(((PqlParser.PredicateExprContext) ctx).predicate());
    }

    private TargetExpr predicate(PqlParser.PredicateContext ctx) {
        if (ctx instanceof PqlParser.ComparePredicateContext c) {
            return new Condition(valueExpr(c.valueExpr()), comparisonOp(c.comparisonOp()), literal(c.literal()));
        }
        if (ctx instanceof PqlParser.InPredicateContext in) {
            List<Literal> items = new ArrayList<>();
            for (PqlParser.LiteralContext lc : in.listLiteral().literal()) items.add(literal(lc));
            Operator op = in.memberOp().NOT() != null ? Operator.NOT_IN : Operator.IN;
            return new Condition(valueExpr(in.valueExpr()), op, Literal.list(items));
        }
        if (ctx instanceof PqlParser.NullPredicateContext n) {
            Operator op = n.NOT() != null ? Operator.IS_NOT_NULL : Operator.IS_NULL;
            return new Condition(valueExpr(n.valueExpr()), op, Literal.NULL);
        }
        return valueExpr(((PqlParser.ValuePredicateContext) ctx).valueExpr());
    }

    private TargetExpr valueExpr(PqlParser.ValueExprContext ctx) {
        if (ctx.aggregation() != null) return aggregation(ctx.aggregation());
        return columnRef(ctx.columnRef());
    }

    private Aggregation aggregation(PqlParser.AggregationContext ctx) {
        AggFunc func = AggFunc.valueOf(ctx.aggFunc().getText().toUpperCase());
        PqlParser.AggOperandContext operand = ctx.aggOperand();
        ColumnRef column = columnRef(operand.columnRef());
        Optional<Filter> filter = operand.expr() == null
                ? Optional.empty() : Optional.of(new Filter(expr(operand.expr())));

        if (ctx.aggWindow() == null) {
            return new Aggregation(func, column, filter, OptionalLong.empty(), 0L, null);
        }
        PqlParser.AggWindowContext w = ctx.aggWindow();
        long start = bound(w.bound(0));
        long end = bound(w.bound(1));
        TimeUnit unit = w.timeUnit() == null
                ? TimeUnit.DAYS
                : TimeUnit.valueOf(w.timeUnit().getText().toUpperCase());
        return new Aggregation(func, column, filter, OptionalLong.of(start), end, unit);
    }

    private long bound(PqlParser.BoundContext ctx) {
        boolean negative = ctx.MINUS() != null;
        if (ctx.INF() != null) return negative ? Aggregation.NEG_INF : Aggregation.POS_INF;
        long v = Long.parseLong(ctx.INT().getText());
        return negative ? -v : v;
    }

    private Operator comparisonOp(PqlParser.ComparisonOpContext ctx) {
        if (ctx.GT() != null) return Operator.GT;
        if (ctx.LT() != null) return Operator.LT;
        if (ctx.EQ() != null || ctx.EQEQ() != null) return Operator.EQ;
        if (ctx.NEQ() != null) return Operator.NEQ;
        if (ctx.GE() != null) return Operator.GE;
        if (ctx.LE() != null) return Operator.LE;
        if (ctx.STARTS() != null) return Operator.STARTS_WITH;
        if (ctx.ENDS() != null) return Operator.ENDS_WITH;
        if (ctx.CONTAINS() != null) {
            return ctx.NOT() != null ? Operator.NOT_CONTAINS : Operator.CONTAINS;
        }
        // LIKE / NOT LIKE
        return ctx.NOT() != null ? Operator.NOT_LIKE : Operator.LIKE;
    }

    private ColumnRef columnRef(PqlParser.ColumnRefContext ctx) {
        String table = ctx.name(0).getText();
        String column = ctx.STAR() != null ? "*" : ctx.name(1).getText();
        return new ColumnRef(table, column);
    }

    private Literal literal(PqlParser.LiteralContext ctx) {
        if (ctx.STRING() != null) return Literal.string(unquote(ctx.STRING().getText()));
        if (ctx.DATE() != null) return Literal.date(parseDate(ctx.DATE().getText()));
        if (ctx.NULL() != null) return Literal.NULL;
        return Literal.number(Double.parseDouble(ctx.number().getText()));
    }

    private static LocalDateTime parseDate(String text) {
        if (text.length() > 10) {
            return LocalDateTime.of(LocalDate.parse(text.substring(0, 10)),
                    LocalTime.parse(text.substring(11)));
        }
        return LocalDate.parse(text).atStartOfDay();
    }

    private static String unquote(String s) {
        char quote = s.charAt(0);
        String body = s.substring(1, s.length() - 1);
        StringBuilder out = new StringBuilder(body.length());
        for (int i = 0; i < body.length(); i++) {
            char c = body.charAt(i);
            if (c == '\\' && i + 1 < body.length()) {
                out.append(body.charAt(++i));
            } else if (c == quote && i + 1 < body.length() && body.charAt(i + 1) == quote) {
                out.append(quote);
                i++;
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }
}
