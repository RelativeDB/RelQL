package com.relativedb.query;

import com.relativedb.schema.RelativeDbSchema;
import com.relativedb.schema.LinkDef;
import com.relativedb.schema.TableDef;
import com.relativedb.schema.ValueType;

/**
 * The semantic rules the grammar deliberately leaves out: tables/columns exist,
 * types line up, window signs, LIST_DISTINCT ⇒ CLASSIFY|RANK, no
 * static/temporal mixing — plus TaskType inference.
 */
final class SemanticValidator {

    private enum Clause { TARGET, WHERE, ASSUMING }

    private final RelativeDbSchema schema;
    private boolean sawTemporal;
    private boolean sawStatic;

    SemanticValidator(RelativeDbSchema schema) { this.schema = schema; }

    TaskType validate(ParsedQuery q) {
        // Entity key: table exists, column is its primary key.
        TableDef entityTable = requireTable(q.entityKey().table());
        if (q.entityKey().isWildcard()) {
            throw new PqlValidationException("entity key must be a primary-key column, not '*'");
        }
        entityTable.primaryKey().ifPresent(pk -> {
            if (!pk.equals(q.entityKey().column())) {
                throw new PqlValidationException("entity key '" + q.entityKey()
                        + "' is not the primary key of table '" + entityTable.name()
                        + "' (expected '" + pk + "')");
            }
        });

        // Target: also drives the static/temporal mixing check.
        sawTemporal = false;
        sawStatic = false;
        checkExpr(q.target(), Clause.TARGET);
        if (sawTemporal && sawStatic) {
            throw new PqlValidationException(
                    "target mixes temporal aggregations with static column references");
        }
        boolean temporalTarget = sawTemporal;

        q.where().ifPresent(w -> checkExpr(w, Clause.WHERE));
        q.assuming().ifPresent(a -> checkExpr(a, Clause.ASSUMING));

        // LIST_DISTINCT ⇒ CLASSIFY | RANK TOP K.
        boolean listDistinctTarget = q.target() instanceof Aggregation agg
                && agg.func() == AggFunc.LIST_DISTINCT;
        if (listDistinctTarget && q.problemType().isEmpty()) {
            throw new PqlValidationException(
                    "LIST_DISTINCT target requires CLASSIFY or RANK TOP K");
        }
        if (q.problemType().isPresent()
                && !listDistinctTarget && !(q.target() instanceof ColumnRef)) {
            throw new PqlValidationException(
                    "CLASSIFY / RANK TOP K only apply to LIST_DISTINCT or a multicategorical column target");
        }
        q.topK().ifPresent(k -> {
            if (k <= 0) throw new PqlValidationException("RANK TOP K requires K >= 1, got " + k);
        });

        // Forecasting is now implied by a target window with HORIZONS > 1; the
        // parser derives num_forecasts from it. Sanity-check the derived count.
        if (q.numForecasts().isPresent() && q.numForecasts().getAsInt() <= 0) {
            throw new PqlValidationException("forecast horizon count must be >= 1");
        }

        TaskType taskType = inferTaskType(q, temporalTarget);
        q.ret().ifPresent(r -> validateReturn(r, taskType));
        return taskType;
    }

    // ------------------------------------------------------------------
    //  RETURN <output> compatibility with the inferred task
    // ------------------------------------------------------------------

    private void validateReturn(ReturnSpec ret, TaskType task) {
        if (!allowsReturn(ret.kind(), task)) {
            throw new PqlValidationException("RETURN " + ret.kind()
                    + " is not compatible with the inferred task type " + task);
        }
        if (ret.kind() == ReturnSpec.Kind.QUANTILES) {
            for (double q : ret.quantiles()) {
                if (!(q > 0.0 && q < 1.0)) {
                    throw new PqlValidationException(
                            "RETURN QUANTILES requires each quantile in (0, 1), got " + q);
                }
            }
        }
        if (ret.kind() == ReturnSpec.Kind.INTERVAL) {
            int pct = ret.interval().orElseThrow(() -> new PqlValidationException(
                    "RETURN INTERVAL requires a percent"));
            if (!(pct > 0 && pct < 100)) {
                throw new PqlValidationException(
                        "RETURN INTERVAL requires a percent in (0, 100), got " + pct);
            }
        }
    }

    private static boolean allowsReturn(ReturnSpec.Kind kind, TaskType task) {
        return switch (kind) {
            case EXPECTED_VALUE -> task == TaskType.REGRESSION || task == TaskType.FORECASTING
                    || task == TaskType.BINARY_CLASSIFICATION;
            case PROBABILITY -> task == TaskType.BINARY_CLASSIFICATION;
            case CLASS, DISTRIBUTION -> task == TaskType.BINARY_CLASSIFICATION
                    || task == TaskType.MULTICLASS_CLASSIFICATION;
            case QUANTILES, INTERVAL -> task == TaskType.REGRESSION || task == TaskType.FORECASTING;
            case MULTILABEL -> task == TaskType.MULTILABEL_RANKING;
            case MULTICLASS -> task == TaskType.MULTICLASS_CLASSIFICATION;
        };
    }

    // ------------------------------------------------------------------
    //  Expression checking
    // ------------------------------------------------------------------

    private void checkExpr(TargetExpr e, Clause clause) {
        if (e instanceof LogicalOp op) {
            checkExpr(op.left(), clause);
            checkExpr(op.right(), clause);
        } else if (e instanceof Not not) {
            checkExpr(not.inner(), clause);
        } else if (e instanceof Condition cond) {
            ValueType left = valueType(cond.left(), clause);
            if (cond.rightExpr().isPresent()) {
                valueType(cond.rightExpr().get(), clause);   // column/expression RHS
            } else {
                checkOperator(cond, left);
            }
        } else if (e instanceof Aggregation agg) {
            checkAggregation(agg, clause);
        } else if (e instanceof ColumnRef ref) {
            resolveColumnType(ref, false);
            if (clause == Clause.TARGET) sawStatic = true;
        } else if (e instanceof Arith a) {
            checkExpr(a.left(), clause);
            checkExpr(a.right(), clause);
        } else if (e instanceof Func f) {
            for (TargetExpr arg : f.args()) checkExpr(arg, clause);
        } else if (e instanceof Case c) {
            for (Case.When w : c.whens()) {
                checkExpr(w.cond(), clause);
                checkExpr(w.then(), clause);
            }
            if (c.elseExpr() != null) checkExpr(c.elseExpr(), clause);
        } else if (e instanceof LitExpr) {
            // a bare literal binds nothing.
        }
    }

    /** Type of a value expression (aggregation or column), checking it as a side effect. */
    private ValueType valueType(TargetExpr e, Clause clause) {
        if (e instanceof Aggregation agg) {
            checkAggregation(agg, clause);
            return aggResultType(agg);
        }
        if (e instanceof ColumnRef ref) {
            if (clause == Clause.TARGET) sawStatic = true;
            return resolveColumnType(ref, false);
        }
        if (e instanceof LitExpr lit) {
            return literalType(lit.value());
        }
        if (e instanceof Arith || e instanceof Func || e instanceof Case) {
            checkExpr(e, clause);          // bind columns / detect temporal
            return ValueType.NUMBER;       // arithmetic / scalar funcs are numeric
        }
        throw new PqlValidationException("expected a value expression, got " + e);
    }

    private static ValueType literalType(Literal lit) {
        return switch (lit.kind()) {
            case NUMBER -> ValueType.NUMBER;
            case BOOLEAN -> ValueType.BOOLEAN;
            case DATE -> ValueType.DATETIME;
            case STRING, NULL, LIST -> ValueType.TEXT;
        };
    }

    private void checkAggregation(Aggregation agg, Clause clause) {
        ValueType operand = resolveColumnType(agg.column(), agg.func().allowsWildcard());
        if (agg.func().requiresNumeric() && operand != ValueType.NUMBER) {
            throw new PqlValidationException(agg.func() + "(" + agg.column()
                    + ") requires a NUMBER column, got " + operand);
        }
        agg.filter().ifPresent(f -> checkExpr(f.condition(), clause));

        // HORIZONS > 1 is only meaningful on the PREDICT target (it drives
        // forecasting); it is not allowed inside WHERE / ASSUMING frames.
        if (agg.isMultiHorizon() && clause != Clause.TARGET) {
            throw new PqlValidationException(
                    "HORIZONS > 1 is only allowed on the PREDICT target, not in " + clause);
        }

        if (!agg.hasWindow()) {
            if (clause == Clause.TARGET) sawStatic = true;
            return;
        }
        if (clause == Clause.TARGET) sawTemporal = true;

        long start = agg.startOr(0);
        long end = agg.end();
        if (start >= end) {
            throw new PqlValidationException("aggregation window (" + fmt(start) + ", "
                    + fmt(end) + "] is empty: start must be < end");
        }
        switch (clause) {
            case TARGET, ASSUMING -> {
                // Target/counterfactual windows look FORWARD from the anchor.
                if (start < 0) {
                    throw new PqlValidationException(clause + " window must be in the future "
                            + "(start >= 0), got start=" + fmt(start));
                }
            }
            case WHERE -> {
                // Temporal-filter windows look BACKWARD (may be negative or -INF).
                if (end > 0) {
                    throw new PqlValidationException("WHERE window must be in the past "
                            + "(end <= 0), got end=" + fmt(end));
                }
            }
        }
    }

    private void checkOperator(Condition cond, ValueType left) {
        Operator op = cond.op();
        Literal right = cond.right();
        if (op.isTextOp() && left != ValueType.TEXT) {
            throw new PqlValidationException(op + " requires a TEXT value, got " + left);
        }
        if (op.isTextOp() && right.kind() != Literal.Kind.STRING) {
            throw new PqlValidationException(op + " requires a string literal, got " + right);
        }
        if (op.isOrdering()) {
            if (left == ValueType.NUMBER && right.kind() != Literal.Kind.NUMBER) {
                throw new PqlValidationException(op + " on a NUMBER value requires a numeric literal, got " + right);
            }
            if (left == ValueType.DATETIME && right.kind() != Literal.Kind.DATE) {
                throw new PqlValidationException(op + " on a DATETIME value requires a date literal, got " + right);
            }
            if (left == ValueType.BOOLEAN || left == ValueType.TEXT) {
                throw new PqlValidationException(op + " requires an orderable value, got " + left);
            }
        }
        if ((op == Operator.EQ || op == Operator.NEQ) && right.kind() == Literal.Kind.NUMBER
                && left == ValueType.TEXT) {
            throw new PqlValidationException("cannot compare TEXT value with numeric literal " + right);
        }
        if (op == Operator.IN || op == Operator.NOT_IN) {
            if (right.kind() != Literal.Kind.LIST) {
                throw new PqlValidationException(op + " requires a list literal");
            }
        }
    }

    private ValueType aggResultType(Aggregation agg) {
        return switch (agg.func()) {
            case COUNT, COUNT_DISTINCT, SUM, AVG -> ValueType.NUMBER;
            case EXISTS -> ValueType.BOOLEAN;
            case MIN, MAX, FIRST, LAST, LIST_DISTINCT ->
                    agg.column().isWildcard() ? ValueType.NUMBER : resolveColumnType(agg.column(), false);
        };
    }

    // ------------------------------------------------------------------
    //  Column resolution
    // ------------------------------------------------------------------

    /**
     * Resolves {@code table.column} to its value type. FK columns (edges) and
     * the primary key resolve as TEXT — categorical identifiers, valid as
     * recommendation targets (e.g. LIST_DISTINCT(transactions.article_id)).
     */
    private ValueType resolveColumnType(ColumnRef ref, boolean allowWildcard) {
        TableDef table = requireTable(ref.table());
        if (ref.isWildcard()) {
            if (!allowWildcard) {
                throw new PqlValidationException("'" + ref + "': '*' is only valid in COUNT(...)");
            }
            return ValueType.NUMBER;
        }
        var declared = table.column(ref.column());
        if (declared.isPresent()) return declared.get().type();
        if (table.timeColumn().map(ref.column()::equals).orElse(false)) return ValueType.DATETIME;
        if (table.primaryKey().map(ref.column()::equals).orElse(false)) return ValueType.TEXT;
        for (LinkDef link : schema.linksFrom(ref.table())) {
            if (link.fkColumn().equals(ref.column())) return ValueType.TEXT;
        }
        throw new PqlValidationException("unknown column '" + ref.column()
                + "' in table '" + ref.table() + "'");
    }

    private TableDef requireTable(String name) {
        return schema.table(name).orElseThrow(() ->
                new PqlValidationException("unknown table '" + name + "'"));
    }

    // ------------------------------------------------------------------
    //  Task type inference
    // ------------------------------------------------------------------

    private TaskType inferTaskType(ParsedQuery q, boolean temporalTarget) {
        if (q.numForecasts().isPresent()) return TaskType.FORECASTING;
        if (q.problemType().isPresent()) {
            return q.problemType().get() == ProblemType.RANK
                    ? TaskType.MULTILABEL_RANKING
                    : TaskType.MULTICLASS_CLASSIFICATION;
        }
        TargetExpr t = q.target();
        if (t instanceof Condition || t instanceof LogicalOp || t instanceof Not) {
            return TaskType.BINARY_CLASSIFICATION;
        }
        if (t instanceof Aggregation agg) {
            if (agg.func() == AggFunc.EXISTS) {
                return TaskType.BINARY_CLASSIFICATION;
            }
            if (agg.func() == AggFunc.FIRST || agg.func() == AggFunc.LAST) {
                ValueType vt = aggResultType(agg);
                return vt == ValueType.NUMBER || vt == ValueType.DATETIME
                        ? TaskType.REGRESSION : TaskType.MULTICLASS_CLASSIFICATION;
            }
            return TaskType.REGRESSION;
        }
        // Arithmetic / scalar-function / CASE targets are numeric ⇒ regression.
        if (t instanceof Arith || t instanceof Func || t instanceof Case) {
            return TaskType.REGRESSION;
        }
        if (t instanceof LitExpr lit) {
            return lit.value().kind() == Literal.Kind.BOOLEAN
                    ? TaskType.BINARY_CLASSIFICATION : TaskType.REGRESSION;
        }
        // Bare static column.
        ValueType vt = resolveColumnType((ColumnRef) t, false);
        return switch (vt) {
            case NUMBER, DATETIME -> TaskType.REGRESSION;
            case BOOLEAN -> TaskType.BINARY_CLASSIFICATION;
            case TEXT -> TaskType.MULTICLASS_CLASSIFICATION;
        };
    }

    private static String fmt(long bound) {
        if (bound == Aggregation.NEG_INF) return "-INF";
        if (bound == Aggregation.POS_INF) return "+INF";
        return String.valueOf(bound);
    }
}
