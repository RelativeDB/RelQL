package com.relativedb;

import com.relativedb.query.AsOf;
import com.relativedb.query.Pql;
import com.relativedb.query.PqlSyntaxException;
import com.relativedb.query.PqlValidationException;
import com.relativedb.query.ReturnSpec;
import com.relativedb.query.TaskType;
import com.relativedb.query.ValidatedQuery;
import com.relativedb.schema.RelativeDbSchema;
import com.relativedb.schema.LinkDef;
import com.relativedb.schema.TableDef;
import org.junit.jupiter.api.Test;

import static com.relativedb.schema.ValueType.*;
import static org.junit.jupiter.api.Assertions.*;

class PqlValidationTest {

    private static final RelativeDbSchema SCHEMA = RelativeDbSchema.newSchema()
            .table(TableDef.newTable("customers")
                    .column("age", NUMBER)
                    .column("industry", TEXT)
                    .column("active", BOOLEAN)
                    .column("date_of_birth", DATETIME)
                    .primaryKey("customer_id")
                    .build())
            .table(TableDef.newTable("transactions")
                    .column("price", NUMBER)
                    .column("status", TEXT)
                    .column("ts", DATETIME)
                    .primaryKey("tx_id")
                    .timeColumn("ts")
                    .build())
            .link(LinkDef.link("transactions", "customer_id", "customers"))
            .link(LinkDef.link("transactions", "article_id", "articles"))
            .table(TableDef.newTable("articles")
                    .column("description", TEXT)
                    .primaryKey("article_id")
                    .build())
            .build();

    private static TaskType taskOf(String q) {
        return Pql.validate(q, SCHEMA).taskType();
    }

    // ---- task type inference -------------------------------------------

    @Test
    void bareAggregationIsRegression() {
        assertEquals(TaskType.REGRESSION,
                taskOf("PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id"));
    }

    @Test
    void comparedAggregationIsBinaryClassification() {
        assertEquals(TaskType.BINARY_CLASSIFICATION,
                taskOf("PREDICT COUNT(transactions.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id"));
    }

    @Test
    void existsTargetIsBinaryClassification() {
        assertEquals(TaskType.BINARY_CLASSIFICATION,
                taskOf("PREDICT EXISTS(transactions.*) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id"));
        assertEquals(TaskType.BINARY_CLASSIFICATION,
                taskOf("PREDICT NOT EXISTS(transactions.*) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id"));
    }

    @Test
    void arithmeticAndFunctionTargetsAreRegression() {
        assertEquals(TaskType.REGRESSION,
                taskOf("PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) * 2 FOR EACH customers.customer_id"));
        assertEquals(TaskType.REGRESSION,
                taskOf("PREDICT COALESCE(SUM(transactions.price) OVER (30 DAYS FOLLOWING), 0) "
                        + "FOR EACH customers.customer_id"));
    }

    @Test
    void caseTargetIsRegression() {
        assertEquals(TaskType.REGRESSION,
                taskOf("PREDICT CASE WHEN COUNT(transactions.*) OVER (30 DAYS FOLLOWING) > 10 THEN 1 ELSE 0 END "
                        + "FOR EACH customers.customer_id"));
    }

    @Test
    void lastOnCategoricalIsMulticlass() {
        assertEquals(TaskType.MULTICLASS_CLASSIFICATION,
                taskOf("PREDICT LAST(transactions.status) OVER (90 DAYS FOLLOWING) FOR EACH customers.customer_id"));
    }

    @Test
    void listDistinctRankIsRanking() {
        assertEquals(TaskType.MULTILABEL_RANKING,
                taskOf("PREDICT LIST_DISTINCT(transactions.article_id) OVER (30 DAYS FOLLOWING) RANK TOP 12 "
                        + "FOR EACH customers.customer_id"));
    }

    @Test
    void horizonsTargetIsForecasting() {
        assertEquals(TaskType.FORECASTING,
                taskOf("PREDICT SUM(transactions.price) OVER (7 DAYS FOLLOWING HORIZONS 4) "
                        + "FOR EACH customers.customer_id"));
    }

    @Test
    void staticColumnTargets() {
        assertEquals(TaskType.MULTICLASS_CLASSIFICATION,
                taskOf("PREDICT customers.industry FOR EACH customers.customer_id"));
        assertEquals(TaskType.REGRESSION,
                taskOf("PREDICT customers.age FOR EACH customers.customer_id"));
        assertEquals(TaskType.BINARY_CLASSIFICATION,
                taskOf("PREDICT customers.active FOR EACH customers.customer_id"));
        assertEquals(TaskType.BINARY_CLASSIFICATION,
                taskOf("PREDICT customers.industry = 'IT' FOR EACH customers.customer_id"));
    }

    // ---- clauses represented, not executed ------------------------------

    @Test
    void asOfAndReturnAreRepresented() {
        ValidatedQuery vq = Pql.validate(
                "PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id "
                + "AS OF :t RETURN EXPECTED VALUE", SCHEMA);
        assertEquals(TaskType.REGRESSION, vq.taskType());
        assertEquals(AsOf.Kind.PARAM, vq.query().asOf().orElseThrow().kind());
        assertEquals(ReturnSpec.Kind.EXPECTED_VALUE, vq.query().ret().orElseThrow().kind());
    }

    @Test
    void explainAndAblateDoNotThrow() {
        assertDoesNotThrow(() -> Pql.validate(
                "EXPLAIN PLAN PREDICT EXISTS(transactions.*) OVER (30 DAYS FOLLOWING) "
                + "FOR EACH customers.customer_id ABLATE TABLE articles RETURN PROBABILITY", SCHEMA));
    }

    // ---- binding errors -------------------------------------------------

    @Test
    void unknownTableRejected() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(orders.price) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id"));
    }

    @Test
    void unknownColumnRejected() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.nope) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id"));
    }

    @Test
    void entityKeyMustBePrimaryKey() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) FOR EACH customers.age"));
    }

    @Test
    void sumRequiresNumericColumn() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.status) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id"));
    }

    @Test
    void wildcardOnlyValidInCountOrExists() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.*) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id"));
    }

    @Test
    void listDistinctRequiresRankOrClassify() {
        PqlValidationException e = assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT LIST_DISTINCT(transactions.article_id) OVER (30 DAYS FOLLOWING) "
                        + "FOR EACH customers.customer_id"));
        assertTrue(e.getMessage().contains("LIST_DISTINCT"));
    }

    @Test
    void targetWindowMustBeFuture() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.price) OVER (30 DAYS PRECEDING) FOR EACH customers.customer_id"));
    }

    @Test
    void whereWindowMustBePast() {
        // Past filter window is fine...
        assertDoesNotThrow(() -> taskOf(
                "PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id "
                + "WHERE COUNT(transactions.*) OVER (30 DAYS PRECEDING) > 0"));
        // ...a future one is not.
        assertThrows(PqlValidationException.class, () -> taskOf(
                "PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id "
                + "WHERE COUNT(transactions.*) OVER (30 DAYS FOLLOWING) > 0"));
    }

    @Test
    void horizonsOnlyAllowedOnTarget() {
        assertThrows(PqlValidationException.class, () -> taskOf(
                "PREDICT SUM(transactions.price) OVER (7 DAYS FOLLOWING) FOR EACH customers.customer_id "
                + "WHERE COUNT(transactions.*) OVER (7 DAYS PRECEDING HORIZONS 3) > 0"));
    }

    @Test
    void emptyWindowRejectedAtParse() {
        // start >= end is rejected by the parser now (was a validation rule).
        assertThrows(PqlSyntaxException.class, () -> taskOf(
                "PREDICT SUM(transactions.price) "
                + "OVER (RANGE BETWEEN 30 DAYS FOLLOWING AND 30 DAYS FOLLOWING) FOR EACH customers.customer_id"));
    }

    @Test
    void staticTemporalMixingRejected() {
        assertThrows(PqlValidationException.class, () -> taskOf(
                "PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) > 10 OR customers.industry = 'IT' "
                + "FOR EACH customers.customer_id"));
    }

    @Test
    void textOperatorRequiresTextColumn() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT customers.age CONTAINS 'x' FOR EACH customers.customer_id"));
    }

    @Test
    void orderingOnTextRejected() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT customers.industry > 5 FOR EACH customers.customer_id"));
    }

    @Test
    void datetimeComparesWithDateLiteral() {
        assertDoesNotThrow(() -> taskOf(
                "PREDICT customers.date_of_birth <= 1990-01-01 FOR EACH customers.customer_id"));
        assertThrows(PqlValidationException.class, () -> taskOf(
                "PREDICT customers.date_of_birth <= 5 FOR EACH customers.customer_id"));
    }

    @Test
    void fkColumnIsValidRecommendationTarget() {
        // article_id is an FK edge, not a declared column — allowed as a categorical target.
        assertDoesNotThrow(() -> taskOf(
                "PREDICT LIST_DISTINCT(transactions.article_id) OVER (30 DAYS FOLLOWING) RANK TOP 5 "
                + "FOR EACH customers.customer_id"));
    }
}
