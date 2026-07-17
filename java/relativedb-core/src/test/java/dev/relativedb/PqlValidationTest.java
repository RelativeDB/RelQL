package dev.relativedb;

import dev.relativedb.query.Pql;
import dev.relativedb.query.PqlValidationException;
import dev.relativedb.query.TaskType;
import dev.relativedb.schema.RelativeDbSchema;
import dev.relativedb.schema.LinkDef;
import dev.relativedb.schema.TableDef;
import org.junit.jupiter.api.Test;

import static dev.relativedb.schema.ValueType.*;
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
                taskOf("PREDICT SUM(transactions.price, 0, 30) FOR EACH customers.customer_id"));
    }

    @Test
    void comparedAggregationIsBinaryClassification() {
        assertEquals(TaskType.BINARY_CLASSIFICATION,
                taskOf("PREDICT COUNT(transactions.*, 0, 90, days) = 0 FOR EACH customers.customer_id"));
    }

    @Test
    void lastOnCategoricalIsMulticlass() {
        assertEquals(TaskType.MULTICLASS_CLASSIFICATION,
                taskOf("PREDICT LAST(transactions.status, 0, 90) FOR EACH customers.customer_id"));
    }

    @Test
    void listDistinctRankIsRanking() {
        assertEquals(TaskType.MULTILABEL_RANKING,
                taskOf("PREDICT LIST_DISTINCT(transactions.article_id, 0, 30) RANK TOP 12 FOR EACH customers.customer_id"));
    }

    @Test
    void forecastIsForecasting() {
        assertEquals(TaskType.FORECASTING,
                taskOf("PREDICT SUM(transactions.price, 0, 7, days) FORECAST 4 TIMEFRAMES FOR EACH customers.customer_id"));
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

    // ---- binding errors -------------------------------------------------

    @Test
    void unknownTableRejected() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(orders.price, 0, 30) FOR EACH customers.customer_id"));
    }

    @Test
    void unknownColumnRejected() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.nope, 0, 30) FOR EACH customers.customer_id"));
    }

    @Test
    void entityKeyMustBePrimaryKey() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.price, 0, 30) FOR EACH customers.age"));
    }

    @Test
    void sumRequiresNumericColumn() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.status, 0, 30) FOR EACH customers.customer_id"));
    }

    @Test
    void wildcardOnlyValidInCount() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.*, 0, 30) FOR EACH customers.customer_id"));
    }

    @Test
    void listDistinctRequiresRankOrClassify() {
        PqlValidationException e = assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT LIST_DISTINCT(transactions.article_id, 0, 30) FOR EACH customers.customer_id"));
        assertTrue(e.getMessage().contains("LIST_DISTINCT"));
    }

    @Test
    void targetWindowMustBeFuture() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.price, -30, 0) FOR EACH customers.customer_id"));
    }

    @Test
    void whereWindowMustBePast() {
        // Past filter window is fine...
        assertDoesNotThrow(() -> taskOf(
                "PREDICT SUM(transactions.price, 0, 30) FOR EACH customers.customer_id "
                + "WHERE COUNT(transactions.*, -30, 0) > 0"));
        // ...a future one is not.
        assertThrows(PqlValidationException.class, () -> taskOf(
                "PREDICT SUM(transactions.price, 0, 30) FOR EACH customers.customer_id "
                + "WHERE COUNT(transactions.*, 0, 30) > 0"));
    }

    @Test
    void emptyWindowRejected() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT SUM(transactions.price, 30, 30) FOR EACH customers.customer_id"));
    }

    @Test
    void staticTemporalMixingRejected() {
        assertThrows(PqlValidationException.class, () -> taskOf(
                "PREDICT SUM(transactions.price, 0, 30) > 10 OR customers.industry = 'IT' "
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
                "PREDICT LIST_DISTINCT(transactions.article_id, 0, 30) RANK TOP 5 FOR EACH customers.customer_id"));
    }

    @Test
    void forecastRequiresWindowedAggregation() {
        assertThrows(PqlValidationException.class,
                () -> taskOf("PREDICT customers.age FORECAST 4 TIMEFRAMES FOR EACH customers.customer_id"));
    }
}
