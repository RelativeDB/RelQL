package dev.relativedb;

import dev.relativedb.query.Aggregation;
import dev.relativedb.query.AggFunc;
import dev.relativedb.query.ColumnRef;
import dev.relativedb.query.Condition;
import dev.relativedb.query.Literal;
import dev.relativedb.query.Operator;
import dev.relativedb.query.ParsedQuery;
import dev.relativedb.query.Pql;
import dev.relativedb.query.PqlSyntaxException;
import dev.relativedb.query.ProblemType;
import dev.relativedb.query.TimeUnit;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class PqlParserTest {

    static Stream<String> corpus() throws Exception {
        try (BufferedReader r = new BufferedReader(new InputStreamReader(
                PqlParserTest.class.getResourceAsStream("/examples.pql"), StandardCharsets.UTF_8))) {
            return r.lines().filter(l -> !l.isBlank()).toList().stream();
        }
    }

    @ParameterizedTest
    @MethodSource("corpus")
    @DisplayName("all 44 corpus queries parse")
    void corpusParses(String query) {
        ParsedQuery parsed = Pql.parse(query);
        assertNotNull(parsed.target());
        assertNotNull(parsed.entityKey());
    }

    @Test
    void corpusHas44Queries() throws Exception {
        assertEquals(44, corpus().count());
    }

    @Test
    void parsesRegressionTarget() {
        ParsedQuery q = Pql.parse("PREDICT SUM(TRANSACTIONS.PRICE, 0, 30) FOR EACH CUSTOMERS.CUSTOMER_ID");
        Aggregation agg = assertInstanceOf(Aggregation.class, q.target());
        assertEquals(AggFunc.SUM, agg.func());
        assertEquals(new ColumnRef("TRANSACTIONS", "PRICE"), agg.column());
        assertEquals(0, agg.start().orElseThrow());
        assertEquals(30, agg.end());
        assertEquals(TimeUnit.DAYS, agg.unit());   // default unit
        assertTrue(agg.hasWindow());
        assertEquals(new ColumnRef("CUSTOMERS", "CUSTOMER_ID"), q.entityKey());
        assertTrue(q.entityIds().isEmpty());
    }

    @Test
    void parsesExplicitUnitAndComparison() {
        ParsedQuery q = Pql.parse(
                "PREDICT COUNT(transactions.*, 0, 30, months) = 0 FOR EACH customers.customer_id");
        Condition cond = assertInstanceOf(Condition.class, q.target());
        assertEquals(Operator.EQ, cond.op());
        Aggregation agg = assertInstanceOf(Aggregation.class, cond.left());
        assertTrue(agg.column().isWildcard());
        assertEquals(TimeUnit.MONTHS, agg.unit());
        assertEquals(Literal.Kind.NUMBER, cond.right().kind());
    }

    @Test
    void parsesRankTopK() {
        ParsedQuery q = Pql.parse(
                "PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID, 0, 30) RANK TOP 12 FOR EACH CUSTOMERS.CUSTOMER_ID");
        assertEquals(ProblemType.RANK, q.problemType().orElseThrow());
        assertEquals(12, q.topK().orElseThrow());
    }

    @Test
    void parsesClassify() {
        ParsedQuery q = Pql.parse(
                "PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID, 0, 30) CLASSIFY FOR EACH CUSTOMERS.CUSTOMER_ID");
        assertEquals(ProblemType.CLASSIFY, q.problemType().orElseThrow());
        assertTrue(q.topK().isEmpty());
    }

    @Test
    void parsesForecast() {
        ParsedQuery q = Pql.parse(
                "PREDICT SUM(usage.count, 0, 1, days) FORECAST 28 TIMEFRAMES FOR EACH accounts.account_id");
        assertEquals(28, q.numForecasts().orElseThrow());
        // "count" used as a column name (soft keyword)
        Aggregation agg = assertInstanceOf(Aggregation.class, q.target());
        assertEquals("count", agg.column().column());
    }

    @Test
    void parsesEntitySelectors() {
        ParsedQuery in = Pql.parse("PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR users.user_id IN (42, 123)");
        assertEquals(2, in.entityIds().size());
        assertEquals(42.0, (Double) in.entityIds().get(0).value());

        ParsedQuery eq = Pql.parse(
                "PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR users.user_id = 42 ASSUMING users.plan = 'premium'");
        assertEquals(1, eq.entityIds().size());
        assertTrue(eq.assuming().isPresent());
    }

    @Test
    void parsesInfBoundAndNegativeWindows() {
        ParsedQuery q = Pql.parse("PREDICT COUNT(transactions.*, -INF, 0) > 0 FOR EACH user.user_id");
        Condition cond = assertInstanceOf(Condition.class, q.target());
        Aggregation agg = assertInstanceOf(Aggregation.class, cond.left());
        assertEquals(Aggregation.NEG_INF, agg.start().orElseThrow());
        assertEquals(0, agg.end());
    }

    @Test
    void parsesWindowlessFilteredAggregation() {
        ParsedQuery q = Pql.parse(
                "PREDICT COUNT(transaction.* WHERE transaction.amount > 100) FOR EACH user.user_id WHERE user.country = 'US'");
        Aggregation agg = assertInstanceOf(Aggregation.class, q.target());
        assertFalse(agg.hasWindow());
        assertTrue(agg.filter().isPresent());
        assertTrue(q.where().isPresent());
    }

    @Test
    void parsesTextOperatorsAndMembership() {
        assertEquals(Operator.NOT_LIKE, ((Condition) Pql.parse(
                "PREDICT LAST(LOAN.STATUS, 0, 30) NOT LIKE '%DENIED' FOR EACH LOAN.id").target()).op());
        assertEquals(Operator.IN, ((Condition) Pql.parse(
                "PREDICT LOAN.STATUS IN ('A', 'C') FOR EACH LOAN.id").target()).op());
        assertEquals(Operator.IN, ((Condition) Pql.parse(
                "PREDICT LOAN.STATUS IS IN ('A', 'C') FOR EACH LOAN.id").target()).op());
        assertEquals(Operator.IS_NULL, ((Condition) Pql.parse(
                "PREDICT ARTICLES.DESCRIPTION IS NULL FOR EACH ARTICLES.id").target()).op());
        assertEquals(Operator.STARTS_WITH, ((Condition) Pql.parse(
                "PREDICT MOVIE.TITLE STARTS WITH 'The' FOR EACH MOVIE.id").target()).op());
        assertEquals(Operator.NOT_CONTAINS, ((Condition) Pql.parse(
                "PREDICT ARTICLES.DESCRIPTION NOT CONTAINS 'refurbished' FOR EACH ARTICLES.id").target()).op());
    }

    @Test
    void keywordsAreCaseInsensitive() {
        assertDoesNotThrow(() -> Pql.parse(
                "predict sum(TRANSACTIONS.PRICE, 0, 30) for each CUSTOMERS.CUSTOMER_ID"));
    }

    @ParameterizedTest
    @ValueSource(strings = {
        "",                                                            // empty
        "SELECT * FROM t",                                             // not PQL
        "PREDICT FOR EACH CUSTOMERS.CUSTOMER_ID",                      // missing target
        "PREDICT SUM(TRANSACTIONS.PRICE, 0, 30)",                      // missing FOR EACH
        "PREDICT SUM(TRANSACTIONS.PRICE, 0, 30) FOR EACH CUSTOMERS",   // no .column
        "PREDICT SUM(TRANSACTIONS.PRICE, 0) FOR EACH CUSTOMERS.CUSTOMER_ID",  // one bound
        "PREDICT SUM(TRANSACTIONS.PRICE 0, 30) FOR EACH CUSTOMERS.CUSTOMER_ID", // missing comma
        "PREDICT LIST_DISTINCT(T.C, 0, 30) RANK TOP FOR EACH T.ID",    // missing K
        "PREDICT COUNT(t.*, 0, 30) = FOR EACH t.id",                   // dangling comparison
        "PREDICT SUM(TRANSACTIONS.PRICE, 0, 30) FOR EACH CUSTOMERS.CUSTOMER_ID extra garbage",
        "PREDICT SUM(t.c, 0, 30, fortnights) FOR EACH t.id",           // bad unit
        "PREDICT SUM(t.c, 0, 30) FORECAST TIMEFRAMES FOR EACH t.id",   // missing N
    })
    void malformedQueriesAreRejected(String bad) {
        assertThrows(PqlSyntaxException.class, () -> Pql.parse(bad));
    }

    @Test
    void syntaxErrorCarriesLocation() {
        PqlSyntaxException e = assertThrows(PqlSyntaxException.class,
                () -> Pql.parse("PREDICT SUM(t.c, 0, 30) FOR EACH t.id trailing"));
        assertEquals(1, e.line());
        assertTrue(e.charPositionInLine() > 0);
    }
}
