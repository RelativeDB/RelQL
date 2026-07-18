package com.relativedb;

import com.relativedb.query.Aggregation;
import com.relativedb.query.AggFunc;
import com.relativedb.query.Arith;
import com.relativedb.query.AsOf;
import com.relativedb.query.ColumnRef;
import com.relativedb.query.Condition;
import com.relativedb.query.Explain;
import com.relativedb.query.Literal;
import com.relativedb.query.Not;
import com.relativedb.query.Operator;
import com.relativedb.query.ParsedQuery;
import com.relativedb.query.Pql;
import com.relativedb.query.PqlSyntaxException;
import com.relativedb.query.ProblemType;
import com.relativedb.query.ReturnSpec;
import com.relativedb.query.TimeUnit;
import com.relativedb.query.Window;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
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
    @DisplayName("all 54 corpus queries parse")
    void corpusParses(String query) {
        ParsedQuery parsed = Pql.parse(query);
        assertNotNull(parsed.target());
        assertNotNull(parsed.entityKey());
    }

    @Test
    void corpusHas54Queries() throws Exception {
        assertEquals(54, corpus().count());
    }

    @Test
    void parsesRegressionTarget() {
        ParsedQuery q = Pql.parse(
                "PREDICT SUM(TRANSACTIONS.PRICE) OVER (30 DAYS FOLLOWING) FOR EACH CUSTOMERS.CUSTOMER_ID");
        Aggregation agg = assertInstanceOf(Aggregation.class, q.target());
        assertEquals(AggFunc.SUM, agg.func());
        assertEquals(new ColumnRef("TRANSACTIONS", "PRICE"), agg.column());
        assertEquals(0, agg.start().orElseThrow());
        assertEquals(30, agg.end());
        assertEquals(TimeUnit.DAYS, agg.unit());   // default unit
        assertEquals(1, agg.horizons());
        assertTrue(agg.step().isEmpty());
        assertTrue(agg.hasWindow());
        assertEquals(new ColumnRef("CUSTOMERS", "CUSTOMER_ID"), q.entityKey());
        assertTrue(q.entityIds().isEmpty());
    }

    @Test
    void parsesExplicitRangeWindow() {
        ParsedQuery q = Pql.parse(
                "PREDICT SUM(TRANSACTIONS.PRICE) OVER (RANGE BETWEEN 10 DAYS FOLLOWING AND 30 DAYS FOLLOWING) "
                + "FOR EACH CUSTOMERS.CUSTOMER_ID");
        Aggregation agg = assertInstanceOf(Aggregation.class, q.target());
        assertEquals(10, agg.start().orElseThrow());
        assertEquals(30, agg.end());
    }

    @Test
    void parsesExplicitUnitAndComparison() {
        ParsedQuery q = Pql.parse(
                "PREDICT COUNT(transactions.*) OVER (30 MONTHS FOLLOWING) = 0 FOR EACH customers.customer_id");
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
                "PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID) OVER (30 DAYS FOLLOWING) RANK TOP 12 "
                + "FOR EACH CUSTOMERS.CUSTOMER_ID");
        assertEquals(ProblemType.RANK, q.problemType().orElseThrow());
        assertEquals(12, q.topK().orElseThrow());
    }

    @Test
    void parsesClassify() {
        ParsedQuery q = Pql.parse(
                "PREDICT LIST_DISTINCT(TRANSACTIONS.ARTICLE_ID) OVER (30 DAYS FOLLOWING) CLASSIFY "
                + "FOR EACH CUSTOMERS.CUSTOMER_ID");
        assertEquals(ProblemType.CLASSIFY, q.problemType().orElseThrow());
        assertTrue(q.topK().isEmpty());
    }

    @Test
    void parsesHorizonsForecast() {
        ParsedQuery q = Pql.parse(
                "PREDICT SUM(usage.count) OVER (1 DAY FOLLOWING HORIZONS 28) FOR EACH accounts.account_id");
        assertEquals(28, q.numForecasts().orElseThrow());
        // "count" used as a column name (soft keyword)
        Aggregation agg = assertInstanceOf(Aggregation.class, q.target());
        assertEquals("count", agg.column().column());
        assertEquals(28, agg.horizons());
        assertTrue(agg.isMultiHorizon());
    }

    @Test
    void parsesNamedWindowStepAndHorizons() {
        ParsedQuery q = Pql.parse(
                "PREDICT SUM(sales.qty) OVER demand_projection FOR EACH stores.store_id "
                + "WINDOW demand_projection AS (30 DAYS FOLLOWING HORIZONS 6 STEP 7 DAYS)");
        Aggregation agg = assertInstanceOf(Aggregation.class, q.target());
        assertEquals(6, agg.horizons());
        assertEquals(7, agg.step().orElseThrow());
        assertEquals(6, q.numForecasts().orElseThrow());
        // The declared template is retained on the query.
        Window w = q.windows().get("demand_projection");
        assertNotNull(w);
        assertEquals(30, w.end());
        assertEquals(6, w.horizons());
        assertEquals(7, w.step().orElseThrow());
    }

    @Test
    void parsesEntitySelectors() {
        ParsedQuery in = Pql.parse(
                "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id IN (42, 123)");
        assertEquals(2, in.entityIds().size());
        assertEquals(42.0, (Double) in.entityIds().get(0).value());

        ParsedQuery eq = Pql.parse(
                "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id = 42 "
                + "ASSUMING users.plan = 'premium'");
        assertEquals(1, eq.entityIds().size());
        assertTrue(eq.assuming().isPresent());
    }

    @Test
    void parsesUnboundedPrecedingWindow() {
        ParsedQuery q = Pql.parse(
                "PREDICT COUNT(transactions.*) OVER (UNBOUNDED PRECEDING) > 0 FOR EACH user.user_id");
        Condition cond = assertInstanceOf(Condition.class, q.target());
        Aggregation agg = assertInstanceOf(Aggregation.class, cond.left());
        assertEquals(Aggregation.NEG_INF, agg.start().orElseThrow());
        assertEquals(0, agg.end());
    }

    @Test
    void parsesWindowlessFilteredAggregation() {
        ParsedQuery q = Pql.parse(
                "PREDICT COUNT(transaction.* WHERE transaction.amount > 100) FOR EACH user.user_id "
                + "WHERE user.country = 'US'");
        Aggregation agg = assertInstanceOf(Aggregation.class, q.target());
        assertFalse(agg.hasWindow());
        assertEquals(1, agg.horizons());
        assertTrue(agg.filter().isPresent());
        assertTrue(q.where().isPresent());
    }

    @Test
    void parsesExistsAndNotExists() {
        // EXISTS aggregation target.
        Aggregation exists = assertInstanceOf(Aggregation.class, Pql.parse(
                "PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id").target());
        assertEquals(AggFunc.EXISTS, exists.func());
        assertTrue(exists.column().isWildcard());

        // NOT EXISTS wraps the aggregation in a Not.
        Not not = assertInstanceOf(Not.class, Pql.parse(
                "PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FOR EACH customers.customer_id").target());
        Aggregation inner = assertInstanceOf(Aggregation.class, not.inner());
        assertEquals(AggFunc.EXISTS, inner.func());
    }

    @Test
    void parsesArithmeticTargetAndLiterals() {
        ParsedQuery q = Pql.parse(
                "PREDICT GREATEST(SUM(orders.revenue) OVER (30 DAYS FOLLOWING), 0) * 2 "
                + "FOR EACH customers.customer_id");
        Arith arith = assertInstanceOf(Arith.class, q.target());
        assertEquals('*', arith.op());
    }

    @Test
    void parsesReturnAndAsOf() {
        ParsedQuery q = Pql.parse(
                "PREDICT SUM(sales.qty) OVER (7 DAYS FOLLOWING) FOR EACH stores.store_id "
                + "AS OF :prediction_time RETURN EXPECTED VALUE");
        AsOf asOf = q.asOf().orElseThrow();
        assertEquals(AsOf.Kind.PARAM, asOf.kind());
        assertEquals("prediction_time", asOf.value());
        assertEquals(ReturnSpec.Kind.EXPECTED_VALUE, q.ret().orElseThrow().kind());
    }

    @Test
    void parsesReturnQuantilesAndInterval() {
        ReturnSpec quantiles = Pql.parse(
                "PREDICT SUM(orders.amount) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING) "
                + "FOR customers.customer_id IN ('C7', 'C9') AS OF :prediction_time "
                + "RETURN QUANTILES (0.10, 0.50, 0.90)").ret().orElseThrow();
        assertEquals(ReturnSpec.Kind.QUANTILES, quantiles.kind());
        assertArrayEquals(new double[] {0.10, 0.50, 0.90}, quantiles.quantiles(), 1e-9);

        ReturnSpec interval = Pql.parse(
                "PREDICT SUM(payments.amount) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id "
                + "AS OF :t RETURN INTERVAL 90%").ret().orElseThrow();
        assertEquals(ReturnSpec.Kind.INTERVAL, interval.kind());
        assertEquals(90, interval.interval().orElseThrow());
    }

    @Test
    void parsesExplainPrefix() {
        ParsedQuery q = Pql.parse(
                "EXPLAIN PLAN FORMAT TEXT PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) "
                + "FOR EACH customers.customer_id ABLATE TABLE support_tickets RETURN PROBABILITY");
        Explain explain = q.explain().orElseThrow();
        assertEquals(Explain.Mode.PLAN, explain.mode());
        assertEquals(Explain.Format.TEXT, explain.format());
        assertEquals(1, q.ablations().size());
        assertEquals("support_tickets", q.ablations().get(0).name());
        assertEquals(ReturnSpec.Kind.PROBABILITY, q.ret().orElseThrow().kind());
    }

    @Test
    void parsesColumnToColumnComparison() {
        Condition cond = assertInstanceOf(Condition.class, Pql.parse(
                "PREDICT customers.spend > customers.budget FOR EACH customers.customer_id").target());
        assertEquals(Operator.GT, cond.op());
        ColumnRef rhs = assertInstanceOf(ColumnRef.class, cond.rightExpr().orElseThrow());
        assertEquals(new ColumnRef("customers", "budget"), rhs);
    }

    @Test
    void parsesTextOperatorsAndMembership() {
        assertEquals(Operator.NOT_LIKE, ((Condition) Pql.parse(
                "PREDICT LAST(LOAN.STATUS) OVER (30 DAYS FOLLOWING) NOT LIKE '%DENIED' FOR EACH LOAN.id").target()).op());
        assertEquals(Operator.IN, ((Condition) Pql.parse(
                "PREDICT LOAN.STATUS IN ('A', 'C') FOR EACH LOAN.id").target()).op());
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
                "predict sum(TRANSACTIONS.PRICE) over (30 days following) for each CUSTOMERS.CUSTOMER_ID"));
    }

    @ParameterizedTest
    @ValueSource(strings = {
        "",                                                            // empty
        "SELECT * FROM t",                                             // not RelQL
        "PREDICT FOR EACH CUSTOMERS.CUSTOMER_ID",                      // missing target
        "PREDICT SUM(TRANSACTIONS.PRICE) OVER (30 DAYS FOLLOWING)",    // missing FOR EACH
        "PREDICT SUM(TRANSACTIONS.PRICE) OVER (30 DAYS FOLLOWING) FOR EACH CUSTOMERS",   // no .column
        "PREDICT SUM(TRANSACTIONS.PRICE, 0, 30) FOR EACH CUSTOMERS.CUSTOMER_ID",         // positional form removed
        "PREDICT COUNT(t.*) OVER (30 DAYS) FOR EACH t.id",            // missing PRECEDING/FOLLOWING
        "PREDICT SUM(t.v) OVER (30 DAYS FOLLOWING HORIZONS 0) FOR EACH t.id",            // HORIZONS 0
        "PREDICT SUM(t.v) OVER undeclared_win FOR EACH t.id",         // undeclared window
        "PREDICT SUM(t.v) OVER (RANGE BETWEEN 1 DAY FOLLOWING AND 2 MONTHS FOLLOWING) FOR EACH t.id", // mixed unit domains
        "PREDICT LIST_DISTINCT(T.C) OVER (30 DAYS FOLLOWING) RANK TOP FOR EACH T.ID",    // missing K
        "PREDICT COUNT(t.*) OVER (30 DAYS FOLLOWING) = FOR EACH t.id",                   // dangling comparison
        "PREDICT SUM(TRANSACTIONS.PRICE) OVER (30 DAYS FOLLOWING) FOR EACH CUSTOMERS.CUSTOMER_ID extra garbage",
        "PREDICT SUM(t.c) OVER (30 fortnights FOLLOWING) FOR EACH t.id",                 // bad unit
        "PREDICT SUM(t.c) OVER (30 DAYS FOLLOWING) FORECAST 4 TIMEFRAMES FOR EACH t.id", // FORECAST removed
    })
    void malformedQueriesAreRejected(String bad) {
        assertThrows(PqlSyntaxException.class, () -> Pql.parse(bad));
    }
}
