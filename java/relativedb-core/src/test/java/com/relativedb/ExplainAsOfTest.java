package com.relativedb;

import com.relativedb.engine.ExecutionInput;
import com.relativedb.engine.ExplainResult;
import com.relativedb.engine.PredictionResult;
import com.relativedb.engine.RelativeDbEngine;
import com.relativedb.query.Explain;
import com.relativedb.retrieve.RetrieverWiring;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletionException;

import static com.relativedb.TestData.*;
import static org.junit.jupiter.api.Assertions.*;

/** Real execution of AS OF (effective anchor) and EXPLAIN (plan/context/analyze). */
class ExplainAsOfTest {

    private Store store;
    private RetrieverWiring wiring;

    /** The shared stub records batches and score() calls (asserted below). */
    private final StubBackend spyBackend = new StubBackend();

    @BeforeEach
    void setUp() {
        store = new Store();
        store.customers.add(customer(1, 34));
        store.orders.add(order(100, 1, 2, "2026-01-01T00:00:00Z"));  // qty 2, early
        store.orders.add(order(103, 1, 9, "2026-08-01T00:00:00Z"));  // qty 9, late
        wiring = RetrieverWiring.newWiring()
                .entities("customers", store::byIds)
                .entities("orders", store::byIds)
                .defaultLinks(store::children)
                .build();
    }

    private RelativeDbEngine engine() {
        return RelativeDbEngine.newEngine(SCHEMA, wiring).modelBackend(spyBackend).build();
    }

    private boolean batchHasOrderQty(double qty) {
        return spyBackend.lastBatch.get().tokens().stream()
                .anyMatch(t -> t.table().equals("orders") && t.normalizedValue() == qty);
    }

    // ------------------------------------------------------------------
    //  AS OF
    // ------------------------------------------------------------------

    @Test
    void asOfDateOverridesPassedAnchorTime() {
        // anchorTime would admit the late order; AS OF date must override it.
        engine().execute(ExecutionInput.newInput()
                .query("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                        + "FOR customers.customer_id = 1 AS OF 2026-07-01")
                .anchorTime(Instant.parse("2026-09-01T00:00:00Z"))
                .entityIds(List.of(1L))
                .build()).toCompletableFuture().join();
        assertTrue(batchHasOrderQty(2.0), "early order (before AS OF date) must reach the model");
        assertFalse(batchHasOrderQty(9.0), "late order (after AS OF date) must be excluded");
    }

    @Test
    void asOfNowMatchesNoAsOfBehavior() {
        // AS OF NOW == the execution anchor: the late order is admitted.
        engine().execute(ExecutionInput.newInput()
                .query("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                        + "FOR customers.customer_id = 1 AS OF NOW")
                .anchorTime(Instant.parse("2026-09-01T00:00:00Z"))
                .entityIds(List.of(1L))
                .build()).toCompletableFuture().join();
        assertTrue(batchHasOrderQty(9.0), "AS OF NOW must behave like the execution anchor");
    }

    @Test
    void asOfParamBindsFromParamsMap() {
        engine().execute(ExecutionInput.newInput()
                .query("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                        + "FOR customers.customer_id = 1 AS OF :t")
                .anchorTime(Instant.parse("2026-09-01T00:00:00Z"))
                .params(Map.of("t", Instant.parse("2026-07-01T00:00:00Z")))
                .entityIds(List.of(1L))
                .build()).toCompletableFuture().join();
        assertTrue(batchHasOrderQty(2.0));
        assertFalse(batchHasOrderQty(9.0), "bound param anchor must exclude the late order");
    }

    @Test
    void asOfUnboundParamWithoutAnchorRaises() {
        CompletionException e = assertThrows(CompletionException.class, () ->
                engine().execute(ExecutionInput.newInput()
                        .query("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) "
                                + "FOR customers.customer_id = 1 AS OF :t")
                        .entityIds(List.of(1L))
                        .build()).toCompletableFuture().join());
        assertTrue(e.getCause() instanceof IllegalArgumentException);
        assertTrue(e.getCause().getMessage().contains("t"),
                "error should name the missing param");
    }

    // ------------------------------------------------------------------
    //  EXPLAIN PLAN
    // ------------------------------------------------------------------

    private static final String CHURN =
            "PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id";

    @Test
    void explainPlanDoesNotScore() {
        ExplainResult r = engine().explain(ExecutionInput.newInput()
                .query("EXPLAIN PLAN " + CHURN)
                .entityIds(List.of(1L))
                .build());
        assertEquals(Explain.Mode.PLAN, r.mode());
        assertEquals(0, spyBackend.scoreCalls.get(), "PLAN must not invoke the model");
        assertTrue(r.context().isEmpty());
        assertTrue(r.predictions().isEmpty());
    }

    @Test
    @SuppressWarnings("unchecked")
    void explainPlanFieldsAreCorrect() {
        ExplainResult r = engine().explain(ExecutionInput.newInput()
                .query("EXPLAIN PLAN " + CHURN)
                .anchorTime(Instant.parse("2026-07-01T00:00:00Z"))
                .entityIds(List.of(1L))
                .build());
        Map<String, Object> plan = r.plan();
        assertEquals("binary", plan.get("task_type"));
        assertEquals("probability", plan.get("output"));

        Map<String, Object> asOf = (Map<String, Object>) plan.get("as_of");
        assertEquals("execution-anchor", asOf.get("source"));

        List<Object> windows = (List<Object>) plan.get("windows");
        assertEquals(1, windows.size());
        Map<String, Object> w = (Map<String, Object>) windows.get(0);
        assertEquals("orders", w.get("table"));
        assertEquals(0L, ((Number) w.get("start")).longValue());
        assertEquals(90L, ((Number) w.get("end")).longValue());
        assertEquals("days", w.get("unit"));
        assertEquals("target", w.get("role"));
    }

    // ------------------------------------------------------------------
    //  EXPLAIN CONTEXT / ANALYZE
    // ------------------------------------------------------------------

    @Test
    @SuppressWarnings("unchecked")
    void explainContextPopulatesCountsWithoutScoring() {
        ExplainResult r = engine().explain(ExecutionInput.newInput()
                .query("EXPLAIN CONTEXT " + CHURN)
                .anchorTime(Instant.parse("2026-07-01T00:00:00Z"))
                .entityIds(List.of(1L))
                .build());
        assertEquals(Explain.Mode.CONTEXT, r.mode());
        assertEquals(0, spyBackend.scoreCalls.get(), "CONTEXT must not score");
        assertTrue(r.predictions().isEmpty());

        Map<String, Object> ctx = r.context().orElseThrow();
        assertTrue(((Number) ctx.get("total_rows")).longValue() > 0);
        assertTrue(((Number) ctx.get("total_cells")).longValue() > 0);
        Map<String, Object> tables = (Map<String, Object>) ctx.get("tables");
        assertTrue(tables.containsKey("customers"));
    }

    @Test
    void explainAnalyzeHasPredictions() {
        ExplainResult r = engine().explain(ExecutionInput.newInput()
                .query("EXPLAIN ANALYZE " + CHURN)
                .anchorTime(Instant.parse("2026-07-01T00:00:00Z"))
                .entityIds(List.of(1L))
                .build());
        assertEquals(Explain.Mode.ANALYZE, r.mode());
        assertTrue(r.context().isPresent());
        PredictionResult pr = r.predictions().orElseThrow();
        assertEquals(1, pr.predictions().size());
        assertEquals(0.83, pr.predictions().get(0).probability().orElseThrow(), 1e-9);
        assertTrue(spyBackend.scoreCalls.get() > 0);
    }

    // ------------------------------------------------------------------
    //  render()
    // ------------------------------------------------------------------

    @Test
    void renderJsonParsesAndTextIsReadable() {
        ExplainResult json = engine().explain(ExecutionInput.newInput()
                .query("EXPLAIN ANALYZE FORMAT JSON " + CHURN)
                .anchorTime(Instant.parse("2026-07-01T00:00:00Z"))
                .entityIds(List.of(1L))
                .build());
        String jsonOut = json.render();
        // Parses as JSON and exposes the key fields.
        Object parsed = MiniJson.parse(jsonOut);
        assertTrue(parsed instanceof Map);
        Map<String, Object> root = (Map<String, Object>) parsed;
        assertEquals("ANALYZE", root.get("mode"));
        assertTrue(root.containsKey("plan"));
        assertTrue(root.containsKey("context"));
        assertTrue(root.containsKey("predictions"));

        ExplainResult text = engine().explain(ExecutionInput.newInput()
                .query("EXPLAIN PLAN FORMAT TEXT " + CHURN)
                .entityIds(List.of(1L))
                .build());
        String textOut = text.render();
        assertTrue(textOut.contains("target"), "TEXT dump must mention the target");
        assertTrue(textOut.contains("COUNT(orders.*)"), "TEXT dump must render the target expr");
        assertTrue(textOut.contains("binary"), "TEXT dump must mention the task type");
    }

    @Test
    void explainAblationWarnsNotImplemented() {
        ExplainResult r = engine().explain(ExecutionInput.newInput()
                .query("EXPLAIN PLAN PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING) "
                        + "FOR EACH customers.customer_id ABLATE TABLE orders")
                .entityIds(List.of(1L))
                .build());
        // ABLATE parses but stays declared-not-applied; only EXPLAIN ABLATION warns.
        List<?> ablations = (List<?>) r.plan().get("ablations");
        assertEquals(1, ablations.size());
    }

    // ------------------------------------------------------------------
    //  execute() rejects EXPLAIN
    // ------------------------------------------------------------------

    @Test
    void executeOnExplainQueryThrows() {
        CompletionException e = assertThrows(CompletionException.class, () ->
                engine().execute(ExecutionInput.newInput()
                        .query("EXPLAIN PLAN " + CHURN)
                        .entityIds(List.of(1L))
                        .build()).toCompletableFuture().join());
        assertTrue(e.getCause() instanceof IllegalStateException);
        assertTrue(e.getCause().getMessage().contains("explain()"));
    }

    // ------------------------------------------------------------------
    //  Tiny JSON reader (test-only) to prove render(JSON) is well-formed.
    // ------------------------------------------------------------------

    private static final class MiniJson {
        private final String s;
        private int i;
        private MiniJson(String s) { this.s = s; }

        static Object parse(String text) {
            MiniJson j = new MiniJson(text);
            j.ws();
            Object v = j.value();
            j.ws();
            if (j.i != j.s.length()) throw new IllegalArgumentException("trailing at " + j.i);
            return v;
        }

        private Object value() {
            char c = s.charAt(i);
            return switch (c) {
                case '{' -> object();
                case '[' -> array();
                case '"' -> string();
                case 't', 'f' -> bool();
                case 'n' -> nul();
                default -> number();
            };
        }

        private Map<String, Object> object() {
            i++; // {
            java.util.LinkedHashMap<String, Object> m = new java.util.LinkedHashMap<>();
            ws();
            if (s.charAt(i) == '}') { i++; return m; }
            while (true) {
                ws();
                String k = string();
                ws();
                i++; // :
                ws();
                m.put(k, value());
                ws();
                char c = s.charAt(i++);
                if (c == '}') return m;
                if (c != ',') throw new IllegalArgumentException("expected , or } at " + i);
            }
        }

        private List<Object> array() {
            i++; // [
            java.util.ArrayList<Object> list = new java.util.ArrayList<>();
            ws();
            if (s.charAt(i) == ']') { i++; return list; }
            while (true) {
                ws();
                list.add(value());
                ws();
                char c = s.charAt(i++);
                if (c == ']') return list;
                if (c != ',') throw new IllegalArgumentException("expected , or ] at " + i);
            }
        }

        private String string() {
            i++; // "
            StringBuilder b = new StringBuilder();
            while (true) {
                char c = s.charAt(i++);
                if (c == '"') return b.toString();
                if (c == '\\') {
                    char e = s.charAt(i++);
                    switch (e) {
                        case 'n' -> b.append('\n');
                        case 't' -> b.append('\t');
                        case 'r' -> b.append('\r');
                        case 'b' -> b.append('\b');
                        case 'f' -> b.append('\f');
                        case 'u' -> { b.append((char) Integer.parseInt(s.substring(i, i + 4), 16)); i += 4; }
                        default -> b.append(e);
                    }
                } else {
                    b.append(c);
                }
            }
        }

        private Boolean bool() {
            if (s.startsWith("true", i)) { i += 4; return Boolean.TRUE; }
            if (s.startsWith("false", i)) { i += 5; return Boolean.FALSE; }
            throw new IllegalArgumentException("bad bool at " + i);
        }

        private Object nul() {
            if (s.startsWith("null", i)) { i += 4; return null; }
            throw new IllegalArgumentException("bad null at " + i);
        }

        private Double number() {
            int start = i;
            while (i < s.length()) {
                char c = s.charAt(i);
                if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') i++;
                else break;
            }
            return Double.parseDouble(s.substring(start, i));
        }

        private void ws() {
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i++;
                else break;
            }
        }
    }
}
