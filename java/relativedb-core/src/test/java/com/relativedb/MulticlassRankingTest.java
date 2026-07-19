package com.relativedb;

import com.relativedb.engine.ExecutionInput;
import com.relativedb.engine.PredictionResult;
import com.relativedb.engine.PredictionResult.EntityPrediction;
import com.relativedb.engine.PredictionResult.RankedItem;
import com.relativedb.engine.RelativeDbEngine;
import com.relativedb.model.ModelBackend;
import com.relativedb.model.ModelCapabilities;
import com.relativedb.model.ModelOutput;
import com.relativedb.model.TokenBatch;
import com.relativedb.query.TaskType;
import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.RetrieverWiring;
import com.relativedb.retrieve.Row;
import com.relativedb.retrieve.TemporalBound;
import com.relativedb.schema.LinkDef;
import com.relativedb.schema.RelativeDbSchema;
import com.relativedb.schema.TableDef;
import com.relativedb.schema.ValueType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Flow;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end routing of MULTICLASS_CLASSIFICATION and MULTILABEL_RANKING through
 * the engine with a recording fake backend: verifies the engine enumerates the
 * class-label domain / candidate pool from the {@code TableScanner}s (temporally
 * bounded, canonically ordered, capped), assembles the masked-text / per-candidate
 * batches, and shapes {@code RETURN}/top-k correctly. Backend model math is
 * covered separately by the native test in relativedb-rt.
 */
class MulticlassRankingTest {

    private static final Instant ANCHOR = Instant.parse("2026-04-01T00:00:00Z");

    private static final RelativeDbSchema SCHEMA = RelativeDbSchema.newSchema()
            .table(TableDef.newTable("customers")
                    .column("age", ValueType.NUMBER)
                    .column("plan", ValueType.TEXT)
                    .primaryKey("customer_id")
                    .build())
            .table(TableDef.newTable("articles")
                    .column("price", ValueType.NUMBER)
                    .primaryKey("article_id")
                    .build())
            .table(TableDef.newTable("orders")
                    .column("qty", ValueType.NUMBER)
                    .primaryKey("order_id")
                    .timeColumn("order_date")
                    .build())
            .link(LinkDef.link("orders", "customer_id", "customers"))
            .link(LinkDef.link("orders", "article_id", "articles"))
            .build();

    private RecordingBackend backend;
    private RetrieverWiring wiring;

    @BeforeEach
    void setUp() {
        Store store = new Store();
        store.customers.add(Row.newRow("customers", EntityId.of(1L))
                .cell("age", 34).cell("plan", "premium").build());
        // Distinct plan values (unsorted on insertion) — the class-label domain.
        store.customers.add(Row.newRow("customers", EntityId.of(2L))
                .cell("age", 20).cell("plan", "basic").build());
        store.customers.add(Row.newRow("customers", EntityId.of(3L))
                .cell("age", 51).cell("plan", "gold").build());

        store.articles.add(Row.newRow("articles", EntityId.of(30L)).cell("price", 9.0).build());
        store.articles.add(Row.newRow("articles", EntityId.of(10L)).cell("price", 5.0).build());
        store.articles.add(Row.newRow("articles", EntityId.of(20L)).cell("price", 7.0).build());

        store.orders.add(order(100, 1, 10, 2, "2026-01-10T00:00:00Z"));
        store.orders.add(order(101, 1, 20, 4, "2026-02-10T00:00:00Z"));
        // After the anchor: must not affect enumeration or context.
        store.orders.add(order(102, 1, 30, 1, "2026-09-10T00:00:00Z"));

        backend = new RecordingBackend();
        wiring = RetrieverWiring.newWiring()
                .entities("customers", store::byIds)
                .entities("articles", store::byIds)
                .entities("orders", store::byIds)
                .defaultLinks(store::children)
                .scanner("customers", store::scan)
                .scanner("articles", store::scan)
                .scanner("orders", store::scan)
                .build();
    }

    private static Row order(long id, long cust, long article, double qty, String time) {
        return Row.newRow("orders", EntityId.of(id))
                .cell("qty", qty)
                .timestamp(Instant.parse(time))
                .parent("customer_id", EntityId.of(cust))
                .parent("article_id", EntityId.of(article))
                .build();
    }

    private EntityPrediction runOne(String pql) {
        RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring)
                .modelBackend(backend)
                .build();
        PredictionResult r = engine.execute(ExecutionInput.newInput()
                .query(pql)
                .anchorTime(ANCHOR)
                .entityIds(List.of(1L))
                .build()).toCompletableFuture().join();
        assertEquals(1, r.predictions().size());
        return r.predictions().get(0);
    }

    // ------------------------------------------------------------------
    //  Multiclass
    // ------------------------------------------------------------------

    @Test
    void multiclassEnumeratesSortedClassDomainAndReturnsDistribution() {
        EntityPrediction p = runOne(
                "PREDICT customers.plan FOR EACH customers.customer_id RETURN DISTRIBUTION");

        // Class-label domain: distinct plan values sorted UTF-8 ascending.
        assertEquals(List.of("basic", "gold", "premium"), backend.lastLabels);
        Map<String, Double> dist = p.classProbs();
        assertEquals(backend.lastLabels, new ArrayList<>(dist.keySet()),
                "distribution keys must be in the canonical class order");
        assertEquals(1.0, dist.values().stream().mapToDouble(Double::doubleValue).sum(), 1e-9);

        // The single context reached the multiclass path (not score()).
        assertEquals(0, backend.scoreCalls);
        assertEquals(1, backend.multiclassCalls);
    }

    @Test
    void multiclassReturnClassPicksArgmaxLabel() {
        EntityPrediction p = runOne(
                "PREDICT customers.plan FOR EACH customers.customer_id RETURN CLASS");
        // RecordingBackend puts the highest probability on the last label.
        assertEquals("premium", p.predictedClass().orElseThrow());
        assertTrue(p.classProbs().isEmpty(), "CLASS returns a label, not the distribution");
    }

    // ------------------------------------------------------------------
    //  Ranking
    // ------------------------------------------------------------------

    @Test
    void rankingEnumeratesSortedCandidatesInjectsLinkAndAppliesTopK() {
        EntityPrediction p = runOne(
                "PREDICT LIST_DISTINCT(orders.article_id) RANK TOP 2 "
                + "FOR EACH customers.customer_id");

        // Candidate pool: distinct article ids, numeric ascending.
        assertEquals(List.of("10", "20", "30"), backend.lastCandidateIds);
        // One context batch per candidate.
        assertEquals(3, backend.lastCandidateBatches.size());
        assertEquals(0, backend.scoreCalls);
        assertEquals(1, backend.rankingCalls);

        // Each candidate batch carries an articles token (the injected candidate).
        for (TokenBatch b : backend.lastCandidateBatches) {
            assertTrue(b.tokens().stream().anyMatch(t -> t.table().equals("articles")),
                    "candidate batch must include the candidate article as a token");
        }
        // The synthetic masked target cell (task.label) is the ONLY target token,
        // and the candidate node id is wired into its f2p alongside the entity
        // node (parity with Python/Rust: f2p = [entity_node, candidate_node]).
        TokenBatch first = backend.lastCandidateBatches.get(0);
        assertEquals(1, first.tokens().stream().filter(TokenBatch.Token::isTarget).count(),
                "exactly one masked target token (the synthetic task.label cell)");
        assertTrue(first.tokens().stream()
                        .anyMatch(t -> t.table().equals("task") && t.column().equals("label")
                                && t.isTarget() && t.parentRowIds().size() >= 2),
                "the synthetic target cell must gain the candidate parent edge (entity + candidate)");

        // RANK TOP 2 → exactly two ranked items, descending score. RecordingBackend
        // scores by ascending id, so the two highest ids come first.
        List<RankedItem> ranked = p.ranked();
        assertEquals(2, ranked.size());
        assertEquals("30", ranked.get(0).item());
        assertEquals("20", ranked.get(1).item());
        assertTrue(ranked.get(0).score() >= ranked.get(1).score());
    }

    // ------------------------------------------------------------------
    //  Fixtures
    // ------------------------------------------------------------------

    /** Records what the engine handed the backend and returns deterministic outputs. */
    private static final class RecordingBackend implements ModelBackend {
        int scoreCalls;
        int multiclassCalls;
        int rankingCalls;
        List<String> lastLabels = List.of();
        List<String> lastCandidateIds = List.of();
        List<TokenBatch> lastCandidateBatches = List.of();

        @Override public ModelCapabilities capabilities() { return ModelCapabilities.all(8192); }

        @Override public CompletionStage<ModelOutput> score(TokenBatch batch, TaskType taskType) {
            scoreCalls++;
            return CompletableFuture.completedFuture(ModelOutput.regression(0.0));
        }

        @Override public ModelOutput classifyMulticlass(TokenBatch context,
                List<String> classLabels, TaskType taskType) {
            multiclassCalls++;
            lastLabels = List.copyOf(classLabels);
            // Increasing probability by index → argmax is the last label.
            Map<String, Double> probs = new LinkedHashMap<>();
            double n = classLabels.size();
            double total = n * (n + 1) / 2.0;
            for (int i = 0; i < classLabels.size(); i++) {
                probs.put(classLabels.get(i), (i + 1) / total);
            }
            return ModelOutput.multiclass(probs);
        }

        @Override public ModelOutput rankCandidates(List<TokenBatch> candidateContexts,
                List<String> candidateIds, TaskType taskType) {
            rankingCalls++;
            lastCandidateIds = List.copyOf(candidateIds);
            lastCandidateBatches = List.copyOf(candidateContexts);
            // Score by ascending numeric id, ordered desc (ties by input order) —
            // mirrors the real backend's ordering contract.
            Integer[] order = new Integer[candidateIds.size()];
            for (int i = 0; i < order.length; i++) order[i] = i;
            double[] s = new double[candidateIds.size()];
            for (int i = 0; i < s.length; i++) s[i] = Double.parseDouble(candidateIds.get(i));
            java.util.Arrays.sort(order, (x, y) -> {
                int c = Double.compare(s[y], s[x]);
                return c != 0 ? c : Integer.compare(x, y);
            });
            Map<String, Double> ranked = new LinkedHashMap<>();
            for (int idx : order) ranked.put(candidateIds.get(idx), s[idx]);
            return ModelOutput.ranking(ranked);
        }
    }

    /** Minimal in-memory store: point retriever, children, and scanner. */
    private static final class Store {
        final List<Row> customers = new ArrayList<>();
        final List<Row> articles = new ArrayList<>();
        final List<Row> orders = new ArrayList<>();

        List<Row> table(String name) {
            return switch (name) {
                case "customers" -> customers;
                case "articles" -> articles;
                default -> orders;
            };
        }

        CompletableFuture<List<Row>> byIds(String table, List<EntityId> ids, TemporalBound bound) {
            List<Row> out = new ArrayList<>();
            for (Row r : table(table)) {
                if (ids.contains(r.id()) && r.timestamp().map(bound::admits).orElse(true)) out.add(r);
            }
            return CompletableFuture.completedFuture(out);
        }

        CompletableFuture<List<Row>> children(LinkDef link, EntityId parent,
                TemporalBound bound, int limit) {
            List<Row> out = table(link.fromTable()).stream()
                    .filter(r -> parent.equals(r.parents().get(link.fkColumn())))
                    .filter(r -> r.timestamp().map(bound::admits).orElse(true))
                    .sorted(Comparator.comparing((Row r) -> r.timestamp().orElse(Instant.MIN)).reversed())
                    .limit(limit)
                    .toList();
            return CompletableFuture.completedFuture(out);
        }

        Flow.Publisher<Row> scan(String table, TemporalBound bound) {
            List<Row> rows = table(table).stream()
                    .filter(r -> r.timestamp().map(bound::admits).orElse(true))
                    .toList();
            return subscriber -> subscriber.onSubscribe(new Flow.Subscription() {
                int next = 0;
                boolean done = false;
                @Override public void request(long n) {
                    while (n-- > 0 && next < rows.size()) subscriber.onNext(rows.get(next++));
                    if (next >= rows.size() && !done) { done = true; subscriber.onComplete(); }
                }
                @Override public void cancel() { done = true; }
            });
        }
    }
}
