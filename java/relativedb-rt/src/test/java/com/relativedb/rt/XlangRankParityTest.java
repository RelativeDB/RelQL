package com.relativedb.rt;

import com.relativedb.engine.ContextPolicy;
import com.relativedb.engine.ExecutionInput;
import com.relativedb.engine.PredictionResult;
import com.relativedb.engine.PredictionResult.EntityPrediction;
import com.relativedb.engine.PredictionResult.RankedItem;
import com.relativedb.engine.RelativeDbEngine;
import com.relativedb.model.ModelConfig;
import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.RetrieverWiring;
import com.relativedb.retrieve.Row;
import com.relativedb.retrieve.TemporalBound;
import com.relativedb.schema.LinkDef;
import com.relativedb.schema.RelativeDbSchema;
import com.relativedb.schema.TableDef;
import com.relativedb.schema.ValueType;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Flow;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Cross-language ranking-parity test (Java side). Loads the shared fixture at
 * {@code benchmarks/xlang_fixture/} — a fixed real MovieLens Top-5 ranking
 * scenario with precomputed MiniLM embeddings (normalize=False) — runs it
 * through the relativedb Java engine + native RT-J backend, and asserts the
 * golden.json invariants:
 *
 * <ul>
 *   <li>the ranking is NOT the degenerate candidate-enumeration order
 *       {@code [1,2,3,50,260]} (the signature of the no-target-token bug);</li>
 *   <li>top1 == movie 593 (Silence of the Lambs) for BOTH users — the stable,
 *       tie-free signal Python and Rust both produce;</li>
 *   <li>at least {@code min_distinct_scores} distinct candidate scores.</li>
 * </ul>
 *
 * <p>Mirrors the Python test's context policy: one hop entity->events, a wide
 * budget so the full per-user history + all candidates enter context.
 */
class XlangRankParityTest {

    private static final Path FIXTURE = Path.of(System.getProperty("xlang.fixture",
        "/Users/henneberger/getasterisk/benchmarks/xlang_fixture"));

    private static final RelativeDbSchema SCHEMA = RelativeDbSchema.newSchema()
        .table(TableDef.newTable("users").primaryKey("user_id").build())
        .table(TableDef.newTable("movies")
            .column("title", ValueType.TEXT).column("genres", ValueType.TEXT)
            .primaryKey("movie_id").build())
        .table(TableDef.newTable("ratings")
            .column("rating", ValueType.NUMBER).column("ts", ValueType.DATETIME)
            .primaryKey("rating_id").timeColumn("ts").build())
        .link(LinkDef.link("ratings", "user_id", "users"))
        .link(LinkDef.link("ratings", "movie_id", "movies"))
        .build();

    // One hop entity->events, wide budget (mirrors the Python WIDE_POLICY).
    private static final ContextPolicy WIDE = ContextPolicy.newPolicy()
        .maxContextCells(5_000_000).bfsWidth(20_000).maxHops(1).build();

    @Test
    void reproducesFixtureInvariants() throws IOException {
        assumeTrue(RtNative.isAvailable(), "librt_c not available");
        assumeTrue(GoldenData.classificationCheckpointPresent(),
            "classification checkpoint not in local HF cache");
        assumeTrue(Files.isDirectory(FIXTURE), "xlang fixture missing: " + FIXTURE);

        Golden g = Golden.load(FIXTURE.resolve("golden.json"));
        PrecomputedEncoder encoder = loadEmbeddings(FIXTURE.resolve("embeddings.tsv"));

        Store store = new Store();
        for (String[] f : readTsv(FIXTURE.resolve("movies.tsv"))) {
            store.movies.add(Row.newRow("movies", EntityId.of(Long.parseLong(f[0])))
                .cell("title", f[1]).cell("genres", f[2]).build());
        }
        TreeSet<Long> userIds = new TreeSet<>();
        for (String[] f : readTsv(FIXTURE.resolve("ratings.tsv"))) {
            long uid = Long.parseLong(f[1]), mid = Long.parseLong(f[2]);
            userIds.add(uid);
            Instant ts = Instant.ofEpochSecond(Long.parseLong(f[4]));
            store.ratings.add(Row.newRow("ratings", EntityId.of(Long.parseLong(f[0])))
                .cell("rating", Double.parseDouble(f[3])).cell("ts", ts).timestamp(ts)
                .parent("user_id", EntityId.of(uid)).parent("movie_id", EntityId.of(mid))
                .build());
        }
        for (long uid : g.users) store.users.add(Row.newRow("users", EntityId.of(uid)).build());

        RetrieverWiring wiring = RetrieverWiring.newWiring()
            .entities("users", store::byIds).entities("movies", store::byIds)
            .entities("ratings", store::byIds).defaultLinks(store::children)
            .scanner("users", store::scan).scanner("movies", store::scan)
            .scanner("ratings", store::scan).build();

        Instant anchor = Instant.ofEpochSecond(g.anchorEpoch);

        try (RtNativeBackend backend = new RtNativeBackend(ModelConfig.defaults(), encoder)) {
            CapturingBackend capture = new CapturingBackend(backend);
            RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring)
                .modelBackend(capture).contextPolicy(WIDE).build();

            PredictionResult result = engine.execute(ExecutionInput.newInput()
                .query(g.query).anchorTime(anchor)
                .entityIds(new ArrayList<>(g.users)).build()).toCompletableFuture().join();

            Map<Long, List<Long>> top5 = new LinkedHashMap<>();
            for (EntityPrediction p : result.predictions()) {
                long uid = ((Number) p.id().raw()).longValue();
                List<Long> ids = new ArrayList<>();
                for (RankedItem it : p.ranked()) ids.add(Long.parseLong(it.item()));
                top5.put(uid, ids);
            }

            System.out.println("\n==== XLANG JAVA PARITY (fixture) ====");
            System.out.println("query : " + g.query);
            System.out.println("anchor: " + anchor);
            for (long uid : g.users) {
                System.out.println("user " + uid + " top5=" + top5.get(uid)
                    + "   python=" + g.pythonGolden.get(uid));
            }

            // Distinct candidate scores come from the FULL ranked score map.
            int maxDistinct = 0;
            for (Map<String, Double> scores : capture.rankedByCall.values()) {
                maxDistinct = Math.max(maxDistinct,
                    (int) scores.values().stream().distinct().count());
            }
            System.out.println("max distinct candidate scores: " + maxDistinct);
            for (var e : capture.rankedByCall.entrySet()) {
                System.out.println("scores call " + e.getKey() + ": " + e.getValue());
            }
            System.out.println("=====================================\n");

            // ---- Assertions on the golden invariants ----
            for (long uid : g.users) {
                List<Long> got = top5.get(uid);
                assertNotEquals(g.degenerate, got,
                    "user " + uid + " returned the degenerate enumeration order — ranking bug");
                assertEquals(g.expectedTop1.get(uid), got.get(0),
                    "user " + uid + " top1=" + got.get(0) + " expected " + g.expectedTop1.get(uid));
            }
            assertTrue(maxDistinct >= g.minDistinctScores,
                "expected >= " + g.minDistinctScores + " distinct candidate scores, got " + maxDistinct);
        }
    }

    // ---- Backend wrapper capturing the full ranked score map per rank call ----
    private static final class CapturingBackend implements com.relativedb.model.ModelBackend {
        private final RtNativeBackend delegate;
        final Map<Integer, Map<String, Double>> rankedByCall = new LinkedHashMap<>();
        private int calls = 0;
        CapturingBackend(RtNativeBackend d) { this.delegate = d; }
        @Override public com.relativedb.model.ModelCapabilities capabilities() {
            return delegate.capabilities();
        }
        @Override public java.util.concurrent.CompletionStage<com.relativedb.model.ModelOutput>
                score(com.relativedb.model.TokenBatch b, com.relativedb.query.TaskType t) {
            return delegate.score(b, t);
        }
        @Override public com.relativedb.model.ModelOutput classifyMulticlass(
                com.relativedb.model.TokenBatch c, List<String> labels,
                com.relativedb.query.TaskType t) {
            return delegate.classifyMulticlass(c, labels, t);
        }
        @Override public com.relativedb.model.ModelOutput rankCandidates(
                List<com.relativedb.model.TokenBatch> ctx, List<String> ids,
                com.relativedb.query.TaskType t) {
            com.relativedb.model.ModelOutput out = delegate.rankCandidates(ctx, ids, t);
            rankedByCall.put(calls++, new LinkedHashMap<>(out.rankedScores()));
            return out;
        }
    }

    private static PrecomputedEncoder loadEmbeddings(Path tsv) throws IOException {
        Map<String, float[]> table = new java.util.HashMap<>();
        for (String line : Files.readAllLines(tsv)) {
            if (line.isBlank()) continue;
            int tab = line.indexOf('\t');
            String key = line.substring(0, tab);
            String[] nums = line.substring(tab + 1).trim().split("\\s+");
            float[] v = new float[TextEncoder.DIMENSION];
            for (int i = 0; i < v.length; i++) v[i] = Float.parseFloat(nums[i]);
            table.put(key, v);
        }
        return new PrecomputedEncoder(table);
    }

    private static final class Store {
        final List<Row> users = new ArrayList<>();
        final List<Row> movies = new ArrayList<>();
        final List<Row> ratings = new ArrayList<>();
        List<Row> table(String n) {
            return switch (n) { case "users" -> users; case "movies" -> movies; default -> ratings; };
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
                .limit(limit).toList();
            return CompletableFuture.completedFuture(out);
        }
        Flow.Publisher<Row> scan(String table, TemporalBound bound) {
            List<Row> rows = table(table).stream()
                .filter(r -> r.timestamp().map(bound::admits).orElse(true)).toList();
            return subscriber -> subscriber.onSubscribe(new Flow.Subscription() {
                int next = 0; boolean done = false;
                @Override public void request(long n) {
                    while (n-- > 0 && next < rows.size()) subscriber.onNext(rows.get(next++));
                    if (next >= rows.size() && !done) { done = true; subscriber.onComplete(); }
                }
                @Override public void cancel() { done = true; }
            });
        }
    }

    /** Minimal golden.json reader (no JSON lib). */
    private static final class Golden {
        long anchorEpoch;
        String query;
        int topK;
        int minDistinctScores;
        List<Long> users = new ArrayList<>();
        List<Long> degenerate = new ArrayList<>();
        Map<Long, Long> expectedTop1 = new LinkedHashMap<>();
        Map<Long, List<Long>> pythonGolden = new LinkedHashMap<>();

        static Golden load(Path path) throws IOException {
            String j = Files.readString(path);
            Golden g = new Golden();
            g.anchorEpoch = Long.parseLong(scalar(j, "anchor_epoch"));
            g.query = strScalar(j, "query");
            g.topK = Integer.parseInt(scalar(j, "top_k"));
            g.minDistinctScores = Integer.parseInt(scalar(j, "min_distinct_scores"));
            // Top-level "users": [1, 2] (the schema also has a "users" key).
            java.util.regex.Matcher um = java.util.regex.Pattern
                .compile("\"users\"\\s*:\\s*\\[([^\\]]*)\\]").matcher(j);
            if (um.find()) for (String u : arr(um.group(1))) g.users.add(Long.parseLong(u));
            for (String u : arr(section(objectBlock(j, "invariants"),
                    "\"must_not_equal_degenerate_order\""))) g.degenerate.add(Long.parseLong(u));
            String top1Block = objectBlock(objectBlock(j, "invariants"), "expected_top1");
            for (String[] kv : intPairs(top1Block)) {
                g.expectedTop1.put(Long.parseLong(kv[0]), Long.parseLong(kv[1]));
            }
            String pyBlock = objectBlock(objectBlock(j, "per_binding_golden"), "python");
            for (long uid : g.users) {
                String a = section(pyBlock, "\"" + uid + "\"");
                List<Long> ids = new ArrayList<>();
                for (String s : arr(a)) ids.add(Long.parseLong(s));
                g.pythonGolden.put(uid, ids);
            }
            return g;
        }
        static String scalar(String j, String key) {
            int i = j.indexOf("\"" + key + "\"");
            int c = j.indexOf(':', i) + 1, e = c;
            while (e < j.length() && ",}\n".indexOf(j.charAt(e)) < 0) e++;
            return j.substring(c, e).trim();
        }
        static String strScalar(String j, String key) {
            int i = j.indexOf("\"" + key + "\"");
            int q1 = j.indexOf('"', j.indexOf(':', i) + 1);
            int q2 = j.indexOf('"', q1 + 1);
            return j.substring(q1 + 1, q2);
        }
        static String section(String j, String key) {
            int i = j.indexOf(key);
            int b = j.indexOf('[', i), e = j.indexOf(']', b);
            return j.substring(b + 1, e);
        }
        static String objectBlock(String j, String key) {
            int i = j.indexOf("\"" + key + "\"");
            int b = j.indexOf('{', i), depth = 0, e = b;
            for (; e < j.length(); e++) {
                if (j.charAt(e) == '{') depth++;
                else if (j.charAt(e) == '}' && --depth == 0) break;
            }
            return j.substring(b + 1, e);
        }
        static List<String> arr(String a) {
            List<String> out = new ArrayList<>();
            java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("[0-9]+").matcher(a);
            while (m.find()) out.add(m.group());
            return out;
        }
        static List<String[]> intPairs(String block) {
            List<String[]> out = new ArrayList<>();
            java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\"([0-9]+)\"\\s*:\\s*([0-9]+)").matcher(block);
            while (m.find()) out.add(new String[]{m.group(1), m.group(2)});
            return out;
        }
    }

    private static List<String[]> readTsv(Path path) throws IOException {
        List<String[]> rows = new ArrayList<>();
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank()) continue;
            rows.add(line.split("\t", -1));
        }
        return rows;
    }
}
