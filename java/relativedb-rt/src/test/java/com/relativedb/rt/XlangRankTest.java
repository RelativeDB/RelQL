package com.relativedb.rt;

import com.relativedb.engine.ExecutionInput;
import com.relativedb.engine.PredictionResult;
import com.relativedb.engine.PredictionResult.EntityPrediction;
import com.relativedb.engine.PredictionResult.RankedItem;
import com.relativedb.engine.RelativeDbEngine;
import com.relativedb.model.ModelConfig;
import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.LinkRetriever;
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Flow;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Cross-language reproduction: run the FIXED MovieLens ranking scenario (shared
 * with the Python + Rust harness) through the relativedb Java engine + native
 * RT-J backend, using the SAME MiniLM embeddings Python recorded, and print the
 * top-5 recommended movies per user. Goal: demonstrate the Java binding produces
 * the same top-5 as Python on identical data + embeddings.
 *
 * <p>Data/embeddings/scenario live in the shared scratchpad {@code xlang/} dir
 * (path via {@code -Dxlang.dir=...}, defaulting to the session scratchpad).
 * Nothing here is modified in library {@code src/main}.
 */
class XlangRankTest {

    private static final Path XLANG = Path.of(System.getProperty("xlang.dir",
        "/private/tmp/claude-501/-Users-henneberger-getasterisk/"
        + "9892aecd-aaa6-41c7-bd26-817485487547/scratchpad/xlang"));

    // ---- Schema (mirrors the Python movielens schema for this scenario) -----
    private static final RelativeDbSchema SCHEMA = RelativeDbSchema.newSchema()
        .table(TableDef.newTable("users")
            .primaryKey("user_id")
            .build())
        .table(TableDef.newTable("movies")
            .column("title", ValueType.TEXT)
            .column("genres", ValueType.TEXT)
            .primaryKey("movie_id")
            .build())
        .table(TableDef.newTable("ratings")
            .column("rating", ValueType.NUMBER)
            .column("ts", ValueType.DATETIME)
            .primaryKey("rating_id")
            .timeColumn("ts")
            .build())
        .link(LinkDef.link("ratings", "user_id", "users"))
        .link(LinkDef.link("ratings", "movie_id", "movies"))
        .build();

    @Test
    void reproducesPythonTop5() throws IOException {
        assumeTrue(RtNative.isAvailable(), "librt_c not available");
        assumeTrue(GoldenData.classificationCheckpointPresent(),
            "classification checkpoint not in local HF cache");
        assumeTrue(Files.isDirectory(XLANG), "shared xlang dir missing: " + XLANG);

        // ---- shared inputs ---------------------------------------------------
        Scenario sc = Scenario.load(XLANG.resolve("scenario.json"));
        MissRecordingEncoder encoder =
            MissRecordingEncoder.fromTsv(XLANG.resolve("embeddings.tsv"));

        Store store = new Store();
        // movies: movie_id \t title \t genres
        for (String[] f : readTsv(XLANG.resolve("movies.tsv"))) {
            store.movies.add(Row.newRow("movies", EntityId.of(Long.parseLong(f[0])))
                .cell("title", f[1])
                .cell("genres", f[2])
                .build());
            encoder.knownTitlesAndGenres.add(f[1]);
            encoder.knownTitlesAndGenres.add(f[2]);
        }
        // ratings: rating_id \t user_id \t movie_id \t rating \t ts_epoch_seconds
        TreeSet<Long> users = new TreeSet<>();
        for (String[] f : readTsv(XLANG.resolve("ratings.tsv"))) {
            long uid = Long.parseLong(f[1]);
            long mid = Long.parseLong(f[2]);
            users.add(uid);
            Instant ts = Instant.ofEpochSecond(Long.parseLong(f[4]));
            store.ratings.add(Row.newRow("ratings", EntityId.of(Long.parseLong(f[0])))
                .cell("rating", Double.parseDouble(f[3]))
                .cell("ts", ts)
                .timestamp(ts)
                .parent("user_id", EntityId.of(uid))
                .parent("movie_id", EntityId.of(mid))
                .build());
        }
        // users table has no feature cells; the seed row must still EXIST so the
        // context frontier is non-empty and its ratings children get expanded.
        for (Long uid : sc.users) {
            store.users.add(Row.newRow("users", EntityId.of(uid)).build());
        }

        // ---- wiring: entities + scanner for all three, one defaultLinks ------
        RetrieverWiring wiring = RetrieverWiring.newWiring()
            .entities("users", store::byIds)
            .entities("movies", store::byIds)
            .entities("ratings", store::byIds)
            .defaultLinks(store::children)
            .scanner("users", store::scan)
            .scanner("movies", store::scan)
            .scanner("ratings", store::scan)
            .build();

        Instant anchor = Instant.ofEpochSecond(sc.anchorEpoch);

        try (RtNativeBackend backend =
                 new RtNativeBackend(ModelConfig.defaults(), encoder)) {

            CapturingBackend capture = new CapturingBackend(backend);
            RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring)
                .modelBackend(capture)
                .build();

            PredictionResult result = engine.execute(ExecutionInput.newInput()
                .query(sc.query)
                .anchorTime(anchor)
                .entityIds(new ArrayList<>(sc.users))
                .build()).toCompletableFuture().join();

            // ---- report ------------------------------------------------------
            StringBuilder out = new StringBuilder();
            out.append("\n==== XLANG JAVA RT-J RANKING ====\n");
            out.append("query : ").append(sc.query).append('\n');
            out.append("anchor: ").append(anchor).append(" (epoch ")
               .append(sc.anchorEpoch).append(")\n");
            out.append("users : ").append(sc.users).append("   top_k=")
               .append(sc.topK).append('\n');

            Map<Long, List<String>> javaResult = new LinkedHashMap<>();
            for (EntityPrediction p : result.predictions()) {
                long uid = ((Number) p.id().raw()).longValue();
                List<String> ids = new ArrayList<>();
                List<String> titles = new ArrayList<>();
                for (RankedItem it : p.ranked()) {
                    ids.add(it.item());
                    titles.add(sc.titleOf.getOrDefault(it.item(), it.item()));
                }
                javaResult.put(uid, ids);
                out.append("user ").append(uid).append(": ")
                   .append(String.join(" | ", titles)).append('\n');
            }

            out.append("\n---- parity vs Python (by movie_id) ----\n");
            boolean allMatch = true;
            for (long uid : sc.users) {
                List<String> py = sc.pythonResult.get(String.valueOf(uid));
                List<String> jv = javaResult.get(uid);
                boolean m = py != null && py.equals(jv);
                allMatch &= m;
                out.append("user ").append(uid).append(": ")
                   .append(m ? "MATCH" : "DIFF").append("\n")
                   .append("   python: ").append(py).append('\n')
                   .append("   java  : ").append(jv).append('\n');
            }
            out.append("ALL USERS MATCH PYTHON: ").append(allMatch).append('\n');

            // ---- embedding misses -------------------------------------------
            List<String> misses = encoder.missesInOrder();
            out.append("\n---- embedding misses ----\n");
            out.append("distinct miss count: ").append(misses.size()).append('\n');
            out.append("examples: ")
               .append(misses.subList(0, Math.min(12, misses.size()))).append('\n');
            // Any miss that is a movie title or genre string is a BUG.
            List<String> badMisses = new ArrayList<>();
            for (String miss : misses) {
                if (encoder.knownTitlesAndGenres.contains(miss)) badMisses.add(miss);
            }
            out.append("title/genre value misses (should be 0): ")
               .append(badMisses.size());
            if (!badMisses.isEmpty()) out.append("  <-- BUG: ").append(badMisses);
            out.append('\n');
            out.append("\n---- raw per-candidate scores (full, pre-topK) ----\n");
            for (var e : capture.rankedByCall.entrySet()) {
                out.append("call ").append(e.getKey()).append(": ").append(e.getValue()).append('\n');
            }

            out.append("\n---- ALL strings Java requested from encoder ----\n");
            out.append(encoder.requestedOrder).append('\n');

            // ---- DIAGNOSTIC: alias bare col names to Python's contextual keys --
            // Python's RtNativeBackend encodes col_name_v as "<col> of <table>"
            // and system tokens as "<x> of task"; Java's encodes the BARE column
            // name, so on the shared table those miss and col_name_v is zeroed.
            // Feed the Python vectors under the bare keys and re-rank to test
            // whether col_name_v is what drives the divergence.
            Map<String, float[]> aliased = new java.util.HashMap<>(encoder.table);
            putAlias(aliased, "title", "title of movies");
            putAlias(aliased, "genres", "genres of movies");
            putAlias(aliased, "rating", "rating of ratings");
            putAlias(aliased, "ts", "ts of ratings");
            MissRecordingEncoder aliasEnc = new MissRecordingEncoder(aliased);
            try (RtNativeBackend backend2 =
                     new RtNativeBackend(ModelConfig.defaults(), aliasEnc)) {
                RelativeDbEngine engine2 = RelativeDbEngine.newEngine(SCHEMA, wiring)
                    .modelBackend(backend2).build();
                PredictionResult r2 = engine2.execute(ExecutionInput.newInput()
                    .query(sc.query).anchorTime(anchor)
                    .entityIds(new ArrayList<>(sc.users)).build())
                    .toCompletableFuture().join();
                out.append("\n---- DIAGNOSTIC: col-name-aliased re-rank ----\n");
                boolean aliasMatch = true;
                for (EntityPrediction p : r2.predictions()) {
                    long uid = ((Number) p.id().raw()).longValue();
                    List<String> ids = new ArrayList<>();
                    for (RankedItem it : p.ranked()) ids.add(it.item());
                    List<String> py = sc.pythonResult.get(String.valueOf(uid));
                    boolean m = py != null && py.equals(ids);
                    aliasMatch &= m;
                    out.append("user ").append(uid).append(": ").append(m ? "MATCH" : "DIFF")
                       .append("  java=").append(ids).append("  python=").append(py).append('\n');
                }
                out.append("ALIASED ALL USERS MATCH PYTHON: ").append(aliasMatch).append('\n');
            }
            // ---- CONFIRMATION: give `users` a target cell so the number head
            // has a masked target token to score (existence prediction). ------
            RelativeDbSchema schema3 = RelativeDbSchema.newSchema()
                .table(TableDef.newTable("users").column("u", ValueType.NUMBER)
                    .primaryKey("user_id").build())
                .table(TableDef.newTable("movies").column("title", ValueType.TEXT)
                    .column("genres", ValueType.TEXT).primaryKey("movie_id").build())
                .table(TableDef.newTable("ratings").column("rating", ValueType.NUMBER)
                    .column("ts", ValueType.DATETIME).primaryKey("rating_id")
                    .timeColumn("ts").build())
                .link(LinkDef.link("ratings", "user_id", "users"))
                .link(LinkDef.link("ratings", "movie_id", "movies"))
                .build();
            Store store3 = new Store();
            store3.movies.addAll(store.movies);
            store3.ratings.addAll(store.ratings);
            for (Long uid : sc.users) {
                store3.users.add(Row.newRow("users", EntityId.of(uid)).cell("u", 0.0).build());
            }
            RetrieverWiring wiring3 = RetrieverWiring.newWiring()
                .entities("users", store3::byIds).entities("movies", store3::byIds)
                .entities("ratings", store3::byIds).defaultLinks(store3::children)
                .scanner("users", store3::scan).scanner("movies", store3::scan)
                .scanner("ratings", store3::scan).build();
            MissRecordingEncoder enc3 =
                MissRecordingEncoder.fromTsv(XLANG.resolve("embeddings.tsv"));
            try (RtNativeBackend b3 = new RtNativeBackend(ModelConfig.defaults(), enc3)) {
                CapturingBackend cap3 = new CapturingBackend(b3);
                PredictionResult r3 = RelativeDbEngine.newEngine(schema3, wiring3)
                    .modelBackend(cap3).build()
                    .execute(ExecutionInput.newInput().query(sc.query).anchorTime(anchor)
                        .entityIds(new ArrayList<>(sc.users)).build())
                    .toCompletableFuture().join();
                out.append("\n---- CONFIRMATION: users given a target cell ----\n");
                for (var e : cap3.rankedByCall.entrySet()) {
                    out.append("scores call ").append(e.getKey()).append(": ")
                       .append(e.getValue()).append('\n');
                }
                for (EntityPrediction p : r3.predictions()) {
                    List<String> ids = new ArrayList<>();
                    for (RankedItem it : p.ranked()) ids.add(it.item());
                    out.append("user ").append(((Number) p.id().raw()).longValue())
                       .append(" top5=").append(ids).append('\n');
                }
            }
            out.append("=================================\n");

            System.out.println(out);
            try {
                Files.writeString(XLANG.resolve("java_result.txt"), out.toString());
            } catch (IOException ignore) { /* reporting only */ }
        }
    }

    // ------------------------------------------------------------------------
    //  Backend wrapper that records the FULL ranked score map per rank call.
    // ------------------------------------------------------------------------
    private static final class CapturingBackend
            implements com.relativedb.model.ModelBackend {
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

    // ------------------------------------------------------------------------
    //  Encoder: precomputed table, ZERO-on-miss, records the missing strings.
    // ------------------------------------------------------------------------
    private static final class MissRecordingEncoder implements TextEncoder {
        final Map<String, float[]> table;
        private final Map<String, Boolean> misses = new ConcurrentHashMap<>();
        private final List<String> missOrder =
            java.util.Collections.synchronizedList(new ArrayList<>());
        final java.util.Set<String> knownTitlesAndGenres = new java.util.HashSet<>();

        MissRecordingEncoder(Map<String, float[]> table) { this.table = table; }

        static MissRecordingEncoder fromTsv(Path tsv) throws IOException {
            Map<String, float[]> table = new java.util.HashMap<>();
            for (String line : Files.readAllLines(tsv)) {
                if (line.isBlank()) continue;
                int tab = line.indexOf('\t');
                String key = line.substring(0, tab);
                String[] nums = line.substring(tab + 1).trim().split("\\s+");
                if (nums.length != TextEncoder.DIMENSION) {
                    throw new IllegalStateException("embedding for '" + key
                        + "' has " + nums.length + " dims, expected " + DIMENSION);
                }
                float[] v = new float[DIMENSION];
                for (int i = 0; i < DIMENSION; i++) v[i] = Float.parseFloat(nums[i]);
                table.put(key, v);
            }
            return new MissRecordingEncoder(table);
        }

        final List<String> requestedOrder =
            java.util.Collections.synchronizedList(new ArrayList<>());
        private final Map<String, Boolean> requested = new ConcurrentHashMap<>();

        @Override public float[] encode(String text) {
            if (requested.putIfAbsent(text, Boolean.TRUE) == null) requestedOrder.add(text);
            float[] v = table.get(text);
            if (v != null) return v;
            if (misses.putIfAbsent(text, Boolean.TRUE) == null) missOrder.add(text);
            return new float[DIMENSION]; // zero vector on miss
        }

        List<String> missesInOrder() { return new ArrayList<>(missOrder); }
    }

    // ------------------------------------------------------------------------
    //  In-memory store (point retriever, newest-first children, scanner).
    // ------------------------------------------------------------------------
    private static final class Store {
        final List<Row> users = new ArrayList<>();
        final List<Row> movies = new ArrayList<>();
        final List<Row> ratings = new ArrayList<>();

        List<Row> table(String name) {
            return switch (name) {
                case "users" -> users;
                case "movies" -> movies;
                default -> ratings;
            };
        }

        CompletableFuture<List<Row>> byIds(String table, List<EntityId> ids,
                                           TemporalBound bound) {
            List<Row> out = new ArrayList<>();
            for (Row r : table(table)) {
                if (ids.contains(r.id())
                        && r.timestamp().map(bound::admits).orElse(true)) {
                    out.add(r);
                }
            }
            return CompletableFuture.completedFuture(out);
        }

        CompletableFuture<List<Row>> children(LinkDef link, EntityId parent,
                                              TemporalBound bound, int limit) {
            List<Row> out = table(link.fromTable()).stream()
                .filter(r -> parent.equals(r.parents().get(link.fkColumn())))
                .filter(r -> r.timestamp().map(bound::admits).orElse(true))
                .sorted(Comparator.comparing(
                    (Row r) -> r.timestamp().orElse(Instant.MIN)).reversed())
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

    // ------------------------------------------------------------------------
    //  Minimal scenario.json parsing (no JSON lib needed).
    // ------------------------------------------------------------------------
    private static final class Scenario {
        long anchorEpoch;
        String query;
        List<Long> users = new ArrayList<>();
        int topK;
        Map<String, String> titleOf = new LinkedHashMap<>();
        Map<String, List<String>> pythonResult = new LinkedHashMap<>();

        static Scenario load(Path path) throws IOException {
            String j = Files.readString(path);
            Scenario s = new Scenario();
            s.anchorEpoch = Long.parseLong(scalar(j, "anchor_epoch"));
            s.query = strScalar(j, "query");
            s.topK = Integer.parseInt(scalar(j, "top_k"));
            for (String u : arrScalars(section(j, "\"users\""))) s.users.add(Long.parseLong(u));
            // title_of: { "1": "Toy Story (1995)", ... }
            String titleBlock = objectBlock(j, "title_of");
            for (String[] kv : keyStringPairs(titleBlock)) s.titleOf.put(kv[0], kv[1]);
            // python_result: { "1": ["593", ...], "2": [...] }
            String prBlock = objectBlock(j, "python_result");
            for (Long uid : s.users) {
                String arr = section(prBlock, "\"" + uid + "\"");
                s.pythonResult.put(String.valueOf(uid), arrStringScalars(arr));
            }
            return s;
        }

        // --- tiny hand-rolled JSON helpers (scenario is well-formed) ---
        static String scalar(String j, String key) {
            int i = j.indexOf("\"" + key + "\"");
            int c = j.indexOf(':', i) + 1;
            int e = c;
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
            int b = j.indexOf('[', i);
            int e = j.indexOf(']', b);
            return j.substring(b + 1, e);
        }
        static String objectBlock(String j, String key) {
            int i = j.indexOf("\"" + key + "\"");
            int b = j.indexOf('{', i);
            int depth = 0, e = b;
            for (; e < j.length(); e++) {
                if (j.charAt(e) == '{') depth++;
                else if (j.charAt(e) == '}' && --depth == 0) break;
            }
            return j.substring(b + 1, e);
        }
        static List<String> arrScalars(String arr) {
            List<String> out = new ArrayList<>();
            for (String p : arr.split(",")) {
                String t = p.trim();
                if (!t.isEmpty()) out.add(t);
            }
            return out;
        }
        static List<String> arrStringScalars(String arr) {
            List<String> out = new ArrayList<>();
            java.util.regex.Matcher m =
                java.util.regex.Pattern.compile("\"([^\"]*)\"").matcher(arr);
            while (m.find()) out.add(m.group(1));
            return out;
        }
        static List<String[]> keyStringPairs(String block) {
            List<String[]> out = new ArrayList<>();
            java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\"([^\"]*)\"\\s*:\\s*\"([^\"]*)\"").matcher(block);
            while (m.find()) out.add(new String[]{m.group(1), m.group(2)});
            return out;
        }
    }

    private static void putAlias(Map<String, float[]> m, String bareKey, String pyKey) {
        float[] v = m.get(pyKey);
        if (v != null) m.put(bareKey, v);
    }

    private static List<String[]> readTsv(Path path) throws IOException {
        List<String[]> rows = new ArrayList<>();
        for (String line : Files.readAllLines(path)) {
            if (line.isBlank()) continue;
            rows.add(line.split("\t", -1));
        }
        return rows;
    }

    // Keep an unused import honest for readers scanning the wiring SPI types.
    @SuppressWarnings("unused")
    private static final Class<?> LINK_RETRIEVER = LinkRetriever.class;
}
