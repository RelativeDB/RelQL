package com.relativedb.rt;

import com.relativedb.model.ModelBackend;
import com.relativedb.model.ModelCapabilities;
import com.relativedb.model.ModelConfig;
import com.relativedb.model.ModelOutput;
import com.relativedb.model.TokenBatch;
import com.relativedb.query.TaskType;
import com.relativedb.schema.ValueType;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@link ModelBackend} over the golden-verified native C++ RT engine
 * ({@code librt_c}). Scores {@link TokenBatch}es with the real Relational
 * Transformer instead of test fakes.
 *
 * <p><b>Routing</b>: classification-family {@link TaskType}s go to the
 * classification checkpoint, regression/forecasting to the regression
 * checkpoint — via {@link ModelConfig#modelUriFor(TaskType)}, with URIs
 * resolved by {@link CheckpointResolver} (file://, plain paths, and the local
 * Hugging Face cache for hf:// URIs). Checkpoints load lazily on first use of
 * their task family and are cached until {@link #close()}.
 *
 * <p><b>TokenBatch mapping</b> (one cell = one token, RAW PRE-SORT order —
 * the engine sorts internally):
 * <ul>
 *   <li>{@code node_idxs} ← {@code rowId}</li>
 *   <li>{@code f2p} ← {@code parentRowIds}, capped at 5, padded with -1</li>
 *   <li>{@code col_idxs}/{@code table_idxs} ← column/table names interned per
 *       forward call</li>
 *   <li>{@code sem_types} ← NUMBER=0, TEXT=1, DATETIME=2, BOOLEAN=3</li>
 *   <li>value channels ← {@code normalizedValue} into number/datetime/boolean;
 *       TEXT cells go through the {@link TextEncoder} into {@code text_v}</li>
 *   <li>{@code col_name_v} ← {@link TextEncoder} of the column name, every token</li>
 * </ul>
 *
 * <p><b>Decoding</b>: classification checkpoints emit logits — this backend
 * applies a sigmoid to fill {@link ModelOutput#probability()}. The regression
 * checkpoint emits NORMALIZED values, returned raw in
 * {@link ModelOutput#value()} (denormalization with train-split stats is the
 * caller's concern), and forecasting returns the same value as a
 * single-element forecast.
 */
public final class RtNativeBackend implements ModelBackend, AutoCloseable {

    private static final int MAX_PARENTS = 5;
    private static final int DIM = TextEncoder.DIMENSION;

    /**
     * Multiclass softmax temperature — a shared contract constant (identical in
     * the Python and Rust bindings). Turns the (uncalibrated) cosine
     * similarities into the DISTRIBUTION return shape; {@code argmax} is
     * invariant to it.
     */
    static final double T_SOFTMAX = 0.1;
    /** L2-normalization epsilon (mandatory, identical across bindings). */
    static final double NORM_EPS = 1e-8;

    private final ModelConfig config;
    private final TextEncoder encoder;
    private final int nThreads;
    private final Map<Path, RtModel> models = new ConcurrentHashMap<>();
    /** label string → L2-normalized MiniLM embedding (fixed per label). */
    private final Map<String, float[]> labelEmbeddings = new ConcurrentHashMap<>();

    public RtNativeBackend(ModelConfig config, TextEncoder encoder) {
        this(config, encoder, 0);
    }

    /** {@code nThreads <= 0} lets the engine pick hardware concurrency. */
    public RtNativeBackend(ModelConfig config, TextEncoder encoder, int nThreads) {
        this.config = Objects.requireNonNull(config, "config");
        this.encoder = Objects.requireNonNull(encoder, "encoder");
        this.nThreads = nThreads;
    }

    @Override
    public ModelCapabilities capabilities() {
        return ModelCapabilities.all(8192);
    }

    @Override
    public CompletionStage<ModelOutput> score(TokenBatch batch, TaskType taskType) {
        try {
            float raw = scoreAll(List.of(batch), taskType)[0];
            return CompletableFuture.completedFuture(decode(raw, taskType));
        } catch (Throwable t) {
            return CompletableFuture.failedFuture(t);
        }
    }

    /**
     * Scores several entities in one native forward pass (one batch row per
     * TokenBatch, padded to the longest). Returns the RAW engine scores —
     * classification logits / normalized regression values — length
     * {@code batches.size()}.
     */
    public float[] scoreAll(List<TokenBatch> batches, TaskType taskType) {
        if (batches.isEmpty()) return new float[0];
        NativeArrays a = buildNative(batches, false);
        RtModel model = modelFor(taskType);
        return model.forward(a.b, a.s, a.nodeIdxs, a.f2p, a.colIdxs, a.tableIdxs,
            a.isPadding, a.semTypes, a.isTarget, a.numberV, a.datetimeV, a.booleanV,
            a.textV, a.colNameV, nThreads);
    }

    // ------------------------------------------------------------------
    //  Multiclass classification (contract §2)
    // ------------------------------------------------------------------

    /**
     * Multiclass decode: the entity context's target cell(s) are masked as TEXT,
     * the TEXT decoder head prediction is L2-normalized, then matched by cosine
     * against the L2-normalized MiniLM embeddings of {@code classLabels}. Returns
     * a {@code softmax(cosine / T_SOFTMAX)} distribution over the labels, in the
     * given label order.
     */
    @Override
    public ModelOutput classifyMulticlass(TokenBatch entityContext,
                                          List<String> classLabels, TaskType taskType) {
        if (classLabels.isEmpty()) {
            throw new IllegalArgumentException("multiclass needs at least one class label");
        }
        NativeArrays a = buildNative(List.of(entityContext), /*maskTargetsAsText=*/true);
        RtModel model = modelFor(taskType);
        float[] text = model.forwardTargetText(a.b, a.s, a.nodeIdxs, a.f2p, a.colIdxs,
            a.tableIdxs, a.isPadding, a.semTypes, a.isTarget, a.numberV, a.datetimeV,
            a.booleanV, a.textV, a.colNameV, nThreads);   // length DIM (B == 1)
        float[] pred = l2normalize(Arrays.copyOf(text, DIM));

        int k = classLabels.size();
        double[] cos = new double[k];
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            cos[i] = dot(pred, labelEmbedding(classLabels.get(i)));
            double logit = cos[i] / T_SOFTMAX;
            if (logit > maxLogit) maxLogit = logit;
        }
        double sumExp = 0.0;
        double[] exp = new double[k];
        for (int i = 0; i < k; i++) {
            exp[i] = Math.exp(cos[i] / T_SOFTMAX - maxLogit);
            sumExp += exp[i];
        }
        Map<String, Double> classProbs = new LinkedHashMap<>();
        for (int i = 0; i < k; i++) {
            classProbs.put(classLabels.get(i), exp[i] / sumExp);
        }
        return ModelOutput.multiclass(classProbs);
    }

    // ------------------------------------------------------------------
    //  Ranking (contract §3)
    // ------------------------------------------------------------------

    /**
     * Ranking decode: score one existence context per candidate (number head),
     * apply a sigmoid, and return the candidate id → probability map ordered by
     * descending probability (ties broken by the input candidate order).
     */
    @Override
    public ModelOutput rankCandidates(List<TokenBatch> candidateContexts,
                                      List<String> candidateIds, TaskType taskType) {
        if (candidateContexts.size() != candidateIds.size()) {
            throw new IllegalArgumentException("candidateContexts/candidateIds size mismatch: "
                + candidateContexts.size() + " != " + candidateIds.size());
        }
        if (candidateContexts.isEmpty()) return ModelOutput.ranking(Map.of());

        float[] logits = scoreAll(candidateContexts, taskType);
        int n = logits.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        double[] prob = new double[n];
        for (int i = 0; i < n; i++) prob[i] = sigmoid(logits[i]);
        // Descending probability; ties broken by ascending candidate index.
        Arrays.sort(order, (x, y) -> {
            int c = Double.compare(prob[y], prob[x]);
            return c != 0 ? c : Integer.compare(x, y);
        });
        Map<String, Double> ranked = new LinkedHashMap<>();
        for (int idx : order) ranked.put(candidateIds.get(idx), prob[idx]);
        return ModelOutput.ranking(ranked);
    }

    // ------------------------------------------------------------------
    //  Native array assembly (shared by all forward paths)
    // ------------------------------------------------------------------

    /** Packed RAW PRE-SORT native input arrays for one forward call. */
    private static final class NativeArrays {
        int b;
        int s;
        long[] nodeIdxs;
        long[] f2p;
        long[] colIdxs;
        long[] tableIdxs;
        byte[] isPadding;
        long[] semTypes;
        byte[] isTarget;
        float[] numberV;
        float[] datetimeV;
        float[] booleanV;
        float[] textV;
        float[] colNameV;
    }

    /**
     * Packs {@code batches} into the native arrays. When {@code maskTargetsAsText}
     * is set, every target cell is forced to {@code sem_type = TEXT} with a
     * zeroed {@code text_v} (the model substitutes its learned mask embedding) —
     * the masked-TEXT target cell the multiclass recipe requires. The column-name
     * embedding is still emitted for every cell.
     */
    private NativeArrays buildNative(List<TokenBatch> batches, boolean maskTargetsAsText) {
        NativeArrays a = new NativeArrays();
        int b = batches.size();
        int s = batches.stream().mapToInt(TokenBatch::size).max().orElse(0);
        if (s == 0) throw new IllegalArgumentException("all TokenBatches are empty");
        a.b = b;
        a.s = s;
        a.nodeIdxs = new long[b * s];
        a.f2p = new long[b * s * MAX_PARENTS];
        Arrays.fill(a.f2p, -1L);
        a.colIdxs = new long[b * s];
        a.tableIdxs = new long[b * s];
        a.isPadding = new byte[b * s];
        a.semTypes = new long[b * s];
        a.isTarget = new byte[b * s];
        a.numberV = new float[b * s];
        a.datetimeV = new float[b * s];
        a.booleanV = new float[b * s];
        a.textV = new float[b * s * DIM];
        a.colNameV = new float[b * s * DIM];

        Map<String, Long> colIntern = new HashMap<>();
        Map<String, Long> tableIntern = new HashMap<>();

        for (int bi = 0; bi < b; bi++) {
            List<TokenBatch.Token> tokens = batches.get(bi).tokens();
            for (int si = 0; si < s; si++) {
                int i = bi * s + si;
                if (si >= tokens.size()) {
                    a.isPadding[i] = 1;
                    continue;
                }
                TokenBatch.Token t = tokens.get(si);
                a.nodeIdxs[i] = t.rowId();
                List<Integer> parents = t.parentRowIds();
                int np = Math.min(parents == null ? 0 : parents.size(), MAX_PARENTS);
                for (int p = 0; p < np; p++) {
                    a.f2p[i * MAX_PARENTS + p] = parents.get(p);
                }
                a.colIdxs[i] = intern(colIntern, t.column());
                a.tableIdxs[i] = intern(tableIntern, t.table());
                boolean target = t.isTarget();
                a.isTarget[i] = target ? (byte) 1 : (byte) 0;
                if (maskTargetsAsText && target) {
                    // Masked TEXT target cell: sem_type=TEXT, text_v = 384 zeros
                    // (the model uses its mask embedding). Value channels ignored.
                    a.semTypes[i] = semType(ValueType.TEXT);
                } else {
                    a.semTypes[i] = semType(t.valueType());
                    switch (t.valueType()) {
                        case NUMBER -> a.numberV[i] = value(t);
                        case DATETIME -> a.datetimeV[i] = value(t);
                        case BOOLEAN -> a.booleanV[i] = value(t);
                        case TEXT -> {
                            if (t.text() != null) {
                                copyEmbedding(encoder.encode(t.text()), a.textV, i,
                                    "text of " + t.column());
                            }
                        }
                    }
                }
                // Schema phrase "<column> of <table>" — identical to the Python
                // (`f"{c} of {t}"`) and Rust (`format!("{} of {}")`) bindings, so
                // the frozen encoder's col_name_v matches across languages.
                String phrase = t.column() + " of " + t.table();
                copyEmbedding(encoder.encode(phrase), a.colNameV, i, "column name " + phrase);
            }
        }
        return a;
    }

    /** L2-normalized MiniLM embedding of a raw class-label string (cached). */
    private float[] labelEmbedding(String label) {
        return labelEmbeddings.computeIfAbsent(label, l -> {
            float[] emb = encoder.encode(l);
            if (emb == null || emb.length != DIM) {
                throw new IllegalArgumentException("TextEncoder returned "
                    + (emb == null ? "null" : emb.length + "-dim")
                    + " embedding for class label '" + l + "'; expected " + DIM);
            }
            return l2normalize(emb.clone());
        });
    }

    /** In-place L2 normalization with the mandatory {@code +1e-8} epsilon. */
    static float[] l2normalize(float[] v) {
        double sq = 0.0;
        for (float x : v) sq += (double) x * x;
        double inv = 1.0 / (Math.sqrt(sq) + NORM_EPS);
        for (int i = 0; i < v.length; i++) v[i] = (float) (v[i] * inv);
        return v;
    }

    static double dot(float[] a, float[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += (double) a[i] * b[i];
        return s;
    }

    /** The lazily-loaded checkpoint serving {@code taskType} (shared, cached). */
    public RtModel modelFor(TaskType taskType) {
        Path path = CheckpointResolver.resolve(config.modelUriFor(taskType));
        return models.computeIfAbsent(path, RtModel::load);
    }

    private static ModelOutput decode(float raw, TaskType taskType) {
        return switch (taskType) {
            case BINARY_CLASSIFICATION -> ModelOutput.binary(sigmoid(raw));
            case FORECASTING -> ModelOutput.forecast(List.of((double) raw));
            case REGRESSION -> ModelOutput.regression(raw);
            // A single scalar cannot carry a multiclass distribution or a ranking.
            // Previously these silently collapsed to a binary sigmoid — the engine
            // now routes them to classifyMulticlass()/rankCandidates() instead.
            case MULTICLASS_CLASSIFICATION, MULTILABEL_RANKING ->
                throw new UnsupportedOperationException("task " + taskType
                    + " must be decoded via classifyMulticlass()/rankCandidates(), "
                    + "not the single-score score() path");
        };
    }

    static double sigmoid(double logit) {
        return 1.0 / (1.0 + Math.exp(-logit));
    }

    /** RT sem-type ids: NUMBER=0, TEXT=1, DATETIME=2, BOOLEAN=3. */
    static long semType(ValueType type) {
        return switch (type) {
            case NUMBER -> 0;
            case TEXT -> 1;
            case DATETIME -> 2;
            case BOOLEAN -> 3;
        };
    }

    private static float value(TokenBatch.Token t) {
        double v = t.normalizedValue();
        return Double.isNaN(v) ? 0f : (float) v;
    }

    private static long intern(Map<String, Long> table, String name) {
        return table.computeIfAbsent(name, k -> (long) table.size());
    }

    private static void copyEmbedding(float[] emb, float[] dest, int tokenIndex, String what) {
        if (emb == null || emb.length != DIM) {
            throw new IllegalArgumentException("TextEncoder returned "
                + (emb == null ? "null" : emb.length + "-dim") + " embedding for "
                + what + "; expected " + DIM);
        }
        System.arraycopy(emb, 0, dest, tokenIndex * DIM, DIM);
    }

    @Override
    public void close() {
        models.values().forEach(RtModel::close);
        models.clear();
    }
}
