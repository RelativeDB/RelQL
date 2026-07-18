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

    private final ModelConfig config;
    private final TextEncoder encoder;
    private final int nThreads;
    private final Map<Path, RtModel> models = new ConcurrentHashMap<>();

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
        int b = batches.size();
        int s = batches.stream().mapToInt(TokenBatch::size).max().orElse(0);
        if (s == 0) throw new IllegalArgumentException("all TokenBatches are empty");

        long[] nodeIdxs = new long[b * s];
        long[] f2p = new long[b * s * MAX_PARENTS];
        Arrays.fill(f2p, -1L);
        long[] colIdxs = new long[b * s];
        long[] tableIdxs = new long[b * s];
        byte[] isPadding = new byte[b * s];
        long[] semTypes = new long[b * s];
        byte[] isTarget = new byte[b * s];
        float[] numberV = new float[b * s];
        float[] datetimeV = new float[b * s];
        float[] booleanV = new float[b * s];
        float[] textV = new float[b * s * DIM];
        float[] colNameV = new float[b * s * DIM];

        Map<String, Long> colIntern = new HashMap<>();
        Map<String, Long> tableIntern = new HashMap<>();

        for (int bi = 0; bi < b; bi++) {
            List<TokenBatch.Token> tokens = batches.get(bi).tokens();
            for (int si = 0; si < s; si++) {
                int i = bi * s + si;
                if (si >= tokens.size()) {
                    isPadding[i] = 1;
                    continue;
                }
                TokenBatch.Token t = tokens.get(si);
                nodeIdxs[i] = t.rowId();
                List<Integer> parents = t.parentRowIds();
                int np = Math.min(parents == null ? 0 : parents.size(), MAX_PARENTS);
                for (int p = 0; p < np; p++) {
                    f2p[i * MAX_PARENTS + p] = parents.get(p);
                }
                colIdxs[i] = intern(colIntern, t.column());
                tableIdxs[i] = intern(tableIntern, t.table());
                semTypes[i] = semType(t.valueType());
                isTarget[i] = t.isTarget() ? (byte) 1 : (byte) 0;
                switch (t.valueType()) {
                    case NUMBER -> numberV[i] = value(t);
                    case DATETIME -> datetimeV[i] = value(t);
                    case BOOLEAN -> booleanV[i] = value(t);
                    case TEXT -> {
                        if (t.text() != null) {
                            copyEmbedding(encoder.encode(t.text()), textV, i, "text of " + t.column());
                        }
                    }
                }
                copyEmbedding(encoder.encode(t.column()), colNameV, i, "column name " + t.column());
            }
        }

        RtModel model = modelFor(taskType);
        return model.forward(b, s, nodeIdxs, f2p, colIdxs, tableIdxs, isPadding,
            semTypes, isTarget, numberV, datetimeV, booleanV, textV, colNameV, nThreads);
    }

    /** The lazily-loaded checkpoint serving {@code taskType} (shared, cached). */
    public RtModel modelFor(TaskType taskType) {
        Path path = CheckpointResolver.resolve(config.modelUriFor(taskType));
        return models.computeIfAbsent(path, RtModel::load);
    }

    private static ModelOutput decode(float raw, TaskType taskType) {
        if (taskType.isClassificationFamily()) {
            return ModelOutput.binary(sigmoid(raw));
        }
        if (taskType == TaskType.FORECASTING) {
            return ModelOutput.forecast(List.of((double) raw));
        }
        return ModelOutput.regression(raw);
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
