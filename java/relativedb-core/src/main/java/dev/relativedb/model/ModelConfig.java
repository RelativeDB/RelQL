package dev.relativedb.model;

import dev.relativedb.query.TaskType;

import java.util.Objects;

/**
 * Which model checkpoints and which frozen text/schema encoder to use.
 * All are URIs/names, resolvable without any bundled storage:
 * <ul>
 *   <li>model: {@code hf://<org>/<repo>[/<subdir>]} (Hugging Face) or {@code file://...}</li>
 *   <li>embeddings: a SentenceTransformers model name or {@code file://...} path</li>
 * </ul>
 *
 * RT-J ships TWO separate checkpoints — a classifier and a regressor — so the
 * config holds one URI per task family and the engine routes by the query's
 * inferred {@link TaskType}:
 * <pre>
 *   BINARY/MULTICLASS CLASSIFICATION, RANKING → classificationModelUri
 *   REGRESSION, FORECASTING                   → regressionModelUri
 * </pre>
 *
 * CONSTRAINT (KB F13/F14): the embedding model must be the one the checkpoints
 * were trained with — rt-j pins "all-MiniLM-L12-v2" (384-dim), shared by both
 * variants. Loaders that find an {@code embedding_model} in a checkpoint config
 * MUST verify it against this setting and fail fast on mismatch unless
 * {@link Builder#allowEmbeddingMismatch(boolean)} is set.
 */
public final class ModelConfig {

    public static final String DEFAULT_CLASSIFICATION_MODEL_URI =
        "hf://stanford-star/rt-j/classification";
    public static final String DEFAULT_REGRESSION_MODEL_URI =
        "hf://stanford-star/rt-j/regression";
    public static final String DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L12-v2";

    private final String classificationModelUri;
    private final String regressionModelUri;
    private final String embeddingModel;
    private final boolean allowEmbeddingMismatch;

    private ModelConfig(BuilderImpl b) {
        this.classificationModelUri = b.classificationModelUri;
        this.regressionModelUri = b.regressionModelUri;
        this.embeddingModel = b.embeddingModel;
        this.allowEmbeddingMismatch = b.allowEmbeddingMismatch;
    }

    public static ModelConfig defaults() { return newConfig().build(); }

    public static Builder newConfig() { return new BuilderImpl(); }

    public interface Builder {
        Builder classificationModelUri(String uri);
        Builder regressionModelUri(String uri);
        /** Convenience: one checkpoint for ALL task types. Sets both URIs. */
        Builder modelUri(String uri);
        Builder embeddingModel(String nameOrUri);
        /** Escape hatch: skip the checkpoint↔encoder compatibility check. */
        Builder allowEmbeddingMismatch(boolean b);
        ModelConfig build();
    }

    /** Routing accessor: which checkpoint serves this task type. */
    public String modelUriFor(TaskType taskType) {
        return taskType.isClassificationFamily() ? classificationModelUri : regressionModelUri;
    }

    public String classificationModelUri() { return classificationModelUri; }
    public String regressionModelUri() { return regressionModelUri; }
    public String embeddingModel() { return embeddingModel; }
    public boolean allowEmbeddingMismatch() { return allowEmbeddingMismatch; }

    /** Embedding dimension of the configured text encoder (384 for MiniLM-L12). */
    public int textDimension() {
        return embeddingModel.contains("MiniLM") ? 384 : 384; // MiniLM family; override point
    }

    private static final class BuilderImpl implements Builder {
        private String classificationModelUri = DEFAULT_CLASSIFICATION_MODEL_URI;
        private String regressionModelUri = DEFAULT_REGRESSION_MODEL_URI;
        private String embeddingModel = DEFAULT_EMBEDDING_MODEL;
        private boolean allowEmbeddingMismatch = false;

        @Override public Builder classificationModelUri(String uri) {
            this.classificationModelUri = Objects.requireNonNull(uri); return this;
        }
        @Override public Builder regressionModelUri(String uri) {
            this.regressionModelUri = Objects.requireNonNull(uri); return this;
        }
        @Override public Builder modelUri(String uri) {
            Objects.requireNonNull(uri);
            this.classificationModelUri = uri;
            this.regressionModelUri = uri;
            return this;
        }
        @Override public Builder embeddingModel(String nameOrUri) {
            this.embeddingModel = Objects.requireNonNull(nameOrUri); return this;
        }
        @Override public Builder allowEmbeddingMismatch(boolean b) {
            this.allowEmbeddingMismatch = b; return this;
        }
        @Override public ModelConfig build() { return new ModelConfig(this); }
    }
}
