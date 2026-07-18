package com.relativedb.engine;

import com.relativedb.model.ModelBackend;
import com.relativedb.model.ModelConfig;
import com.relativedb.model.ModelOutput;
import com.relativedb.model.TokenBatch;
import com.relativedb.query.Pql;
import com.relativedb.query.TaskType;
import com.relativedb.query.ValidatedQuery;
import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.RetrieverWiring;
import com.relativedb.retrieve.Row;
import com.relativedb.retrieve.StatsProvider;
import com.relativedb.retrieve.TemporalBound;
import com.relativedb.schema.RelativeDbSchema;
import com.relativedb.schema.TableDef;
import com.relativedb.schema.ValueType;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

/**
 * The engine: owns parsing, planning, context assembly, and model invocation —
 * never touches a database. (The design doc's {@code RqlEngine}.)
 */
public final class RelativeDbEngine {

    private final RelativeDbSchema schema;
    private final RetrieverWiring wiring;
    private final ModelBackend backend;          // nullable until resolved
    private final ModelConfig modelConfig;
    private final ContextPolicy policy;
    private final SamplerMode samplerMode;
    private final Instrumentation instrumentation;
    private volatile CscIndex cscIndex;          // lazy, CSC mode only

    private RelativeDbEngine(BuilderImpl b) {
        this.schema = b.schema;
        this.wiring = b.wiring;
        this.backend = b.backend;
        this.modelConfig = b.modelConfig != null ? b.modelConfig : ModelConfig.defaults();
        this.policy = b.policy != null ? b.policy : ContextPolicy.defaults();
        this.samplerMode = b.samplerMode;
        this.instrumentation = b.instrumentation != null ? b.instrumentation : Instrumentation.NOOP;
    }

    public static Builder newEngine(RelativeDbSchema schema, RetrieverWiring wiring) {
        return new BuilderImpl(schema, wiring);
    }

    public interface Builder {
        Builder modelBackend(ModelBackend backend);
        Builder modelConfig(ModelConfig config);
        Builder contextPolicy(ContextPolicy policy);
        Builder samplerMode(SamplerMode mode);
        Builder instrumentation(Instrumentation instr);
        RelativeDbEngine build();
    }

    public RelativeDbSchema schema() { return schema; }
    public ContextPolicy contextPolicy() { return policy; }
    public SamplerMode samplerMode() { return samplerMode; }
    public ModelConfig modelConfig() { return modelConfig; }

    // ------------------------------------------------------------------
    //  Execution
    // ------------------------------------------------------------------

    public CompletionStage<PredictionResult> execute(ExecutionInput input) {
        return CompletableFuture.supplyAsync(() -> executeSync(input));
    }

    public CompletionStage<EvaluationResult> evaluate(ExecutionInput input, List<Metric> metrics) {
        CompletableFuture<EvaluationResult> f = new CompletableFuture<>();
        f.completeExceptionally(new UnsupportedOperationException(
                "evaluate() is not implemented yet — planned alongside the reference model backends"));
        return f;
    }

    /**
     * Assemble the in-context subgraph for one entity — the testable core of
     * both sampler modes.
     */
    public ContextGraph assembleContext(String table, EntityId id, TemporalBound bound) {
        return new ContextAssembler(schema, contextSource(bound), policy, instrumentation)
                .assemble(table, id, bound);
    }

    /** CSC mode only: rebuild the materialized index from the TableScanners. */
    public void refresh() {
        if (samplerMode == SamplerMode.CSC) {
            cscIndex = CscIndex.build(schema, wiring, TemporalBound.unbounded());
        }
    }

    private ContextSource contextSource(TemporalBound bound) {
        if (samplerMode == SamplerMode.RETRIEVER) return new RetrieverContextSource(wiring);
        CscIndex index = cscIndex;
        if (index == null) {
            synchronized (this) {
                if (cscIndex == null) {
                    // Load everything once; per-query bounds are applied at read time.
                    cscIndex = CscIndex.build(schema, wiring, TemporalBound.unbounded());
                }
                index = cscIndex;
            }
        }
        return index;
    }

    private PredictionResult executeSync(ExecutionInput input) {
        ValidatedQuery vq = input.validatedQuery()
                .orElseGet(() -> Pql.validate(input.pql().orElseThrow(), schema));
        instrumentation.onQueryValidated(vq);
        TaskType taskType = vq.taskType();

        String entityTable = vq.query().entityKey().table();
        List<EntityId> ids = resolveEntityIds(input, vq, entityTable);

        Instant anchor = input.anchorTime().orElse(null);
        Instant contextAnchor = input.contextAnchorTime().orElse(anchor);
        TemporalBound bound = contextAnchor == null
                ? TemporalBound.unbounded() : TemporalBound.atOrBefore(contextAnchor);

        ModelBackend resolved = resolveBackend(taskType);

        List<PredictionResult.EntityPrediction> predictions = new ArrayList<>(ids.size());
        for (EntityId id : ids) {
            TemporalBound entityBound = bound;
            Instant effectiveAnchor = contextAnchor;
            if (input.perEntityAnchor()) {
                Optional<Instant> entityTime = contextSource(bound)
                        .byIds(entityTable, List.of(id), TemporalBound.unbounded()).stream()
                        .findFirst().flatMap(Row::timestamp);
                if (entityTime.isPresent()) {
                    effectiveAnchor = entityTime.get();
                    entityBound = TemporalBound.atOrBefore(effectiveAnchor);
                }
            }
            ContextGraph context = assembleContext(entityTable, id, entityBound);
            TokenBatch batch = toTokenBatch(context);
            instrumentation.onModelInvoked(id, batch.size());
            ModelOutput out = resolved.score(batch, taskType).toCompletableFuture().join();
            predictions.add(decode(id, taskType, out, vq.query(), context, effectiveAnchor));
        }
        return new PredictionResult(taskType, predictions);
    }

    private List<EntityId> resolveEntityIds(ExecutionInput input, ValidatedQuery vq, String entityTable) {
        if (!input.entityIds().isEmpty()) {
            return input.entityIds().stream().map(EntityId::of).toList();
        }
        if (!vq.query().entityIds().isEmpty()) {
            return vq.query().entityIds().stream().map(l -> EntityId.of(rawId(l.value()))).toList();
        }
        // FOR EACH over all entities: only enumerable with a TableScanner.
        Optional<com.relativedb.retrieve.TableScanner> scanner = wiring.scanner(entityTable);
        if (scanner.isPresent()) {
            List<EntityId> all = new ArrayList<>();
            CscIndex index = (CscIndex) (samplerMode == SamplerMode.CSC
                    ? contextSource(TemporalBound.unbounded()) : null);
            if (index != null) {
                index.cohort(entityTable, EntityId.of(new Object()), TemporalBound.unbounded(),
                        Integer.MAX_VALUE).ifPresent(all::addAll);
                return all;
            }
        }
        throw new IllegalArgumentException("FOR EACH over all entities of '" + entityTable
                + "' requires explicit entityIds(...) on the ExecutionInput (RETRIEVER mode "
                + "cannot enumerate a table) or a TableScanner + SamplerMode.CSC");
    }

    private static Object rawId(Object literalValue) {
        // Integer-valued numeric ids surface as longs (FOR users.user_id = 42).
        if (literalValue instanceof Double d && d == Math.floor(d) && !d.isInfinite()) {
            return d.longValue();
        }
        return literalValue;
    }

    private ModelBackend resolveBackend(TaskType taskType) {
        if (backend != null) return backend;
        throw new IllegalStateException("no ModelBackend configured. The core engine does not "
                + "bundle a model runtime: add a backend artifact (e.g. relativedb-model-rt) or "
                + "pass Builder.modelBackend(...). Routing was ready to use checkpoint '"
                + modelConfig.modelUriFor(taskType) + "' with embedding model '"
                + modelConfig.embeddingModel() + "'.");
    }

    // ------------------------------------------------------------------
    //  Token batch construction (one token per cell, F10)
    // ------------------------------------------------------------------

    private TokenBatch toTokenBatch(ContextGraph context) {
        StatsProvider stats = wiring.stats().orElse(null);
        TokenBatch.Builder batch = TokenBatch.newBatch();
        Map<String, Integer> rowIds = new HashMap<>();
        for (Row row : context.rows()) {
            rowIds.putIfAbsent(row.table() + " " + row.id().raw(), rowIds.size());
        }
        for (Row row : context.rows()) {
            int rowId = rowIds.get(row.table() + " " + row.id().raw());
            List<Integer> parentIds = new ArrayList<>();
            for (var e : row.parents().entrySet()) {
                schema.linksFrom(row.table()).stream()
                        .filter(l -> l.fkColumn().equals(e.getKey()))
                        .findFirst()
                        .map(l -> rowIds.get(l.toTable() + " " + e.getValue().raw()))
                        .filter(Objects::nonNull)
                        .ifPresent(parentIds::add);
            }
            boolean isSeed = row.table().equals(context.seedTable()) && row.id().equals(context.seedId());
            TableDef table = schema.table(row.table()).orElse(null);
            for (var cell : row.cells().entrySet()) {
                ValueType type = table == null ? null
                        : table.column(cell.getKey()).map(com.relativedb.schema.ColumnDef::type).orElse(null);
                Object v = cell.getValue();
                if (v instanceof Double d) {
                    double norm = stats != null
                            ? stats.numericStats(row.table(), cell.getKey()).normalize(d) : d;
                    batch.numeric(rowId, parentIds, row.table(), cell.getKey(),
                            type != null ? type : ValueType.NUMBER, norm, isSeed);
                } else if (v instanceof Boolean b) {
                    batch.numeric(rowId, parentIds, row.table(), cell.getKey(),
                            ValueType.BOOLEAN, b ? 1.0 : 0.0, isSeed);
                } else if (v instanceof Instant t) {
                    double norm = stats != null ? stats.datetimeStats().normalize(t)
                            : t.getEpochSecond();
                    batch.numeric(rowId, parentIds, row.table(), cell.getKey(),
                            ValueType.DATETIME, norm, isSeed);
                } else {
                    batch.text(rowId, parentIds, row.table(), cell.getKey(),
                            String.valueOf(v), isSeed);
                }
            }
        }
        return batch.build();
    }

    // ------------------------------------------------------------------
    //  Output decoding
    // ------------------------------------------------------------------

    private final ReturnShaper returnShaper = new ReturnShaper();

    private PredictionResult.EntityPrediction decode(EntityId id, TaskType taskType, ModelOutput out,
            com.relativedb.query.ParsedQuery query, ContextGraph context, Instant anchor) {
        return returnShaper.shape(id, taskType, out, query, context, anchor);
    }

    private static final class BuilderImpl implements Builder {
        final RelativeDbSchema schema;
        final RetrieverWiring wiring;
        ModelBackend backend;
        ModelConfig modelConfig;
        ContextPolicy policy;
        SamplerMode samplerMode = SamplerMode.RETRIEVER;
        Instrumentation instrumentation;

        BuilderImpl(RelativeDbSchema schema, RetrieverWiring wiring) {
            this.schema = Objects.requireNonNull(schema, "schema");
            this.wiring = Objects.requireNonNull(wiring, "wiring");
        }
        @Override public Builder modelBackend(ModelBackend backend) { this.backend = backend; return this; }
        @Override public Builder modelConfig(ModelConfig config) { this.modelConfig = config; return this; }
        @Override public Builder contextPolicy(ContextPolicy policy) { this.policy = policy; return this; }
        @Override public Builder samplerMode(SamplerMode mode) {
            this.samplerMode = Objects.requireNonNull(mode); return this;
        }
        @Override public Builder instrumentation(Instrumentation instr) {
            this.instrumentation = instr; return this;
        }
        @Override public RelativeDbEngine build() { return new RelativeDbEngine(this); }
    }
}
