package com.relativedb.engine;

import com.relativedb.model.ModelBackend;
import com.relativedb.model.ModelConfig;
import com.relativedb.model.ModelOutput;
import com.relativedb.model.TokenBatch;
import com.relativedb.query.Ablation;
import com.relativedb.query.Aggregation;
import com.relativedb.query.Arith;
import com.relativedb.query.AsOf;
import com.relativedb.query.Case;
import com.relativedb.query.ColumnRef;
import com.relativedb.query.Condition;
import com.relativedb.query.Explain;
import com.relativedb.query.Func;
import com.relativedb.query.LitExpr;
import com.relativedb.query.LogicalOp;
import com.relativedb.query.Not;
import com.relativedb.query.ParsedQuery;
import com.relativedb.query.Pql;
import com.relativedb.query.ReturnSpec;
import com.relativedb.query.TargetExpr;
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
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
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
        return CompletableFuture.supplyAsync(() -> {
            ValidatedQuery vq = resolve(input);
            if (vq.query().explain().isPresent()) {
                throw new IllegalStateException(
                        "this is an EXPLAIN query — call explain() instead of execute()");
            }
            return executeScored(input, vq);
        });
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

    private ValidatedQuery resolve(ExecutionInput input) {
        return input.validatedQuery()
                .orElseGet(() -> Pql.validate(input.pql().orElseThrow(), schema));
    }

    private PredictionResult executeScored(ExecutionInput input, ValidatedQuery vq) {
        instrumentation.onQueryValidated(vq);
        TaskType taskType = vq.taskType();

        String entityTable = vq.query().entityKey().table();
        List<EntityId> ids = resolveEntityIds(input, vq, entityTable);

        Instant anchor = effectiveAnchor(input, vq.query());
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
            predictions.add(decode(id, taskType, out, vq.query()));
        }
        return new PredictionResult(taskType, predictions);
    }

    // ------------------------------------------------------------------
    //  AS OF — effective anchor resolution
    // ------------------------------------------------------------------

    /**
     * The anchor actually used for this execution, resolved from the query's
     * {@code AS OF} clause layered over {@link ExecutionInput#anchorTime()}:
     * absent/NOW use the execution anchor; DATE parses and overrides; PARAM binds
     * from {@link ExecutionInput#params()}, falling back to the execution anchor,
     * and raising if neither is available.
     */
    private Instant effectiveAnchor(ExecutionInput input, ParsedQuery query) {
        Instant base = input.anchorTime().orElse(null);
        Optional<AsOf> asOf = query.asOf();
        if (asOf.isEmpty()) return base;
        AsOf a = asOf.get();
        return switch (a.kind()) {
            case NOW -> base;
            case DATE -> parseAnchorDate(a.value());
            case PARAM -> {
                Instant bound = input.params().get(a.value());
                if (bound != null) yield bound;
                if (base != null) yield base;
                throw new IllegalArgumentException("AS OF :" + a.value()
                        + " is unbound — supply it via ExecutionInput.params(\"" + a.value()
                        + "\", ...) or set an anchorTime() fallback");
            }
        };
    }

    /** Parse an {@code AS OF <date>} literal (UTC): {@code YYYY-MM-DD} or {@code YYYY-MM-DD HH:MM:SS}. */
    static Instant parseAnchorDate(String text) {
        String t = text.trim();
        try {
            if (t.length() <= 10) {
                return LocalDate.parse(t).atStartOfDay(ZoneOffset.UTC).toInstant();
            }
            String iso = t.indexOf('T') >= 0 ? t : t.replace(' ', 'T');
            return LocalDateTime.parse(iso).toInstant(ZoneOffset.UTC);
        } catch (DateTimeParseException e) {
            throw new IllegalArgumentException("AS OF date '" + text
                    + "' is not a valid YYYY-MM-DD or YYYY-MM-DD HH:MM:SS timestamp", e);
        }
    }

    // ------------------------------------------------------------------
    //  EXPLAIN
    // ------------------------------------------------------------------

    /**
     * Compile (and, per mode, assemble and score) an EXPLAIN request. PLAN parses
     * and validates without touching the model; CONTEXT additionally assembles the
     * per-entity context and reports on it (still no scoring); ANALYZE assembles
     * and scores; ABLATION returns PLAN with a not-implemented warning. Works on a
     * non-EXPLAIN query too, defaulting to PLAN.
     */
    public ExplainResult explain(ExecutionInput input) {
        ValidatedQuery vq = resolve(input);
        ParsedQuery q = vq.query();
        Explain explain = q.explain().orElse(new Explain(Explain.Mode.PLAN, Explain.Format.TEXT));
        Explain.Mode mode = explain.mode();
        Explain.Format format = explain.format();

        Instant anchor = effectiveAnchor(input, q);
        Map<String, Object> plan = buildPlan(vq, input, anchor);

        Map<String, Object> context = null;
        PredictionResult predictions = null;
        switch (mode) {
            case PLAN -> { }
            case CONTEXT -> context = buildContext(input, vq, anchor);
            case ANALYZE -> {
                context = buildContext(input, vq, anchor);
                predictions = executeScored(input, vq);
            }
            case ABLATION -> addWarning(plan, "ablation not implemented");
        }
        return new ExplainResult(mode, format, plan, context, predictions);
    }

    @SuppressWarnings("unchecked")
    private static void addWarning(Map<String, Object> plan, String warning) {
        ((List<Object>) plan.get("warnings")).add(warning);
    }

    private Map<String, Object> buildPlan(ValidatedQuery vq, ExecutionInput input, Instant anchor) {
        ParsedQuery q = vq.query();
        Map<String, Object> plan = new LinkedHashMap<>();
        plan.put("target", renderExpr(q.target()));
        plan.put("task_type", taskTypeName(vq.taskType()));

        Map<String, Object> entity = new LinkedHashMap<>();
        entity.put("table", q.entityKey().table());
        entity.put("pk", q.entityKey().column());
        List<Object> overrideIds = input.entityIds();
        if (!overrideIds.isEmpty()) {
            entity.put("selector", overrideIds.stream().map(String::valueOf).toList());
        } else {
            entity.put("selector", "FOR EACH");
        }
        plan.put("entity", entity);

        plan.put("output", outputForm(q, vq.taskType()));

        List<Object> windows = new ArrayList<>();
        collectWindows(q.target(), "target", windows);
        q.where().ifPresent(w -> collectWindows(w, "where", windows));
        q.assuming().ifPresent(a -> collectWindows(a, "assuming", windows));
        plan.put("windows", windows);

        plan.put("where_present", q.where().isPresent());
        Map<String, Object> assuming = new LinkedHashMap<>();
        assuming.put("present", q.assuming().isPresent());
        assuming.put("note", "carried, not applied");
        plan.put("assuming", assuming);

        plan.put("as_of", asOfInfo(q, anchor));

        List<Object> ablations = new ArrayList<>();
        for (Ablation ab : q.ablations()) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("kind", ab.kind());
            m.put("name", ab.name());
            m.put("note", "declared, not applied");
            ablations.add(m);
        }
        plan.put("ablations", ablations);

        plan.put("warnings", new ArrayList<>());
        return plan;
    }

    private Map<String, Object> asOfInfo(ParsedQuery q, Instant anchor) {
        Map<String, Object> info = new LinkedHashMap<>();
        String source = "execution-anchor";
        if (q.asOf().isPresent()) {
            source = switch (q.asOf().get().kind()) {
                case DATE -> "query-date";
                case PARAM -> "query-param";
                case NOW -> "execution-anchor";
            };
        }
        info.put("source", source);
        info.put("value", anchor == null ? null : anchor.toString());
        return info;
    }

    private String outputForm(ParsedQuery q, TaskType taskType) {
        Optional<ReturnSpec> ret = q.ret();
        if (ret.isPresent()) {
            return switch (ret.get().kind()) {
                case EXPECTED_VALUE -> "expected_value";
                case PROBABILITY -> "probability";
                case CLASS -> "class";
                case DISTRIBUTION -> "distribution";
                case QUANTILES -> "quantiles";
                case INTERVAL -> "interval";
                case MULTILABEL -> "multilabel";
                case MULTICLASS -> "multiclass";
            };
        }
        return switch (taskType) {
            case REGRESSION -> "value";
            case BINARY_CLASSIFICATION -> "probability";
            case MULTICLASS_CLASSIFICATION -> "class";
            case MULTILABEL_RANKING -> "ranked";
            case FORECASTING -> "value-per-horizon";
        };
    }

    private static String taskTypeName(TaskType t) {
        return switch (t) {
            case REGRESSION -> "regression";
            case BINARY_CLASSIFICATION -> "binary";
            case MULTICLASS_CLASSIFICATION -> "multiclass";
            case MULTILABEL_RANKING -> "ranking";
            case FORECASTING -> "forecasting";
        };
    }

    /** Collect every windowed aggregation in {@code e}, tagging each with {@code role}. */
    private void collectWindows(TargetExpr e, String role, List<Object> out) {
        if (e == null) return;
        if (e instanceof Aggregation agg) {
            if (agg.hasWindow()) {
                Map<String, Object> w = new LinkedHashMap<>();
                w.put("table", agg.column().table());
                w.put("time_column", schema.table(agg.column().table())
                        .flatMap(TableDef::timeColumn).orElse(null));
                w.put("start", bound(agg.startOr(0)));
                w.put("end", bound(agg.end()));
                w.put("unit", agg.unit().name().toLowerCase(java.util.Locale.ROOT));
                w.put("horizons", agg.horizons());
                w.put("step", agg.step().isPresent() ? (Object) agg.step().getAsLong() : null);
                w.put("role", role);
                out.add(w);
            }
            agg.filter().ifPresent(f -> collectWindows(f.condition(), role, out));
            return;
        }
        if (e instanceof LogicalOp op) { collectWindows(op.left(), role, out); collectWindows(op.right(), role, out); }
        else if (e instanceof Not not) collectWindows(not.inner(), role, out);
        else if (e instanceof Arith a) { collectWindows(a.left(), role, out); collectWindows(a.right(), role, out); }
        else if (e instanceof Condition cond) {
            collectWindows(cond.left(), role, out);
            cond.rightExpr().ifPresent(r -> collectWindows(r, role, out));
        } else if (e instanceof Func f) {
            for (TargetExpr arg : f.args()) collectWindows(arg, role, out);
        } else if (e instanceof Case c) {
            for (Case.When wn : c.whens()) { collectWindows(wn.cond(), role, out); collectWindows(wn.then(), role, out); }
            if (c.elseExpr() != null) collectWindows(c.elseExpr(), role, out);
        }
    }

    /** Bound value for the plan: {@code -INF}/{@code +INF} as strings, else the long. */
    private static Object bound(long v) {
        if (v == Aggregation.NEG_INF) return "-INF";
        if (v == Aggregation.POS_INF) return "+INF";
        return v;
    }

    // ------------------------------------------------------------------
    //  EXPLAIN CONTEXT — assemble and report (no scoring)
    // ------------------------------------------------------------------

    private Map<String, Object> buildContext(ExecutionInput input, ValidatedQuery vq, Instant anchor) {
        String entityTable = vq.query().entityKey().table();
        List<EntityId> ids = resolveEntityIds(input, vq, entityTable);
        Instant contextAnchor = input.contextAnchorTime().orElse(anchor);
        TemporalBound bound = contextAnchor == null
                ? TemporalBound.unbounded() : TemporalBound.atOrBefore(contextAnchor);

        CapturingInstrumentation capture = new CapturingInstrumentation();
        ContextAssembler assembler =
                new ContextAssembler(schema, contextSource(bound), policy, capture);

        Map<String, long[]> perTable = new LinkedHashMap<>();   // table -> [rows, cells]
        Map<String, Instant[]> perTableTime = new LinkedHashMap<>(); // table -> [min, max]
        long totalRows = 0;
        long totalCells = 0;
        long linksTraversed = 0;

        for (EntityId id : ids) {
            TemporalBound entityBound = bound;
            if (input.perEntityAnchor()) {
                Optional<Instant> entityTime = contextSource(bound)
                        .byIds(entityTable, List.of(id), TemporalBound.unbounded()).stream()
                        .findFirst().flatMap(Row::timestamp);
                if (entityTime.isPresent()) {
                    entityBound = TemporalBound.atOrBefore(entityTime.get());
                }
            }
            ContextGraph graph = assembler.assemble(entityTable, id, entityBound);
            for (Row row : graph.rows()) {
                long[] tc = perTable.computeIfAbsent(row.table(), k -> new long[2]);
                tc[0] += 1;
                tc[1] += row.cellCount();
                totalRows += 1;
                totalCells += row.cellCount();
                linksTraversed += row.parents().size();
                row.timestamp().ifPresent(ts -> {
                    Instant[] mm = perTableTime.computeIfAbsent(row.table(), k -> new Instant[2]);
                    if (mm[0] == null || ts.isBefore(mm[0])) mm[0] = ts;
                    if (mm[1] == null || ts.isAfter(mm[1])) mm[1] = ts;
                });
            }
        }

        Map<String, Object> ctx = new LinkedHashMap<>();
        ctx.put("entities_covered", ids.size());
        ctx.put("total_rows", totalRows);
        ctx.put("total_cells", totalCells);
        ctx.put("links_traversed", linksTraversed);

        Map<String, Object> tables = new LinkedHashMap<>();
        for (var e : perTable.entrySet()) {
            Map<String, Object> t = new LinkedHashMap<>();
            t.put("rows", e.getValue()[0]);
            t.put("cells", e.getValue()[1]);
            Instant[] mm = perTableTime.get(e.getKey());
            t.put("min_time", mm != null && mm[0] != null ? mm[0].toString() : null);
            t.put("max_time", mm != null && mm[1] != null ? mm[1].toString() : null);
            tables.put(e.getKey(), t);
        }
        ctx.put("tables", tables);

        Map<String, Object> rejections = new LinkedHashMap<>();
        capture.temporalRejections.forEach((tbl, n) -> rejections.put(tbl, n));
        ctx.put("temporal_rejections", rejections);

        List<Object> unreachable = new ArrayList<>();
        for (TableDef td : schema.tables()) {
            if (!perTable.containsKey(td.name())) unreachable.add(td.name());
        }
        ctx.put("unreachable_tables", unreachable);
        return ctx;
    }

    /** Captures per-table temporal-bound rejections during a single EXPLAIN CONTEXT assembly. */
    private static final class CapturingInstrumentation implements Instrumentation {
        final Map<String, Integer> temporalRejections = new LinkedHashMap<>();
        @Override public void onTemporalViolationDropped(EntityId entity, String table) {
            temporalRejections.merge(table, 1, Integer::sum);
        }
    }

    // ------------------------------------------------------------------
    //  Target expression rendering (human-readable, no scoring)
    // ------------------------------------------------------------------

    private static String renderExpr(TargetExpr e) {
        if (e instanceof Aggregation agg) {
            StringBuilder sb = new StringBuilder();
            sb.append(agg.func().name()).append('(').append(agg.column());
            agg.filter().ifPresent(f -> sb.append(" WHERE ").append(renderExpr(f.condition())));
            sb.append(')');
            if (agg.hasWindow()) {
                sb.append(" OVER (").append(bound(agg.startOr(0))).append(", ")
                        .append(bound(agg.end())).append(' ')
                        .append(agg.unit().name());
                if (agg.horizons() > 1) sb.append(" HORIZONS ").append(agg.horizons());
                sb.append(')');
            }
            return sb.toString();
        }
        if (e instanceof ColumnRef ref) return ref.toString();
        if (e instanceof Condition cond) {
            String rhs = cond.rightExpr().map(RelativeDbEngine::renderExpr)
                    .orElseGet(() -> String.valueOf(cond.right()));
            return renderExpr(cond.left()) + " " + cond.op().name() + " " + rhs;
        }
        if (e instanceof LogicalOp op) {
            return "(" + renderExpr(op.left()) + " " + op.op().name() + " " + renderExpr(op.right()) + ")";
        }
        if (e instanceof Not not) return "NOT (" + renderExpr(not.inner()) + ")";
        if (e instanceof Arith a) {
            return "(" + renderExpr(a.left()) + " " + a.op() + " " + renderExpr(a.right()) + ")";
        }
        if (e instanceof Func f) {
            StringBuilder sb = new StringBuilder(f.name()).append('(');
            for (int i = 0; i < f.args().size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(renderExpr(f.args().get(i)));
            }
            return sb.append(')').toString();
        }
        if (e instanceof Case c) {
            StringBuilder sb = new StringBuilder("CASE");
            for (Case.When w : c.whens()) {
                sb.append(" WHEN ").append(renderExpr(w.cond())).append(" THEN ").append(renderExpr(w.then()));
            }
            if (c.elseExpr() != null) sb.append(" ELSE ").append(renderExpr(c.elseExpr()));
            return sb.append(" END").toString();
        }
        if (e instanceof LitExpr lit) return String.valueOf(lit.value());
        return String.valueOf(e);
    }

    private List<EntityId> resolveEntityIds(ExecutionInput input, ValidatedQuery vq, String entityTable) {
        if (!input.entityIds().isEmpty()) {
            return input.entityIds().stream().map(EntityId::of).toList();
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

    private ModelBackend resolveBackend(TaskType taskType) {
        if (backend != null) return backend;
        throw new IllegalStateException("Engine requires a model backend (e.g. RtNativeBackend); "
                + "there is no built-in model-free scorer. Add a backend artifact "
                + "(e.g. relativedb-rt) or pass Builder.modelBackend(...). Routing was ready to use "
                + "checkpoint '" + modelConfig.modelUriFor(taskType) + "' with embedding model '"
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
            com.relativedb.query.ParsedQuery query) {
        return returnShaper.shape(id, taskType, out, query);
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
