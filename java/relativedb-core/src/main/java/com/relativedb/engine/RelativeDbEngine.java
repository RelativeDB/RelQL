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

        // Class-label domain (multiclass) and candidate pool (ranking) are
        // query-global under the temporal bound — enumerated once and shared
        // across entities. (Contract §2.5 / §3.1.)
        List<String> classLabels = taskType == TaskType.MULTICLASS_CLASSIFICATION
                ? multiclassLabels(vq, bound) : List.of();
        List<Row> rankCandidates = taskType == TaskType.MULTILABEL_RANKING
                ? rankCandidateRows(vq, bound) : List.of();

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
            ModelOutput out = switch (taskType) {
                case MULTICLASS_CLASSIFICATION -> {
                    TokenBatch batch = buildBatch(context, effectiveAnchor, taskType);
                    instrumentation.onModelInvoked(id, batch.size());
                    yield resolved.classifyMulticlass(batch, classLabels, taskType);
                }
                case MULTILABEL_RANKING -> {
                    List<String> candidateIds = new ArrayList<>(rankCandidates.size());
                    for (Row candidate : rankCandidates) {
                        candidateIds.add(String.valueOf(candidate.id().raw()));
                    }
                    List<TokenBatch> batches =
                            buildRankingBatches(context, effectiveAnchor, rankCandidates, taskType);
                    instrumentation.onModelInvoked(id, batches.isEmpty() ? 0 : batches.get(0).size());
                    ModelOutput scored = resolved.rankCandidates(batches, candidateIds, taskType);
                    yield truncateRanking(scored, vq.query().topK());
                }
                default -> {
                    TokenBatch batch = buildBatch(context, effectiveAnchor, taskType);
                    instrumentation.onModelInvoked(id, batch.size());
                    yield resolved.score(batch, taskType).toCompletableFuture().join();
                }
            };
            predictions.add(decode(id, taskType, out, vq.query()));
        }
        return new PredictionResult(taskType, predictions);
    }

    // ------------------------------------------------------------------
    //  Multiclass class-domain & ranking candidate enumeration
    // ------------------------------------------------------------------

    /** Cap on distinct class labels / ranking candidates (contract §2.5 / §3.1). */
    static final int MAX_MULTICLASS_CLASSES = 1000;
    static final int MAX_RANK_CANDIDATES = 1000;

    /**
     * The distinct observed values of the target categorical column, temporally
     * bounded, sorted lexicographically (UTF-8 byte order, ascending) and capped
     * at {@link #MAX_MULTICLASS_CLASSES}. Requires a {@link com.relativedb.retrieve.TableScanner}
     * over the target's table.
     */
    private List<String> multiclassLabels(ValidatedQuery vq, TemporalBound bound) {
        ColumnRef target = categoricalTargetColumn(vq.query().target());
        var scanner = wiring.scanner(target.table()).orElseThrow(() ->
                new IllegalStateException("multiclass classification of '" + target
                        + "' requires a TableScanner over table '" + target.table()
                        + "' to enumerate the class-label domain"));
        java.util.TreeSet<String> distinct = new java.util.TreeSet<>(RelativeDbEngine::compareUtf8);
        for (Row row : drain(scanner.scan(target.table(), bound))) {
            if (!row.timestamp().map(bound::admits).orElse(true)) continue;
            Object v = row.cells().get(target.column());
            if (v != null) distinct.add(String.valueOf(v));
        }
        List<String> labels = new ArrayList<>(distinct);
        return labels.size() > MAX_MULTICLASS_CLASSES
                ? labels.subList(0, MAX_MULTICLASS_CLASSES) : labels;
    }

    /**
     * The distinct rows of the parent table referenced by {@code LIST_DISTINCT(table.fk)},
     * temporally bounded, deduplicated by id, sorted (numeric ascending if the id
     * type is integral, else lexicographic UTF-8 ascending on the stringified id)
     * and capped at {@link #MAX_RANK_CANDIDATES}. Requires a
     * {@link com.relativedb.retrieve.TableScanner} over the parent table.
     */
    private List<Row> rankCandidateRows(ValidatedQuery vq, TemporalBound bound) {
        if (!(vq.query().target() instanceof Aggregation agg)) {
            throw new IllegalStateException("ranking target must be LIST_DISTINCT(table.fk), got "
                    + vq.query().target());
        }
        ColumnRef fk = agg.column();
        String parentTable = schema.linksFrom(fk.table()).stream()
                .filter(l -> l.fkColumn().equals(fk.column()))
                .map(com.relativedb.schema.LinkDef::toTable)
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("ranking target '" + fk
                        + "' is not a declared foreign key of table '" + fk.table() + "'"));
        var scanner = wiring.scanner(parentTable).orElseThrow(() ->
                new IllegalStateException("ranking requires a TableScanner over the parent table '"
                        + parentTable + "' to enumerate link candidates"));
        Map<EntityId, Row> byId = new LinkedHashMap<>();
        for (Row row : drain(scanner.scan(parentTable, bound))) {
            if (!row.timestamp().map(bound::admits).orElse(true)) continue;
            byId.putIfAbsent(row.id(), row);
        }
        List<Row> candidates = new ArrayList<>(byId.values());
        boolean integral = candidates.stream().allMatch(r -> isIntegral(r.id().raw()));
        candidates.sort(integral
                ? java.util.Comparator.comparing(r -> ((Number) r.id().raw()).longValue())
                : (x, y) -> compareUtf8(String.valueOf(x.id().raw()), String.valueOf(y.id().raw())));
        return candidates.size() > MAX_RANK_CANDIDATES
                ? candidates.subList(0, MAX_RANK_CANDIDATES) : candidates;
    }

    /** Apply {@code RANK TOP k}: keep the first {@code k} of the already-ranked map. */
    private static ModelOutput truncateRanking(ModelOutput scored, java.util.OptionalInt topK) {
        Map<String, Double> ranked = scored.rankedScores();
        int k = topK.orElse(ranked.size());
        if (k >= ranked.size()) return scored;
        Map<String, Double> kept = new LinkedHashMap<>();
        for (var e : ranked.entrySet()) {
            if (kept.size() >= k) break;
            kept.put(e.getKey(), e.getValue());
        }
        return ModelOutput.ranking(kept);
    }

    private static ColumnRef categoricalTargetColumn(TargetExpr target) {
        if (target instanceof ColumnRef c) return c;
        if (target instanceof Aggregation agg) return agg.column();
        throw new IllegalStateException("multiclass target is not a categorical column: " + target);
    }

    private static boolean isIntegral(Object raw) {
        return raw instanceof Long || raw instanceof Integer
                || raw instanceof Short || raw instanceof Byte
                || raw instanceof java.math.BigInteger;
    }

    /** Lexicographic comparison by UTF-8 bytes (unsigned) — identical across bindings. */
    static int compareUtf8(String a, String b) {
        byte[] x = a.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] y = b.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        int n = Math.min(x.length, y.length);
        for (int i = 0; i < n; i++) {
            int cx = x[i] & 0xFF, cy = y[i] & 0xFF;
            if (cx != cy) return Integer.compare(cx, cy);
        }
        return Integer.compare(x.length, y.length);
    }

    private static List<Row> drain(java.util.concurrent.Flow.Publisher<Row> publisher) {
        List<Row> rows = new ArrayList<>();
        CompletableFuture<Void> done = new CompletableFuture<>();
        publisher.subscribe(new java.util.concurrent.Flow.Subscriber<>() {
            @Override public void onSubscribe(java.util.concurrent.Flow.Subscription s) {
                s.request(Long.MAX_VALUE);
            }
            @Override public void onNext(Row row) { rows.add(row); }
            @Override public void onError(Throwable t) { done.completeExceptionally(t); }
            @Override public void onComplete() { done.complete(null); }
        });
        done.join();
        return rows;
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
    //
    //  Parity with the Python (rt_native._build_ctx_seq/_normalize) and Rust
    //  (native.rs build_ctx_seq) bindings. Every entity's sequence carries a
    //  SYNTHETIC masked target token — a "task" row ("__target__") with a masked
    //  `label` cell wired to the entity node — which is the ONLY target token
    //  (the seed entity's own feature cells are ordinary, non-target features).
    //  Without it a cell-less entity table (e.g. MovieLens `users`) would emit
    //  no target token and every candidate would score sigmoid(0)=0.5, producing
    //  the degenerate enumeration-order ranking.
    // ------------------------------------------------------------------

    /** Shared cross-binding synthetic-task constants (identical in Python & Rust). */
    private static final String TASK_TABLE = "task";
    private static final String TASK_TIME_COL = "timestamp";
    private static final String TASK_LABEL_COL = "label";
    private static final String TARGET_ROW_KEY = "__target__";

    /** A raw (pre-normalization) token: {@code value} is Double/Boolean/Instant/String, or null for the masked target. */
    private static final class RawTok {
        final int node;
        List<Integer> parents;
        final String col;
        final String table;
        final ValueType sem;
        final Object value;
        final boolean target;
        RawTok(int node, List<Integer> parents, String col, String table,
               ValueType sem, Object value, boolean target) {
            this.node = node; this.parents = parents; this.col = col; this.table = table;
            this.sem = sem; this.value = value; this.target = target;
        }
        RawTok copy() {
            return new RawTok(node, new ArrayList<>(parents), col, table, sem, value, target);
        }
    }

    /** One entity's assembled raw sequence plus the handles ranking needs to rewire the target f2p. */
    private static final class BaseSeq {
        final List<RawTok> toks = new ArrayList<>();
        final Map<String, Integer> nodeOf = new LinkedHashMap<>();
        int entityNode;
        int tgtLabelIdx;
    }

    private static String nodeKey(String table, Object raw) { return table + " " + raw; }

    private static int nodeId(BaseSeq b, String table, Object raw) {
        return b.nodeOf.computeIfAbsent(nodeKey(table, raw), k -> b.nodeOf.size());
    }

    /** Prefer the schema-declared type; else infer the sem-type from the value (nulls/unsupported -> no token). */
    private static ValueType semOf(Object v, ValueType declared) {
        if (declared != null) return declared;
        if (v instanceof Boolean) return ValueType.BOOLEAN;
        if (v instanceof Number) return ValueType.NUMBER;
        if (v instanceof Instant) return ValueType.DATETIME;
        if (v instanceof String) return ValueType.TEXT;
        return null;
    }

    /**
     * Assemble one entity's context into a raw token sequence with the synthetic
     * masked target token (Python {@code _build_ctx_seq}). {@code targetSem} is
     * TEXT for multiclass (the masked-TEXT target cell, §2.1), NUMBER otherwise.
     *
     * <p>Self-label history (F65) is not emitted: the Java engine has no target
     * evaluator, and for the ranking target ({@code LIST_DISTINCT(...)}) both the
     * Python and Rust bindings gather ZERO self-labels anyway, so parity holds on
     * the ranking path. (Binary/regression self-labels remain a documented gap.)
     */
    private BaseSeq buildBaseSeq(ContextGraph context, Instant anchor, ValueType targetSem) {
        BaseSeq b = new BaseSeq();
        // Context rows claim node ids first so f2p links resolve in any order.
        for (Row r : context.rows()) nodeId(b, r.table(), r.id().raw());
        b.entityNode = nodeId(b, context.seedTable(), context.seedId().raw());

        // -- the synthetic target task row (masked label) --
        int tgtNode = nodeId(b, TASK_TABLE, TARGET_ROW_KEY);
        if (anchor != null) {
            b.toks.add(new RawTok(tgtNode, new ArrayList<>(List.of(b.entityNode)),
                    TASK_TIME_COL, TASK_TABLE, ValueType.DATETIME, anchor, false));
        }
        b.tgtLabelIdx = b.toks.size();
        b.toks.add(new RawTok(tgtNode, new ArrayList<>(List.of(b.entityNode)),
                TASK_LABEL_COL, TASK_TABLE, targetSem, null, true));

        // -- one token per feature cell of every context row (none are targets) --
        for (Row r : context.rows()) {
            List<Integer> parents = resolveParentNodes(b, r);
            int rnode = b.nodeOf.get(nodeKey(r.table(), r.id().raw()));
            emitFeatureCells(b.toks, r, rnode, parents);
        }
        return b;
    }

    /** Resolve a row's declared FK edges to the context node ids (dropping dangling/undeclared ones). */
    private List<Integer> resolveParentNodes(BaseSeq b, Row row) {
        List<Integer> parents = new ArrayList<>();
        for (var e : row.parents().entrySet()) {
            schema.linksFrom(row.table()).stream()
                    .filter(l -> l.fkColumn().equals(e.getKey()))
                    .findFirst()
                    .map(l -> b.nodeOf.get(nodeKey(l.toTable(), e.getValue().raw())))
                    .filter(Objects::nonNull)
                    .ifPresent(parents::add);
        }
        return parents;
    }

    /** Emit one raw token per feature cell of {@code row}, all non-target. */
    private void emitFeatureCells(List<RawTok> out, Row row, int node, List<Integer> parents) {
        TableDef table = schema.table(row.table()).orElse(null);
        for (var cell : row.cells().entrySet()) {
            ValueType declared = table == null ? null
                    : table.column(cell.getKey()).map(com.relativedb.schema.ColumnDef::type).orElse(null);
            ValueType sem = semOf(cell.getValue(), declared);
            if (sem == null) continue;
            out.add(new RawTok(node, parents, cell.getKey(), row.table(), sem, cell.getValue(), false));
        }
    }

    /** Score/multiclass path: one entity, one normalized token batch. */
    private TokenBatch buildBatch(ContextGraph context, Instant anchor, TaskType taskType) {
        ValueType targetSem = taskType == TaskType.MULTICLASS_CLASSIFICATION
                ? ValueType.TEXT : ValueType.NUMBER;
        BaseSeq b = buildBaseSeq(context, anchor, targetSem);
        return normalizeToBatches(List.of(b.toks), 0.0, 1.0).get(0);
    }

    /**
     * Ranking path (§3.2): one existence context per candidate. Each candidate's
     * feature cells are emitted (as a fresh node when the candidate isn't already
     * a context row) and the candidate node is wired into the synthetic target
     * label cell's f2p alongside the entity node — the per-candidate link the
     * model scores. All candidate sequences are normalized together (batch-internal
     * stats, label stats fixed to 0/1) exactly as Python's {@code _score_ranking}.
     */
    private List<TokenBatch> buildRankingBatches(ContextGraph context, Instant anchor,
                                                 List<Row> candidates, TaskType taskType) {
        BaseSeq b = buildBaseSeq(context, anchor, ValueType.NUMBER);
        List<List<RawTok>> seqs = new ArrayList<>(candidates.size());
        for (Row cand : candidates) {
            List<RawTok> seq = new ArrayList<>(b.toks.size() + 4);
            for (RawTok t : b.toks) seq.add(t.copy());
            Integer existing = b.nodeOf.get(nodeKey(cand.table(), cand.id().raw()));
            int candNode;
            if (existing == null) {
                candNode = b.nodeOf.size();   // fresh node id (per-candidate, not shared)
                TableDef ct = schema.table(cand.table()).orElse(null);
                for (var cell : cand.cells().entrySet()) {
                    ValueType declared = ct == null ? null
                            : ct.column(cell.getKey()).map(com.relativedb.schema.ColumnDef::type).orElse(null);
                    ValueType sem = semOf(cell.getValue(), declared);
                    if (sem == null) continue;
                    seq.add(new RawTok(candNode, List.of(), cell.getKey(), cand.table(),
                            sem, cell.getValue(), false));
                }
            } else {
                candNode = existing;
            }
            seq.get(b.tgtLabelIdx).parents = new ArrayList<>(List.of(b.entityNode, candNode));
            seqs.add(seq);
        }
        return normalizeToBatches(seqs, 0.0, 1.0);
    }

    private static double asNumber(Object v) {
        if (v instanceof Boolean bool) return bool ? 1.0 : 0.0;
        return ((Number) v).doubleValue();
    }

    /** Fractional days since the epoch (Python {@code _days}: {@code timestamp()/86400}). */
    private static double days(Instant t) {
        return (t.getEpochSecond() + t.getNano() / 1_000_000_000.0) / 86400.0;
    }

    /** Population mean and (std + 1e-8), matching numpy {@code std(ddof=0)}. */
    private static double[] meanStd(List<Double> vals) {
        double mu = 0.0;
        for (double v : vals) mu += v;
        mu /= vals.size();
        double var = 0.0;
        for (double v : vals) var += (v - mu) * (v - mu);
        var /= vals.size();
        return new double[] { mu, Math.sqrt(var) + 1e-8 };
    }

    /**
     * Normalize raw sequences into token batches (Python {@code _normalize}):
     * numbers/booleans z-scored per (column,table) over the batch's in-context
     * values; datetimes share one global stat (fractional days); the task label
     * column uses {@code (labelMu,labelSd)}; the masked target cell -> 0.0.
     */
    private List<TokenBatch> normalizeToBatches(List<List<RawTok>> seqs,
                                                double labelMu, double labelSd) {
        Map<String, List<Double>> numVals = new HashMap<>();
        List<Double> dtVals = new ArrayList<>();
        for (List<RawTok> seq : seqs) {
            for (RawTok t : seq) {
                if (t.target || t.value == null) continue;
                if (t.sem == ValueType.DATETIME) {
                    dtVals.add(days((Instant) t.value));
                } else if (t.sem == ValueType.NUMBER || t.sem == ValueType.BOOLEAN) {
                    numVals.computeIfAbsent(nodeKey(t.col, t.table), k -> new ArrayList<>())
                            .add(asNumber(t.value));
                }
            }
        }
        Map<String, double[]> stats = new HashMap<>();
        numVals.forEach((k, vals) -> stats.put(k, meanStd(vals)));
        stats.put(nodeKey(TASK_LABEL_COL, TASK_TABLE), new double[] { labelMu, labelSd });
        double[] dt = dtVals.isEmpty() ? new double[] { 0.0, 1.0 } : meanStd(dtVals);

        List<TokenBatch> out = new ArrayList<>(seqs.size());
        for (List<RawTok> seq : seqs) {
            TokenBatch.Builder batch = TokenBatch.newBatch();
            for (RawTok t : seq) {
                if (t.target) {
                    if (t.sem == ValueType.TEXT) {
                        batch.text(t.node, t.parents, t.table, t.col, null, true);
                    } else {
                        batch.numeric(t.node, t.parents, t.table, t.col, ValueType.NUMBER, 0.0, true);
                    }
                    continue;
                }
                switch (t.sem) {
                    case TEXT -> batch.text(t.node, t.parents, t.table, t.col,
                            String.valueOf(t.value), false);
                    case DATETIME -> batch.numeric(t.node, t.parents, t.table, t.col,
                            ValueType.DATETIME, (days((Instant) t.value) - dt[0]) / dt[1], false);
                    case NUMBER, BOOLEAN -> {
                        double[] s = stats.get(nodeKey(t.col, t.table));
                        double x = asNumber(t.value);
                        double norm = s == null ? x : (x - s[0]) / s[1];
                        batch.numeric(t.node, t.parents, t.table, t.col, t.sem, norm, false);
                    }
                }
            }
            out.add(batch.build());
        }
        return out;
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
