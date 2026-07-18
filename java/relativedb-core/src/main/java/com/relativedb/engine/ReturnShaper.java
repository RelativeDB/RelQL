package com.relativedb.engine;

import com.relativedb.engine.PredictionResult.EntityPrediction;
import com.relativedb.engine.PredictionResult.Interval;
import com.relativedb.engine.PredictionResult.RankedItem;
import com.relativedb.engine.PredictionResult.TimeframeValue;
import com.relativedb.model.ModelOutput;
import com.relativedb.query.AggFunc;
import com.relativedb.query.Aggregation;
import com.relativedb.query.Arith;
import com.relativedb.query.Case;
import com.relativedb.query.Condition;
import com.relativedb.query.Func;
import com.relativedb.query.LogicalOp;
import com.relativedb.query.Not;
import com.relativedb.query.ParsedQuery;
import com.relativedb.query.ReturnSpec;
import com.relativedb.query.TargetExpr;
import com.relativedb.query.TaskType;
import com.relativedb.query.TimeUnit;
import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.Row;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Applies the parsed {@code RETURN <output>} clause to a scored entity,
 * selecting the output object (hard label, class distribution, empirical
 * quantiles, prediction interval, expected value, ...) without changing task
 * routing.
 *
 * <p>The base prediction (probability {@code p}, regression {@code value},
 * multiclass distribution, ranking, forecast series) comes from the model
 * backend's {@link ModelOutput}. {@code QUANTILES}/{@code INTERVAL} dispersion
 * is derived from the entity's own trailing history — the target's temporal
 * aggregation re-evaluated over {@code numHistoryWindows} pseudo-anchors
 * ({@code anchor - span*k}) on the assembled context (the model-free
 * "self-labels" signal). This mirrors the reference {@code HistoryBaselineBackend}
 * in the Python/Rust bindings; the native RT backend keeps returning its default
 * object, so RETURN shaping lives here on the engine side.
 *
 * <p>When {@code query.ret} is absent the output is byte-for-byte the legacy
 * default for the task type.
 */
final class ReturnShaper {

    /** Trailing windows sampled for empirical quantiles/intervals. */
    static final int DEFAULT_HISTORY_WINDOWS = 3;

    private final int numHistoryWindows;

    ReturnShaper() { this(DEFAULT_HISTORY_WINDOWS); }

    ReturnShaper(int numHistoryWindows) {
        this.numHistoryWindows = Math.max(1, numHistoryWindows);
    }

    EntityPrediction shape(EntityId id, TaskType taskType, ModelOutput out,
                           ParsedQuery query, ContextGraph context, Instant anchor) {
        ReturnSpec ret = query == null ? null : query.ret().orElse(null);
        ReturnSpec.Kind kind = ret == null ? null : ret.kind();

        return switch (taskType) {
            case REGRESSION -> regression(id, out.value(), kind, ret, query, context, anchor);
            case FORECASTING -> forecasting(id, out, kind, ret, query, context, anchor);
            case BINARY_CLASSIFICATION -> binary(id, out.probability(), kind);
            case MULTICLASS_CLASSIFICATION -> multiclass(id, out.classProbs(), kind);
            case MULTILABEL_RANKING -> ranking(id, out);
        };
    }

    // ------------------------------------------------------------------
    //  Per-task shaping
    // ------------------------------------------------------------------

    private EntityPrediction regression(EntityId id, double value, ReturnSpec.Kind kind,
                                        ReturnSpec ret, ParsedQuery query,
                                        ContextGraph context, Instant anchor) {
        if (kind == ReturnSpec.Kind.QUANTILES || kind == ReturnSpec.Kind.INTERVAL) {
            List<Double> values = historyValues(query, context, anchor);
            return new EntityPrediction(id, OptionalDouble.of(value), OptionalDouble.empty(),
                    Map.of(), List.of(), List.of(),
                    Optional.empty(), quantilesFor(ret, values), intervalFor(ret, values));
        }
        // EXPECTED_VALUE and default are both the regression value.
        return new EntityPrediction(id, OptionalDouble.of(value), OptionalDouble.empty(),
                Map.of(), List.of(), List.of());
    }

    private EntityPrediction forecasting(EntityId id, ModelOutput out, ReturnSpec.Kind kind,
                                         ReturnSpec ret, ParsedQuery query,
                                         ContextGraph context, Instant anchor) {
        List<TimeframeValue> forecast = new ArrayList<>();
        for (int i = 0; i < out.forecastValues().size(); i++) {
            forecast.add(new TimeframeValue(i + 1, out.forecastValues().get(i)));
        }
        if (kind == ReturnSpec.Kind.QUANTILES || kind == ReturnSpec.Kind.INTERVAL) {
            List<Double> values = historyValues(query, context, anchor);
            OptionalDouble base = mean(values);
            return new EntityPrediction(id, base, OptionalDouble.empty(),
                    Map.of(), List.of(), List.copyOf(forecast),
                    Optional.empty(), quantilesFor(ret, values), intervalFor(ret, values));
        }
        return new EntityPrediction(id, OptionalDouble.empty(), OptionalDouble.empty(),
                Map.of(), List.of(), List.copyOf(forecast));
    }

    private EntityPrediction binary(EntityId id, double p, ReturnSpec.Kind kind) {
        if (kind == ReturnSpec.Kind.CLASS) {
            String label = p >= 0.5 ? "true" : "false";
            return new EntityPrediction(id, OptionalDouble.empty(), OptionalDouble.empty(),
                    Map.of(), List.of(), List.of(), Optional.of(label), Map.of(), Optional.empty());
        }
        if (kind == ReturnSpec.Kind.DISTRIBUTION) {
            Map<String, Double> dist = new LinkedHashMap<>();
            dist.put("true", p);
            dist.put("false", 1.0 - p);
            return new EntityPrediction(id, OptionalDouble.empty(), OptionalDouble.empty(),
                    dist, List.of(), List.of());
        }
        if (kind == ReturnSpec.Kind.EXPECTED_VALUE) {
            // Expected value of the 0/1 indicator is p.
            return new EntityPrediction(id, OptionalDouble.of(p), OptionalDouble.empty(),
                    Map.of(), List.of(), List.of());
        }
        // PROBABILITY (explicit) or default.
        return new EntityPrediction(id, OptionalDouble.empty(), OptionalDouble.of(p),
                Map.of(), List.of(), List.of());
    }

    private EntityPrediction multiclass(EntityId id, Map<String, Double> classProbs,
                                        ReturnSpec.Kind kind) {
        if (kind == ReturnSpec.Kind.CLASS) {
            Optional<String> label = classProbs.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey);
            return new EntityPrediction(id, OptionalDouble.empty(), OptionalDouble.empty(),
                    Map.of(), List.of(), List.of(), label, Map.of(), Optional.empty());
        }
        // DISTRIBUTION / MULTICLASS / default: the model's distribution.
        return new EntityPrediction(id, OptionalDouble.empty(), OptionalDouble.empty(),
                classProbs, List.of(), List.of());
    }

    private EntityPrediction ranking(EntityId id, ModelOutput out) {
        List<RankedItem> ranked = out.rankedScores().entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .map(e -> new RankedItem(e.getKey(), e.getValue()))
                .toList();
        return new EntityPrediction(id, OptionalDouble.empty(), OptionalDouble.empty(),
                Map.of(), ranked, List.of());
    }

    // ------------------------------------------------------------------
    //  RETURN QUANTILES / INTERVAL from trailing history
    // ------------------------------------------------------------------

    private Map<Double, Double> quantilesFor(ReturnSpec ret, List<Double> values) {
        if (ret == null || ret.kind() != ReturnSpec.Kind.QUANTILES || values.isEmpty()) {
            return Map.of();
        }
        Map<Double, Double> out = new LinkedHashMap<>();
        for (double q : ret.quantiles()) {
            out.put(q, empiricalQuantile(values, q));
        }
        return out;
    }

    private Optional<Interval> intervalFor(ReturnSpec ret, List<Double> values) {
        if (ret == null || ret.kind() != ReturnSpec.Kind.INTERVAL || values.isEmpty()) {
            return Optional.empty();
        }
        double pct = ret.interval().orElse(0);
        double lo = (1.0 - pct / 100.0) / 2.0;
        double hi = 1.0 - lo;
        return Optional.of(new Interval(empiricalQuantile(values, lo),
                empiricalQuantile(values, hi)));
    }

    /**
     * Linear-interpolation empirical quantile (numpy/percentile "linear"
     * semantics) over an unsorted sample; {@code q} clamped to {@code [0,1]}.
     */
    static double empiricalQuantile(List<Double> values, double q) {
        List<Double> xs = new ArrayList<>(values);
        xs.sort(Comparator.naturalOrder());
        int n = xs.size();
        if (n == 1) return xs.get(0);
        double qc = Math.max(0.0, Math.min(1.0, q));
        double idx = qc * (n - 1);
        int lo = (int) Math.floor(idx);
        int hi = Math.min(lo + 1, n - 1);
        double frac = idx - lo;
        return xs.get(lo) + (xs.get(hi) - xs.get(lo)) * frac;
    }

    private static OptionalDouble mean(List<Double> values) {
        if (values.isEmpty()) return OptionalDouble.empty();
        double sum = 0.0;
        for (double v : values) sum += v;
        return OptionalDouble.of(sum / values.size());
    }

    /**
     * Per-pseudo-anchor numeric evaluations of the target's temporal
     * aggregation (booleans surface as 0.0/1.0), oldest first. Empty when the
     * target has no windowed aggregation to re-evaluate.
     */
    List<Double> historyValues(ParsedQuery query, ContextGraph context, Instant anchor) {
        Aggregation agg = query == null ? null : findWindowedAgg(query.target());
        if (agg == null || context == null) return List.of();
        List<Double> vals = new ArrayList<>();
        for (Instant pa : pseudoAnchors(agg, anchor)) {
            Double v = evalAggregation(agg, context, pa);
            if (v != null && !Double.isNaN(v)) vals.add(v);
        }
        return vals;
    }

    private List<Instant> pseudoAnchors(Aggregation agg, Instant anchor) {
        Duration span = span(agg);
        if (anchor == null || span == null) {
            List<Instant> single = new ArrayList<>();
            single.add(anchor);          // may be null -> aggregate over all context rows
            return single;
        }
        List<Instant> anchors = new ArrayList<>(numHistoryWindows);
        for (int k = numHistoryWindows; k >= 1; k--) {    // oldest first
            anchors.add(anchor.minus(span.multipliedBy(k)));
        }
        return anchors;
    }

    /** Window width, or {@code null} for an unbounded ({@code ±INF}) frame. */
    private static Duration span(Aggregation agg) {
        long start = agg.startOr(0);
        long end = agg.end();
        if (isInfinite(start) || isInfinite(end)) return null;
        return agg.unit().duration().multipliedBy(end - start);
    }

    private static boolean isInfinite(long bound) {
        return bound == Aggregation.NEG_INF || bound == Aggregation.POS_INF;
    }

    /** Evaluate the aggregation over context rows in {@code (lo, hi]} around {@code anchor}. */
    private static Double evalAggregation(Aggregation agg, ContextGraph context, Instant anchor) {
        String table = agg.column().table();
        String col = agg.column().column();
        boolean wildcard = agg.column().isWildcard();
        TimeUnit unit = agg.unit();

        Instant lo = null;
        Instant hi = null;
        boolean windowed = anchor != null && unit != null;
        if (windowed) {
            long start = agg.startOr(0);
            long end = agg.end();
            lo = isInfinite(start) ? null : anchor.plus(unit.duration().multipliedBy(start));
            hi = isInfinite(end) ? null : anchor.plus(unit.duration().multipliedBy(end));
        }

        List<Row> rows = new ArrayList<>();
        for (Row r : context.rows()) {
            if (!r.table().equals(table)) continue;
            if (windowed) {
                Instant ts = r.timestamp().orElse(null);
                if (ts == null) continue;
                if (lo != null && !ts.isAfter(lo)) continue;      // start EXCLUDED
                if (hi != null && ts.isAfter(hi)) continue;       // end INCLUDED
            }
            rows.add(r);
        }
        rows.sort(Comparator.comparing(r -> r.timestamp().orElse(Instant.MIN)));

        return switch (agg.func()) {
            case EXISTS -> rows.isEmpty() ? 0.0 : 1.0;
            case COUNT -> wildcard
                    ? (double) rows.size()
                    : (double) rows.stream().filter(r -> r.cells().get(col) != null).count();
            case COUNT_DISTINCT -> (double) distinctValues(rows, col).size();
            case SUM -> rows.stream().map(r -> numeric(r.cells().get(col)))
                    .filter(v -> v != null).mapToDouble(Double::doubleValue).sum();
            case AVG -> {
                List<Double> nums = numericValues(rows, col);
                yield nums.isEmpty() ? null : nums.stream().mapToDouble(Double::doubleValue).average().orElseThrow();
            }
            case MIN -> {
                List<Double> nums = numericValues(rows, col);
                yield nums.isEmpty() ? null : nums.stream().mapToDouble(Double::doubleValue).min().orElseThrow();
            }
            case MAX -> {
                List<Double> nums = numericValues(rows, col);
                yield nums.isEmpty() ? null : nums.stream().mapToDouble(Double::doubleValue).max().orElseThrow();
            }
            case FIRST -> {
                List<Double> nums = numericValues(rows, col);
                yield nums.isEmpty() ? null : nums.get(0);
            }
            case LAST -> {
                List<Double> nums = numericValues(rows, col);
                yield nums.isEmpty() ? null : nums.get(nums.size() - 1);
            }
            case LIST_DISTINCT -> null;   // not a numeric summary
        };
    }

    private static List<Double> numericValues(List<Row> rows, String col) {
        List<Double> nums = new ArrayList<>();
        for (Row r : rows) {
            Double v = numeric(r.cells().get(col));
            if (v != null) nums.add(v);
        }
        return nums;
    }

    private static LinkedHashSet<Object> distinctValues(List<Row> rows, String col) {
        LinkedHashSet<Object> seen = new LinkedHashSet<>();
        for (Row r : rows) {
            Object v = r.cells().get(col);
            if (v != null) seen.add(v);
        }
        return seen;
    }

    private static Double numeric(Object v) {
        if (v instanceof Double d) return d;
        if (v instanceof Number n) return n.doubleValue();
        if (v instanceof Boolean b) return b ? 1.0 : 0.0;
        return null;
    }

    /** First aggregation carrying a temporal window anywhere in the target tree. */
    private static Aggregation findWindowedAgg(TargetExpr e) {
        if (e instanceof Aggregation agg) {
            return agg.hasWindow() ? agg : null;
        }
        if (e instanceof LogicalOp op) {
            Aggregation l = findWindowedAgg(op.left());
            return l != null ? l : findWindowedAgg(op.right());
        }
        if (e instanceof Not not) return findWindowedAgg(not.inner());
        if (e instanceof Arith a) {
            Aggregation l = findWindowedAgg(a.left());
            return l != null ? l : findWindowedAgg(a.right());
        }
        if (e instanceof Condition cond) {
            Aggregation l = findWindowedAgg(cond.left());
            if (l != null) return l;
            return cond.rightExpr().map(ReturnShaper::findWindowedAgg).orElse(null);
        }
        if (e instanceof Func f) {
            for (TargetExpr arg : f.args()) {
                Aggregation l = findWindowedAgg(arg);
                if (l != null) return l;
            }
            return null;
        }
        if (e instanceof Case c) {
            for (Case.When w : c.whens()) {
                Aggregation l = findWindowedAgg(w.cond());
                if (l != null) return l;
                Aggregation t = findWindowedAgg(w.then());
                if (t != null) return t;
            }
            return c.elseExpr() != null ? findWindowedAgg(c.elseExpr()) : null;
        }
        return null;
    }
}
