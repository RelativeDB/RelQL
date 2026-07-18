package com.relativedb.engine;

import com.relativedb.engine.PredictionResult.EntityPrediction;
import com.relativedb.engine.PredictionResult.RankedItem;
import com.relativedb.engine.PredictionResult.TimeframeValue;
import com.relativedb.model.ModelOutput;
import com.relativedb.query.ParsedQuery;
import com.relativedb.query.ReturnSpec;
import com.relativedb.query.TaskType;
import com.relativedb.retrieve.EntityId;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Applies the parsed {@code RETURN <output>} clause to a scored entity,
 * selecting the output object (hard label, class distribution, expected value,
 * ...) without changing task routing.
 *
 * <p>Shaping operates purely on the model backend's {@link ModelOutput}
 * (probability {@code p}, regression {@code value}, multiclass distribution,
 * ranking, forecast series); it never fabricates a distribution:
 * <ul>
 *   <li>{@code CLASS} → hard label (binary: threshold {@code 0.5}).</li>
 *   <li>{@code DISTRIBUTION} → {@code {"true":p,"false":1-p}} (binary).</li>
 *   <li>{@code PROBABILITY}, {@code EXPECTED VALUE} → the probability / value.</li>
 *   <li>{@code QUANTILES}, {@code INTERVAL} → need a quantile/distribution head
 *       that a single-score checkpoint does not expose, so they raise at
 *       execution (validation may still accept them syntactically).</li>
 * </ul>
 *
 * <p>When {@code query.ret} is absent the output is the default for the task type.
 */
final class ReturnShaper {

    static final String UNSUPPORTED_DISPERSION =
            "RETURN QUANTILES/INTERVAL requires a quantile/distribution head the "
            + "current checkpoint does not expose";

    EntityPrediction shape(EntityId id, TaskType taskType, ModelOutput out, ParsedQuery query) {
        ReturnSpec ret = query == null ? null : query.ret().orElse(null);
        ReturnSpec.Kind kind = ret == null ? null : ret.kind();

        if (kind == ReturnSpec.Kind.QUANTILES || kind == ReturnSpec.Kind.INTERVAL) {
            throw new UnsupportedOperationException(UNSUPPORTED_DISPERSION);
        }

        return switch (taskType) {
            case REGRESSION -> regression(id, out.value());
            case FORECASTING -> forecasting(id, out);
            case BINARY_CLASSIFICATION -> binary(id, out.probability(), kind);
            case MULTICLASS_CLASSIFICATION -> multiclass(id, out.classProbs(), kind);
            case MULTILABEL_RANKING -> ranking(id, out);
        };
    }

    // ------------------------------------------------------------------
    //  Per-task shaping (over the model's output)
    // ------------------------------------------------------------------

    private EntityPrediction regression(EntityId id, double value) {
        // EXPECTED_VALUE and default are both the regression value.
        return new EntityPrediction(id, OptionalDouble.of(value), OptionalDouble.empty(),
                Map.of(), List.of(), List.of());
    }

    private EntityPrediction forecasting(EntityId id, ModelOutput out) {
        List<TimeframeValue> forecast = new ArrayList<>();
        for (int i = 0; i < out.forecastValues().size(); i++) {
            forecast.add(new TimeframeValue(i + 1, out.forecastValues().get(i)));
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
}
