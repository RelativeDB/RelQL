package com.relativedb.engine;

import com.relativedb.query.TaskType;
import com.relativedb.retrieve.EntityId;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/** Decoded predictions, one per scored entity. */
public final class PredictionResult {
    private final TaskType taskType;
    private final List<EntityPrediction> predictions;

    public PredictionResult(TaskType taskType, List<EntityPrediction> predictions) {
        this.taskType = taskType;
        this.predictions = List.copyOf(predictions);
    }

    public TaskType taskType() { return taskType; }
    public List<EntityPrediction> predictions() { return predictions; }

    public record RankedItem(String item, double score) { }
    public record TimeframeValue(int timeframe, double value) { }

    /** A closed prediction interval {@code [lower, upper]} for {@code RETURN INTERVAL}. */
    public record Interval(double lower, double upper) { }

    public record EntityPrediction(EntityId id,
                                   OptionalDouble value,           // regression / score
                                   OptionalDouble probability,     // binary clf
                                   Map<String, Double> classProbs, // multiclass
                                   List<RankedItem> ranked,        // RANK TOP K
                                   List<TimeframeValue> forecast,  // FORECAST N
                                   Optional<String> predictedClass,   // RETURN CLASS: hard label
                                   Map<Double, Double> quantiles,     // RETURN QUANTILES: q -> value (ordered)
                                   Optional<Interval> interval)       // RETURN INTERVAL: [lo, hi]
    {
        /** Legacy 6-arg shape: no RETURN-specific outputs. */
        public EntityPrediction(EntityId id, OptionalDouble value, OptionalDouble probability,
                                Map<String, Double> classProbs, List<RankedItem> ranked,
                                List<TimeframeValue> forecast) {
            this(id, value, probability, classProbs, ranked, forecast,
                    Optional.empty(), Map.of(), Optional.empty());
        }
    }
}
