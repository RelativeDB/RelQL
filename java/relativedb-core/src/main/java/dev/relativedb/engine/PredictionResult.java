package dev.relativedb.engine;

import dev.relativedb.query.TaskType;
import dev.relativedb.retrieve.EntityId;

import java.util.List;
import java.util.Map;
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

    public record EntityPrediction(EntityId id,
                                   OptionalDouble value,           // regression / score
                                   OptionalDouble probability,     // binary clf
                                   Map<String, Double> classProbs, // multiclass
                                   List<RankedItem> ranked,        // RANK TOP K
                                   List<TimeframeValue> forecast)  // FORECAST N
    { }
}
