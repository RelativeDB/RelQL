package com.relativedb.model;

import java.util.List;
import java.util.Map;

/**
 * Raw model output for one scored entity, decoded by the engine per task type:
 * regression → {@code value}; binary → {@code probability};
 * multiclass → {@code classProbs}; ranking → {@code rankedScores};
 * forecasting → {@code forecastValues}.
 */
public record ModelOutput(
    double value,
    double probability,
    Map<String, Double> classProbs,
    Map<String, Double> rankedScores,
    List<Double> forecastValues
) {
    public static ModelOutput regression(double value) {
        return new ModelOutput(value, Double.NaN, Map.of(), Map.of(), List.of());
    }
    public static ModelOutput binary(double probability) {
        return new ModelOutput(Double.NaN, probability, Map.of(), Map.of(), List.of());
    }
    public static ModelOutput multiclass(Map<String, Double> classProbs) {
        return new ModelOutput(Double.NaN, Double.NaN, classProbs, Map.of(), List.of());
    }
    public static ModelOutput ranking(Map<String, Double> rankedScores) {
        return new ModelOutput(Double.NaN, Double.NaN, Map.of(), rankedScores, List.of());
    }
    public static ModelOutput forecast(List<Double> values) {
        return new ModelOutput(Double.NaN, Double.NaN, Map.of(), Map.of(), values);
    }
}
