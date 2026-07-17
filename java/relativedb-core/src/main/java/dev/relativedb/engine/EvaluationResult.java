package dev.relativedb.engine;

import java.util.Map;

/** Metric name → value, from {@code evaluate(...)}. */
public record EvaluationResult(Map<Metric, Double> metrics) { }
