package com.relativedb.retrieve;

/**
 * Normalization statistics — training-split-only by contract (F11, F12).
 * Deliberately a retriever: stats are data-owner knowledge, not engine state.
 */
public interface StatsProvider {

    /** Per-column numeric normalization stats (mean, std). */
    ColumnStats numericStats(String table, String column);

    /** Global datetime mean/std (F12). */
    DatetimeStats datetimeStats();

    record ColumnStats(double mean, double std) {
        public double normalize(double value) { return std == 0 ? 0 : (value - mean) / std; }
    }

    /** Mean/std of epoch seconds across all timestamped rows in the training split. */
    record DatetimeStats(double meanEpochSeconds, double stdSeconds) {
        public double normalize(java.time.Instant t) {
            return stdSeconds == 0 ? 0 : (t.getEpochSecond() - meanEpochSeconds) / stdSeconds;
        }
    }
}
