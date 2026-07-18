package com.relativedb.query;

import java.util.OptionalLong;

/**
 * A normalized temporal frame: an {@code (start, end]} offset window in a single
 * {@code unit}, plus multi-horizon fields. Used for declared {@code WINDOW ... AS}
 * templates and as the shape an aggregation's inline window decodes into.
 *
 * <p>{@code -INF}/{@code +INF} bounds are encoded as
 * {@link Aggregation#NEG_INF}/{@link Aggregation#POS_INF}. {@code horizons} is
 * {@code >= 1} (1 = single frame); {@code step} is empty when it defaults to the
 * frame width.
 */
public record Window(OptionalLong start, long end, TimeUnit unit,
                     long horizons, OptionalLong step) {

    public long startOr(long fallback) { return start.orElse(fallback); }
}
