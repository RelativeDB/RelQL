package dev.relativedb.query;

import java.util.Optional;
import java.util.OptionalLong;

/**
 * Temporal aggregation over a fact column across a (start, end] window.
 * {@code start} is excluded, {@code end} included; unit defaults to DAYS.
 *
 * <p>Window encoding: a windowed aggregation always has a non-null {@code unit}
 * (DAYS when unspecified) — {@code unit == null} means the window was omitted
 * entirely (a filtered static aggregation, e.g.
 * {@code COUNT(t.* WHERE t.amount > 100)}). {@code -INF}/{@code +INF} bounds
 * are encoded as {@link Long#MIN_VALUE}/{@link Long#MAX_VALUE}.
 */
public record Aggregation(AggFunc func, ColumnRef column, Optional<Filter> filter,
                          OptionalLong start, long end, TimeUnit unit) implements TargetExpr {

    /** Sentinel for an unbounded past window bound ({@code -INF}). */
    public static final long NEG_INF = Long.MIN_VALUE;
    /** Sentinel for an unbounded future window bound ({@code +INF}). */
    public static final long POS_INF = Long.MAX_VALUE;

    /** True when the aggregation carries a temporal window. */
    public boolean hasWindow() { return unit != null; }

    public long startOr(long fallback) { return start.orElse(fallback); }
}
