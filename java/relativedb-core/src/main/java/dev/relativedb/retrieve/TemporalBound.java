package dev.relativedb.retrieve;

import java.time.Instant;
import java.util.Objects;
import java.util.Optional;

/** "Nothing newer than this" — the engine's temporal-leakage guard (F24). */
public final class TemporalBound {
    private static final TemporalBound UNBOUNDED = new TemporalBound(null);

    private final Instant asOf; // nullable = unbounded

    private TemporalBound(Instant asOf) { this.asOf = asOf; }

    public static TemporalBound atOrBefore(Instant t) {
        return new TemporalBound(Objects.requireNonNull(t, "asOf"));
    }

    /** For static tables without time. */
    public static TemporalBound unbounded() { return UNBOUNDED; }

    public Optional<Instant> asOf() { return Optional.ofNullable(asOf); }

    /** True if a row stamped {@code t} is admissible under this bound (t ≤ asOf). */
    public boolean admits(Instant t) {
        return asOf == null || t == null || !t.isAfter(asOf);
    }

    @Override public boolean equals(Object o) {
        return o instanceof TemporalBound b && Objects.equals(b.asOf, asOf);
    }
    @Override public int hashCode() { return Objects.hashCode(asOf); }
    @Override public String toString() { return asOf == null ? "unbounded" : "atOrBefore(" + asOf + ")"; }
}
