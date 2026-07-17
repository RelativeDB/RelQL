package dev.relativedb.retrieve;

import java.util.Objects;

/** Opaque row identity. Wraps whatever the user's storage uses. */
public final class EntityId {
    private final Object raw;

    private EntityId(Object raw) { this.raw = Objects.requireNonNull(raw, "raw id"); }

    public static EntityId of(Object raw) { return new EntityId(raw); }

    public Object raw() { return raw; }

    @Override public boolean equals(Object o) { return o instanceof EntityId e && e.raw.equals(raw); }
    @Override public int hashCode() { return raw.hashCode(); }
    @Override public String toString() { return String.valueOf(raw); }
}
