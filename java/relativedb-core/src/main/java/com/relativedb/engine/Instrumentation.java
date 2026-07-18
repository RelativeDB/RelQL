package com.relativedb.engine;

import com.relativedb.query.ValidatedQuery;
import com.relativedb.retrieve.EntityId;

/** Optional tracing/metrics hooks. All methods default to no-ops. */
public interface Instrumentation {
    Instrumentation NOOP = new Instrumentation() { };

    default void onQueryValidated(ValidatedQuery query) { }
    default void onContextAssembled(EntityId entity, int rows, int cells) { }
    default void onTemporalViolationDropped(EntityId entity, String table) { }
    default void onModelInvoked(EntityId entity, int tokens) { }
}
