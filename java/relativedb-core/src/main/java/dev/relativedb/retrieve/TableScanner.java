package dev.relativedb.retrieve;

import java.util.concurrent.Flow;

/**
 * OPTIONAL: bulk access to a table, for strategies that materialize an
 * in-memory index (CSC sampling). Streams the SAME {@link Row} type the point
 * retrievers return. Implementations iterate a DataFrame export, a Parquet
 * file, a REST page-scroll — anything.
 */
@FunctionalInterface
public interface TableScanner {
    /** Stream every row of {@code table} with time ≤ bound. Order irrelevant. */
    Flow.Publisher<Row> scan(String table, TemporalBound bound);
}
