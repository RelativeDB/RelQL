package com.relativedb;

import com.relativedb.model.ModelBackend;
import com.relativedb.model.ModelCapabilities;
import com.relativedb.model.ModelOutput;
import com.relativedb.model.TokenBatch;
import com.relativedb.query.TaskType;
import com.relativedb.retrieve.EntityId;
import com.relativedb.retrieve.Row;
import com.relativedb.retrieve.TemporalBound;
import com.relativedb.schema.RelativeDbSchema;
import com.relativedb.schema.LinkDef;
import com.relativedb.schema.TableDef;
import com.relativedb.schema.ValueType;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Flow;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** Shared fixture: a customers/orders graph with an in-memory backing store. */
final class TestData {

    static final RelativeDbSchema SCHEMA = RelativeDbSchema.newSchema()
            .table(TableDef.newTable("customers")
                    .column("age", ValueType.NUMBER)
                    .primaryKey("customer_id")
                    .build())
            .table(TableDef.newTable("orders")
                    .column("qty", ValueType.NUMBER)
                    .primaryKey("order_id")
                    .timeColumn("order_date")
                    .build())
            .link(LinkDef.link("orders", "customer_id", "customers"))
            .build();

    static Instant t(String iso) { return Instant.parse(iso); }

    static Row customer(long id, double age) {
        return Row.newRow("customers", EntityId.of(id)).cell("age", age).build();
    }

    static Row order(long id, long customerId, double qty, String time) {
        return Row.newRow("orders", EntityId.of(id))
                .cell("qty", qty)
                .timestamp(t(time))
                .parent("customer_id", EntityId.of(customerId))
                .build();
    }

    /** Simple honest in-memory store used by retriever and scanner fixtures. */
    static final class Store {
        final List<Row> customers = new ArrayList<>();
        final List<Row> orders = new ArrayList<>();

        List<Row> table(String name) { return name.equals("customers") ? customers : orders; }

        CompletableFuture<List<Row>> byIds(String table, List<EntityId> ids, TemporalBound bound) {
            List<Row> out = new ArrayList<>();
            for (Row r : table(table)) {
                if (ids.contains(r.id()) && r.timestamp().map(bound::admits).orElse(true)) out.add(r);
            }
            return CompletableFuture.completedFuture(out);
        }

        /** Children newest-first, honoring bound and limit. */
        CompletableFuture<List<Row>> children(LinkDef link, EntityId parent,
                                              TemporalBound bound, int limit) {
            List<Row> out = table(link.fromTable()).stream()
                    .filter(r -> parent.equals(r.parents().get(link.fkColumn())))
                    .filter(r -> r.timestamp().map(bound::admits).orElse(true))
                    .sorted(Comparator.comparing((Row r) -> r.timestamp().orElse(Instant.MIN)).reversed())
                    .limit(limit)
                    .toList();
            return CompletableFuture.completedFuture(out);
        }

        Flow.Publisher<Row> scan(String table, TemporalBound bound) {
            List<Row> rows = table(table).stream()
                    .filter(r -> r.timestamp().map(bound::admits).orElse(true))
                    .toList();
            return subscriber -> subscriber.onSubscribe(new Flow.Subscription() {
                int next = 0;
                boolean done = false;
                @Override public void request(long n) {
                    while (n-- > 0 && next < rows.size()) subscriber.onNext(rows.get(next++));
                    if (next >= rows.size() && !done) { done = true; subscriber.onComplete(); }
                }
                @Override public void cancel() { done = true; }
            });
        }
    }

    /**
     * The one shared test-only {@link ModelBackend}: deterministic scores
     * ({@code binary 0.83} / {@code regression 12.5}) with no model artifacts.
     * Records the last batch and the number of score() calls so engine/plumbing
     * tests can assert what reached the model and whether it was invoked.
     */
    static final class StubBackend implements ModelBackend {
        final AtomicReference<TokenBatch> lastBatch = new AtomicReference<>();
        final AtomicInteger scoreCalls = new AtomicInteger();

        @Override public ModelCapabilities capabilities() { return ModelCapabilities.all(8192); }

        @Override public CompletionStage<ModelOutput> score(TokenBatch batch, TaskType taskType) {
            lastBatch.set(batch);
            scoreCalls.incrementAndGet();
            return CompletableFuture.completedFuture(
                    taskType == TaskType.BINARY_CLASSIFICATION
                            ? ModelOutput.binary(0.83) : ModelOutput.regression(12.5));
        }
    }

    private TestData() { }
}
