package dev.relativedb;

import dev.relativedb.engine.RelativeDbEngine;
import dev.relativedb.engine.ExecutionInput;
import dev.relativedb.engine.PredictionResult;
import dev.relativedb.model.ModelBackend;
import dev.relativedb.model.ModelCapabilities;
import dev.relativedb.model.ModelOutput;
import dev.relativedb.model.TokenBatch;
import dev.relativedb.query.TaskType;
import dev.relativedb.retrieve.EntityId;
import dev.relativedb.retrieve.RetrieverWiring;
import dev.relativedb.retrieve.Row;
import dev.relativedb.schema.RelativeDbSchema;
import dev.relativedb.schema.LinkDef;
import dev.relativedb.schema.TableDef;
import dev.relativedb.schema.ValueType;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Industry example: streaming-service inactivity churn, mirroring
 * examples/industry/growth_churn.py.
 *
 * PREDICT COUNT(events.*, 0, 30, days) = 0 FOR EACH users.user_id
 *
 * Data plants the signal: "engaged" users stream weekly up to the anchor;
 * "fading" users stopped ~45 days before it. The backend here is a
 * transparent context-evidence baseline (churn risk decays with the amount
 * of recent activity that reached the model's context window) — it stands in
 * for a real RT backend while exercising the full pipeline: PQL parse →
 * validation → retriever hop loop → temporal guard → tokenization → scoring.
 */
class GrowthChurnExampleTest {

    static final Instant ANCHOR = Instant.parse("2026-07-01T00:00:00Z");

    static final RelativeDbSchema SCHEMA = RelativeDbSchema.newSchema()
            .table(TableDef.newTable("users")
                    .column("age", ValueType.NUMBER)
                    .primaryKey("user_id")
                    .build())
            .table(TableDef.newTable("events")
                    .column("minutes", ValueType.NUMBER)
                    .primaryKey("event_id")
                    .timeColumn("ts")
                    .build())
            .link(LinkDef.link("events", "user_id", "users"))
            .build();

    /** Context-evidence baseline: p(churn) = 1 / (1 + recent event tokens). */
    static final ModelBackend BASELINE = new ModelBackend() {
        @Override public ModelCapabilities capabilities() { return ModelCapabilities.all(8192); }
        @Override public CompletionStage<ModelOutput> score(TokenBatch batch, TaskType taskType) {
            long evidence = batch.tokens().stream()
                    .filter(t -> t.table().equals("events") && !t.isTarget())
                    .count();
            return CompletableFuture.completedFuture(
                    ModelOutput.binary(1.0 / (1.0 + evidence)));
        }
    };

    @Test
    void fadingUsersScoreRiskierThanEngagedUsers() {
        TestData.Store store = new TestData.Store();
        List<EntityId> engaged = new ArrayList<>();
        List<EntityId> fading = new ArrayList<>();
        long eventId = 0;

        for (int u = 0; u < 10; u++) {
            EntityId uid = EntityId.of((long) u);
            (u < 5 ? engaged : fading).add(uid);
            store.customers.add(Row.newRow("users", uid).cell("age", 30 + u).build());
            int lastActiveDays = u < 5 ? 2 : 45;      // planted signal
            int events = u < 5 ? 20 : 8;
            for (int k = 0; k < events; k++) {
                store.orders.add(Row.newRow("events", EntityId.of(1000 + eventId++))
                        .cell("minutes", 30.0 + k)
                        .timestamp(ANCHOR.minus(lastActiveDays + 7L * k, ChronoUnit.DAYS))
                        .parent("user_id", uid)
                        .build());
            }
        }

        RetrieverWiring wiring = RetrieverWiring.newWiring()
                .entities("users", (t, ids, b) -> store.byIds("customers", ids, b))
                .entities("events", (t, ids, b) -> store.byIds("orders", ids, b))
                .defaultLinks((link, parent, bound, limit) ->
                        store.children(link, parent, bound, limit))
                .build();

        RelativeDbEngine engine = RelativeDbEngine.newEngine(SCHEMA, wiring)
                .modelBackend(BASELINE)
                .build();

        List<EntityId> all = new ArrayList<>(engaged);
        all.addAll(fading);
        PredictionResult result = engine.execute(ExecutionInput.newInput()
                .query("PREDICT COUNT(events.*, 0, 30, days) = 0 FOR EACH users.user_id")
                .anchorTime(ANCHOR)
                .entityIds(all.stream().map(EntityId::raw).toList())
                .build()).toCompletableFuture().join();

        double engagedMean = mean(result, engaged);
        double fadingMean = mean(result, fading);
        assertTrue(fadingMean > engagedMean,
                "fading users must score riskier: fading=" + fadingMean
                        + " engaged=" + engagedMean);
    }

    private static double mean(PredictionResult r, List<EntityId> ids) {
        return r.predictions().stream()
                .filter(p -> ids.contains(p.id()))
                .mapToDouble(p -> p.probability().orElseThrow())
                .average().orElseThrow();
    }
}
