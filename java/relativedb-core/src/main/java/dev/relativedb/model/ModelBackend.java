package dev.relativedb.model;

import dev.relativedb.query.TaskType;

import java.util.concurrent.CompletionStage;

/** Anything that can score a {@link TokenBatch} can back the engine. */
public interface ModelBackend {
    /** Which TaskTypes this backend serves, max context cells, etc. */
    ModelCapabilities capabilities();

    CompletionStage<ModelOutput> score(TokenBatch batch, TaskType taskType);
}
