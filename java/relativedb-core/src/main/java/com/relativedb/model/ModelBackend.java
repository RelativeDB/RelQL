package com.relativedb.model;

import com.relativedb.query.TaskType;

import java.util.List;
import java.util.concurrent.CompletionStage;

/** Anything that can score a {@link TokenBatch} can back the engine. */
public interface ModelBackend {
    /** Which TaskTypes this backend serves, max context cells, etc. */
    ModelCapabilities capabilities();

    /**
     * Scores a single {@link TokenBatch} to one scalar output (binary logit /
     * regression value / forecast). Multiclass and ranking cannot be produced
     * from a single scalar — the engine routes those to
     * {@link #classifyMulticlass} / {@link #rankCandidates} instead.
     */
    CompletionStage<ModelOutput> score(TokenBatch batch, TaskType taskType);

    /**
     * Multiclass classification: score the entity's context (the target cell
     * masked as TEXT) and match the predicted embedding against the {@code
     * classLabels}, returning a {@link ModelOutput#classProbs()} distribution
     * over exactly those labels (in the given order). Backends that do not
     * expose the TEXT decoder head reject this.
     */
    default ModelOutput classifyMulticlass(TokenBatch entityContext,
                                           List<String> classLabels, TaskType taskType) {
        throw new UnsupportedOperationException(
                "this ModelBackend does not implement multiclass classification");
    }

    /**
     * Ranking: score one existence context per candidate (already assembled by
     * the engine, one {@link TokenBatch} per candidate, aligned with {@code
     * candidateIds}) and return a {@link ModelOutput#rankedScores()} map of
     * candidate id → score, ordered by descending score (ties broken by the
     * input candidate order). Backends that cannot score candidates reject this.
     */
    default ModelOutput rankCandidates(List<TokenBatch> candidateContexts,
                                       List<String> candidateIds, TaskType taskType) {
        throw new UnsupportedOperationException(
                "this ModelBackend does not implement ranking");
    }
}
