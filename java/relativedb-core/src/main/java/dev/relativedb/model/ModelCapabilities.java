package dev.relativedb.model;

import dev.relativedb.query.TaskType;

import java.util.Set;

/** What a backend can do. */
public record ModelCapabilities(Set<TaskType> supportedTasks, int maxContextCells) {
    public static ModelCapabilities all(int maxContextCells) {
        return new ModelCapabilities(Set.of(TaskType.values()), maxContextCells);
    }
    public boolean supports(TaskType t) { return supportedTasks.contains(t); }
}
