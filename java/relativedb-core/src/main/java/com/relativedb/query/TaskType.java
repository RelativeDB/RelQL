package com.relativedb.query;

public enum TaskType {
    REGRESSION,
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_RANKING,
    FORECASTING;

    /** The model-checkpoint family this task routes to (RT-J best_clf / best_reg). */
    public boolean isClassificationFamily() {
        return this == BINARY_CLASSIFICATION || this == MULTICLASS_CLASSIFICATION
                || this == MULTILABEL_RANKING;
    }
}
