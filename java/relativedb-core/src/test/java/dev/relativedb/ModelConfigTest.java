package dev.relativedb;

import dev.relativedb.model.ModelConfig;
import dev.relativedb.query.TaskType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ModelConfigTest {

    @Test
    void defaultsPinTheRtJCheckpoints() {
        ModelConfig c = ModelConfig.defaults();
        assertEquals("hf://stanford-star/rt-j/classification", c.classificationModelUri());
        assertEquals("hf://stanford-star/rt-j/regression", c.regressionModelUri());
        assertEquals("all-MiniLM-L12-v2", c.embeddingModel());
        assertFalse(c.allowEmbeddingMismatch());
    }

    @Test
    void classificationFamilyRoutesToClassifierCheckpoint() {
        ModelConfig c = ModelConfig.defaults();
        assertEquals(ModelConfig.DEFAULT_CLASSIFICATION_MODEL_URI,
                c.modelUriFor(TaskType.BINARY_CLASSIFICATION));
        assertEquals(ModelConfig.DEFAULT_CLASSIFICATION_MODEL_URI,
                c.modelUriFor(TaskType.MULTICLASS_CLASSIFICATION));
        assertEquals(ModelConfig.DEFAULT_CLASSIFICATION_MODEL_URI,
                c.modelUriFor(TaskType.MULTILABEL_RANKING));
    }

    @Test
    void regressionFamilyRoutesToRegressorCheckpoint() {
        ModelConfig c = ModelConfig.defaults();
        assertEquals(ModelConfig.DEFAULT_REGRESSION_MODEL_URI,
                c.modelUriFor(TaskType.REGRESSION));
        assertEquals(ModelConfig.DEFAULT_REGRESSION_MODEL_URI,
                c.modelUriFor(TaskType.FORECASTING));
    }

    @Test
    void modelUriSetsBothVariants() {
        ModelConfig c = ModelConfig.newConfig().modelUri("file:///models/unified").build();
        assertEquals("file:///models/unified", c.modelUriFor(TaskType.REGRESSION));
        assertEquals("file:///models/unified", c.modelUriFor(TaskType.BINARY_CLASSIFICATION));
    }

    @Test
    void singleVariantOverrideKeepsTheOther() {
        ModelConfig c = ModelConfig.newConfig()
                .regressionModelUri("file:///models/my-finetuned-reg")
                .build();
        assertEquals("file:///models/my-finetuned-reg", c.modelUriFor(TaskType.FORECASTING));
        assertEquals(ModelConfig.DEFAULT_CLASSIFICATION_MODEL_URI,
                c.modelUriFor(TaskType.MULTILABEL_RANKING));
    }

    @Test
    void miniLmTextDimensionIs384() {
        assertEquals(384, ModelConfig.defaults().textDimension());
    }
}
