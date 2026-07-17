//! Model configuration + routing.
//!
//! RT-J ships TWO separate checkpoints — a classifier and a regressor — so the
//! config holds one URI per task family and the engine routes by the query's
//! inferred [`TaskType`]:
//!
//! * BINARY/MULTICLASS classification, ranking  -> `classification_model_uri`
//! * regression, forecasting                    -> `regression_model_uri`
//!
//! CONSTRAINT (KB F13/F14): the embedding model must be the one the checkpoints
//! were trained with — rt-j pins `all-MiniLM-L12-v2` (384-dim), shared by both
//! variants.
//!
//! The [`ModelBackend`](crate::engine::ModelBackend) trait (which consumes
//! assembled contexts) lives in [`crate::engine`] alongside the context types.

use std::fmt;

use crate::pql::ast::TaskType;

pub const DEFAULT_CLASSIFICATION_MODEL_URI: &str = "hf://stanford-star/rt-j/classification";
pub const DEFAULT_REGRESSION_MODEL_URI: &str = "hf://stanford-star/rt-j/regression";
pub const DEFAULT_EMBEDDING_MODEL: &str = "all-MiniLM-L12-v2";

/// A checkpoint pins a different text encoder than the config.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EmbeddingMismatchError(pub String);

impl fmt::Display for EmbeddingMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "embedding mismatch: {}", self.0)
    }
}
impl std::error::Error for EmbeddingMismatchError {}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ModelConfig {
    pub classification_model_uri: String,
    pub regression_model_uri: String,
    pub embedding_model: String,
    pub allow_embedding_mismatch: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            classification_model_uri: DEFAULT_CLASSIFICATION_MODEL_URI.to_string(),
            regression_model_uri: DEFAULT_REGRESSION_MODEL_URI.to_string(),
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            allow_embedding_mismatch: false,
        }
    }
}

impl ModelConfig {
    pub fn defaults() -> ModelConfig {
        ModelConfig::default()
    }

    /// One checkpoint for ALL task types (e.g. a custom unified model).
    pub fn with_model_uri(mut self, uri: impl Into<String>) -> ModelConfig {
        let uri = uri.into();
        self.classification_model_uri = uri.clone();
        self.regression_model_uri = uri;
        self
    }

    pub fn classification_model_uri(mut self, uri: impl Into<String>) -> ModelConfig {
        self.classification_model_uri = uri.into();
        self
    }

    pub fn regression_model_uri(mut self, uri: impl Into<String>) -> ModelConfig {
        self.regression_model_uri = uri.into();
        self
    }

    pub fn embedding_model(mut self, uri: impl Into<String>) -> ModelConfig {
        self.embedding_model = uri.into();
        self
    }

    pub fn allow_embedding_mismatch(mut self, b: bool) -> ModelConfig {
        self.allow_embedding_mismatch = b;
        self
    }

    /// Routing accessor: which checkpoint serves this task type.
    pub fn model_uri_for(&self, task_type: TaskType) -> &str {
        match task_type {
            TaskType::Regression | TaskType::Forecasting => &self.regression_model_uri,
            _ => &self.classification_model_uri,
        }
    }

    /// Text/schema embedding width (384 for MiniLM).
    pub fn d_text(&self) -> usize {
        match self.embedding_model.as_str() {
            "all-MiniLM-L12-v2" | "all-MiniLM-L6-v2" => 384,
            _ => 384,
        }
    }

    /// Fail fast on encoder mismatch (F13/F14) unless overridden.
    pub fn check_checkpoint_embedding(
        &self,
        checkpoint_embedding_model: &str,
    ) -> Result<(), EmbeddingMismatchError> {
        if checkpoint_embedding_model != self.embedding_model && !self.allow_embedding_mismatch {
            return Err(EmbeddingMismatchError(format!(
                "checkpoint was trained with embedding model {:?} but the config pins {:?}; \
                 set allow_embedding_mismatch=true to override",
                checkpoint_embedding_model, self.embedding_model
            )));
        }
        Ok(())
    }
}
