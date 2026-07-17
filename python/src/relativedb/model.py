"""Model configuration + routing.

RT-J ships TWO separate checkpoints — a classifier and a regressor — so the
config holds one URI per task family and the engine routes by the query's
inferred :class:`~relativedb.pql.ast.TaskType`:

* BINARY/MULTICLASS classification, ranking  -> ``classification_model_uri``
* regression, forecasting                    -> ``regression_model_uri``

CONSTRAINT (KB F13/F14): the embedding model must be the one the checkpoints
were trained with — rt-j pins ``all-MiniLM-L12-v2`` (384-dim), shared by both
variants. Loaders that find an ``embedding_model`` in a checkpoint config must
verify it against this setting and fail fast unless
``allow_embedding_mismatch`` is set.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .pql.ast import TaskType

__all__ = ["ModelConfig", "EmbeddingMismatchError",
           "DEFAULT_CLASSIFICATION_MODEL_URI", "DEFAULT_REGRESSION_MODEL_URI",
           "DEFAULT_EMBEDDING_MODEL"]

DEFAULT_CLASSIFICATION_MODEL_URI = "hf://stanford-star/rt-j/classification"
DEFAULT_REGRESSION_MODEL_URI = "hf://stanford-star/rt-j/regression"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L12-v2"

_EMBEDDING_DIMS = {"all-MiniLM-L12-v2": 384, "all-MiniLM-L6-v2": 384}


class EmbeddingMismatchError(ValueError):
    """A checkpoint pins a different text encoder than the config."""


@dataclass(frozen=True)
class ModelConfig:
    classification_model_uri: str = DEFAULT_CLASSIFICATION_MODEL_URI
    regression_model_uri: str = DEFAULT_REGRESSION_MODEL_URI
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    allow_embedding_mismatch: bool = False

    @staticmethod
    def defaults() -> "ModelConfig":
        return ModelConfig()

    def with_model_uri(self, uri: str) -> "ModelConfig":
        """One checkpoint for ALL task types (e.g. a custom unified model)."""
        return replace(self, classification_model_uri=uri,
                       regression_model_uri=uri)

    def model_uri_for(self, task_type: TaskType) -> str:
        """Routing accessor: which checkpoint serves this task type."""
        if task_type in (TaskType.REGRESSION, TaskType.FORECASTING):
            return self.regression_model_uri
        return self.classification_model_uri

    @property
    def d_text(self) -> int:
        """Text/schema embedding width (384 for MiniLM)."""
        return _EMBEDDING_DIMS.get(self.embedding_model, 384)

    def check_checkpoint_embedding(self, checkpoint_embedding_model: str) -> None:
        """Fail fast on encoder mismatch (F13/F14) unless overridden."""
        if (checkpoint_embedding_model != self.embedding_model
                and not self.allow_embedding_mismatch):
            raise EmbeddingMismatchError(
                f"checkpoint was trained with embedding model "
                f"{checkpoint_embedding_model!r} but the config pins "
                f"{self.embedding_model!r}; set allow_embedding_mismatch=True "
                f"to override")

    def to_json_dict(self) -> dict:
        return {
            "classification_model_uri": self.classification_model_uri,
            "regression_model_uri": self.regression_model_uri,
            "embedding_model": self.embedding_model,
            "allow_embedding_mismatch": self.allow_embedding_mismatch,
        }
