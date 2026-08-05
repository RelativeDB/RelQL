"""relativedb — predictive queries (RelQL) over your own data.

GraphQL-style execution: the engine owns the query language, planning,
context assembly, and model routing — all data access goes through
user-defined retrievers. No bundled database connectors.
"""
from .schema import ColumnDef, LinkDef, Schema, SchemaError, TableDef, ValueType
from .retrieve import (CohortRetriever, EntityRetriever, LinkRetriever, Row,
                       RetrieverWiring, TableScanner, TemporalBound,
                       WiringError)
from .relql import (MissingParameterError, ParsedQuery, RelqlSyntaxError,
                  RelqlValidationError, TaskType, UnreferencedParameterError,
                  ValidatedQuery, parse, validate)
from .model import (DEFAULT_CLASSIFICATION_MODEL_URI, DEFAULT_EMBEDDING_MODEL,
                    DEFAULT_REGRESSION_MODEL_URI, EmbeddingMismatchError,
                    ModelConfig, NormalizationMode)
from .task import TaskSpec, TaskSpecFactory, canonical_target
from .traversal import (BreadthFirstTraversal, GraphAccess, GraphTraversal,
                        ReferenceTraversal, TraversalResult)
from .engine import (AssumptionNotAppliedWarning,
                     ContextCompositionWarning, ContextPolicy,
                     ContextTruncationWarning, Engine, EntityContext,
                     EntityPrediction, ExecutionError, ExecutionInput,
                     ExplainResult, InvisibleTableWarning, ModelBackend,
                     PredictionResult, SamplerMode)
from .csc import CscIndex
from .scoring import (ColumnStats, ContextConnectivityWarning, ForwardResult,
                      Scorer, ScoringError, SequenceBackend, TokenBatch)
from .remote import RemoteBackend, RemoteScorer, RemoteScoringError

# The local model engine lives in relativedb.rt, which imports torch. These
# names resolve lazily so `import relativedb` stays light for clients that
# score through a cloud backend URL.
_ENGINE_EXPORTS = ("RtBackend", "RtNativeBackend", "RtNativeUnavailableError",
                   "TextEmbedder", "FineTunedHead", "FineTunedCheckpoint")


def __getattr__(name):
    """Lazy exports that live in the torch-backed relativedb.rt subpackage."""
    if name in _ENGINE_EXPORTS:
        from . import rt
        return getattr(rt, name)
    raise AttributeError(f"module 'relativedb' has no attribute {name!r}")


__version__ = "0.1.3"

__all__ = [
    "Schema", "TableDef", "ColumnDef", "LinkDef", "ValueType", "SchemaError",
    "Row", "TemporalBound", "RetrieverWiring", "EntityRetriever",
    "LinkRetriever", "CohortRetriever", "TableScanner", "WiringError",
    "parse", "validate", "ParsedQuery", "ValidatedQuery", "TaskType",
    "RelqlSyntaxError", "RelqlValidationError", "MissingParameterError",
    "UnreferencedParameterError",
    "ModelConfig", "NormalizationMode", "EmbeddingMismatchError",
    "TaskSpec", "TaskSpecFactory", "canonical_target",
    "GraphAccess", "GraphTraversal", "TraversalResult",
    "BreadthFirstTraversal", "ReferenceTraversal",
    "DEFAULT_CLASSIFICATION_MODEL_URI", "DEFAULT_REGRESSION_MODEL_URI",
    "DEFAULT_EMBEDDING_MODEL",
    "Engine", "ExecutionInput", "ExecutionError", "ContextPolicy",
    "ContextTruncationWarning", "AssumptionNotAppliedWarning",
    "ContextCompositionWarning", "InvisibleTableWarning",
    "SamplerMode", "PredictionResult", "ExplainResult",
    "EntityPrediction", "EntityContext",
    "ModelBackend", "CscIndex",
    "RemoteBackend", "RemoteScorer", "RemoteScoringError",
    "SequenceBackend", "Scorer", "TokenBatch", "ForwardResult",
    "ScoringError", "ContextConnectivityWarning", "ColumnStats",
    "RtBackend", "RtNativeBackend", "RtNativeUnavailableError", "TextEmbedder",
    "FineTunedHead", "FineTunedCheckpoint",
]
