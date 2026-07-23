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
                  RelqlValidationError, TaskType, ValidatedQuery, parse,
                  validate)
from .model import (DEFAULT_CLASSIFICATION_MODEL_URI, DEFAULT_EMBEDDING_MODEL,
                    DEFAULT_REGRESSION_MODEL_URI, EmbeddingMismatchError,
                    ModelConfig, NormalizationMode)
from .task import TaskSpec, TaskSpecFactory, canonical_target
from .traversal import (BreadthFirstTraversal, GraphAccess, GraphTraversal,
                        ReferenceTraversal, TraversalResult)
from .engine import (AssumptionNotAppliedWarning, ContextPolicy,
                     ContextTruncationWarning, Engine, EntityContext,
                     EntityPrediction, ExecutionError, ExecutionInput,
                     ExplainResult, ModelBackend, PredictionResult,
                     SamplerMode)
from .csc import CscIndex


def __getattr__(name):
    """Lazy exports with optional runtime deps (librt_c, MiniLM encoder)."""
    if name in ("RtNativeBackend", "RtNativeUnavailableError", "TextEmbedder",
                "ContextConnectivityWarning", "FineTunedHead",
                "FineTunedCheckpoint", "ColumnStats"):
        from . import rt_native
        return getattr(rt_native, name)
    raise AttributeError(f"module 'relativedb' has no attribute {name!r}")


__version__ = "0.1.1"

__all__ = [
    "Schema", "TableDef", "ColumnDef", "LinkDef", "ValueType", "SchemaError",
    "Row", "TemporalBound", "RetrieverWiring", "EntityRetriever",
    "LinkRetriever", "CohortRetriever", "TableScanner", "WiringError",
    "parse", "validate", "ParsedQuery", "ValidatedQuery", "TaskType",
    "RelqlSyntaxError", "RelqlValidationError", "MissingParameterError",
    "ModelConfig", "NormalizationMode", "EmbeddingMismatchError",
    "TaskSpec", "TaskSpecFactory", "canonical_target",
    "GraphAccess", "GraphTraversal", "TraversalResult",
    "BreadthFirstTraversal", "ReferenceTraversal",
    "DEFAULT_CLASSIFICATION_MODEL_URI", "DEFAULT_REGRESSION_MODEL_URI",
    "DEFAULT_EMBEDDING_MODEL",
    "Engine", "ExecutionInput", "ExecutionError", "ContextPolicy",
    "ContextTruncationWarning", "AssumptionNotAppliedWarning",
    "SamplerMode", "PredictionResult", "ExplainResult",
    "EntityPrediction", "EntityContext",
    "ModelBackend", "CscIndex",
    "RtNativeBackend", "RtNativeUnavailableError", "TextEmbedder",
    "ContextConnectivityWarning", "ColumnStats", "FineTunedHead",
    "FineTunedCheckpoint",
]
