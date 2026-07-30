"""relativedb-engine — the optional in-process native engine for relativedb.

The base ``relativedb`` package is pure Python: it parses RelQL, assembles
contexts and token batches, and can score through a remote backend URL. This
package adds the local alternative: librt_c (the golden-verified C++ RT-J
engine with its native MiniLM text encoder) behind the same
:class:`relativedb.scoring.Scorer` protocol, plus the adaptation paths that
are native by design — task-head fitting, full fine-tuning, and the optional
XGBoost flat-feature backend (``pip install relativedb-engine[xgboost]``).

    from relativedb import Engine
    from relativedb_engine import RtNativeBackend

    engine = Engine(schema, wiring,
                    model_backend=RtNativeBackend(schema=schema))
"""
from .native import (FineTunedCheckpoint, RtLib, RtModel, RtNativeError,
                     RtNativeUnavailableError, load_lib, resolve_model_path,
                     RT_DEVICE_CPU, RT_DEVICE_CUDA, RT_DEVICE_MPS)
from .backend import FineTunedHead, RtNativeBackend
from .scorer import NativeScorer, NativeTextEncoder, resolve_minilm_snapshot

# The historical name: TextEmbedder embedded through sentence-transformers in
# Python. Embedding is native-only now; the alias keeps old call sites working
# against the encoder that actually runs.
TextEmbedder = NativeTextEncoder


def __getattr__(name):
    """Lazy exports with optional runtime deps (xgboost)."""
    if name in ("XgboostBackend", "XgboostUnavailableError", "fit_xgboost"):
        from . import xgb
        return getattr(xgb, name)
    raise AttributeError(f"module 'relativedb_engine' has no attribute {name!r}")


__version__ = "0.1.3"

__all__ = [
    "RtNativeBackend", "NativeScorer", "NativeTextEncoder", "TextEmbedder",
    "FineTunedHead", "FineTunedCheckpoint",
    "RtLib", "RtModel", "RtNativeError", "RtNativeUnavailableError",
    "load_lib", "resolve_model_path", "resolve_minilm_snapshot",
    "RT_DEVICE_CPU", "RT_DEVICE_MPS", "RT_DEVICE_CUDA",
    "XgboostBackend", "XgboostUnavailableError", "fit_xgboost",
]
