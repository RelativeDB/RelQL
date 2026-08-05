"""relativedb.rt — local model execution over relational-transformers.

The base ``relativedb`` package is pure Python: it parses RelQL, assembles
contexts and token batches, and can score through a remote backend URL. This
package adds the local alternative: RT-J inference through the shared
``relational-transformers`` runtime (Triton on CUDA, torch on MPS/CPU, ONNX
for exported graphs), MiniLM text encoding in torch, and the adaptation paths
— frozen task-head fitting and full fine-tuning.

    from relativedb import Engine
    from relativedb.rt import RtBackend

    engine = Engine(schema, wiring, model_backend=RtBackend(schema=schema))
"""
from .backend import FineTunedHead, RtBackend, RtNativeBackend
from .scorer import (RT_DEVICE_CPU, RT_DEVICE_CUDA, RT_DEVICE_MPS,
                     EngineError, EngineUnavailableError,
                     NativeScorer, NativeTextEncoder, RelationalScorer,
                     RtNativeError, RtNativeUnavailableError, TextEncoder,
                     resolve_minilm_snapshot, resolve_model_path)

# The historical name from when embedding ran through a separate package.
TextEmbedder = TextEncoder

__version__ = "0.1.3"

__all__ = [
    "RtBackend", "RtNativeBackend", "RelationalScorer", "NativeScorer",
    "TextEncoder", "NativeTextEncoder", "TextEmbedder",
    "FineTunedHead", "EngineError", "EngineUnavailableError",
    "RtNativeError", "RtNativeUnavailableError",
    "resolve_model_path", "resolve_minilm_snapshot",
    "RT_DEVICE_CPU", "RT_DEVICE_MPS", "RT_DEVICE_CUDA",
]
