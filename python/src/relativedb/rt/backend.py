"""The in-process model backend: :class:`relativedb.scoring.SequenceBackend`
over a :class:`~relativedb.rt.scorer.RelationalScorer`. Sequence assembly
stays in the base package; the model runtime is the shared
relational-transformers package, and head FITTING lives in
relational-transformers-utils (``fit_feature_head``) — this backend only
serves fitted heads.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from relational_transformers import FineTunedHead
from relational_transformers_utils.schema import Schema

from relativedb.model import NormalizationMode
from relativedb.retrieve import RetrieverWiring
from relativedb.scoring import ColumnStats, SequenceBackend
from relativedb.task import TaskSpecFactory

from .scorer import RelationalScorer, TextEncoder

__all__ = ["RtBackend", "RtNativeBackend", "FineTunedHead"]


class RtNativeBackend(SequenceBackend):
    """The in-process :class:`~relativedb.engine.ModelBackend`: sequence
    assembly from the base package, MiniLM text embedding in torch, and the
    RT-J forward through relational-transformers (Triton on CUDA for target
    scores, torch elsewhere, ONNX when selected).

    Beyond scoring it carries the adaptation surface: :meth:`fit_head`
    (frozen-backbone task heads) and the collation plumbing used by
    :func:`relativedb.training.finetune`.
    """

    def __init__(self, *, schema: Optional[Schema] = None,
                 wiring: Optional[RetrieverWiring] = None,
                 lib_path: Optional[str] = None,
                 embedder: Optional[TextEncoder] = None,
                 n_threads: int = 0,
                 num_history_windows: int = 3,
                 max_seq_len: int = 2048,      # reference eval uses 8192
                 column_stats: Optional["ColumnStats"] = None,
                 normalization_mode: Optional[NormalizationMode | str] = None,
                 task_spec_factory: Optional[TaskSpecFactory] = None,
                 device: Optional[int] = None,
                 cuda_backend: str = "triton",
                 inference_backend: str = "auto",
                 onnx_model_path: Optional[str] = None,
                 head: Optional[Any] = None,
                 batch_size: Optional[int] = None):
        self._lib_path = lib_path
        if isinstance(head, (str, os.PathLike)):
            head = FineTunedHead.load(str(head))
        if head is not None and isinstance(head.column_stats, dict):
            head.column_stats = ColumnStats.from_dict(head.column_stats)
        scorer = RelationalScorer(embedder=embedder,
                                  n_threads=n_threads, device=device,
                                  cuda_backend=cuda_backend,
                                  inference_backend=inference_backend,
                                  onnx_model_path=onnx_model_path)
        super().__init__(scorer, schema=schema, wiring=wiring,
                         n_threads=n_threads,
                         num_history_windows=num_history_windows,
                         max_seq_len=max_seq_len, column_stats=column_stats,
                         normalization_mode=normalization_mode,
                         task_spec_factory=task_spec_factory, head=head,
                         batch_size=batch_size)

    @property
    def embedder(self) -> TextEncoder:
        return self.scorer.embedder

    @property
    def device(self) -> Optional[int]:
        return self.scorer.device

    @device.setter
    def device(self, value: Optional[int]) -> None:
        self.scorer.device = value


# Primary public name. The historical name remains source-compatible.
RtBackend = RtNativeBackend
