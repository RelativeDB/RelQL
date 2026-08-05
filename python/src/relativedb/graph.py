"""Array-backed context assembly, re-exported from relational-transformers-utils.

The walk buffer lives in :mod:`relational_transformers_utils.graph`; the
committed sampling fingerprints pin its output.
"""
from __future__ import annotations

from relational_transformers_utils.graph import ContextGraph, ContextTruncated

__all__ = ["ContextGraph", "ContextTruncated"]
