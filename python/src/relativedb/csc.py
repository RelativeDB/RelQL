"""CSC context-collection index, adapted to the retriever wiring.

The adjacency and the snapshot index live in
:mod:`relational_transformers_utils.csc`; this module keeps the
wiring-facing ``build`` that feeds them from :class:`TableScanner`
implementations.
"""
from __future__ import annotations

from relational_transformers_utils.csc import CscAdjacency
from relational_transformers_utils.csc import CscIndex as _CscIndex

from .retrieve import RetrieverWiring, TemporalBound
from .schema import Schema

__all__ = ["CscIndex", "CscAdjacency"]


class CscIndex(_CscIndex):
    """Snapshot index over scanner-provided tables. Rebuild via a new build()."""

    @staticmethod
    def build(schema: Schema, wiring: RetrieverWiring,
              bound: TemporalBound = TemporalBound.unbounded(), *,
              allow_missing_scanners: bool = False) -> "CscIndex":
        tables = {}
        for table in schema.tables:
            scanner = wiring.scanners.get(table.name)
            if scanner is None:
                if not allow_missing_scanners:
                    scanner = wiring.scanner(table.name)  # raises precise error
                else:
                    continue  # empty table in the snapshot
            tables[table.name] = scanner(table.name, bound)
        built = _CscIndex.build(schema, tables, bound)
        index = CscIndex()
        index.rows = built.rows
        index.dense = built.dense
        index.adjacency = built.adjacency
        return index
