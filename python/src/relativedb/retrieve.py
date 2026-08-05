"""The retriever SPI — the heart of the design.

Users implement these small callables (structural ``typing.Protocol``s, so any
function with the right shape works). All receive a :class:`TemporalBound` —
the engine's leakage guard (F24) — which implementations must honor and the
engine re-checks defensively.

Mirrors ``dev.rql.retrieve`` from the Java API design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable

from relational_transformers_utils.csc import CscAdjacency  # noqa: F401
from relational_transformers_utils.csc import CscIndex as _CscIndex
from relational_transformers_utils.rows import Row, TemporalBound
from relational_transformers_utils.schema import LinkDef, Schema


__all__ = [
    "TemporalBound", "Row", "CscAdjacency", "CscIndex",
    "EntityRetriever", "LinkRetriever",
    "CohortRetriever", "TableScanner", "RetrieverWiring", "WiringError",
]


@runtime_checkable
class EntityRetriever(Protocol):
    """Batched point lookup: rows of one table by id (DataFetcher analog)."""

    def __call__(self, table: str, ids: Sequence[Any],
                 bound: TemporalBound) -> list[Row]: ...


@runtime_checkable
class LinkRetriever(Protocol):
    """Children of a parent row along one P→F link, newest-first, capped at
    ``limit``. MUST NOT return rows newer than ``bound``."""

    def __call__(self, link: LinkDef, parent_id: Any,
                 bound: TemporalBound, limit: int) -> list[Row]: ...


@runtime_checkable
class CohortRetriever(Protocol):
    """OPTIONAL: similar/other entity ids of the same table for in-context
    examples (RT-J Tier 1/2). Without one, context is target-entity-local."""

    def __call__(self, table: str, anchor: Any,
                 bound: TemporalBound, limit: int) -> list[Any]: ...


@runtime_checkable
class TableScanner(Protocol):
    """OPTIONAL: stream every row of ``table`` with time <= bound (any order).
    Required for :class:`~relativedb.engine.SamplerMode.CSC`."""

    def __call__(self, table: str, bound: TemporalBound) -> Iterable[Row]: ...


class WiringError(ValueError):
    """Raised when the wiring is missing a required retriever."""


@dataclass
class RetrieverWiring:
    """Schema element -> implementation. GraphQL RuntimeWiring analog."""

    entities: dict[str, EntityRetriever] = field(default_factory=dict)
    links: dict[str, LinkRetriever] = field(default_factory=dict)
    default_link_retriever: Optional[LinkRetriever] = None
    cohorts: dict[str, CohortRetriever] = field(default_factory=dict)
    scanners: dict[str, TableScanner] = field(default_factory=dict)

    @staticmethod
    def new_wiring() -> "RetrieverWiring.Builder":
        return RetrieverWiring.Builder()

    def entity_retriever(self, table: str) -> EntityRetriever:
        r = self.entities.get(table)
        if r is None:
            raise WiringError(f"no EntityRetriever wired for table {table!r}")
        return r

    def link_retriever(self, from_table: str) -> LinkRetriever:
        r = self.links.get(from_table, self.default_link_retriever)
        if r is None:
            raise WiringError(
                f"no LinkRetriever wired for table {from_table!r} "
                f"and no default_links set")
        return r

    def cohort_retriever(self, table: str) -> Optional[CohortRetriever]:
        return self.cohorts.get(table)

    def scanner(self, table: str) -> TableScanner:
        s = self.scanners.get(table)
        if s is None:
            raise WiringError(
                f"no TableScanner wired for table {table!r} (required for "
                f"SamplerMode.CSC)")
        return s

    class Builder:
        def __init__(self) -> None:
            self._w = RetrieverWiring()

        def entities(self, table: str, retriever: EntityRetriever) -> "RetrieverWiring.Builder":
            self._w.entities[table] = retriever
            return self

        def links(self, from_table: str, retriever: LinkRetriever) -> "RetrieverWiring.Builder":
            self._w.links[from_table] = retriever
            return self

        def default_links(self, retriever: LinkRetriever) -> "RetrieverWiring.Builder":
            self._w.default_link_retriever = retriever
            return self

        def cohort(self, table: str, retriever: CohortRetriever) -> "RetrieverWiring.Builder":
            self._w.cohorts[table] = retriever
            return self

        def scanner(self, table: str, scanner: TableScanner) -> "RetrieverWiring.Builder":
            self._w.scanners[table] = scanner
            return self

        def build(self) -> "RetrieverWiring":
            return self._w


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
