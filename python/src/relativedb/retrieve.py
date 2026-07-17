"""The retriever SPI — the heart of the design.

Users implement these small callables (structural ``typing.Protocol``s, so any
function with the right shape works). All receive a :class:`TemporalBound` —
the engine's leakage guard (F24) — which implementations must honor and the
engine re-checks defensively.

Mirrors ``dev.rql.retrieve`` from the Java API design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable

from .schema import LinkDef

__all__ = [
    "TemporalBound", "Row", "EntityRetriever", "LinkRetriever",
    "CohortRetriever", "TableScanner", "RetrieverWiring", "WiringError",
]


def _to_utc(t: datetime) -> datetime:
    if t.tzinfo is None:
        return t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


@dataclass(frozen=True)
class TemporalBound:
    """"Nothing newer than this" — the temporal-leakage guard (F24).

    ``as_of is None`` means unbounded (static tables without time).
    """

    as_of: Optional[datetime] = None

    @staticmethod
    def at_or_before(t: datetime) -> "TemporalBound":
        return TemporalBound(_to_utc(t))

    @staticmethod
    def unbounded() -> "TemporalBound":
        return TemporalBound(None)

    @property
    def is_unbounded(self) -> bool:
        return self.as_of is None

    def admits(self, timestamp: Optional[datetime]) -> bool:
        """A row with no timestamp is static and always admitted."""
        if self.as_of is None or timestamp is None:
            return True
        return _to_utc(timestamp) <= self.as_of

    def admits_row(self, row: "Row") -> bool:
        return self.admits(row.timestamp)


@dataclass(frozen=True)
class Row:
    """One row's typed feature cells.

    IDs and FK values are NOT cells (F17) — links are reported separately via
    ``parents`` so the engine can traverse without ever tokenizing identifiers.
    Missing/null values: simply omit the cell — nulls emit no token.
    """

    table: str
    id: Any
    cells: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    parents: dict[str, Any] = field(default_factory=dict)  # fk column -> parent id

    def __post_init__(self) -> None:
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", _to_utc(self.timestamp))

    @property
    def key(self) -> tuple[str, Any]:
        return (self.table, self.id)

    def to_json_dict(self) -> dict:
        """The Row JSON shape shared with the ``relativedb-ffi`` C ABI."""
        cells = {}
        for k, v in self.cells.items():
            if isinstance(v, datetime):
                cells[k] = v.isoformat()
            else:
                cells[k] = v
        return {
            "table": self.table,
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "cells": cells,
            "parents": dict(self.parents),
        }

    @staticmethod
    def from_json_dict(d: dict) -> "Row":
        ts = d.get("timestamp")
        return Row(
            table=d["table"],
            id=d["id"],
            cells=dict(d.get("cells") or {}),
            timestamp=datetime.fromisoformat(ts) if ts else None,
            parents=dict(d.get("parents") or {}),
        )


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
