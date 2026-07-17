"""Schema declaration: tables, columns, links, value types.

Only *shape* lives here — no URLs, no credentials, no connectors.
Mirrors ``dev.rql.schema`` from the Java API design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

__all__ = ["ValueType", "ColumnDef", "TableDef", "LinkDef", "Schema", "SchemaError"]


class ValueType(Enum):
    """Semantic value types — exactly RT's sem types (F10–F13)."""

    NUMBER = "number"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


class SchemaError(ValueError):
    """Raised when a schema is internally inconsistent."""


@dataclass(frozen=True)
class ColumnDef:
    """A typed feature column. IDs / FK columns are *not* columns here (F17)."""

    name: str
    type: ValueType

    @staticmethod
    def of(name: str, type: ValueType) -> "ColumnDef":
        return ColumnDef(name, type)


@dataclass(frozen=True)
class LinkDef:
    """A foreign-key link: ``from_table.fk_column -> to_table.primary_key``."""

    from_table: str
    fk_column: str
    to_table: str

    @staticmethod
    def link(from_table: str, fk_column: str, to_table: str) -> "LinkDef":
        return LinkDef(from_table, fk_column, to_table)


@dataclass(frozen=True)
class TableDef:
    """A table: typed feature columns + identity (PK) + optional row time.

    The primary key is identity only — never surfaced as a cell (F17).
    ``time_column`` drives temporal filtering (F24) and windows.
    """

    name: str
    columns: tuple[ColumnDef, ...] = ()
    primary_key: Optional[str] = None
    time_column: Optional[str] = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for c in self.columns:
            if c.name in seen:
                raise SchemaError(
                    f"table {self.name!r}: duplicate column {c.name!r}")
            seen.add(c.name)
        if self.time_column is not None and self.time_column not in seen:
            raise SchemaError(
                f"table {self.name!r}: time_column {self.time_column!r} "
                f"is not a declared column")

    @staticmethod
    def new_table(name: str) -> "TableDef.Builder":
        return TableDef.Builder(name)

    def column(self, name: str) -> Optional[ColumnDef]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    class Builder:
        def __init__(self, name: str) -> None:
            self._name = name
            self._columns: list[ColumnDef] = []
            self._pk: Optional[str] = None
            self._time: Optional[str] = None

        def column(self, name_or_def, type: Optional[ValueType] = None) -> "TableDef.Builder":
            if isinstance(name_or_def, ColumnDef):
                self._columns.append(name_or_def)
            else:
                if type is None:
                    raise SchemaError("column(name, type): type is required")
                self._columns.append(ColumnDef(name_or_def, type))
            return self

        def primary_key(self, column: str) -> "TableDef.Builder":
            self._pk = column
            return self

        def time_column(self, column: str) -> "TableDef.Builder":
            self._time = column
            return self

        def build(self) -> "TableDef":
            return TableDef(self._name, tuple(self._columns), self._pk, self._time)


@dataclass(frozen=True)
class Schema:
    """The declared relational graph. Validates on construction."""

    tables: tuple[TableDef, ...] = ()
    links: tuple[LinkDef, ...] = ()
    _by_name: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_name: dict[str, TableDef] = {}
        for t in self.tables:
            if t.name in by_name:
                raise SchemaError(f"duplicate table {t.name!r}")
            by_name[t.name] = t
        for l in self.links:
            if l.from_table not in by_name:
                raise SchemaError(
                    f"link {l}: unknown from_table {l.from_table!r}")
            if l.to_table not in by_name:
                raise SchemaError(f"link {l}: unknown to_table {l.to_table!r}")
            if by_name[l.to_table].primary_key is None:
                raise SchemaError(
                    f"link {l}: to_table {l.to_table!r} has no primary key")
        object.__setattr__(self, "_by_name", by_name)

    @staticmethod
    def new_schema() -> "Schema.Builder":
        return Schema.Builder()

    def table(self, name: str) -> Optional[TableDef]:
        return self._by_name.get(name)

    def require_table(self, name: str) -> TableDef:
        t = self._by_name.get(name)
        if t is None:
            raise SchemaError(f"unknown table {name!r}")
        return t

    def links_from(self, table: str) -> list[LinkDef]:
        """F→P links whose *from* side is ``table`` (its parents)."""
        return [l for l in self.links if l.from_table == table]

    def links_to(self, table: str) -> list[LinkDef]:
        """P→F links whose *to* side is ``table`` (its children edges)."""
        return [l for l in self.links if l.to_table == table]

    def to_json_dict(self) -> dict:
        """JSON-friendly form, shared with the ``relativedb-ffi`` C ABI."""
        return {
            "tables": [
                {
                    "name": t.name,
                    "columns": [{"name": c.name, "type": c.type.value}
                                for c in t.columns],
                    "primary_key": t.primary_key,
                    "time_column": t.time_column,
                }
                for t in self.tables
            ],
            "links": [
                {"from_table": l.from_table, "fk_column": l.fk_column,
                 "to_table": l.to_table}
                for l in self.links
            ],
        }

    class Builder:
        def __init__(self) -> None:
            self._tables: list[TableDef] = []
            self._links: list[LinkDef] = []

        def table(self, table: TableDef) -> "Schema.Builder":
            self._tables.append(table)
            return self

        def link(self, link_or_from, fk_column: Optional[str] = None,
                 to_table: Optional[str] = None) -> "Schema.Builder":
            if isinstance(link_or_from, LinkDef):
                self._links.append(link_or_from)
            else:
                self._links.append(LinkDef(link_or_from, fk_column, to_table))
            return self

        def build(self) -> "Schema":
            return Schema(tuple(self._tables), tuple(self._links))
