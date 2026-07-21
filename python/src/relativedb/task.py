"""Stable task identities and target-token schema.

RelQL text is user-facing syntax, not a stable model identifier: whitespace,
parentheses, and bind values can change while the prediction task remains the
same.  ``TaskSpec`` canonicalizes the bound AST and gives derived tasks stable
table/column names.  Bare entity-column autocomplete keeps the real physical
table and column, matching the reference implementation's masked-cell shape.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from .relql.ast import ColumnRef, ParsedQuery, TaskType

__all__ = ["TaskSpec", "TaskSpecFactory", "canonical_target"]


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            f.name: _canonical(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.compare
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    return value


def canonical_target(target: Any) -> str:
    """Canonical JSON for a target AST, independent of source formatting."""
    return json.dumps(_canonical(target), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _slug(text: str, limit: int = 52) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (value or "target")[:limit].rstrip("_")


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """Stable model-facing identity and target cell location for one task."""

    id: str
    entity_table: str
    task_type: TaskType
    table_name: str
    target_column: str
    time_column: str = "timestamp"
    direct_target: bool = False
    canonical: str = ""

    @classmethod
    def from_query(cls, query: ParsedQuery, task_type: TaskType) -> "TaskSpec":
        target_json = canonical_target(query.target)
        identity = json.dumps({
            "entity_table": query.entity_key.table,
            "target": json.loads(target_json),
            "task_type": task_type.value,
        }, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        target = query.target
        if (isinstance(target, ColumnRef)
                and target.table == query.entity_key.table):
            return cls(
                id=f"column:{target.table}.{target.column}:{task_type.value}",
                entity_table=query.entity_key.table,
                task_type=task_type,
                table_name=target.table,
                target_column=target.column,
                direct_target=True,
                canonical=identity,
            )
        hint = _slug(target_json)
        return cls(
            id=f"relql:{digest}",
            entity_table=query.entity_key.table,
            task_type=task_type,
            table_name=f"task_{_slug(query.entity_key.table, 32)}",
            target_column=f"{hint}_{digest[:8]}",
            direct_target=False,
            canonical=identity,
        )

    def to_json_dict(self) -> dict:
        return {
            "id": self.id,
            "entity_table": self.entity_table,
            "task_type": self.task_type.value,
            "table_name": self.table_name,
            "target_column": self.target_column,
            "time_column": self.time_column,
            "direct_target": self.direct_target,
            "canonical": self.canonical,
        }


class TaskSpecFactory(Protocol):
    def __call__(self, query: ParsedQuery, task_type: TaskType) -> TaskSpec: ...
