"""Schema declaration: tables, columns, links, value types.

Only *shape* lives here — no URLs, no credentials, no connectors. The types
are shared with the relational-transformers utility package, so a schema
built here is directly usable by its context-collection and normalization
tools.
"""
from __future__ import annotations

from relational_transformers_utils.schema import (ColumnDef, LinkDef, Schema,
                                                  SchemaError, TableDef,
                                                  ValueType)

__all__ = ["ValueType", "ColumnDef", "TableDef", "LinkDef", "Schema", "SchemaError"]
