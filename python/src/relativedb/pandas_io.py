"""DataFrames in, predictions out — the "bring your own data" story.

``relativedb.from_dataframes({...}, links=[...])`` derives a :class:`Schema`
and wires in-memory retrievers + scanners over the frames automatically.
No connectors: the frames came from wherever *you* got them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence, Union

from .engine import (ContextPolicy, Engine, EntityContext, ExecutionInput,
                     ModelBackend, PredictionResult, SamplerMode)
from .model import ModelConfig
from .retrieve import RetrieverWiring, Row, TemporalBound
from .schema import ColumnDef, LinkDef, Schema, SchemaError, TableDef, ValueType

__all__ = ["from_dataframes", "Dataset"]

_TIME_NAME_HINTS = ("timestamp", "time", "date", "datetime", "created_at",
                    "created", "event_time", "occurred_at")


def _import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "from_dataframes requires pandas: pip install 'relativedb[pandas]'"
        ) from e


def _as_link(spec) -> LinkDef:
    if isinstance(spec, LinkDef):
        return spec
    if isinstance(spec, (tuple, list)) and len(spec) == 3:
        return LinkDef(*spec)
    raise SchemaError(
        f"link spec must be a LinkDef or (from_table, fk_column, to_table) "
        f"triple, got {spec!r}")


def _infer_primary_key(table: str, columns: Sequence[str]) -> Optional[str]:
    singular = table[:-1] if table.endswith("s") else table
    for cand in (f"{singular}_id", f"{table}_id", "id"):
        if cand in columns:
            return cand
    return None


def _value_type(pd, dtype) -> ValueType:
    if pd.api.types.is_bool_dtype(dtype):
        return ValueType.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return ValueType.DATETIME
    if pd.api.types.is_numeric_dtype(dtype):
        return ValueType.NUMBER
    return ValueType.TEXT


def _cell_value(pd, v, vtype: ValueType):
    if v is None or (pd.isna(v) if not isinstance(v, (list, tuple)) else False):
        return None
    if vtype is ValueType.DATETIME:
        return pd.Timestamp(v).to_pydatetime()
    if vtype is ValueType.NUMBER:
        return float(v)
    if vtype is ValueType.BOOLEAN:
        return bool(v)
    return str(v)


@dataclass
class _TableFrame:
    name: str
    df: Any
    pk: Optional[str]
    time_column: Optional[str]
    fk_columns: dict[str, str]           # fk column -> parent table
    feature_types: dict[str, ValueType]  # feature column -> type

    def rows(self) -> list[Row]:
        pd = _import_pandas()
        out: list[Row] = []
        for i, rec in enumerate(self.df.to_dict("records")):
            rid = rec[self.pk] if self.pk else i
            cells = {}
            for col, vt in self.feature_types.items():
                v = _cell_value(pd, rec.get(col), vt)
                if v is not None:
                    cells[col] = v
            parents = {}
            for fk in self.fk_columns:
                v = rec.get(fk)
                if v is not None and not pd.isna(v):
                    parents[fk] = v
            ts = None
            if self.time_column is not None:
                tv = rec.get(self.time_column)
                if tv is not None and not pd.isna(tv):
                    ts = pd.Timestamp(tv).to_pydatetime()
            out.append(Row(table=self.name, id=rid, cells=cells,
                           timestamp=ts, parents=parents))
        return out


class Dataset:
    """Schema + in-memory retrievers/scanners over a dict of DataFrames,
    with a Kumo-style ``predict(query, anchor_time=...)`` returning rows."""

    def __init__(self, schema: Schema, wiring: RetrieverWiring,
                 rows: dict[str, list[Row]]):
        self.schema = schema
        self.wiring = wiring
        self._rows = rows

    def engine(self, *, model_config: Optional[ModelConfig] = None,
               model_backend: Optional[ModelBackend] = None,
               context_policy: Optional[ContextPolicy] = None,
               sampler_mode: SamplerMode = SamplerMode.RETRIEVER) -> Engine:
        return Engine(self.schema, self.wiring, model_config=model_config,
                      model_backend=model_backend,
                      context_policy=context_policy,
                      sampler_mode=sampler_mode)

    def predict(self, query: str, *, anchor_time: Optional[datetime] = None,
                indices: Optional[Sequence[Any]] = None,
                sampler_mode: SamplerMode = SamplerMode.RETRIEVER,
                **engine_kwargs):
        """Run a PQL query; returns a ``pandas.DataFrame`` of predictions."""
        pd = _import_pandas()
        if anchor_time is not None:
            anchor_time = pd.Timestamp(anchor_time).to_pydatetime()
        eng = self.engine(sampler_mode=sampler_mode, **engine_kwargs)
        result = eng.execute(ExecutionInput(query=query,
                                            anchor_time=anchor_time,
                                            entity_ids=indices))
        return result.to_dataframe()

    def assemble_context(self, entity_table: str, entity_id: Any,
                         anchor_time: Optional[datetime] = None,
                         **engine_kwargs) -> EntityContext:
        pd = _import_pandas()
        if anchor_time is not None:
            anchor_time = pd.Timestamp(anchor_time).to_pydatetime()
        return self.engine(**engine_kwargs).assemble_context(
            entity_table, entity_id, anchor_time)


def from_dataframes(dataframes: dict[str, Any],
                    links: Sequence[Union[LinkDef, tuple]] = (),
                    *,
                    primary_keys: Optional[dict[str, str]] = None,
                    time_columns: Optional[dict[str, str]] = None) -> Dataset:
    """Build a :class:`Dataset` (schema + in-memory retrievers) from frames.

    - ``links``: ``(from_table, fk_column, to_table)`` triples (FK -> PK).
    - ``primary_keys``: per-table PK override; otherwise inferred from
      ``<singular>_id`` / ``<table>_id`` / ``id`` column names.
    - ``time_columns``: per-table row-time override; otherwise inferred when a
      table has exactly one datetime column, or one whose name is a common
      time name (``timestamp``, ``date``, ``created_at``, ...).

    PK and FK columns never become feature cells (F17) — FKs surface as
    parent edges, PKs as row identity.
    """
    pd = _import_pandas()
    link_defs = [_as_link(l) for l in links]
    primary_keys = dict(primary_keys or {})
    time_columns = dict(time_columns or {})

    tables: dict[str, _TableFrame] = {}
    for name, df in dataframes.items():
        cols = list(df.columns)
        pk = primary_keys.get(name) or _infer_primary_key(name, cols)
        fk_columns = {l.fk_column: l.to_table for l in link_defs
                      if l.from_table == name}
        for fk in fk_columns:
            if fk not in cols:
                raise SchemaError(
                    f"link fk column {fk!r} not found in table {name!r}")
        feature_cols = [c for c in cols if c != pk and c not in fk_columns]
        feature_types = {c: _value_type(pd, df[c].dtype) for c in feature_cols}
        tc = time_columns.get(name)
        if tc is None:
            dt_cols = [c for c in feature_cols
                       if feature_types[c] is ValueType.DATETIME]
            if len(dt_cols) == 1:
                tc = dt_cols[0]
            elif len(dt_cols) > 1:
                hinted = [c for c in dt_cols
                          if c.lower() in _TIME_NAME_HINTS
                          or any(c.lower().endswith(h) for h in _TIME_NAME_HINTS)]
                if len(hinted) == 1:
                    tc = hinted[0]
        tables[name] = _TableFrame(name, df, pk, tc, fk_columns, feature_types)

    builder = Schema.new_schema()
    for tf in tables.values():
        tb = TableDef.new_table(tf.name)
        for col, vt in tf.feature_types.items():
            tb.column(ColumnDef(col, vt))
        if tf.pk:
            tb.primary_key(tf.pk)
        if tf.time_column:
            tb.time_column(tf.time_column)
        builder.table(tb.build())
    for l in link_defs:
        builder.link(l)
    schema = builder.build()

    rows = {name: tf.rows() for name, tf in tables.items()}
    by_id = {name: {r.id: r for r in rs} for name, rs in rows.items()}
    children_by_parent: dict[tuple, dict[Any, list[Row]]] = {}
    for l in link_defs:
        bucket: dict[Any, list[Row]] = {}
        for r in rows[l.from_table]:
            pid = r.parents.get(l.fk_column)
            if pid is not None:
                bucket.setdefault(pid, []).append(r)
        for kids in bucket.values():  # newest-first
            kids.sort(key=lambda r: (r.timestamp is None,
                                     -(r.timestamp.timestamp()
                                       if r.timestamp else 0.0)))
        children_by_parent[(l.from_table, l.fk_column, l.to_table)] = bucket

    def make_entity_retriever(table: str):
        idx = by_id[table]

        def fetch(t: str, ids, bound: TemporalBound) -> list[Row]:
            return [idx[i] for i in ids
                    if i in idx and bound.admits_row(idx[i])]
        return fetch

    def link_retriever(link: LinkDef, parent_id, bound: TemporalBound,
                       limit: int) -> list[Row]:
        bucket = children_by_parent.get(
            (link.from_table, link.fk_column, link.to_table), {})
        out = []
        for r in bucket.get(parent_id, ()):  # already newest-first
            if bound.admits_row(r):
                out.append(r)
                if len(out) >= limit:
                    break
        return out

    def make_scanner(table: str):
        def scan(t: str, bound: TemporalBound):
            return (r for r in rows[table] if bound.admits_row(r))
        return scan

    def make_cohort(table: str):
        def cohort(t: str, anchor, bound: TemporalBound, limit: int):
            out = []
            for r in rows[table]:
                if r.id != anchor and bound.admits_row(r):
                    out.append(r.id)
                    if len(out) >= limit:
                        break
            return out
        return cohort

    wb = RetrieverWiring.new_wiring().default_links(link_retriever)
    for name in tables:
        wb.entities(name, make_entity_retriever(name))
        wb.scanner(name, make_scanner(name))
        wb.cohort(name, make_cohort(name))
    return Dataset(schema, wb.build(), rows)
