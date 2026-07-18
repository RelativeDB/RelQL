"""Example-only pandas connector.

This module is deliberately outside the ``relationdb`` package. It shows how
an application can translate its own DataFrames into the public ``Row`` and
``RetrieverWiring`` APIs. Production connectors should push bounds and limits
into their backing store instead of materializing every row as this demo does.
"""
from __future__ import annotations

import pandas as pd

from relativedb import RetrieverWiring, Row, TemporalBound


def wire_pandas_frames(schema, frames):
    """Wire explicitly declared ``schema`` tables to user-owned DataFrames."""
    rows = {}
    for table in schema.tables:
        frame = frames[table.name]
        parent_columns = {link.fk_column for link in schema.links_from(table.name)}
        converted = []
        for record in frame.to_dict("records"):
            timestamp = record.get(table.time_column) if table.time_column else None
            if timestamp is not None and not pd.isna(timestamp):
                timestamp = pd.Timestamp(timestamp).to_pydatetime()
            else:
                timestamp = None
            cells = {
                column.name: record[column.name]
                for column in table.columns
                if record.get(column.name) is not None
                and not pd.isna(record[column.name])
            }
            parents = {
                column: record[column]
                for column in parent_columns
                if record.get(column) is not None and not pd.isna(record[column])
            }
            converted.append(Row(table.name, record[table.primary_key], cells,
                                 timestamp=timestamp, parents=parents))
        rows[table.name] = converted

    by_id = {table: {row.id: row for row in table_rows}
             for table, table_rows in rows.items()}

    def entities(table, ids, bound: TemporalBound):
        return [by_id[table][row_id] for row_id in ids
                if row_id in by_id[table]
                and bound.admits_row(by_id[table][row_id])]

    def children(link, parent_id, bound: TemporalBound, limit):
        matches = [row for row in rows[link.from_table]
                   if row.parents.get(link.fk_column) == parent_id
                   and bound.admits_row(row)]
        matches.sort(key=lambda row: row.timestamp.timestamp()
                     if row.timestamp else float("-inf"), reverse=True)
        return matches[:limit]

    def scanner(table, bound: TemporalBound):
        return (row for row in rows[table] if bound.admits_row(row))

    builder = RetrieverWiring.new_wiring().default_links(children)
    for table in rows:
        builder.entities(table, entities).scanner(table, scanner)
    return builder.build()


def predictions_frame(result):
    """Application-owned presentation adapter for a ``PredictionResult``."""
    records = []
    for prediction in result.predictions:
        record = {"entity_id": prediction.id}
        for field in ("value", "probability"):
            value = getattr(prediction, field)
            if value is not None:
                record[field] = value
        if prediction.ranked:
            record["ranked"] = list(prediction.ranked)
        if prediction.forecast:
            record["forecast"] = list(prediction.forecast)
        if getattr(prediction, "predicted_class", None) is not None:
            record["predicted_class"] = prediction.predicted_class
        if getattr(prediction, "quantiles", None):
            record["quantiles"] = dict(prediction.quantiles)
        if getattr(prediction, "interval", None) is not None:
            record["interval"] = tuple(prediction.interval)
        records.append(record)
    return pd.DataFrame.from_records(records)
