"""End-to-end RelQL evaluation utilities for the hosted RelBench F1 data.

Unlike ``run_native_on_reference.py``, this module does not accept reference-
generated model tensors.  It converts the relational database into RelativeDB
``Row`` objects, builds a real ``Schema`` and ``RetrieverWiring``, and executes
queries through ``Engine.execute``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import json
import ml_dtypes

from relativedb import (ColumnDef, ContextPolicy, Engine, ExecutionInput,
                        LinkDef, ModelConfig, ReferenceTraversal,
                        RetrieverWiring, Row, Schema, TableDef, TemporalBound,
                        TaskSpec, ValueType, NormalizationMode, parse, validate)
from relativedb.rt_native import (RT_DEVICE_MPS, ColumnStats,
                                  RtNativeBackend)


DATASET = "relbench/core/rel-f1"


@dataclass(frozen=True)
class RelqlTask:
    name: str
    query: str
    id_column: str
    target_column: str
    classification: bool
    per_entity_anchor: bool
    # RelBench autocomplete tasks can explicitly mask correlated columns in
    # addition to the target.  Apply that contract to the database presented
    # to both the query engine and the baseline.
    remove_columns: tuple[tuple[str, str], ...] = ()


TASKS = {
    "driver-dnf": RelqlTask(
        name="driver-dnf",
        query=(
            "PREDICT COUNT(results.* WHERE results.statusId != 1) "
            "OVER (30 DAYS FOLLOWING) > 0 FROM drivers "
            "WHERE drivers.driverId IN :ids RETURN PROBABILITY"
        ),
        id_column="driverId",
        target_column="did_not_finish",
        classification=True,
        per_entity_anchor=False,
    ),
    "qualifying-position": RelqlTask(
        name="qualifying-position",
        query=(
            "PREDICT qualifying.position FROM qualifying "
            "WHERE qualifying.qualifyId IN :ids RETURN EXPECTED VALUE"
        ),
        id_column="qualifyId",
        target_column="position",
        classification=False,
        per_entity_anchor=True,
    ),
    "results-position": RelqlTask(
        name="results-position",
        query=(
            "PREDICT results.position FROM results "
            "WHERE results.resultId IN :ids RETURN EXPECTED VALUE"
        ),
        id_column="resultId",
        target_column="position",
        classification=False,
        per_entity_anchor=True,
        remove_columns=(
            ("results", "statusId"),
            ("results", "positionOrder"),
            ("results", "points"),
            ("results", "laps"),
            ("results", "milliseconds"),
            ("results", "fastestLap"),
            ("results", "rank"),
        ),
    ),
}


def _value_type(series: pd.Series) -> ValueType:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return ValueType.DATETIME
    if pd.api.types.is_bool_dtype(series.dtype):
        return ValueType.BOOLEAN
    if pd.api.types.is_numeric_dtype(series.dtype):
        return ValueType.NUMBER
    return ValueType.TEXT


def _python_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def build_product_database(dataset, *, remove_columns=()):
    """Return a real RelativeDB schema/wiring over the hosted F1 tables."""
    db = dataset.get_db(upto_test_timestamp=False)
    removed = set(remove_columns)
    tables: list[TableDef] = []
    links: list[LinkDef] = []
    rows: dict[str, list[Row]] = {}

    # rustler preprocessing sorts database parquet paths before assigning
    # global node ids. Preserve that alphabetical table order so the native
    # reference traversal receives the same per-row RNG seeds.
    for table_name in sorted(db.table_dict):
        table = db.table_dict[table_name]
        frame = table.df
        feature_columns: list[ColumnDef] = []
        fkeys = table.fkey_col_to_pkey_table
        for column in frame.columns:
            if column == table.pkey_col or column in fkeys:
                continue
            if (table_name, column) in removed:
                continue
            feature_columns.append(ColumnDef(column, _value_type(frame[column])))
        tables.append(TableDef(
            table_name, tuple(feature_columns), table.pkey_col, table.time_col))
        links.extend(LinkDef(table_name, fk, parent)
                     for fk, parent in fkeys.items())

        table_rows: list[Row] = []
        feature_names = [c.name for c in feature_columns]
        for record in frame.to_dict("records"):
            rid = _python_value(record[table.pkey_col])
            parents = {fk: _python_value(record[fk]) for fk in fkeys
                       if not pd.isna(record[fk])}
            cells = {column: _python_value(record[column])
                     for column in feature_names
                     if not pd.isna(record[column])}
            timestamp = None
            if table.time_col and not pd.isna(record[table.time_col]):
                timestamp = _python_value(record[table.time_col])
            table_rows.append(Row(table_name, rid, cells, timestamp, parents))
        rows[table_name] = table_rows

    schema = Schema(tuple(tables), tuple(links))
    by_id = {name: {row.id: row for row in table_rows}
             for name, table_rows in rows.items()}

    def entity(table_name, ids, bound: TemporalBound):
        return [row for rid in ids
                if (row := by_id[table_name].get(rid)) is not None
                and bound.admits_row(row)]

    def children(link, parent_id, bound: TemporalBound, limit: int):
        found = [row for row in rows[link.from_table]
                 if row.parents.get(link.fk_column) == parent_id
                 and bound.admits_row(row)]
        found.sort(key=lambda row: (
            row.timestamp is None,
            -(row.timestamp.timestamp() if row.timestamp else 0.0)))
        return found[:limit]

    def scanner(table_name, bound: TemporalBound) -> Iterable[Row]:
        return (row for row in rows[table_name] if bound.admits_row(row))

    builder = RetrieverWiring.new_wiring().default_links(children)
    for table_name in rows:
        builder.entities(table_name, entity)
        builder.scanner(table_name, scanner)
    return schema, builder.build(), rows


def build_engine(dataset, task: RelqlTask, *, context_size: int = 128,
                 batch_size: int = 4, library: str | None = None) -> Engine:
    schema, wiring, physical_rows = build_product_database(
        dataset, remove_columns=task.remove_columns)
    parsed = validate(parse(task.query), schema).query

    def task_spec_factory(query, task_type):
        # The released checkpoint saw explicit RelBench task-table names even
        # for autocomplete tasks. Keep the public RelQL target while making
        # its model-facing task cell identical to that training/eval contract.
        return TaskSpec(
            id=f"rel-f1/{task.name}",
            entity_table=query.entity_key.table,
            task_type=task_type,
            table_name=task.name,
            target_column=task.target_column,
            time_column="date",
            direct_target=False,
        )

    task_spec = task_spec_factory(parsed, parsed.task_type(schema))
    from relbench import load_task
    benchmark_task = load_task(DATASET, task.name)
    train_values = benchmark_task.get_table(
        "train", mask_input_cols=False).df[task.target_column].to_numpy()
    column_stats = ColumnStats.fit(schema, wiring).with_task_values(
        task_spec, train_values)
    task_dir = Path(benchmark_task._task_dir)
    manifest = yaml.safe_load((task_dir / "manifest.yaml").read_text())
    manifest_fkeys = {}
    if manifest.get("entity_col") and manifest.get("entity_table"):
        manifest_fkeys[manifest["entity_col"]] = manifest["entity_table"]
    if manifest.get("src_entity_col") and manifest.get("src_entity_table"):
        manifest_fkeys[manifest["src_entity_col"]] = manifest["src_entity_table"]
    if manifest.get("dst_entity_col") and manifest.get("dst_entity_table"):
        manifest_fkeys[manifest["dst_entity_col"]] = manifest["dst_entity_table"]

    # Reproduce rustler/pre.rs node numbering: database parquet files first,
    # then every task parquet, lexicographically within each group. These ids
    # seed the reference sampler, so a convenient local numbering is not
    # equivalent.
    dataset_dir = Path(dataset.dataset_dir)
    offset = 0
    for path in sorted((dataset_dir / "db").glob("*.parquet")):
        offset += pq.ParquetFile(path).metadata.num_rows
    task_offsets = {}
    for path in sorted((dataset_dir / "tasks").glob("*/*.parquet")):
        task_offsets[path.resolve()] = offset
        offset += pq.ParquetFile(path).metadata.num_rows

    task_rows: list[Row] = []
    task_node_ids: dict[tuple[str, object], int] = {}
    focal_lookup: dict[tuple[object, object], tuple[str, object]] = {}
    train_frame = None
    for one_task_dir in sorted((dataset_dir / "tasks").iterdir()):
        if not one_task_dir.is_dir():
            continue
        one_name = one_task_dir.name
        one_manifest = yaml.safe_load(
            (one_task_dir / "manifest.yaml").read_text())
        one_fkeys = {}
        for column_key, table_key in (
                ("entity_col", "entity_table"),
                ("src_entity_col", "src_entity_table"),
                ("dst_entity_col", "dst_entity_table")):
            if one_manifest.get(column_key) and one_manifest.get(table_key):
                one_fkeys[one_manifest[column_key]] = one_manifest[table_key]
        for split in ("test", "train", "val"):
            path = (one_task_dir / f"{split}.parquet").resolve()
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if one_name == task.name and split == "train":
                train_frame = frame
            base = task_offsets[path]
            for row_i, record in enumerate(frame.to_dict("records")):
                row_id = (split, row_i)
                parents = {}
                for column, parent_table in one_fkeys.items():
                    if column not in record:
                        continue
                    raw_parent = record[column]
                    if (not isinstance(raw_parent, (list, tuple, np.ndarray))
                            and pd.isna(raw_parent)):
                        continue
                    if isinstance(raw_parent, (list, tuple, np.ndarray)):
                        parent_value = tuple(_python_value(value)
                                             for value in raw_parent)
                    else:
                        parent_value = _python_value(raw_parent)
                    # The selected task's entity edge has a dedicated runtime
                    # name; auxiliary task edges retain their parent table so
                    # graph walks can traverse them without schema fallbacks.
                    if (one_name == task.name
                            and column == one_manifest.get("entity_col")):
                        parents["__entity__"] = parent_value
                    else:
                        parents[f"__parent__:{parent_table}"] = parent_value
                cells = {
                    column: _python_value(value)
                    for column, value in record.items()
                    if column not in one_fkeys and not pd.isna(value)
                }
                timestamp = None
                time_column = one_manifest.get("time_col")
                if time_column and not pd.isna(record.get(time_column)):
                    timestamp = _python_value(record[time_column])
                row = Row(one_name, row_id, cells, timestamp, parents)
                task_rows.append(row)
                task_node_ids[row.key] = base + row_i
                if one_name == task.name and split == "test":
                    focal_lookup[(_python_value(record[task.id_column]),
                                  _python_value(record["date"]))] = row.key

    # Same-table fallback samples offsets from the task table's single global
    # node-id range. Its row list must therefore be in that exact order. Edge
    # order is supplied independently by the strict graph-order fixture below.
    task_rows.sort(key=lambda row: task_node_ids[row.key])

    # Runtime random walks consume the exact p2f array order stored by the
    # released reference artifact. Equal-time edges cannot be reconstructed
    # from the current preprocessor source because its table HashMap order is
    # process-randomized. This compact fixture contains only node-id adjacency
    # order (no features, labels, contexts, or model tensors).
    order_path = (Path(__file__).parent / "fixtures"
                  / "rel-f1-reference-graph-order.npz")
    if not order_path.exists():
        raise RuntimeError(
            f"required reference graph-order fixture is missing: {order_path}")
    with np.load(order_path) as order_data:
        p2f_offsets = np.asarray(order_data["p2f_offsets"], dtype=np.int64)
        p2f_children = np.asarray(order_data["p2f_children"], dtype=np.int32)
        f2p_offsets = np.asarray(order_data["f2p_offsets"], dtype=np.int64)
        f2p_parents = np.asarray(order_data["f2p_parents"], dtype=np.int32)
    if (p2f_offsets.ndim != 1 or p2f_offsets.size != offset + 1
            or p2f_offsets[0] != 0
            or p2f_offsets[-1] != p2f_children.size
            or np.any(p2f_offsets[1:] < p2f_offsets[:-1])
            or f2p_offsets.ndim != 1 or f2p_offsets.size != offset + 1
            or f2p_offsets[0] != 0
            or f2p_offsets[-1] != f2p_parents.size
            or np.any(f2p_offsets[1:] < f2p_offsets[:-1])):
        raise RuntimeError("reference graph-order fixture has an invalid shape")
    node_keys: list[tuple[str, object] | None] = [None] * offset
    physical_index = 0
    for table_name in sorted(physical_rows):
        for row in physical_rows[table_name]:
            node_keys[physical_index] = row.key
            physical_index += 1
    for key, node_id in task_node_ids.items():
        node_keys[node_id] = key
    if any(key is None for key in node_keys):
        raise RuntimeError("reference graph-order fixture has unmapped node ids")
    reference_p2f_order = {}
    for parent_id in range(physical_index):
        left = int(p2f_offsets[parent_id])
        right = int(p2f_offsets[parent_id + 1])
        if left != right:
            reference_p2f_order[node_keys[parent_id]] = tuple(
                node_keys[int(child_id)]
                for child_id in p2f_children[left:right])
    reference_f2p_order = {}
    for child_id in range(offset):
        left = int(f2p_offsets[child_id])
        right = int(f2p_offsets[child_id + 1])
        if left != right:
            reference_f2p_order[node_keys[child_id]] = tuple(
                node_keys[int(parent_id)]
                for parent_id in f2p_parents[left:right])

    if train_frame is None:
        raise RuntimeError(f"{task.name} has no materialized train split")
    for column in train_frame.columns:
        if column in manifest_fkeys or column == task.target_column:
            continue
        series = train_frame[column]
        if pd.api.types.is_numeric_dtype(series.dtype):
            column_stats = column_stats.with_column_values(
                task.name, column, series.dropna().to_numpy())

    # rustler uses one population mean/std over every datetime cell in every
    # database and task parquet, including task tables not selected here.
    datetime_days: list[float] = []
    all_parquets = (list((dataset_dir / "db").glob("*.parquet"))
                    + list((dataset_dir / "tasks").glob("*/*.parquet")))
    for path in all_parquets:
        parquet = pq.ParquetFile(path)
        datetime_columns = [field.name for field in parquet.schema_arrow
                            if pd.api.types.is_datetime64_any_dtype(
                                field.type.to_pandas_dtype())]
        for column in datetime_columns:
            series = pd.read_parquet(path, columns=[column])[column].dropna()
            datetime_days.extend(
                series.astype("int64").to_numpy(dtype=np.float64)
                / (86_400.0 * 1_000_000_000.0))
    column_stats = column_stats.with_datetime_values(datetime_days)

    backend = RtNativeBackend(
        schema=schema, wiring=wiring, lib_path=library,
        max_seq_len=context_size, batch_size=batch_size,
        device=RT_DEVICE_MPS,
        task_spec_factory=task_spec_factory,
        column_stats=column_stats,
        normalization_mode=NormalizationMode.REFERENCE)

    # Use the exact preprocessing-time MiniLM table consumed by rustler. A
    # newly encoded vector can differ across accelerator/library versions even
    # for the same phrase, and that is a different model input.
    from huggingface_hub import snapshot_download
    pre_root = Path(snapshot_download(
        repo_id="stanford-star/relbench-preprocessed", repo_type="dataset",
        allow_patterns=["rel-f1/text.json",
                        "rel-f1/text_emb_all-MiniLM-L12-v2.bin"]))
    pre_dataset = pre_root / "rel-f1"
    text_vocab = json.loads((pre_dataset / "text.json").read_text())
    text_values = np.fromfile(
        pre_dataset / "text_emb_all-MiniLM-L12-v2.bin",
        dtype=ml_dtypes.bfloat16).astype(np.float32)
    if text_values.size != len(text_vocab) * 384:
        raise RuntimeError("reference embedding table has an invalid shape")
    text_values = text_values.reshape(len(text_vocab), 384)
    backend.embedder.install_precomputed(
        dict(zip(text_vocab, text_values)), strict=True)

    def task_graph_factory(spec, entity_id, anchor):
        key = focal_lookup.get((_python_value(entity_id),
                                _python_value(anchor)))
        if key is None:
            raise RuntimeError(
                f"no materialized test task row for {task.name} "
                f"entity={entity_id!r} anchor={anchor!r}")
        materialized = []
        for row in task_rows:
            if row.key != key:
                materialized.append(row)
                continue
            cells = dict(row.cells)
            cells.pop(task.target_column, None)
            materialized.append(Row(row.table, row.id, cells,
                                    row.timestamp, row.parents))
        return (materialized, task_node_ids, key, reference_p2f_order,
                reference_f2p_order)

    return Engine(
        schema, wiring,
        model_config=ModelConfig(normalization_mode=NormalizationMode.REFERENCE),
        model_backend=backend,
        context_policy=ContextPolicy(
            max_context_cells=context_size,
            local_context_cells=64,
            bfs_width=32,
            num_walks=10_000,
            walk_length=20,
            seed=0,
        ),
        traversal=ReferenceTraversal(
            task_spec_factory=task_spec_factory,
            task_graph_factory=task_graph_factory),
    )


def execute_group(engine: Engine, task: RelqlTask, ids, anchor: datetime):
    return engine.execute(ExecutionInput(
        query=task.query,
        anchor_time=_python_value(anchor),
        per_entity_anchor=task.per_entity_anchor,
        params={"ids": [_python_value(value) for value in ids]},
    ))
