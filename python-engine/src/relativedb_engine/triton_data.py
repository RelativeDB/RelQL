"""RT-J input sorting and relational work-list construction for Triton."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


FIELDS = (
    "node_idxs",
    "f2p_nbr_idxs",
    "col_name_idxs",
    "table_name_idxs",
    "is_padding",
    "sem_types",
    "is_targets",
    "number_values",
    "datetime_values",
    "boolean_values",
    "text_values",
    "col_name_values",
)


@dataclass
class WorkList:
    qidx: np.ndarray
    kidx: np.ndarray
    qstart: np.ndarray
    kstart: np.ndarray
    nq: np.ndarray
    nk: np.ndarray
    logkv: np.ndarray


def load_npz(path):
    with np.load(path) as values:
        return {name: values[name].copy() for name in FIELDS}, values["labels"].copy()


def sort_batch(batch):
    batch = dict(batch)
    bsz, seq = batch["node_idxs"].shape
    orders = np.empty((bsz, seq), np.int64)
    for b in range(bsz):
        keys = np.where(
            batch["is_padding"][b].astype(bool),
            np.iinfo(np.int64).max,
            batch["col_name_idxs"][b],
        )
        orders[b] = np.lexsort((np.arange(seq), keys))
    for name in FIELDS:
        if name in ("text_values", "col_name_values"):
            continue
        values = batch[name]
        batch[name] = np.stack([values[b, orders[b]] for b in range(bsz)])
    return batch, orders


def _bf16_round(value):
    bits = np.asarray([value], np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return float(bits.view(np.float32)[0])


def build_worklists(batch, query_tile=16, column_cap=512):
    node = batch["node_idxs"]
    f2p = batch["f2p_nbr_idxs"]
    col = batch["col_name_idxs"]
    table = batch["table_name_idxs"]
    padding = batch["is_padding"].astype(bool)
    bsz, seq = node.shape
    grouped = [[], [], []]

    for b in range(bsz):
        by_coltab = defaultdict(list)
        by_node = defaultdict(list)
        nbr_of = defaultdict(list)
        valid = np.flatnonzero(~padding[b])
        for s in valid:
            si = int(s)
            by_coltab[(int(col[b, si]), int(table[b, si]))].append(si)
            by_node[int(node[b, si])].append(si)
            for parent in f2p[b, si].tolist():
                if parent >= 0:
                    nbr_of[parent].append(si)

        for members in by_coltab.values():
            keys = members[:column_cap] if column_cap else members
            grouped[0].append((b, members, keys))

        by_feature = defaultdict(list)
        for s in valid:
            si = int(s)
            key = (int(node[b, si]), *f2p[b, si].tolist())
            by_feature[key].append(si)
        for key, members in by_feature.items():
            seen = set()
            keys = []
            for parent in key:
                if parent < 0 or parent in seen:
                    continue
                seen.add(parent)
                keys.extend(by_node.get(parent, ()))
            grouped[1].append((b, members, keys))

        for nid, members in by_node.items():
            if nid in nbr_of:
                grouped[2].append((b, members, nbr_of[nid]))

    result = []
    for groups in grouped:
        qflat = []
        kflat = []
        qstart = []
        kstart = []
        nq = []
        nk = []
        logkv = []
        for b, queries, keys in groups:
            key_start = len(kflat)
            kflat.extend(b * seq + k for k in keys)
            rounded_log = math.log(max(_bf16_round(float(len(keys))), 1.0))
            for q0 in range(0, len(queries), query_tile):
                tile = queries[q0:q0 + query_tile]
                qstart.append(len(qflat))
                qflat.extend(b * seq + q for q in tile)
                kstart.append(key_start)
                nq.append(len(tile))
                nk.append(len(keys))
                logkv.append(rounded_log)
        result.append(WorkList(
            qidx=np.asarray(qflat, np.int32),
            kidx=np.asarray(kflat, np.int32),
            qstart=np.asarray(qstart, np.int32),
            kstart=np.asarray(kstart, np.int32),
            nq=np.asarray(nq, np.int32),
            nk=np.asarray(nk, np.int32),
            logkv=np.asarray(logkv, np.float32),
        ))
    return result
