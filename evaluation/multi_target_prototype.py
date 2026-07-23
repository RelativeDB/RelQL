"""Shared-context multi-target scoring prototype (driver-dnf).

One race-anchored context scores every driver whose test task row appears in
it: peers' label cells are flipped to masked targets and each target's score
is read from the number head at its own token via rt_forward_tokens_device.
Falls back to additional seed contexts until the cohort is covered.
"""
import ctypes
import math
import sys
import time
import warnings

import numpy as np
from numpy.ctypeslib import ndpointer

sys.path.insert(0, "python/src")
sys.path.insert(0, ".")

from relbench import load_dataset, load_task
from evaluation.f1_relql import DATASET, TASKS, build_engine, _python_value
from relativedb import parse, validate
from relativedb.rt_native import RT_DEVICE_MPS, load_lib

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
LIB = "cpp/build/librt_c.dylib"

dataset = load_dataset(DATASET)
spec = TASKS["driver-dnf"]
task = load_task(DATASET, "driver-dnf")
target_df = task.get_table("test", mask_input_cols=False).df.copy()
engine = build_engine(dataset, spec, context_size=CTX, batch_size=4,
                      library=LIB)
backend = engine.model_backend
schema = engine.schema
pq_template = validate(parse(spec.query), schema).query
task_type = pq_template.task_type(schema)
mode = backend._mode()
model = backend._model_for(engine.model_config.model_uri_for(task_type))

lib = load_lib(LIB)._lib
i64p = ndpointer(np.int64, flags="C_CONTIGUOUS")
u8p = ndpointer(np.uint8, flags="C_CONTIGUOUS")
f32p = ndpointer(np.float32, flags="C_CONTIGUOUS")
lib.rt_forward_tokens_device.restype = ctypes.c_int
lib.rt_forward_tokens_device.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
    i64p, i64p, i64p, i64p, u8p, i64p, u8p,
    f32p, f32p, f32p, f32p, f32p,
    ctypes.c_int32, ctypes.c_int32, f32p,
    ctypes.c_char_p, ctypes.c_size_t]


def forward_tokens(seq):
    kw = backend._collate([seq])
    B, S = kw["node_idxs"].shape
    out = np.zeros(B * S, np.float32)
    err = ctypes.create_string_buffer(512)
    rc = lib.rt_forward_tokens_device(
        model._handle, B, S,
        np.ascontiguousarray(kw["node_idxs"], np.int64).reshape(-1),
        np.ascontiguousarray(kw["f2p"], np.int64).reshape(-1),
        np.ascontiguousarray(kw["col_idxs"], np.int64).reshape(-1),
        np.ascontiguousarray(kw["table_idxs"], np.int64).reshape(-1),
        np.ascontiguousarray(kw["is_padding"], np.uint8).reshape(-1),
        np.ascontiguousarray(kw["sem_types"], np.int64).reshape(-1),
        np.ascontiguousarray(kw["is_target"], np.uint8).reshape(-1),
        np.ascontiguousarray(kw["number_v"], np.float32).reshape(-1),
        np.ascontiguousarray(kw["datetime_v"], np.float32).reshape(-1),
        np.ascontiguousarray(kw["boolean_v"], np.float32).reshape(-1),
        np.ascontiguousarray(kw["text_v"], np.float32).reshape(-1),
        np.ascontiguousarray(kw["col_name_v"], np.float32).reshape(-1),
        backend.n_threads, RT_DEVICE_MPS, out, err, len(err))
    if rc != 0:
        raise RuntimeError(f"rt_forward_tokens_device failed: "
                           f"{err.value.decode('utf-8', 'replace')}")
    return out[:S]   # B == 1


keyed = {}
n_forwards = 0
n_seeds_total = 0
groups = list(target_df.groupby("date", sort=False))
started = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for gi, (date, frame) in enumerate(groups, 1):
        anchor = _python_value(date)
        ids = [_python_value(v) for v in frame[spec.id_column]]
        remaining = list(ids)
        seeds_used = 0
        while remaining:
            seed = remaining[0]
            pq = pq_template.bind_params({"ids": [seed]})
            ctx = engine.assemble_context(pq.entity_key.table, seed, anchor,
                                          query=pq)
            # Coverage-aware seeding: the cohort's test task rows (and their
            # entity rows, so the masked targets keep parent edges) are part
            # of the context by construction. They go right after the focal
            # target, so the sequence budget truncates low-relevance tail
            # rows instead of them.
            state = getattr(engine.traversal, "_shared_state", None)
            if state is not None:
                by_key = state["by_key"]
                have = {r.key for r in ctx.rows}
                inject = []
                task_rows = state["rows_by_table"][spec.name]
                cohort = set(remaining)
                # Each cohort member's own labeled task history (self-labels)
                # is the strongest per-entity signal; collect the most recent
                # few per member along with its test row and entity row.
                history_of: dict = {}
                for row in task_rows:
                    did = row.parents.get("__entity__")
                    if (did in cohort and row.timestamp is not None
                            and row.timestamp < ctx.anchor
                            and spec.target_column in row.cells):
                        history_of.setdefault(did, []).append(row)
                for row in task_rows:
                    if (isinstance(row.id, tuple) and row.id
                            and row.id[0] == "test"
                            and row.timestamp == ctx.anchor
                            and row.parents.get("__entity__") in cohort):
                        did = row.parents.get("__entity__")
                        ekey = (pq.entity_key.table, did)
                        if row.key not in have:
                            inject.append(by_key[row.key])
                            have.add(row.key)
                        if ekey not in have and ekey in by_key:
                            inject.append(by_key[ekey])
                            have.add(ekey)
                        hist = sorted(history_of.get(did, ()),
                                      key=lambda r: r.timestamp, reverse=True)
                        for h in hist[:3]:
                            if h.key not in have:
                                inject.append(by_key[h.key])
                                have.add(h.key)
                if inject:
                    ctx.rows = [ctx.rows[0]] + inject + ctx.rows[1:]
            fk_to_parent = backend._fk_to_parent()
            task_spec = backend.task_spec(pq, task_type)
            labels = []
            seq, node_of, _, tgt_idx = backend._build_ctx_seq(
                pq, task_type, ctx, fk_to_parent, labels, task_spec=task_spec)
            stats = backend._label_stats(seq, labels, task_spec, mode)
            backend._normalize_one(seq, task_spec, stats, mode)

            # Locate co-scorable peers: their test task rows present in this
            # context WITH a label cell.
            peer_row = {}
            for row in ctx.rows:
                if (row.table == task_spec.table_name
                        and isinstance(row.id, tuple) and row.id
                        and row.id[0] == "test"
                        and row.timestamp == ctx.anchor
                        and spec.target_column in row.cells):
                    did = row.parents.get("__entity__")
                    if did in remaining and did != seed:
                        peer_row[did] = row.key

            # Cell index per node for the target column.
            want_nodes = {node_of[k]: did for did, k in peer_row.items()
                          if k in node_of}
            cell_of = {}
            tcol = (task_spec.target_column, task_spec.table_name)
            for s in range(len(seq)):
                did = want_nodes.get(seq.node[s])
                if did is not None and seq.col[s] == tcol and not seq.is_tgt[s]:
                    cell_of[did] = s
            for did, s in cell_of.items():
                seq.is_tgt[s] = True
                seq.value[s] = 0.0

            yhat = forward_tokens(seq)
            n_forwards += 1
            seeds_used += 1
            keyed[(anchor, seed)] = 1.0 / (1.0 + math.exp(-float(yhat[tgt_idx])))
            for did, s in cell_of.items():
                keyed[(anchor, did)] = 1.0 / (1.0 + math.exp(-float(yhat[s])))
            covered = {seed} | set(cell_of)
            remaining = [d for d in remaining if d not in covered]
        n_seeds_total += seeds_used
        if gi == 1 or gi % 10 == 0 or gi == len(groups):
            print(f"group {gi}/{len(groups)}: {len(ids)} drivers, "
                  f"{seeds_used} forwards", flush=True)

elapsed = time.perf_counter() - started
preds = np.asarray([
    keyed[(_python_value(r.date), _python_value(getattr(r, spec.id_column)))]
    for r in target_df.itertuples(index=False)], dtype=np.float32)
scores = task.evaluate(preds)
print(f"\nMULTI-TARGET ctx={CTX}: {dict(scores)}")
print(f"forwards: {n_forwards} (vs {len(preds)} single-target), "
      f"avg seeds/group {n_seeds_total/len(groups):.2f}")
print(f"elapsed: {elapsed:.1f}s for {len(preds)} predictions")
