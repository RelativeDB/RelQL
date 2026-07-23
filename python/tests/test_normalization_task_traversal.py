from __future__ import annotations

import numpy as np
import pytest

from relativedb import (ContextPolicy, Engine, NormalizationMode,
                        ReferenceTraversal, TaskSpec, ValueType)
from relativedb.relql.ast import TaskType
from relativedb.relql.parser import parse, validate
from relativedb.rt_native import ColumnStats, RtNativeBackend
from relativedb.traversal import TraversalResult, _StdRng, _rand_sample

from conftest import dt, in_memory_wiring, churn_rows


QUERY = ("PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 "
         "FROM customers")


def _query(schema, text=QUERY):
    return validate(parse(text), schema).query


def test_zero_shot_normalization_is_batch_invariant(churn_schema):
    wiring = in_memory_wiring(churn_rows())
    backend = RtNativeBackend(schema=churn_schema, wiring=wiring)
    engine = Engine(churn_schema, wiring,
                    context_policy=ContextPolicy(cohort_size=0))
    pq = _query(churn_schema)
    one = engine.assemble_context("customers", "C1", dt("2026-07-01"))
    other = engine.assemble_context("customers", "C7", dt("2026-07-01"))
    seq_one = backend._build_sequences(pq, TaskType.BINARY_CLASSIFICATION,
                                       [one])[0][0]
    seq_batch = backend._build_sequences(pq, TaskType.BINARY_CLASSIFICATION,
                                         [one, other])[0][0]
    assert seq_one.col == seq_batch.col
    assert seq_one.value == pytest.approx(seq_batch.value)


def test_reference_normalization_uses_persisted_column_and_task_stats(
        churn_schema):
    wiring = in_memory_wiring(churn_rows())
    pq = _query(churn_schema)
    spec = TaskSpec.from_query(pq, TaskType.BINARY_CLASSIFICATION)
    stats = ColumnStats.fit(churn_schema, wiring).with_task_values(
        spec, [0.0, 1.0, 1.0])
    backend = RtNativeBackend(
        schema=churn_schema, wiring=wiring, column_stats=stats,
        normalization_mode=NormalizationMode.REFERENCE)
    engine = Engine(churn_schema, wiring,
                    context_policy=ContextPolicy(cohort_size=0))
    ctx = engine.assemble_context("customers", "C1", dt("2026-07-01"))
    seq = backend._build_sequences(
        pq, TaskType.BINARY_CLASSIFICATION, [ctx],
        normalization_mode=NormalizationMode.REFERENCE)[0][0]
    idx = next(i for i, (c, t) in enumerate(seq.col)
               if (c, t) == ("age", "customers"))
    mu, sd = stats.stats[("customers", "age")]
    assert seq.value[idx] == pytest.approx((34.0 - mu) / sd)


def test_task_identity_is_format_stable_and_target_sensitive(churn_schema):
    a = _query(churn_schema, QUERY)
    b = _query(churn_schema,
               "PREDICT   COUNT(orders.*) OVER (90 DAYS FOLLOWING)=0\n"
               "FROM customers")
    c = _query(churn_schema,
               "PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) = 0 "
               "FROM customers")
    sa = TaskSpec.from_query(a, TaskType.BINARY_CLASSIFICATION)
    sb = TaskSpec.from_query(b, TaskType.BINARY_CLASSIFICATION)
    sc = TaskSpec.from_query(c, TaskType.BINARY_CLASSIFICATION)
    assert sa == sb
    assert sa.id != sc.id
    assert sa.target_column != sc.target_column


class _RecordingTraversal:
    def __init__(self):
        self.query = None
    def traverse(self, schema, graph, entity_table, entity_id, bound, policy,
                 *, query=None):
        self.query = query
        rows = graph.entities(entity_table, [entity_id], bound)
        return TraversalResult(tuple(rows),
                               frozenset(r.key for r in rows))


def test_engine_uses_pluggable_traversal_and_passes_query(churn_schema):
    traversal = _RecordingTraversal()
    wiring = in_memory_wiring(churn_rows())
    engine = Engine(churn_schema, wiring, traversal=traversal)
    pq = _query(churn_schema)
    ctx = engine.assemble_context("customers", "C1", dt("2026-07-01"),
                                  query=pq)
    assert traversal.query is pq
    assert ctx.row_keys == {("customers", "C1")}


def test_reference_traversal_is_deterministic_and_temporally_safe(churn_schema):
    wiring = in_memory_wiring(churn_rows(), honor_bound=False)
    policy = ContextPolicy(max_context_cells=40, local_context_cells=8,
                           cohort_size=3, num_walks=16, walk_length=5, seed=7)
    a = Engine(churn_schema, wiring, context_policy=policy,
               traversal=ReferenceTraversal()).assemble_context(
                   "customers", "C7", dt("2026-07-01"))
    b = Engine(churn_schema, wiring, context_policy=policy,
               traversal=ReferenceTraversal()).assemble_context(
                   "customers", "C7", dt("2026-07-01"))
    assert [r.key for r in a.rows] == [r.key for r in b.rows]
    assert ("orders", "O4") not in a.row_keys
    assert ("customers", "C7") in a.focal_row_keys


@pytest.mark.parametrize("seed, expected", [
    (0, (3442241407, 3140108210, 14267822071968393595, 13)),
    (1, (3543144545, 4184349284, 12751046405260142922, 3)),
    (42, (572990626, 2261546851, 10011513049433592189, 14)),
])
def test_reference_rng_matches_rand_0_9_1_stdrng(seed, expected):
    """Values generated by a Rust rand=0.9.1 oracle on arm64."""
    rng = _StdRng(seed)
    assert (rng.u32(), rng.u32(), rng.u64(), rng.range(17)) == expected


def test_native_reference_walk_is_exact_not_a_fallback():
    from relativedb.rt_native import load_lib

    # 0 -> [1,2], 1 -> [0,2], 2 -> [2]. Count visits to non-target nodes.
    offsets = np.asarray([0, 2, 4, 5], np.int32)
    neighbors = np.asarray([1, 2, 0, 2, 2], np.int32)
    eligible = np.asarray([0, 1, 1], np.uint8)
    expected = np.zeros(3, np.uint32)
    rng = _StdRng(123456789)
    for _ in range(37):
        current = 0
        for _ in range(11):
            if eligible[current]:
                expected[current] += 1
            begin, end = offsets[current], offsets[current + 1]
            if begin == end:
                break
            current = neighbors[begin + rng.range(int(end - begin))]

    actual = np.zeros(3, np.uint32)
    lib = load_lib()._lib
    assert lib.rt_reference_walk_counts(
        3, offsets, neighbors, 0, eligible, 123456789, 37, 11, actual) == 0
    assert actual.tolist() == expected.tolist()

    seeds = np.asarray([0, 1, 42], np.uint64)
    first = np.empty(3, np.uint64)
    assert lib.rt_stdrng_first_u64_batch(seeds, len(seeds), first) == 0
    assert first.tolist() == [_StdRng(int(seed)).u64() for seed in seeds]


@pytest.mark.parametrize("length, amount, expected", [
    (20, 5, [2, 8, 4, 10, 17]),
    (200, 20, [26, 105, 51, 109, 174, 129, 198, 85, 194, 15,
               127, 89, 77, 150, 48, 172, 131, 41, 1, 19]),
])
def test_reference_index_sampling_matches_rand_oracle(length, amount, expected):
    assert _rand_sample(_StdRng(42), length, amount) == expected


def test_reference_defaults_match_evaluator_geometry():
    p = ContextPolicy()
    assert (p.max_context_cells, p.local_context_cells, p.bfs_width,
            p.num_walks, p.walk_length) == (8192, 256, 32, 10_000, 20)


def test_reference_graph_is_an_immutable_engine_snapshot(churn_schema):
    rows = churn_rows()
    wiring = in_memory_wiring(rows)
    engine = Engine(churn_schema, wiring,
                    context_policy=ContextPolicy(num_walks=0))
    before = engine.assemble_context("customers", "C7", dt("2026-07-01"))
    original_age = before.entity_cells("customers")["age"]
    rows["customers"][1].cells["age"] = 9999
    rows["customers"].clear()
    rows["orders"].clear()
    after = engine.assemble_context("customers", "C7", dt("2026-07-01"))
    assert [r.key for r in after.rows] == [r.key for r in before.rows]
    assert after.node_ids == before.node_ids
    assert after.entity_cells("customers")["age"] == original_age


def test_materialized_task_target_is_first_and_node_ids_are_global(churn_schema):
    wiring = in_memory_wiring(churn_rows())
    policy = ContextPolicy(num_walks=8, walk_length=4, num_history_windows=2)
    engine = Engine(churn_schema, wiring, context_policy=policy)
    backend = RtNativeBackend(schema=churn_schema, wiring=wiring)
    pq = _query(churn_schema)
    c1 = engine.assemble_context("customers", "C1", dt("2026-07-01"), query=pq)
    c7 = engine.assemble_context("customers", "C7", dt("2026-07-01"), query=pq)
    spec = TaskSpec.from_query(pq, TaskType.BINARY_CLASSIFICATION)
    assert any(r.table == spec.table_name and spec.target_column in r.cells
               for r in c1.rows)
    shared = c1.row_keys & c7.row_keys
    assert all(c1.node_ids[k] == c7.node_ids[k] for k in shared)
    seq = backend._build_sequences(
        pq, TaskType.BINARY_CLASSIFICATION, [c1])[0][0]
    assert seq.is_tgt[0]
    assert seq.col[0] == (spec.target_column, spec.table_name)


def test_fk_feature_is_opt_in_while_pk_never_emits(churn_schema):
    from relativedb import LinkDef, Schema
    links = tuple(LinkDef(l.from_table, l.fk_column, l.to_table,
                          ValueType.TEXT if l.fk_column == "customer_id" else None)
                  for l in churn_schema.links)
    opted = Schema(churn_schema.tables, links)
    rows = churn_rows()
    # A malicious/legacy retriever-provided PK cell must still be suppressed.
    rows["customers"][0].cells["customer_id"] = "C1"
    wiring = in_memory_wiring(rows)
    engine = Engine(opted, wiring, context_policy=ContextPolicy(num_walks=0))
    backend = RtNativeBackend(schema=opted, wiring=wiring)
    pq = _query(opted)
    ctx = engine.assemble_context("customers", "C1", dt("2026-07-01"), query=pq)
    seq = backend._build_sequences(pq, TaskType.BINARY_CLASSIFICATION, [ctx])[0][0]
    assert ("customer_id", "orders") in seq.col
    assert ("customer_id", "customers") not in seq.col
    assert opted.require_table("orders").column("customer_id") is None


def test_history_window_labels_are_scoped_to_the_owning_entity(churn_schema):
    """Self-label windows must aggregate only the entity's own child rows.

    Regression: eval_* aggregates over every row it is handed, so unscoped
    windows labeled NOT EXISTS(orders.*) as 0 for every customer whenever
    anyone in the sampled cohort had ordered — degenerate in-context
    examples. C9 has never ordered (every window must be 1.0); C7 ordered in
    Mar and May 2026 (windows covering those months must be 0.0).
    """
    from relativedb import parse, validate

    wiring = in_memory_wiring(churn_rows())
    engine = Engine(churn_schema, wiring,
                    context_policy=ContextPolicy(num_history_windows=3),
                    traversal=ReferenceTraversal())
    pq = validate(parse(
        "PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) "
        "FROM customers WHERE customers.customer_id IN :ids"),
        churn_schema).query.bind_params({"ids": ["C7"]})
    ctx = engine.assemble_context("customers", "C7", dt("2026-07-01"),
                                  query=pq)

    def labels(cid):
        out = {}
        for r in ctx.rows:
            if r.table == "task_customers" and r.id[0] == cid and r.cells:
                (value,) = r.cells.values()
                out[r.timestamp] = value
        return out

    c9 = labels("C9")
    assert c9 and all(v == 1.0 for v in c9.values()), c9
    c7 = labels("C7")
    # windows anchored 90/180 days before 2026-07-01 cover C7's May and
    # Mar orders respectively
    assert c7[dt("2026-04-02")] == 0.0
    assert c7[dt("2026-01-02")] == 0.0
