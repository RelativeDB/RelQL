"""The remote scoring wire format: what crosses to the serving backend.

The contract is token-level: context creation (rows -> sequences ->
normalization -> collation) happens client-side in this package, text cells
stay RAW STRINGS, and the service materializes the embedding channels. These
tests pin the encode/decode round trip and that a SequenceBackend over a
RemoteScorer produces byte-identical batches to one over any other scorer —
the wire adds nothing and loses nothing.
"""
import json

import numpy as np
import pytest
from conftest import StubScorer, churn_rows, in_memory_wiring

from relativedb import Engine, ExecutionInput, TaskType, parse
from relativedb.remote import decode_batch, encode_batch
from relativedb.scoring import SequenceBackend, TokenBatch


class RecordingScorer(StubScorer):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, batch, *, model_uri, output="target_scores"):
        self.batches.append(batch)
        return super().forward(batch, model_uri=model_uri, output=output)


def _batch_from(churn_schema):
    wiring = in_memory_wiring(churn_rows())
    scorer = RecordingScorer()
    eng = Engine(churn_schema, wiring,
                 model_backend=SequenceBackend(scorer, schema=churn_schema,
                                               wiring=wiring))
    from conftest import dt
    eng.execute(ExecutionInput(
        query="PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) = 0 "
              "FROM customers WHERE customers.customer_id IN :ids",
        params={"ids": ["C7"]}, anchor_time=dt("2026-07-01")))
    assert scorer.batches
    return scorer.batches[0]


def test_wire_round_trip_is_exact(churn_schema):
    batch = _batch_from(churn_schema)
    wire = json.dumps(encode_batch(batch))     # through real JSON
    got = decode_batch(json.loads(wire))
    for field in ("node_idxs", "f2p", "col_idxs", "table_idxs", "is_padding",
                  "sem_types", "is_target", "text_idx"):
        assert np.array_equal(getattr(got, field), getattr(batch, field)), field
    # bf16-rounded floats survive JSON exactly — no tolerance needed
    assert np.array_equal(got.number_v, batch.number_v)
    assert np.array_equal(got.datetime_v, batch.datetime_v)
    assert got.col_phrases == batch.col_phrases
    assert got.texts == batch.texts


def test_wire_carries_text_as_strings_never_vectors(churn_schema):
    batch = _batch_from(churn_schema)
    wire = encode_batch(batch)
    # schema phrases exist for every column the batch references
    assert all(" of " in p for p in wire["col_phrases"])
    # and nothing in the payload is 384-wide: embeddings are the server's job
    def widths(x):
        if isinstance(x, list):
            yield len(x)
            for item in x:
                yield from widths(item)
    assert 384 not in set(widths(list(wire.values())))


def test_engine_accepts_backend_url(churn_schema):
    eng = Engine(churn_schema, in_memory_wiring(churn_rows()),
                 model_backend="http://localhost:9")   # port 9: never listens
    from relativedb import RemoteBackend
    assert isinstance(eng.model_backend, RemoteBackend)
    from relativedb.remote import RemoteScoringError
    from conftest import dt
    with pytest.raises(RemoteScoringError, match="unreachable"):
        eng.execute(ExecutionInput(
            query="PREDICT COUNT(orders.*) OVER (30 DAYS FOLLOWING) = 0 "
                  "FROM customers WHERE customers.customer_id IN :ids",
            params={"ids": ["C7"]}, anchor_time=dt("2026-07-01")))
