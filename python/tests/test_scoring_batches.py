import pytest

from relativedb.scoring import SequenceBackend, _physical_batches


@pytest.mark.parametrize(("cells", "expected"), [
    (1024, [128, 128]),
    (2048, [64, 64, 64, 64]),
    (4096, [32, 32, 32, 32, 32, 32, 32, 32]),
])
def test_physical_batches_adapt_to_context_cells(cells, expected):
    sequences = [range(cells) for _ in range(256)]

    batches = _physical_batches(
        sequences, max_items=128, max_cells=128 * 1024)

    assert [len(batch) for batch in batches] == expected


def test_physical_batches_account_for_padding_to_longest_context():
    sequences = [range(1024) for _ in range(64)]
    sequences += [range(2048) for _ in range(64)]

    batches = _physical_batches(
        sequences, max_items=128, max_cells=128 * 1024)

    assert [len(batch) for batch in batches] == [64, 64]


def test_max_batch_cells_must_be_positive():
    with pytest.raises(ValueError, match="max_batch_cells"):
        SequenceBackend(object(), max_batch_cells=0)
