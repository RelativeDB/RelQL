"""Sampling regression harness: pin the sampled context, not just the answer.

Context assembly is a seeded walk, and the seeds derive from the context policy
and from node indices. A refactor that reorders a loop, renumbers nodes, or
changes which strategy assembles a context can therefore change every
prediction while every behavioural test still passes -- that is precisely how
relativedb.columnar diverged from the reference without anything going red.

The fingerprint is the ORDERED sequence of context rows per entity, plus the
order execute() delivers those contexts to the backend. Order is what the model
consumes and what the RNG determines, so it is the property worth freezing.

The committed fingerprints are a deliberate tripwire, not a convenience. When
one changes, that is a real behavioural change in sampling: work out why before
regenerating. Regenerate only when the change is understood and intended:

    RELATIVEDB_UPDATE_SAMPLING_GOLDEN=1 pytest python/tests/test_sampling_regression.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import sampling_harness as harness

GOLDEN = Path(__file__).parent / "data" / "sampling_fingerprints.json"
UPDATE_ENV = "RELATIVEDB_UPDATE_SAMPLING_GOLDEN"


@pytest.fixture(scope="module")
def computed() -> dict:
    return harness.compute_all()


def _load_golden() -> dict:
    if not GOLDEN.is_file():
        pytest.fail(
            f"{GOLDEN} is missing. Generate it with "
            f"{UPDATE_ENV}=1 pytest {Path(__file__).name}")
    return json.loads(GOLDEN.read_text())


def _describe(case: str, want: dict, got: dict) -> str:
    """A readable account of what moved -- a bare hash mismatch tells a future
    reader nothing about whether the change was benign."""
    lines = [f"sampling changed for case {case!r}:"]
    for eid in sorted(set(want) | set(got)):
        w, g = want.get(eid), got.get(eid)
        if w == g:
            continue
        if w is None or g is None:
            lines.append(f"  {eid}: {'added' if w is None else 'removed'}")
            continue
        if w["rows"] != g["rows"]:
            lines.append(f"  {eid}: {len(w['rows'])} rows -> {len(g['rows'])}")
            for i, (a, b) in enumerate(zip(w["rows"], g["rows"])):
                if a != b:
                    lines.append(f"    first divergence at index {i}: "
                                 f"{a} -> {b}")
                    break
        for key in ("cells", "truncated_children", "hit_cell_budget"):
            if w[key] != g[key]:
                lines.append(f"    {key}: {w[key]!r} -> {g[key]!r}")
        if w["focal"] != g["focal"]:
            # A set diff, not two dumped lists: focal sets run to dozens of
            # keys and printing both buries the one line that matters.
            gone, new = set(w["focal"]) - set(g["focal"]), \
                set(g["focal"]) - set(w["focal"])
            lines.append(f"    focal: -{len(gone)} +{len(new)}")
            for k in sorted(gone)[:3]:
                lines.append(f"      - {k}")
            for k in sorted(new)[:3]:
                lines.append(f"      + {k}")
    return "\n".join(lines)


def test_regenerate_golden_when_asked(computed):
    """Writes the fixture. Skipped unless the env var is set, so a normal run
    can never quietly overwrite the tripwire it is supposed to trip."""
    if not os.environ.get(UPDATE_ENV):
        pytest.skip(f"set {UPDATE_ENV}=1 to regenerate")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(computed, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {GOLDEN} ({len(computed)} cases, "
          f"digest {harness.digest(computed)[:16]})")


def test_sampling_is_deterministic_within_a_process():
    """If this fails, every other assertion here is meaningless. Dict or set
    iteration leaking into the sampler would show up here first."""
    a = harness.compute_all()
    b = harness.compute_all()
    assert harness.digest(a) == harness.digest(b)


def test_every_case_is_covered_by_the_fixture(computed):
    """A new case with no committed fingerprint is unprotected; a stale entry
    means a case was renamed or dropped without regenerating."""
    if os.environ.get(UPDATE_ENV):
        pytest.skip("regenerating")
    golden = _load_golden()
    assert sorted(golden) == sorted(computed)


@pytest.mark.parametrize("case", sorted(harness.cases()))
def test_context_sampling_matches_the_fixture(case, computed):
    if os.environ.get(UPDATE_ENV):
        pytest.skip("regenerating")
    golden = _load_golden()
    want = golden[case]["context"]
    got = computed[case]["context"]
    assert want == got, _describe(case, want, got)


@pytest.mark.parametrize("case", sorted(harness.cases()))
def test_execution_order_matches_the_fixture(case, computed):
    """Contexts must reach the backend in the same order, and predictions must
    come back in that order -- a strategy refactor is exactly what would
    perturb this."""
    if os.environ.get(UPDATE_ENV):
        pytest.skip("regenerating")
    golden = _load_golden()
    assert golden[case]["execution"] == computed[case]["execution"], (
        f"execution order/values changed for {case!r}")


# --------------------------------------------------------------------------
# invariants that must hold whatever the fixture says
# --------------------------------------------------------------------------

def test_sampler_mode_does_not_change_sampling(computed):
    """RETRIEVER and CSC are two ways to reach the same graph. If they ever
    disagree, one of them is wrong -- and the fixture alone would not say so,
    since it would happily record both."""
    assert computed["bfs-retriever"] == computed["bfs-csc"]


def test_pipelining_does_not_change_sampling(computed):
    """The pipelined path overlaps assembly with the forward pass on a
    producer thread. It is an optimization, so it must be invisible in the
    output -- same contexts, same order, same values."""
    assert computed["reference-pipelined"] == computed["reference-default"]


def test_predictions_follow_the_scored_order(computed):
    for case, fp in computed.items():
        ex = fp["execution"]
        assert ex["scored_order"] == ex["prediction_order"], case


def test_cases_are_not_accidentally_identical(computed):
    """A fingerprint that duplicates another adds no detection power. The two
    intended coincidences are the invariants above; anything else means a case
    is not exercising what its name claims."""
    expected_duplicates = {
        frozenset({"bfs-retriever", "bfs-csc"}),
        frozenset({"reference-pipelined", "reference-default"}),
    }
    by_digest: dict[str, list[str]] = {}
    for name, fp in computed.items():
        by_digest.setdefault(harness.digest(fp), []).append(name)
    dupes = {frozenset(v) for v in by_digest.values() if len(v) > 1}
    assert dupes == expected_duplicates
