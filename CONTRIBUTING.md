# Contributing

The repository ships one Python package:

- `python/` — the `relativedb` package: RelQL planning, retriever wiring,
  context assembly, and model routing.
- `python/src/relativedb/rt/` — local model execution over the shared
  [relational-transformers](https://relationaltransformers.com) runtime
  (torch on CPU/MPS/CUDA, Triton CUDA serving, ONNX) and MiniLM text
  encoding in torch. Imported lazily, so the query-planning side of the
  package never pays the torch import. Training lives in the
  relational-transformers package; relativedb only serves fitted heads.

`evaluation/` and `website/` are not part of the published packages.

Supported platforms are Linux x86_64, Linux aarch64, and macOS arm64.
Windows is out of scope. Python 3.10 or newer.

## Get a working tree

```bash
git clone https://github.com/RelativeDB/RelQL.git
cd RelQL
python3 -m venv python/.venv
python/.venv/bin/pip install -e "./python[dev]"
```

## Tests

The suite is split in two tiers.

```bash
# unit tier — offline, no checkpoint. Runs on a bare machine.
python/.venv/bin/python -m pytest python/tests -m "not integration" -q

# integration tier — the real RT-J checkpoint and MiniLM snapshot.
RELATIVEDB_REQUIRE_NATIVE=1 \
  python/.venv/bin/python -m pytest python/tests -m integration -q
```

The first integration run downloads ~570 MB of checkpoints from Hugging Face
(RT-J plus the pinned `all-MiniLM-L12-v2` encoder) into `~/.cache/huggingface`.
Later runs are cache-first and offline.

`RELATIVEDB_REQUIRE_NATIVE=1` turns every "skipped: no checkpoint" into a hard
failure. Use it whenever you believe the prerequisites are present — without
it a failed download reports a green run that verified nothing. CI sets it on
the integration lane for exactly that reason.

Marker discipline: `--strict-markers` is on, and a marker expression that
selects zero tests is a `UsageError`, not a pass. The `checkpoint` sub-marker
refines the integration tier for tests that need a downloaded model. When a
test needs a prerequisite, route the decision through `require_checkpoint` /
`require_text_embedder` in `python/tests/conftest.py` — never call
`pytest.skip` for these conditions directly, or strict mode stops working.

### The sampling regression harness

Context assembly is a seeded walk, so a refactor can change which rows land in
a context — and therefore every prediction — while every behavioural test still
passes. `python/tests/test_sampling_regression.py` freezes the *ordered* row
sequence each entity is given, plus the order `execute()` hands contexts to the
backend, across ten engine configurations. It needs no model and runs in the
unit tier.

`python/tests/data/sampling_fingerprints.json` is a tripwire, not a
convenience. If it fails, sampling changed: the failure names the case, the
entity and the index of the first divergent row. Work out why before touching
the fixture — a diff there is a real behavioural change, and the last time this
class of divergence went unnoticed it cost a whole module. Regenerate only once
the change is understood and intended:

```sh
RELATIVEDB_UPDATE_SAMPLING_GOLDEN=1 pytest python/tests/test_sampling_regression.py
```

Two of the twelve fingerprints are deliberately identical, and a test asserts
exactly which: `bfs-retriever` must equal `bfs-csc` (the sampler mode is a way
of reaching the graph, not a change to it) and `reference-pipelined` must equal
`reference-default` (pipelined assembly is an optimization and must be
invisible in the output). Any *other* pair of identical fingerprints fails the
suite, because a case that duplicates another adds no detection power.

`reference-direct-target-products` looks redundant and is not. A direct target
takes the traversal's non-shared path, where the walk seeds from the *order*
rows are enumerated in rather than their count — and `products` is the only
entity table here whose node indices actually move when that order changes,
since `customers` is both declared first and sorted first. Without it, renaming
or reordering schema tables silently changes sampling and nothing goes red.

## Coverage

The exact invocation CI uses:

```bash
RELATIVEDB_REQUIRE_NATIVE=1 \
  python/.venv/bin/python -m pytest python/tests \
    --cov=relativedb --cov-report=xml:coverage-python.xml --cov-report=term
```

Run both tiers together. Measuring only one reports a number that is wrong in
an obvious direction.

Coverage gates are informational: they annotate a pull request with the delta
and never block a merge.

## Building distributions

```bash
python -m build python
```

The package is pure Python and produces a `py3-none-any` wheel.

See `RELEASING.md` for the full release procedure.

## Pull requests

- CI must be green. The single required check is named `CI`; it aggregates the
  unit matrix and the integration tier.
- New behaviour needs a test in the tier that can actually reach it. Anything
  needing a checkpoint is `@pytest.mark.integration`.
- Add a `CHANGELOG.md` entry under `Unreleased` for user-visible changes.

## Style

Comments explain *why*, not *what*. The engine has a lot of decisions that
look arbitrary until you know what they are compensating for — a pinned
word-piece window, a `.bfloat16()` rounding matched to the upstream reference.
When you make one of those, say what would go wrong otherwise. The Python
sources are written that way; match them.
