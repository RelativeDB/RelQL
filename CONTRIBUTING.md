# Contributing

The repository is two halves that ship as one package:

- `cpp/` — `librt_c`, a dependency-light C++20 implementation of the RT-J
  relational transformer, plus the RelQL parser and the CSC adjacency index.
  No torch, no Python at inference. See `cpp/README.md`.
- `python/` — the `relativedb` package: RelQL planning, retriever wiring,
  context assembly, and model routing. It loads `librt_c` through `ctypes`.

`evaluation/` and `website/` are not part of the published package.

Supported platforms are Linux x86_64, Linux aarch64, and macOS arm64.
Windows is out of scope. Python 3.10 or newer.

## Get a working tree

```bash
git clone https://github.com/RelativeDB/RelQL.git
cd RelQL
```

### Build the engine

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

On Apple silicon this builds the Metal/MPS backend by default (`RT_METAL=ON`);
`-DRT_METAL=OFF` falls back to the Accelerate CPU path. `-DRT_CUDA=ON` builds
the CUDA backend and needs the CUDA toolkit. The artifact is
`cpp/build/librt_c.dylib` on macOS, `cpp/build/librt_c.so` on Linux.

### Install the Python package

```bash
python3 -m venv python/.venv
python/.venv/bin/pip install -e "./python[dev]"
```

`rt_native` finds the library by checking, in order: a copy bundled inside the
installed package (release wheels only), `$RELATIVEDB_RT_LIB`, and then a
monorepo `cpp/build` tree relative to the source. An editable install in this
repository picks up `cpp/build` on its own; anywhere else, point at it
explicitly:

```bash
export RELATIVEDB_RT_LIB=$PWD/cpp/build/librt_c.dylib   # .so on Linux
```

## Tests

### C++

```bash
ctest --test-dir cpp/build --output-on-failure --no-tests=error
```

Three tests, a couple of seconds, no network and no checkpoint: the CSC
adjacency batteries, the RelQL parser corpus, and head fine-tuning (its Metal
sections self-skip when no MPS device is present). `--no-tests=error` matters —
plain `ctest` exits 0 when a build registers no tests at all.

The golden forward-pass test is registered but `DISABLED` by default: it needs
a real checkpoint and a dumped reference batch, neither of which is in the
repository. To run it, regenerate `cpp/testdata/*.bin` with
`cpp/tools/dump_golden.py` and configure with
`-DRT_TEST_CHECKPOINT=/path/to/model.safetensors`, then `ctest -L integration`.

### Python

The suite is split in two tiers.

```bash
# unit tier — offline, no librt_c, no checkpoint. Runs on a bare machine.
python/.venv/bin/python -m pytest python/tests -m "not integration" -q

# integration tier — the real engine and the real RT-J checkpoint.
RELATIVEDB_RT_LIB=$PWD/cpp/build/librt_c.dylib \
RELATIVEDB_REQUIRE_NATIVE=1 \
  python/.venv/bin/python -m pytest python/tests -m integration -q
```

The first integration run downloads ~570 MB of checkpoints from Hugging Face
(RT-J plus the pinned `all-MiniLM-L12-v2` encoder) into `~/.cache/huggingface`.
Later runs are cache-first and offline.

`RELATIVEDB_REQUIRE_NATIVE=1` turns every "skipped: no library / no
checkpoint" into a hard failure. Use it whenever you believe the prerequisites
are present — without it a broken build or a failed download reports a green
run that verified nothing. CI sets it on the integration lane for exactly that
reason.

Marker discipline: `--strict-markers` is on, and a marker expression that
selects zero tests is a `UsageError`, not a pass. Sub-markers `native` (needs
`librt_c` only) and `checkpoint` (also needs a downloaded model) refine the
integration tier. When a test needs a prerequisite, route the decision through
`require_native` / `require_native_csc` / `require_checkpoint` /
`require_text_embedder` in `python/tests/conftest.py` — never call
`pytest.skip` for these conditions directly, or strict mode stops working.

## Coverage

Python — the exact invocation CI uses:

```bash
RELATIVEDB_RT_LIB=$PWD/cpp/build/librt_c.dylib \
RELATIVEDB_REQUIRE_NATIVE=1 \
  python/.venv/bin/python -m pytest python/tests \
    --cov=relativedb --cov-report=xml:coverage-python.xml --cov-report=term
```

Run both tiers together. Measuring only one reports a number that is wrong in
an obvious direction.

C++ on Linux (GCC), which is what CI does and what `codecov.yml` reads:

```bash
cmake -S cpp -B cpp/build-coverage -DCMAKE_BUILD_TYPE=Debug -DRT_COVERAGE=ON
cmake --build cpp/build-coverage -j
ctest --test-dir cpp/build-coverage --output-on-failure --no-tests=error
gcovr --root cpp --filter cpp/src --exclude '.*test_.*' --exclude '.*bench.*' \
      --xml --output coverage-cpp.xml --print-summary
```

On macOS `RT_COVERAGE=ON` selects Clang's `-fprofile-instr-generate
-fcoverage-mapping` instead, which emits `.profraw` rather than the `.gcda`
files gcovr reads. Use the LLVM tools:

```bash
cmake -S cpp -B cpp/build-coverage -DCMAKE_BUILD_TYPE=Debug -DRT_COVERAGE=ON
cmake --build cpp/build-coverage -j
LLVM_PROFILE_FILE="$PWD/cpp/build-coverage/%p.profraw" \
  ctest --test-dir cpp/build-coverage --output-on-failure --no-tests=error
xcrun llvm-profdata merge -sparse cpp/build-coverage/*.profraw \
  -o cpp/build-coverage/all.profdata
xcrun llvm-cov report cpp/build-coverage/csc_test \
  -instr-profile=cpp/build-coverage/all.profdata
```

Coverage gates are informational: they annotate a pull request with the delta
and never block a merge.

## Building distributions

```bash
PYTHON=python/.venv/bin/python sh python/build_wheel.sh
```

Builds a pure sdist first, then compiles `librt_c` into
`cpp/build-wheel/`, copies it next to the package sources, and produces a
platform wheel tagged `py3-none-<platform>` — one wheel per platform, not per
interpreter, because the library is loaded with `ctypes`.

Do not use a bare `python -m build python` as a substitute. It builds the
sdist from your working tree as-is, so a `librt_c` left over from a
development build ends up inside the source distribution.
`build_wheel.sh` removes it first.

See `RELEASING.md` for the full release procedure.

## Pull requests

- CI must be green. The single required check is named `CI`; it aggregates the
  unit matrix, the integration tier, and C++ coverage.
- New behaviour needs a test in the tier that can actually reach it. Anything
  needing `librt_c` or a checkpoint is `@pytest.mark.integration`.
- Add a `CHANGELOG.md` entry under `Unreleased` for user-visible changes.
- Wheel builds do not run on pull requests by default — they are a
  three-platform, ~90-minute matrix. Add the `build-wheels` label to a PR that
  touches `cpp/`, `python/build_wheel.sh`, `python/setup.py`, or packaging
  metadata.

## Style

Comments explain *why*, not *what*. The engine has a lot of decisions that
look arbitrary until you know what they are compensating for — a pinned
word-piece window, a `.bfloat16()` rounding matched to the upstream reference,
a static libstdc++ link that exists because of manylinux policy. When you make
one of those, say what would go wrong otherwise. Both `cpp/README.md` and the
Python sources are written that way; match them.
