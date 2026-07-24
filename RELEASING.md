# Releasing

Cutting a release of the `relativedb` PyPI package. The engine ships inside
the wheel, so a release is a binary release: three platform wheels plus one
sdist, all built from one commit, all verified before anything is uploaded.

**Publishing is manual.** No workflow uploads to PyPI on a tag push or a
merge. `release-libraries.yml` runs only from `workflow_dispatch` with
`publish` explicitly checked, and the upload step sits behind the `pypi`
GitHub environment, whose reviewers must approve the job before it starts.
Do not run it without a sign-off from whoever owns the release.

## What builds what

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` | every push to `main`, every PR | `cpp/` build + `ctest`, Python unit tier on Linux x86_64/aarch64 and macOS arm64; integration tier + coverage on Linux x86_64 and macOS arm64 |
| `wheels.yml` | `workflow_dispatch`, `v*` tag push, weekly cron, or a PR labelled `build-wheels` | builds and verifies the sdist and the three platform wheels; uploads them as run artifacts. Never publishes |
| `release-libraries.yml` | `workflow_dispatch` only | downloads a `wheels.yml` run's artifacts, re-verifies, and (optionally) uploads to TestPyPI and PyPI |

`wheels.yml` builds by calling `python/build_wheel.sh` — the same script a
developer runs locally — inside the manylinux_2_28 container and on the macOS
runner, so the released binary comes off the same code path as a local one.
Wheel tags are `py3-none-<platform>`: `librt_c` is loaded through `ctypes`, so
one wheel per platform covers every CPython 3.

Targets are manylinux_2_28 x86_64, manylinux_2_28 aarch64, and macOS
`universal2` at deployment target 13.0. Windows is out of scope. The sdist
carries no native library at all; installing it gives a pure `py3-none-any`
wheel that falls back to `RELATIVEDB_RT_LIB` or a monorepo `cpp/build` tree at
runtime. That is by design — `cpp/` is not in the sdist and a source install
can never build the engine on its own.

## Checklist

### 1. Bump the version

Two files, and they must agree:

- `python/pyproject.toml` — `version = "X.Y.Z"`
- `python/src/relativedb/__init__.py` — `__version__ = "X.Y.Z"`

Check nothing else drifted:

```bash
grep -rn "$(grep -m1 '^version' python/pyproject.toml | cut -d'"' -f2)" \
  python/pyproject.toml python/src/relativedb/__init__.py README.md python/README.md
```

The package is pre-1.0, so a breaking API or grammar change is a minor bump.

### 2. Update the changelog

Move the `Unreleased` entries in `CHANGELOG.md` into a new `## [X.Y.Z] — YYYY-MM-DD`
section, add a fresh empty `Unreleased`, and update the link definitions at
the bottom. Write user-facing changes, not commit subjects.

### 3. Green CI is a precondition

The commit you intend to release must have a green `CI` check on `main`. That
single check aggregates every lane — the unit matrix, the integration tier
(run with `RELATIVEDB_REQUIRE_NATIVE=1`, so a missing engine or checkpoint is
a failure rather than a skip), and C++ coverage. Do not release off a commit
whose CI was skipped, cancelled, or is still running.

### 4. Build the artifacts

Run **Wheels** (`wheels.yml`) via `workflow_dispatch` on the release commit,
or push the tag (see step 6) and let the `v*` trigger start it. Note the run
id from the run's URL — the publish workflow needs it.

That run must be green. It already does the artifact verification that matters:

- `twine check` on every distribution
- the sdist installs into a clean venv **outside the repository** and
  `load_lib()` fails loudly there, proving the sdist is genuinely pure and is
  not silently picking up a library from the build machine
- each wheel's tag is asserted: manylinux (never a bare `linux_*`, which PyPI
  rejects) and never interpreter-specific; on macOS, `lipo`/`otool` assert
  both slices and `minos 13.0`
- `auditwheel show` on the repaired Linux wheels
- `python/release_smoke.py` runs from a scratch directory against each wheel
  installed in a clean venv — it asserts the native library resolves from
  `site-packages` (not a repo build tree) and that a real RT-J churn
  prediction comes out finite

If you want the same locally before committing to a release:

```bash
PYTHON=python/.venv/bin/python sh python/build_wheel.sh
python/.venv/bin/python -m twine check python/dist/*

# clean-venv smoke, outside the repo, so no build tree can mask a bad bundle
work=$(mktemp -d); cp python/release_smoke.py "$work/"
python3 -m venv "$work/venv"
"$work/venv/bin/pip" install python/dist/*.whl
(cd "$work" && ./venv/bin/python release_smoke.py)
```

`build_wheel.sh` deletes any stray `librt_c` from `python/src/relativedb/`
before building the sdist. Do not substitute a bare `python -m build python`
for it: that builds the sdist from whatever is in your working tree, and a
leftover development `librt_c` ends up inside the source distribution.

### 5. Rehearse on TestPyPI (optional, recommended for a first release of a new
platform target)

Run **Release libraries** with `wheels_run_id` set, `test_pypi` checked and
`publish` unchecked. It downloads, verifies with `twine check --strict`, and
uploads to TestPyPI only. Nothing reaches PyPI.

### 6. Tag

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

Tag the exact commit the artifacts were built from. Pushing a `v*` tag starts
`wheels.yml`; it still does not publish anything.

### 7. Publish

Run **Release libraries** with:

- `wheels_run_id` — the run id from step 4
- `publish` — checked

The `verify` job re-downloads the artifacts, asserts exactly one sdist and at
least one wheel arrived, rejects anything named `relationdb-*` (the project is
`relativedb`), and runs `twine check --strict`. The upload job then waits on
the `pypi` environment's reviewers.

Upload uses PyPI trusted publishing over OIDC. There is no API token in this
repository, and there should never be one. The PyPI project must have a
trusted publisher configured for this repository, the workflow filename
`release-libraries.yml`, and the `pypi` environment.

### 8. Verify what users get

```bash
work=$(mktemp -d); cp python/release_smoke.py "$work/"
python3 -m venv "$work/venv"
"$work/venv/bin/pip" install "relativedb==X.Y.Z"
(cd "$work" && ./venv/bin/python release_smoke.py)
```

From a directory outside the repository, so the `cpp/build` fallback cannot
hide a broken bundle. Confirm `pip` chose a platform wheel and not the sdist.

### 9. GitHub release

Create a release on the `vX.Y.Z` tag with the changelog section as the body.

## If a bad artifact ships

**PyPI does not allow re-uploading a version.** Deleting a release does not
free the version number either — the filename stays permanently taken. There
is no way to fix `X.Y.Z` in place.

1. **Yank** `X.Y.Z` on PyPI. A yanked version disappears from resolution for
   anyone who has not pinned it exactly, while existing pins and lockfiles
   keep working. This is the right first move for a broken wheel: it stops
   new installs without breaking anyone already depending on it.
2. Fix, bump to `X.Y.Z+1`, and run the whole checklist again.
3. Delete the release only if it leaked a secret or shipped something that
   must not be downloadable. Deletion breaks every existing pin; yanking does
   not.

If only one platform's wheel is bad, yanking is still version-wide — PyPI
yanks a release, not a file. There is no per-wheel remedy.
