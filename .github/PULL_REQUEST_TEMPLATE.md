## What changes and why

<!-- The "why" is the part reviewers cannot reconstruct from the diff. If this
     compensates for something — a reference implementation's rounding, a
     platform policy, a checkpoint's training distribution — say so here and in
     a comment at the site. -->

## Numerical impact

<!-- Anything touching cpp/src or the samplers: does golden parity still hold,
     and did any score move? "None" is a fine answer; silence is not. -->

## Checklist

- [ ] `ctest --test-dir cpp/build --output-on-failure --no-tests=error` passes
- [ ] `pytest python/tests -m "not integration"` passes
- [ ] Ran the integration tier with `RELATIVEDB_REQUIRE_NATIVE=1`, or this
      change cannot reach it
- [ ] New behaviour has a test in a tier that actually reaches it
- [ ] `CHANGELOG.md` updated under `Unreleased`, or this is not user-visible
- [ ] Added the `build-wheels` label if this touches `cpp/`, `build_wheel.sh`,
      `setup.py`, or packaging metadata
