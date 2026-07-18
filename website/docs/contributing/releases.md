---
title: Releasing the libraries
description: Build and publish the relationdb Python, Java, and Rust packages.
---

# Releasing the libraries

The manual **Release libraries** GitHub Actions workflow builds and verifies
all three distributions. It does not publish by default. Run it with
`publish` left off to test the complete packaging path and retain the Python,
Java, and Rust artifacts on the workflow run.

The public distribution coordinates are:

| Ecosystem | Distribution |
|---|---|
| PyPI | `relationdb` (import `relativedb`) |
| Maven Central | `com.relativedb:relationdb` and `com.relativedb:relationdb-rt` |
| crates.io | `relationdb` (crate API `relativedb`) |

## One-time registry setup

Before the first publish:

1. Create protected GitHub environments named `pypi`, `crates-io`, and
   `maven-central`, ideally with required reviewers.
2. Register `.github/workflows/release-libraries.yml` as a PyPI trusted
   publisher for the `relationdb` project and the `pypi` environment. No PyPI
   API token is needed.
3. Add a crates.io token as the `CARGO_REGISTRY_TOKEN` environment secret on
   `crates-io`.
4. Verify ownership of the `com.relativedb` namespace in the Sonatype Central
   Portal. Add `MAVEN_CENTRAL_USERNAME`, `MAVEN_CENTRAL_PASSWORD`,
   `MAVEN_SIGNING_KEY`, and `MAVEN_SIGNING_PASSWORD` as `maven-central`
   environment secrets. The signing key is the ASCII-armored private key.

## Publishing

Update the matching version in `python/pyproject.toml`,
`rust/relativedb/Cargo.toml`, and `java/build.gradle`, then run tests locally.
Dispatch **Release libraries** with `publish` off first. Inspect its three
uploaded artifacts. When they are ready, dispatch the same commit again with
`publish` enabled and approve each protected environment.

Publishing is manual-only: pushing a branch or tag cannot trigger it.
