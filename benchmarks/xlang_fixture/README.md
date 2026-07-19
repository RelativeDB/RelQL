# Cross-language ranking-parity fixture

A fixed, real **MovieLens** (`ml-latest-small`) ranking scenario that every
language binding runs through the native RT-J backend to prove it produces the
*right* (correct, non-degenerate) ranking. It is a regression guard against the
two ranking bugs found on **2026-07-18** — see [The bug class](#the-bug-class).

## What's in here

| File | Contents |
|------|----------|
| `movies.tsv` | `movie_id \t title \t genres` — the 10 candidate movies. |
| `ratings.tsv` | `rating_id \t user_id \t movie_id \t rating \t ts_epoch_seconds` — 2 users' rating history. |
| `embeddings.tsv` | `string \t f0 f1 … f383` — precomputed MiniLM (`all-MiniLM-L12-v2`, `normalize=False`) vectors, the *exact* vectors Python produced. Bindings without a built-in MiniLM (Rust, Java) feed these so every language scores the identical text. |
| `golden.json` | Schema, query, `anchor_epoch`, users, `top_k`, `candidate_ids`, `title_of`, the `invariants`, and `per_binding_golden` (each binding's captured top-5). |

The scenario:

- **Query** — `PREDICT LIST_DISTINCT(ratings.movie_id) OVER (60 DAYS FOLLOWING) RANK TOP 5 FOR EACH users.user_id`
- **Anchor** — `2014-12-28T00:00:00Z` (`anchor_epoch = 1419724800`)
- **Users** — `1` and `2`
- **Candidates** — 10 movies: `[1, 2, 3, 50, 260, 296, 318, 356, 593, 858]`

## The invariants (`golden.json.invariants`)

Cross-language checks that hold despite float-level tie ordering:

1. **Not degenerate** — the returned top-5 (by `movie_id`) must **not** equal
   `must_not_equal_degenerate_order = [1, 2, 3, 50, 260]`. That order is exactly
   the candidate-enumeration order both bugs produced.
2. **`top1 == 593`** (*Silence of the Lambs*) for **both** users
   (`expected_top1 = {1: 593, 2: 593}`). This is the strongest, tie-free signal
   and is stable across bindings.
3. **Candidate discrimination** — at least `min_distinct_scores = 5` candidates
   are genuinely distinguished. The `ranked` API exposes only ordered ids (no
   per-candidate scores), so each parity test proves this by additionally
   running `RANK TOP 10` (all candidates) and asserting the full order is **not**
   the sorted candidate-id order and that `top1` is still 593.
4. **Regression guard** — the per-user top-5 reproduces
   `per_binding_golden.<binding>`. Bindings agree on `top1` and the top-5 *set*
   but differ within sub-0.004 score ties at lower ranks (a property of the weak
   model signal on this task, not a bug), so each binding pins its own captured
   order.

## The bug class

Both 2026-07-18 bugs collapsed the ranking to the degenerate
candidate-enumeration order `[1, 2, 3, 50, 260]`:

- **Python** emitted no candidate cells, so every candidate context was
  identical and scored the same.
- **Java** emitted no target token for cell-less entity tables (`users`), with
  the same effect.

When candidates are indistinguishable the engine falls back to enumeration
order — which is precisely what invariants (1) and (3) catch.

## Running the parity test

All three bindings load *this* fixture, build the same schema/wiring/policy, run
at the same anchor for users `[1, 2]`, and assert invariants 1–4.

### Python — `python/tests/test_xlang_parity.py`

Python computes MiniLM itself (matching `embeddings.tsv`), so it does not read
`embeddings.tsv`.

```sh
cd python && SSL_CERT_FILE=/etc/ssl/cert.pem \
  RELATIVEDB_RT_LIB=/Users/henneberger/getasterisk/cpp/build/librt_c.dylib \
  .venv/bin/python -m pytest tests/test_xlang_parity.py -q
```

### Rust — `rust/relativedb/tests/xlang_parity.rs`

Rust has no built-in MiniLM, so it builds a `PrecomputedEncoder` from
`embeddings.tsv` (zero vector on a miss).

```sh
cd rust && RELATIVEDB_RT_LIB=/Users/henneberger/getasterisk/cpp/build/librt_c.dylib \
  cargo test --test xlang_parity
```

### Java — `XlangRankParityTest`

Like Rust, Java feeds the precomputed `embeddings.tsv`. (Added separately under
the Java effort.)

All three skip cleanly when the native dylib / RT-J checkpoint is unavailable.
