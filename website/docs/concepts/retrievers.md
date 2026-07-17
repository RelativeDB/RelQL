---
title: Retrievers
description: The data-access contract between the engine and your storage.
---

# Retrievers

The engine never touches a database. It asks **your** code for rows through
five small interfaces (Java: async interfaces; Python: plain callables; Rust:
traits, usually closures):

| Interface | Signature (conceptual) | Role |
|---|---|---|
| `EntityRetriever` | `(table, ids, bound) → rows` | Batched point lookup: seed rows, parents |
| `LinkRetriever` | `(link, parent_id, bound, limit) → rows` | Children along one FK link, newest-first |
| `CohortRetriever` *(optional)* | `(table, anchor, bound, limit) → ids` | Similar entities for in-context examples |
| `TableScanner` *(optional)* | `(table, bound) → row stream` | Bulk streaming; enables `FOR EACH` and CSC mode |
| `StatsProvider` *(optional)* | — | Normalization statistics |

## Rows

A `Row` carries typed cells, an optional timestamp, and **parent edges**
(`{fk_column: parent_id}`). IDs and FK values are never cells — they surface
only as identity and edges.

## Wiring

A `RetrieverWiring` binds retrievers to tables and links, with a
`default_links` catch-all. It is validated against the schema when the engine
is built, so a missing retriever fails fast, not mid-query.

## Contract essentials

- Return **nothing newer** than the `TemporalBound` you are given. (The
  engine re-checks anyway — see
  [Temporal correctness](temporal-correctness).)
- `LinkRetriever` returns children **newest-first**, capped at `limit`.
- Retrievers own their I/O: batching, caching, auth, and concurrency are
  yours. In Java the SPI is async (`CompletionStage`); Python and Rust are
  deliberately synchronous.
