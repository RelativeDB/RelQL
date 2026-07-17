---
title: Architecture
description: The four-stage execution model.
---

# Architecture

Every query runs through the same four stages, in every language:

```
 PQL string
    │  parse       → typed AST (syntax errors here)
    ▼
 ParsedQuery
    │  validate    → bind names/types/windows against the Schema; infer TaskType
    ▼
 ValidatedQuery
    │  assemble    → hop loop through YOUR retrievers, bounded by anchor time
    ▼
 per-entity contexts
    │  score       → ModelBackend routed by TaskType
    ▼
 PredictionResult  → one prediction per entity (value and/or probability)
```

## Inputs

- **Query** — a PQL string. See the [PQL docs](/pql/).
- **Anchor time** — the "as of" instant t₀. Context may only contain data at
  or before it; the prediction concerns the window after it.
- **Entities** — an explicit ID list, or `FOR EACH` over the whole table
  (enumerated via a `TableScanner`).

## The schema carries shape only

Tables, typed columns (`NUMBER | TEXT | DATETIME | BOOLEAN`), primary keys,
per-table time columns, and FK links. No URLs, no credentials.

Validation enforces that links resolve, link targets have primary keys, and
that **PK/FK columns are never feature columns** — IDs are graph edges, not
values. There is no way to hand the model an identifier as a feature.

## Execution is GraphQL-style

The engine owns the language, planning, context assembly, and model routing.
It never connects to a database: all data access goes through
[retrievers](retrievers) you implement. The same query runs against JDBC, a
REST service, a feature store, or an in-memory test double — only the wiring
changes.
