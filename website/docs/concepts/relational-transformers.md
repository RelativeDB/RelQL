---
title: Relational transformers
description: What a relational transformer is and why it beats per-task models.
---

# Relational transformers

relativedb's default real model is **RT-J** (`stanford-star/rt-j`), a
relational transformer — a foundation model for relational data.

## The idea

A transformer normally attends over a sequence of word tokens. A relational
transformer attends over a **small subgraph of your database**: each token is
one cell (a feature value of some row), and attention is masked along the
structure that relates them:

- **column attention** — cells of the same column, across rows
- **feature attention** — cells of the same row and its FK parents
- **neighbor attention** — cells of a row's FK children

There are no positional encodings — structure is carried entirely by the
masks. The model reads the entity, its neighborhood, and a handful of labeled
in-context examples (including the entity's own past outcomes), and predicts
the masked target cell in a single forward pass.

## Why this beats the alternatives

**vs. classical tabular ML (GBDTs on feature tables).** No hand-built
features, no per-task training, no train/serve skew, and no temporal-leakage
bugs hiding in feature SQL. Change the question, change the query string.

**vs. graph neural networks.** GNNs are also structure-aware but are trained
per task and per schema. A relational transformer is pretrained across many
schemas and predicts **in-context** — the relational analogue of prompting an
LLM instead of fine-tuning one.

**vs. LLMs on serialized rows.** Flattening tables into text throws away
types, keys, and time, and burns context on formatting. The relational
transformer consumes typed cells and real graph structure directly.

## In relativedb

The engine assembles the temporally-bounded context; the model scores it.
Checkpoints are routed per task type (classification vs regression), text
cells embed with a pinned MiniLM encoder, and inference runs in a
[dependency-light C++ engine](../libraries/cpp) verified against the PyTorch
reference. A model-free history baseline is built in, so nothing requires the
model to be present.
