# STYLE

Base style guide: **microsoft**.

## Terminology

| Use | Not | Because |
|---|---|---|
| `RelQL` | PQL, Relql, RELQL, relQL | The query language. Always this casing in prose. |
| `RelativeDB` | Relativedb, relative-db | The product. |
| `relationdb` | relativedb (on PyPI) | The **distribution** name. `pip install relationdb`, `import relativedb`. |
| `RT-J` | RTJ, rt-j (in prose) | The pretrained relational transformer checkpoint. |

Language and engine vocabulary — prefer the first, avoid the rest:

| Use | Not |
|---|---|
| population | entity set, cohort *(the thing `FROM` names)* |
| cohort | id list, entity ids *(the subset actually scored)* |
| as-of time | anchor time, cutoff, t0 |
| window | frame *(when referring to what `OVER` takes)* |
| retriever | connector, adapter, driver |
| bind parameter | placeholder, variable |
| counterfactual | what-if *(for `ASSUMING`)* |
| snapshot | cache *(the CSC index)* |

## Formatting
- Code voice: identifiers, clause keywords, and file paths in backticks —
  `FROM`, `RtNativeBackend`, `cpp/src/relql.cpp`.
- Reference code as `path:line` when pointing at an implementation.
- Tables for comparisons; prose for reasoning. Do not write a table of prose.
- One `h1` per document. The consolidated docs are single pages, so headings carry
  the table of contents — keep levels contiguous.

## Approved phrasings

## Forbidden phrasings

- "simply", "just", "easy to" — the reader decides what is easy.
- "should work", "in theory" — verify and state the result, or say it is untested.
- Describing an unimplemented feature in the present tense. If it does not execute,
  the sentence says so.
