---
id: index
title: RelQL — the language
slug: /
description: The complete RelQL language — grammar, semantics, and patterns, in one page.
---

# RelQL — the Predictive Query Language

RelQL expresses predictions the way SQL expresses lookups. One statement names a
**target** (what to predict), a **population** (who to predict it for), and an
**anchor-relative time window**:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FROM customers
```

*For every customer, will they place zero orders in the 90 days after the
anchor time?*

## Why a language?

- **Declarative** — changing the question means changing the string, not a
  pipeline.
- **Validated** — every query is bound against your schema before execution:
  unknown names, type mismatches, and backwards time windows are rejected up
  front.
- **Self-routing** — the query's shape determines the
  [task type](#task-types) (classification, regression, ranking,
  forecasting), which selects the model checkpoint and output form.

One grammar, single-sourced in a C++ parser and decoded by the Python, Java,
and Rust bindings — all tested against a shared 67-query corpus.

## How to read this page

This is the whole language in one document.

- New to RelQL? Start with the [tutorial](#relql-tutorial) — it builds a query
  up clause by clause.
- Looking something up? Jump to [query structure](#query-structure),
  [aggregations and windows](#aggregations-and-time-windows),
  [conditions](#conditions-and-operators), or [task types](#task-types).
- Want patterns to copy? Go to the [cookbook](#cookbook).

The engine that runs these queries — installation, retrievers, model backends,
and the language libraries — is documented in
[the engine guide](/docs/).


## RelQL tutorial

We'll build up a real query step by step, on a two-table schema:
`customers (customer_id, age, signup_date)` and
`orders (order_id, customer_id, qty, order_date)`, linked by
`orders.customer_id → customers`.

### Step 1: predict an aggregate

Start with the target — an aggregation over linked rows in a future window:

```sql
PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FROM customers
```

`OVER (30 DAYS FOLLOWING)` is a frame relative to the **anchor time** (the "as
of" instant you pass at execution): it covers the 30 days *after* the anchor,
start excluded, end included. This predicts each customer's total order
quantity over the next 30 days — a **regression**.

### Step 2: turn it into a yes/no question

Compare the aggregate to a literal and the task becomes **binary
classification** — the result is a probability:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING) FROM customers
```

"Will this customer place zero orders in the next 90 days?" — churn.

### Step 3: narrow the population

`WHERE` filters *who* gets predicted. Filter frames look **backwards**
(`PRECEDING`), so this restricts to customers active in the last 90 days:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FROM customers
WHERE EXISTS(orders.*) OVER (90 DAYS PRECEDING)
```

Static attributes work too: `WHERE customers.age >= 18`.

### Step 4: target specific entities

`FROM` names the population by table; the primary key comes from the schema.
To score only a specific subset, either constrain them with a `WHERE` predicate
on the key:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FROM customers
WHERE customers.customer_id IN :ids
```

```python
engine.execute(ExecutionInput(query=q, params={"ids": ["C7", "C9"]}))
```

`:ids` is a **bind parameter** — the cohort lives in `params`, not in the query
text, so the same query string is reusable across cohorts. A literal list
(`IN ('C7', 'C9')`) is also valid when the cohort really is fixed.

The engine reads a primary-key predicate as the cohort itself and scores only
those entities, so a pinned query needs no `TableScanner`.

### Step 5: filter the aggregated rows

Aggregations accept an inline row filter — different from `WHERE`, which
filters entities:

```sql
PREDICT SUM(orders.qty WHERE orders.qty > 1) OVER (30 DAYS FOLLOWING)
FROM customers
```

### Step 6: forecast over multiple horizons

Add `HORIZONS N` to a target frame and the single window repeats back to back
— a multi-horizon window *is* a forecast:

```sql
PREDICT SUM(orders.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)
FROM customers
```

Four weekly predictions per customer. (There is no separate `FORECAST` clause;
the horizons on the window imply it.)

### Step 7: rank a set of items

`LIST_DISTINCT` predicts *which* linked IDs will appear; `RANK TOP K` ranks
them:

```sql
PREDICT LIST_DISTINCT(orders.product_id) OVER (30 DAYS FOLLOWING RANK TOP 3)
FROM customers
```

### Step 8: ask "what if"

`ASSUMING` states a counterfactual condition carried with the query:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FROM customers
WHERE customers.customer_id = 'C7'
ASSUMING customers.plan = 'premium'
```

:::note
`ASSUMING` is parsed and validated but not yet applied to assembled context.
:::

### What you've learned

Target → population → filters → horizons → ranking → counterfactuals. Every
query you can write is validated against the schema before it runs, and its
shape determines the [task type](#task-types). Continue with the
[reference](#query-structure) or the [cookbook](#cookbook).


## Query structure

Clause order is significant:

```sql
[EXPLAIN [PLAN|CONTEXT|ANALYZE] [FORMAT TEXT|JSON]]  -- optional: inspect, don't (necessarily) run
PREDICT   <target> [CLASSIFY]                  -- required: what to predict
[FROM      <table> [[AS] <alias>]]             -- the population; inferred if omitted
[WHERE     <condition>]                        -- optional: entity filter (past-facing)
[ASSUMING  <condition>]                        -- optional: counterfactual
[AS OF     <anchor>]                           -- optional: bind the anchor time
[RETURN    <return_spec>]                       -- optional: choose the output form
[WINDOW    <name> AS (<window_spec>)]           -- optional, repeatable: named frames
```

The **trailing clauses** — `WHERE`, `ASSUMING`, `AS OF`, `RETURN`, `WINDOW` —
may appear in any order after `FROM`. Each may appear at
most once, except `WINDOW`, which repeats (one per named frame).

There is no `FORECAST N TIMEFRAMES` clause. To forecast, give the target's
window multiple horizons (`... OVER (7 DAYS FOLLOWING HORIZONS 4)`); a
multi-horizon window *implies* [forecasting](#task-types). See
[Aggregations & windows](#aggregations-and-time-windows).

### Clauses

- **`PREDICT <target>`** — a static column reference
  (`customers.age`, `articles.description IS NULL`), an
  [aggregation](#aggregations-and-time-windows) over linked rows in an `OVER` frame, or a
  richer expression (arithmetic, `CASE WHEN … END`, `COALESCE`, column-to-column
  comparison), optionally compared to a literal. `CLASSIFY` is a target
  directive; ranking is a frame directive, `OVER (… RANK TOP k)` (see
  [task types](#task-types)).
- **`FROM <table> [[AS] <alias>]`** — names the population. The primary key
  comes from the schema, so you write the table, not the key. An alias lets the
  rest of the query use the short name (`FROM customers c … c.plan`).
  Enumerating every entity requires a `TableScanner`; to score a specific
  subset, constrain the key in `WHERE` — `WHERE table.pk IN :ids`. The engine
  reads that as the cohort and scores only those entities, so a pinned query
  needs no scanner.

  `FROM` may be omitted when the target names exactly one table and is not an
  aggregation — then the population is that table:

  ```sql
  PREDICT issues.label WHERE issues.label IS NULL   -- population: issues
  ```

  An aggregate target names a *linked* table rather than the population, so it
  always needs an explicit `FROM`.
- **Column references** — `table.column`, `alias.column`, or a bare `column`,
  which binds to the population:

  ```sql
  PREDICT label FROM issues WHERE label IS NULL     -- both are issues.label
  ```
- **`WHERE <condition>`** — filters the population using static attributes
  and past-facing aggregations. See [Conditions](#conditions-and-operators).
- **`ASSUMING <condition>`** — a counterfactual assumption, parsed and
  validated and carried on the query (not yet applied to context assembly).
- **`AS OF <anchor>`** — binds the anchor time (the instant `NOW` and every
  frame are measured from). The anchor is a `DATE` literal (`2026-07-01`), a
  parameter (`:prediction_time`, bound at execution time), or `NOW`. A `DATE`
  or bound parameter takes precedence over the execution anchor; `NOW` (or no
  `AS OF`) uses the execution anchor.
- **`RETURN <return_spec>`** — selects the output form (see below).
- **`WINDOW <name> AS (<window_spec>)`** — declares a reusable named frame,
  referenced elsewhere as `OVER <name>`. Declared exactly once; referencing an
  undeclared name is an error. See [Aggregations & windows](#aggregations-and-time-windows).

### RETURN — output form

`RETURN` overrides the default output implied by the task type:

```
EXPECTED VALUE | PROBABILITY | CLASS | DISTRIBUTION
| QUANTILES (<num>, ...) | INTERVAL <int> [%] | MULTILABEL | MULTICLASS
```

```sql
PREDICT SUM(payments.amount) OVER (30 DAYS FOLLOWING)
FROM customers
AS OF :t
RETURN INTERVAL 90%
```

### EXPLAIN — inspect without (necessarily) running

An `EXPLAIN` prefix asks the engine to describe what it *would* do. The engine's
`explain()` entry point returns a result you can render as text or JSON
(`FORMAT TEXT | JSON`):

```
EXPLAIN [PLAN | CONTEXT | ANALYZE] [FORMAT TEXT | JSON]
```

- **`PLAN`** — the default (bare `EXPLAIN` == `EXPLAIN PLAN`). Describes the
  query from parsing and validation alone: the normalized target, inferred task
  type, entity selector, resolved output form, each aggregation's normalized
  window, and the resolved anchor source. Does **not** assemble context or
  invoke the model.
- **`CONTEXT`** — additionally assembles the per-entity context and reports
  row/cell counts, links traversed, time ranges, and rows dropped by the
  temporal bound. Does **not** score the model.
- **`ANALYZE`** — assembles and scores, returning the predictions with the plan.

```sql
EXPLAIN PLAN FORMAT TEXT
PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING)
FROM customers
RETURN PROBABILITY
```

### Lexical rules

- Keywords are **case-insensitive**: `PREDICT`, `OVER`, `FOLLOWING`,
  `PRECEDING`, `RANGE`, `BETWEEN`, `HORIZONS`, `STEP`, `WINDOW`, `AS OF`,
  `RETURN`, `EXPLAIN`, `FROM`, `WHERE`, `ASSUMING`, `CLASSIFY`, `RANK`,
  `TOP`.
- Aggregation and condition words (`COUNT`, `SUM`, `AND`, `LIKE`, ...) are
  **soft keywords** — still usable as column names (`usage.count` parses). In
  the `FROM` alias slot the clause words (`AS`, `WHERE`, `ASSUMING`, `ABLATE`,
  `RETURN`, `WINDOW`) are not treated as aliases.
- Column references are `table.column`, `alias.column`, or a bare `column`
  bound to the population; `table.*` counts rows.
- Literals: numbers, `'quoted strings'`, booleans, `DATE`s (`2026-07-01`).
  Frame bounds use `UNBOUNDED PRECEDING` for all history.
- Comments are supported.


## Aggregations and time windows

```
AGG( table.column | table.* [WHERE <row filter>] ) [OVER ( <window_spec> )]
```

An aggregation names a column (or `table.*`), an optional inline row filter,
and a **frame** introduced by `OVER`. The frame — not positional offsets —
carries the time window.

`OVER` is optional. Without it the frame is **unbounded in the direction of the
clause**: the future in `PREDICT` and `ASSUMING`, the past in `WHERE`.

```sql
PREDICT NOT EXISTS(orders.*)      -- (NOW, +inf]  will they ever order again?
FROM customers
WHERE COUNT(orders.*) > 5         -- (-inf, NOW]  have they ever ordered 5+ times?
```

### Functions

`SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `COUNT_DISTINCT`, `LIST_DISTINCT`,
`ARRAY_AGG`, `FIRST`, `LAST`, `EXISTS`, `NOT EXISTS`.

- `COUNT(table.*)` counts rows.
- `FIRST` / `LAST` pick a value by row time — useful for status columns.
- `LIST_DISTINCT` predicts the *set* of values that will appear (usually FK
  IDs); duplicates collapse.
- `ARRAY_AGG` predicts the values in order and **keeps duplicates** — use it
  when "bought twice" should count twice.
- Either can be ranked with the frame's `RANK TOP K` directive, or turned into
  a per-value yes/no with `CLASSIFY`.
- `EXISTS(table.*)` / `NOT EXISTS(table.*)` is a boolean existence test — true
  when any matching row falls in the frame. It reads more directly than the
  `COUNT(...) > 0` idiom (which is still valid): `EXISTS(orders.*) OVER (90 DAYS
  PRECEDING)`.

### The OVER frame

A frame is measured relative to the anchor time (`NOW`); membership is
start-**exclusive**, end-**inclusive**. Direction comes from `PRECEDING`
(past) / `FOLLOWING` (future), and durations are always **positive**.

```
window_spec := frame [HORIZONS <positive-int> [STEP <positive-duration>]]
                     [RANK TOP <positive-int>]

frame := RANGE BETWEEN <bound> AND <bound>
       | <positive-duration> PRECEDING        -- shorthand: (NOW - dur, NOW]
       | <positive-duration> FOLLOWING        -- shorthand: (NOW, NOW + dur]
       | UNBOUNDED PRECEDING                   -- all history up to NOW

bound := NOW
       | <positive-duration> PRECEDING
       | <positive-duration> FOLLOWING
       | UNBOUNDED PRECEDING
       | UNBOUNDED FOLLOWING

duration := <positive-number> <unit>
```

The single-bound forms are **shorthand** for a frame with one endpoint at
`NOW`. Use the full `RANGE BETWEEN` form when neither endpoint is `NOW`:

```sql
COUNT(orders.*) OVER (30 DAYS FOLLOWING)      -- (NOW, NOW+30d]  : the next 30 days
COUNT(orders.*) OVER (90 DAYS PRECEDING)      -- (NOW-90d, NOW]  : the last 90 days
COUNT(orders.*) OVER (UNBOUNDED PRECEDING)    -- all history up to NOW
SUM(sales.qty)  OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING)
                                              -- a future window not starting now
```

- Units: `SECONDS`, `MINUTES`, `HOURS`, `DAYS`, `WEEKS`, `MONTHS`, `YEARS`
  (singular or plural, case-insensitive; a month is a 30-day approximation).
- **Target** frames face the future (`FOLLOWING`). **Filter** frames (inside
  `WHERE`) face the past (`PRECEDING` / `UNBOUNDED PRECEDING`). The validator
  enforces both directions.

### Multiple horizons (forecasting)

Append `HORIZONS N` to repeat the frame N times back to back — a multi-horizon
window *is* a forecast (there is no separate `FORECAST` clause). `STEP`
optionally sets the stride between horizons; it defaults to the frame width, so
give a smaller `STEP` for overlapping horizons:

```sql
SUM(usage.count) OVER (1 DAY FOLLOWING HORIZONS 28)          -- 28 daily steps
SUM(sales.qty)   OVER (30 DAYS FOLLOWING HORIZONS 6 STEP 7 DAYS)  -- overlapping
```

### RANK TOP — ranking within a frame

`RANK TOP K` keeps the K most likely values from the frame, turning a set-valued
aggregation into a [ranking](#task-types). It is part of the frame, so *when* and
*how many* stay independent:

```sql
PREDICT ARRAY_AGG(transactions.article_id) OVER (30 DAYS FOLLOWING RANK TOP 12)
FROM customers
```

*The 12 articles each customer is most likely to buy in the next 30 days.* Drop
the frame's duration to rank over the whole future:

```sql
PREDICT ARRAY_AGG(transactions.article_id) OVER (RANK TOP 12)
FROM customers
```

### Named windows

Declare a frame once with a trailing `WINDOW` clause and reference it by name
as `OVER <name>` — handy when several aggregations share one frame:

```sql
PREDICT SUM(orders.revenue) OVER w - SUM(orders.cost) OVER w
FROM customers
WINDOW w AS (30 DAYS FOLLOWING)
```

A window name is declared exactly once and accepts every frame form, including
`HORIZONS` / `STEP`. Referencing an undeclared name is an error.

### Inline row filters

Filter the rows being aggregated (distinct from `WHERE`, which filters
entities):

```sql
COUNT(transactions.* WHERE transactions.amount > 10) OVER (30 DAYS FOLLOWING)
```


## Conditions and operators

Conditions appear in three places: comparing the target
(`PREDICT COUNT(...) = 0`), filtering entities (`WHERE`), filtering aggregated
rows (inline `WHERE` inside an aggregation), and stating counterfactuals
(`ASSUMING`).

### Comparison operators

`=` `==` `!=` `>` `>=` `<` `<=`

Either side of a comparison may be a literal, a static column, an aggregation
over an `OVER` frame, or a richer expression (arithmetic, `CASE WHEN … END`,
`COALESCE`, `NULLIF`, `ABS`/`LOG`/`EXP`/`LEAST`/`GREATEST`). Column-to-column
comparisons are allowed — e.g. `orders.shipped_at > orders.ordered_at`.

### Boolean composition

`AND`, `OR`, `NOT`, with parentheses.

### Membership and null tests

```sql
customers.location IN ('NY', 'CA')
customers.location NOT IN ('ALASKA', 'HAWAII')
articles.description IS NULL
articles.description IS NOT NULL
```

### String predicates

```sql
loan.status LIKE '%DENIED'        -- SQL % wildcards
movie.title STARTS WITH 'The'
movie.title ENDS WITH 'Returns'
movie.title CONTAINS 'Star'
```

### Bind parameters

Anywhere a literal is allowed, `:name` stands in for a value supplied at
execution time. With `IN`, one parameter binds the **whole list**, so a single
query text serves any cohort size:

```sql
WHERE customers.customer_id = :id
WHERE customers.customer_id IN :ids
WHERE customers.plan LIKE :pattern AND customers.age > :min_age
```

```python
engine.execute(ExecutionInput(query=q, params={"ids": ["C7", "C9"]}))
```

Values come from `params` on the execution input — the same place `AS OF :t`
reads its anchor. A `:name` with no supplied value is an error, never a silent
NULL.

A parameter on the **primary key** does double duty: it also selects the cohort
(see [query structure](#query-structure)), so the engine scores just those
entities instead of enumerating the table.

### Examples

```sql
-- entity filter mixing a static attribute and a past-facing aggregation
WHERE customers.age >= 18 AND EXISTS(orders.*) OVER (90 DAYS PRECEDING)

-- cohort pinned by a bound parameter
WHERE customers.customer_id IN :ids

-- predicate target: multiclass-style question on a status column
PREDICT LAST(loan.status) OVER (30 DAYS FOLLOWING) NOT LIKE '%DENIED' FROM loan
```


## Task types

The validator infers a task type from the target's shape. The task type
selects the model checkpoint and the output form — you never declare it.

| Target shape | Task type | Output |
|---|---|---|
| bare aggregation — `SUM(...)`, `COUNT(...)` | regression | value |
| aggregation vs literal — `COUNT(...) = 0` | binary classification | probability |
| `EXISTS(...)` / `NOT EXISTS(...)` (boolean target) | binary classification | probability |
| `FIRST` / `LAST` / static categorical column | multiclass classification | class + probabilities |
| `LIST_DISTINCT(...) OVER (... RANK TOP K)` | ranking | ranked ID list |
| any target whose window has `HORIZONS > 1` | forecasting | value per horizon |

### Model routing

`ModelConfig` maps task types to checkpoints — by default the classification
family routes to `hf://stanford-star/rt-j/classification` and
regression/forecasting to `hf://stanford-star/rt-j/regression`.

The output column above is the *logical* form each task produces. The RT-J
backend (`RtNativeBackend`) is the only scoring path. It executes **binary
classification**, **regression**, **multiclass classification**, and
**ranking**. Multiclass reuses the checkpoint's text head: the masked target
cell is predicted as a 384-d embedding and matched by cosine similarity to the
class labels' `all-MiniLM-L12-v2` embeddings, returning a predicted class plus
approximate class probabilities (a softmax over the cosine scores — the argmax
is the reference-exact class, the probabilities are an uncalibrated
approximation, not a trained softmax head). Ranking scores each candidate parent
ID with the existence head and returns the top *k*. `RETURN QUANTILES`/`INTERVAL`
remain unsupported — the checkpoint has no variance/quantile head — and raise a
clear error (see [Model backends](/docs/#model-backends)).

### Checking a query

Every library exposes the inference without executing:

```python
pq = relativedb.parse("PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FROM customers")
pq.task_type()    # TaskType.REGRESSION
```


## Cookbook

Copy-paste starting points, drawn from the shared 67-query test corpus.

### Churn (binary classification)

```sql
PREDICT NOT EXISTS(transactions.*) OVER (30 DAYS FOLLOWING)
FROM customers
WHERE EXISTS(transactions.*) OVER (90 DAYS PRECEDING)
```

Add `RETURN PROBABILITY` to get calibrated scores instead of the default output:

```sql
PREDICT NOT EXISTS(transactions.*) OVER (30 DAYS FOLLOWING)
FROM customers
WHERE EXISTS(transactions.*) OVER (90 DAYS PRECEDING)
RETURN PROBABILITY
```

### Spend / LTV slice (regression)

```sql
PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) FROM customers
```

### Recommendations (ranking)

```sql
PREDICT LIST_DISTINCT(transactions.article_id) OVER (30 DAYS FOLLOWING RANK TOP 12)
FROM customers
```

### Daily demand, 4 weeks out (forecasting)

```sql
PREDICT SUM(usage.count) OVER (1 DAY FOLLOWING HORIZONS 28)
FROM accounts
```

The `HORIZONS 28` on the window makes this a 28-step forecast — one prediction
per day.

### Specific entities

`FROM` is the only entity clause; narrow to specific ids with a `WHERE`
predicate on the primary key. Bind the ids as a parameter so one query text
serves any cohort:

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FROM users
WHERE users.user_id IN :ids
```

```python
engine.execute(ExecutionInput(query=q, params={"ids": [42, 123]}))
```

A literal list (`IN (42, 123)`) works too, but hard-codes the cohort.

### Counterfactual

```sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FROM users
WHERE users.user_id = 42
ASSUMING users.plan = 'premium'
```

### Status prediction (string predicate)

```sql
PREDICT LAST(loan.status) OVER (30 DAYS FOLLOWING) NOT LIKE '%DENIED' FROM loan
```

### Missing-attribute prediction (static target)

```sql
PREDICT articles.description IS NULL FROM articles
```

### Population carve-outs

```sql
PREDICT SUM(transactions.value) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING) > 100
FROM customers
WHERE customers.location NOT IN ('ALASKA', 'HAWAII')
```

### As-of a fixed anchor, with quantiles

```sql
PREDICT SUM(orders.amount) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING)
FROM customers
WHERE customers.customer_id IN ('C7', 'C9')
AS OF :prediction_time
RETURN QUANTILES (0.10, 0.50, 0.90)
```

### Reusable named window

```sql
PREDICT SUM(orders.revenue) OVER w - SUM(orders.cost) OVER w
FROM customers
WINDOW w AS (30 DAYS FOLLOWING)
```
