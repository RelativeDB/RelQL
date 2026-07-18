# PQL evolution proposal

## 1. Temporal window frames

Temporal bounds should be removed from aggregation argument lists and expressed
with a compact SQL-window-like OVER clause:

~~~sql
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FOR EACH customers.customer_id
WHERE EXISTS(orders.*) OVER (90 DAYS PRECEDING)
AS OF :prediction_time
~~~

For a Boolean target, probability output is inferred when RETURN is omitted;
`RETURN PROBABILITY` remains available when the author wants to make it
explicit.

In the full form, the only frame component is RANGE BETWEEN. The shorthand may
omit it when one endpoint is NOW. PQL does not need SQL's PARTITION BY or
ORDER BY syntax:

- FOR EACH already supplies the entity partition.
- The referenced table's schema-declared time column supplies temporal order.
- A missing or ambiguous time column is a validation error.
- EXPLAIN displays the inferred partition key and time column.

NOW is the prediction anchor, not wall-clock time and not an input fact row.
AS OF binds NOW explicitly; otherwise the execution input's anchor time binds
it.

Common frames are expressed using standard interval and window-bound
vocabulary:

~~~sql
-- Next 30 days
RANGE BETWEEN NOW
          AND 30 DAYS FOLLOWING

-- Previous 24 hours
RANGE BETWEEN 24 HOURS PRECEDING
          AND NOW

-- Future days 15 through 45
RANGE BETWEEN 15 DAYS FOLLOWING
          AND 45 DAYS FOLLOWING

-- All available history
RANGE BETWEEN UNBOUNDED PRECEDING
          AND NOW
~~~

### Single-bound shorthand

Frames anchored on NOW may omit RANGE BETWEEN and the implied NOW bound:

~~~sql
-- Expands to: RANGE BETWEEN NOW
--                         AND 90 DAYS FOLLOWING
OVER (90 DAYS FOLLOWING)

-- Expands to: RANGE BETWEEN 90 DAYS PRECEDING
--                         AND NOW
OVER (90 DAYS PRECEDING)

-- Expands to: RANGE BETWEEN UNBOUNDED PRECEDING
--                         AND NOW
OVER (UNBOUNDED PRECEDING)
~~~

The full and shortened forms are semantically identical. The shorthand is
preferred for ordinary future and historical horizons. Full RANGE BETWEEN
remains available—and is required—when neither endpoint is implied by NOW:

~~~sql
OVER (
    RANGE BETWEEN 15 DAYS FOLLOWING
              AND 45 DAYS FOLLOWING
)
~~~

Named windows accept either form:

~~~sql
WINDOW next_90_days AS (
    90 DAYS FOLLOWING
)

WINDOW days_15_to_45 AS (
    RANGE BETWEEN 15 DAYS FOLLOWING
              AND 45 DAYS FOLLOWING
)
~~~

### Duration literals

Durations use a numeric value followed directly by a unit:

~~~sql
1 DAY
2 DAYS
90 DAYS
~~~

Units may be singular or plural and are case-insensitive. The same form is used
anywhere PQL expects a duration, including window bounds, STEP, WITHIN, and
anchor recurrence. Because these positions already require a duration, no
wrapper keyword or quoted value is needed.

### Surface grammar

The conceptual grammar is:

~~~text
over_clause
    := OVER '(' window_spec ')'
     | OVER window_name

window_declaration
    := WINDOW window_name AS '(' window_spec ')'

window_spec
    := frame [HORIZONS positive_integer [STEP positive_duration]]

frame
    := RANGE BETWEEN bound AND bound
     | positive_duration PRECEDING
     | positive_duration FOLLOWING
     | UNBOUNDED PRECEDING

bound
    := NOW
     | positive_duration PRECEDING
     | positive_duration FOLLOWING
     | UNBOUNDED PRECEDING
     | UNBOUNDED FOLLOWING

positive_duration
    := <positive_number> <unit>
~~~

Negative or zero durations are invalid; PRECEDING and FOLLOWING determine
direction. Window names must be unique within a query and must be declared
exactly once.

### Frame normalization and membership

Every window is normalized to a lower and upper offset from the query anchor:

~~~text
NOW                                      =>  0
<duration> PRECEDING                     => -duration
<duration> FOLLOWING                     => +duration
UNBOUNDED PRECEDING                      => -infinity
UNBOUNDED FOLLOWING                      => +infinity
~~~

Let a be the instant bound to NOW, and let l and u be the normalized lower and
upper offsets. A row with event time t belongs to the frame exactly when:

~~~text
t >  a + l
t <= a + u
~~~

An unbounded endpoint omits its corresponding comparison. This preserves PQL's
start-exclusive/end-inclusive convention: (a + l, a + u]. A frame is invalid
when l >= u.

The shorthand is part of normalization rather than a different frame type:

~~~text
<duration> FOLLOWING
    => RANGE BETWEEN NOW AND <duration> FOLLOWING

<duration> PRECEDING
    => RANGE BETWEEN <duration> PRECEDING AND NOW

UNBOUNDED PRECEDING
    => RANGE BETWEEN UNBOUNDED PRECEDING AND NOW
~~~

Seconds, minutes, hours, days, and weeks are fixed-duration intervals. Months
and years use calendar arithmetic in the query timezone; they are not converted
to a fixed number of days. The anchor, row timestamps, and computed endpoints
must be compared as instants after timezone normalization.

### Partition, time, and relationship binding

FOR EACH supplies the entity partition. Each table referenced by a framed
expression must have:

- One schema-declared event-time column.
- A schema-resolvable relationship to the FOR EACH entity.
- Complete rows for exact target or population evaluation.

The schema binder determines which fact rows belong to each entity. If that
binding is missing or ambiguous, validation fails rather than choosing a route
arbitrarily. The window itself remains free of PARTITION BY, ORDER BY, and join
syntax.

Window membership needs only the event timestamp. Order-sensitive operations
such as FIRST, LAST, NEXT, or FOLLOWED BY use event time followed by the row's
primary key as a deterministic tie-breaker. If the schema cannot provide a
stable tie-breaker, those operations fail validation.

Rows without an event time do not belong to a temporal frame. A named WINDOW is
an immutable, table-independent frame template: each expression using it binds
the template to its own referenced table, event-time column, and entity
relationship.

Target-label and population-filter windows are evaluated over all qualifying
rows. Model-context fanout or sampling must not affect their result. At
inference time, a future target window is a symbolic label definition and is
not read as model context.

The positional form:

~~~sql
COUNT(orders.*, 0, 90, days)
~~~

is replaced by:

~~~sql
COUNT(orders.*) OVER (90 DAYS FOLLOWING)
~~~

Multi-horizon projection belongs to the window specification:

~~~sql
PREDICT SUM(sales.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)
FOR EACH stores.store_id
AS OF :prediction_time
RETURN EXPECTED VALUE
~~~

A window has one horizon by default. With HORIZONS N, the engine evaluates N
shifted copies of the complete frame. Formally, for horizon h numbered from 1
through N:

~~~text
W_h = (
    NOW + lower + (h - 1) * step,
    NOW + upper + (h - 1) * step
]
~~~

STEP controls the distance between horizon starts. When omitted, it defaults
to the frame width, upper - lower. The example therefore produces:

~~~text
horizon 1: (NOW,       NOW + 7d]
horizon 2: (NOW + 7d,  NOW + 14d]
horizon 3: (NOW + 14d, NOW + 21d]
horizon 4: (NOW + 21d, NOW + 28d]
~~~

STEP permits overlapping projections:

~~~sql
PREDICT SUM(sales.qty) OVER demand_projection
FOR EACH stores.store_id

WINDOW demand_projection AS (
    30 DAYS FOLLOWING
    HORIZONS 6
    STEP 7 DAYS
)
~~~

This produces six rolling 30-day predictions whose starts are one week apart.

Horizon validation rules:

- HORIZONS defaults to 1 and must be a positive integer.
- HORIZONS greater than 1 is valid only on a PREDICT target, not in a population
  WHERE condition.
- A multi-horizon frame must have finite lower and upper bounds.
- STEP is valid only with HORIZONS greater than 1 and must be a positive
  interval.
- If STEP is omitted, the finite frame width is used.
- If the lower and upper bounds cannot be subtracted into one unambiguous
  interval—for example, mixed calendar and fixed-duration bounds—STEP is
  required explicitly.

### Window result types and expression alignment

A framed expression with one horizon produces a scalar value. A framed
expression with multiple horizons produces a horizon series. Prediction results
represent that series relationally:

~~~text
entity_key | horizon | horizon_start | horizon_end | prediction
~~~

Operators and functions apply elementwise to a horizon series. Literals and
unframed entity attributes are broadcast across every horizon. If one target
combines two horizon series, their horizon count, lower/upper offsets, and STEP
must be identical; otherwise validation fails. Named windows are the normal way
to guarantee this alignment.

A multi-horizon window implies forecasting; no separate FORECAST clause is
needed.

### EXPLAIN representation

EXPLAIN always expands shorthand and named windows into their normalized form.
For every framed expression it reports:

- The bound NOW source: query AS OF, execution anchor, or fine-tuning anchor.
- Entity partition and resolved relationship.
- Table, event-time column, and deterministic tie-breaker.
- Normalized lower and upper bounds with endpoint inclusivity.
- HORIZONS, effective STEP, and every concrete horizon start/end.
- Scalar or horizon-series result type.
- Whether evaluation is exact label/population work or sampled model context.

EXPLAIN ANALYZE additionally reports qualifying and rejected row counts per
horizon, including rows rejected for missing timestamps or boundary failures.

## 2. Richer target expressions

PQL should allow arithmetic and familiar scalar expressions:

~~~sql
PREDICT
    SUM(orders.revenue) OVER next_30_days
    -
    SUM(orders.cost) OVER next_30_days
FOR EACH customers.customer_id

WINDOW next_30_days AS (
    30 DAYS FOLLOWING
)
~~~

A named WINDOW contains only the reusable temporal frame. FOR EACH still
provides the entity partition, and each referenced table's schema supplies its
time column.

Initial expression set:

- Arithmetic: +, -, *, /, and parentheses.
- Column-to-column and expression-to-expression comparisons.
- CASE WHEN ... THEN ... ELSE ... END.
- COALESCE and NULLIF.
- ABS, LOG, EXP, LEAST, and GREATEST.
- Typed TRUE and FALSE literals.

SQL expression behavior should be borrowed where practical, especially for
NULL propagation and division by zero.

## 5. Explicit output intent

Task inference remains a convenient default, but RETURN lets the author request
the desired predictive object:

~~~sql
RETURN EXPECTED VALUE
RETURN PROBABILITY
RETURN CLASS
RETURN DISTRIBUTION
RETURN QUANTILES (0.10, 0.50, 0.90)
RETURN INTERVAL 90%
~~~

Validation must reject an output incompatible with the target. EXPLAIN should
show both the inferred target type and the final output schema.

Set targets should distinguish multiclass and multilabel behavior. MULTILABEL
is the more precise spelling for classification on a set target.

## 9. EXPLAIN

EXPLAIN is a prefix so it remains visually and conceptually familiar:

~~~sql
EXPLAIN
PREDICT NOT EXISTS(orders.*) OVER (90 DAYS FOLLOWING)
FOR EACH customers.customer_id
AS OF :prediction_time
RETURN PROBABILITY
~~~

EXPLAIN without a mode is equivalent to EXPLAIN PLAN and does not invoke the
model.

### EXPLAIN PLAN

Static parse, binding, and planning information:

- Normalized target expression.
- Prediction grain and entity selector.
- Inferred task type and requested RETURN type.
- Target, population-filter, and context time bounds.
- Inferred or explicit relationship paths.
- Tables, features, and links eligible for context.
- Explicit ablations and transitively unreachable objects.
- Retriever/index operations expected by hop.
- Fanout, context-cell budget, and sampler mode.
- Model family/checkpoint route.
- Output schema.
- Validation warnings and inferred defaults.

Example:

~~~sql
EXPLAIN PLAN FORMAT TEXT
PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING)
FOR EACH customers.customer_id
ABLATE TABLE support_tickets
RETURN PROBABILITY
~~~

### EXPLAIN CONTEXT

Retrieves and assembles context for specified entities but does not score the
model:

~~~sql
EXPLAIN CONTEXT
PREDICT EXISTS(orders.*) OVER (30 DAYS FOLLOWING)
FOR customers.customer_id = 'C7'
AS OF 2026-07-01
~~~

Report:

- Row and cell counts by table.
- Traversed link counts.
- Minimum/maximum event times.
- Rows rejected by temporal bounds.
- Rows truncated by fanout or context budget.
- Tables that were unreachable.
- Cells masked and rows dropped by ablation.
- Retriever calls, cache hits, and timings.

By default, EXPLAIN CONTEXT should report metadata rather than raw cell values.
An explicit verbose/debug option can expose values where access policy permits.

### EXPLAIN ANALYZE

Executes the query and reports actual planning, retrieval, assembly, model, and
decoding behavior:

~~~sql
EXPLAIN ANALYZE FORMAT JSON
PREDICT SUM(orders.amount) OVER (30 DAYS FOLLOWING)
FOR customers.customer_id IN ('C7', 'C9')
AS OF :prediction_time
RETURN QUANTILES (0.10, 0.50, 0.90)
~~~

In addition to CONTEXT fields, include:

- Backend and model actually used.
- Batch/token sizes.
- Prediction output per entity/horizon.
- Time spent in each execution stage.
- Fallbacks and unsupported-head routing.
- Determinism/sampling seed.

### Output formats

TEXT is optimized for people. JSON is a versioned, machine-readable schema
suitable for tests, notebooks, and visualization:

~~~sql
EXPLAIN PLAN FORMAT JSON ...
EXPLAIN CONTEXT FORMAT JSON ...
EXPLAIN ANALYZE FORMAT JSON ...
EXPLAIN ABLATION FORMAT JSON ...
~~~

The JSON form should carry stable IDs for tables, links, expressions, plan
nodes, ablation variants, and warnings.
