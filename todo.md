## Highest-value improvements

### 1. Make time windows readable

The positional (0, 90, days) syntax is compact but opaque. Add a semantic form
while retaining it for compatibility:

PREDICT COUNT(orders.*) DURING NEXT 90 DAYS = 0
FOR EACH customers.customer_id
WHERE COUNT(orders.*) DURING PREVIOUS 90 DAYS > 0
AS OF :prediction_time

This gives you room for nuance:

DURING DAYS 15 THROUGH 45
DURING NEXT CALENDAR MONTH
DURING ALL HISTORY
DURING PREVIOUS 24 HOURS
DURING (:start, :end]

AS OF would make backtests reproducible in the query itself.


### 2. Add predictive concepts, not merely more aggregates

Several common questions are awkward as COUNT(...) > 0:

PREDICT EXISTS(orders.*) DURING NEXT 30 DAYS
PREDICT TIME UNTIL FIRST orders.*
PREDICT NEXT orders.status
PREDICT QUANTILE(0.90, sales.qty) DURING NEXT 7 DAYS

Particularly valuable additions:

- EXISTS / NOT EXISTS
- TIME UNTIL FIRST
- NEXT
- ANY / ALL
- quantiles and prediction intervals
- event sequences such as event_a FOLLOWED BY event_b
- censoring for survival questions

For example:

PREDICT TIME UNTIL FIRST subscriptions.cancelled
CENSORED AFTER 180 DAYS
FOR EACH customers.customer_id

That captures something meaningfully different from six separate churn queries.

### 3. Support arithmetic and richer expressions

Targets should compose:

PREDICT
SUM(orders.revenue) - SUM(orders.cost)
DURING NEXT 30 DAYS
FOR EACH customers.customer_id

Useful expression support includes:

COALESCE(...)
CASE WHEN ... THEN ... ELSE ... END
SUM(...) / NULLIF(COUNT(...), 0)
column_a = column_b
ABS(...)
LOG(...)

This is one place where borrowing SQL expression semantics makes sense without
turning the entire language into SQL.

### 4. Make relationship paths explicit only when needed

Implicit schema traversal is pleasant when there is one path. When there are
multiple paths, introduce VIA:

PREDICT SUM(payments.amount) DURING NEXT 30 DAYS
FOR EACH customers.customer_id
VIA payments.order_id -> orders.customer_id

The rule could be:

- One valid path: infer it.
- No path: validation error.
- Multiple paths: require VIA.

That preserves the terse common case while making complex schemas unambiguous.