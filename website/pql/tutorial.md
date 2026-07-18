---
title: Tutorial
description: Build a predictive query clause by clause.
---

# RelQL tutorial

We'll build up a real query step by step, on a two-table schema:
`customers (customer_id, age, signup_date)` and
`orders (order_id, customer_id, qty, order_date)`, linked by
`orders.customer_id → customers`.

## Step 1: predict an aggregate

Start with the target — an aggregation over linked rows in a future window:

```sql
PREDICT SUM(orders.qty) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id
```

`OVER (30 DAYS FOLLOWING)` is a frame relative to the **anchor time** (the "as
of" instant you pass at execution): it covers the 30 days *after* the anchor,
start excluded, end included. This predicts each customer's total order
quantity over the next 30 days — a **regression**.

## Step 2: turn it into a yes/no question

Compare the aggregate to a literal and the task becomes **binary
classification** — the result is a probability:

```sql
PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR EACH customers.customer_id
```

"Will this customer place zero orders in the next 90 days?" — churn.

## Step 3: narrow the population

`WHERE` filters *who* gets predicted. Filter frames look **backwards**
(`PRECEDING`), so this restricts to customers active in the last 90 days:

```sql
PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0
FOR EACH customers.customer_id
WHERE COUNT(orders.*) OVER (90 DAYS PRECEDING) > 0
```

Static attributes work too: `WHERE customers.age >= 18`.

## Step 4: target specific entities

Replace `FOR EACH` with an explicit selection:

```sql
PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR customers.customer_id IN ('C7', 'C9')
```

## Step 5: filter the aggregated rows

Aggregations accept an inline row filter — different from `WHERE`, which
filters entities:

```sql
PREDICT SUM(orders.qty WHERE orders.qty > 1) OVER (30 DAYS FOLLOWING)
FOR EACH customers.customer_id
```

## Step 6: forecast over multiple horizons

Add `HORIZONS N` to a target frame and the single window repeats back to back
— a multi-horizon window *is* a forecast:

```sql
PREDICT SUM(orders.qty) OVER (7 DAYS FOLLOWING HORIZONS 4)
FOR EACH customers.customer_id
```

Four weekly predictions per customer. (There is no separate `FORECAST` clause;
the horizons on the window imply it.)

## Step 7: rank a set of items

`LIST_DISTINCT` predicts *which* linked IDs will appear; `RANK TOP K` ranks
them:

```sql
PREDICT LIST_DISTINCT(orders.product_id) OVER (30 DAYS FOLLOWING) RANK TOP 3
FOR EACH customers.customer_id
```

## Step 8: ask "what if"

`ASSUMING` states a counterfactual condition carried with the query:

```sql
PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0
FOR customers.customer_id = 'C7'
ASSUMING customers.plan = 'premium'
```

:::note
`ASSUMING` is parsed and validated but not yet applied to assembled context.
:::

## What you've learned

Target → population → filters → horizons → ranking → counterfactuals. Every
query you can write is validated against the schema before it runs, and its
shape determines the [task type](reference/task-types). Continue with the
[reference](reference/query-structure) or the [cookbook](cookbook).
