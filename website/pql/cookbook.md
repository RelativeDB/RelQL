---
title: Cookbook
description: Real queries from the shared test corpus, by use case.
---

# Cookbook

Copy-paste starting points, drawn from the shared 44-query test corpus.

## Churn (binary classification)

```sql
PREDICT COUNT(transactions.*) OVER (30 DAYS FOLLOWING) = 0
FOR EACH customers.customer_id
WHERE COUNT(transactions.*) OVER (90 DAYS PRECEDING) > 0
```

Or state the same active-in-the-last-90-days filter with an existence test:

```sql
PREDICT NOT EXISTS(transactions.*) OVER (30 DAYS FOLLOWING)
FOR EACH customers.customer_id
WHERE EXISTS(transactions.*) OVER (90 DAYS PRECEDING)
RETURN PROBABILITY
```

## Spend / LTV slice (regression)

```sql
PREDICT SUM(transactions.price) OVER (30 DAYS FOLLOWING) FOR EACH customers.customer_id
```

## Recommendations (ranking)

```sql
PREDICT LIST_DISTINCT(transactions.article_id) OVER (30 DAYS FOLLOWING) RANK TOP 12
FOR EACH customers.customer_id
```

## Daily demand, 4 weeks out (forecasting)

```sql
PREDICT SUM(usage.count) OVER (1 DAY FOLLOWING HORIZONS 28)
FOR EACH accounts.account_id
```

The `HORIZONS 28` on the window makes this a 28-step forecast — one prediction
per day.

## Specific entities

```sql
PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id IN (42, 123)
```

## Counterfactual

```sql
PREDICT COUNT(orders.*) OVER (90 DAYS FOLLOWING) = 0 FOR users.user_id = 42
ASSUMING users.plan = 'premium'
```

## Status prediction (string predicate)

```sql
PREDICT LAST(loan.status) OVER (30 DAYS FOLLOWING) NOT LIKE '%DENIED' FOR EACH loan.id
```

## Missing-attribute prediction (static target)

```sql
PREDICT articles.description IS NULL FOR EACH articles.id
```

## Population carve-outs

```sql
PREDICT SUM(transactions.value) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING) > 100
FOR EACH customers.customer_id
WHERE customers.location NOT IN ('ALASKA', 'HAWAII')
```

## As-of a fixed anchor, with quantiles

```sql
PREDICT SUM(orders.amount) OVER (RANGE BETWEEN 15 DAYS FOLLOWING AND 45 DAYS FOLLOWING)
FOR customers.customer_id IN ('C7', 'C9')
AS OF :prediction_time
RETURN QUANTILES (0.10, 0.50, 0.90)
```

## Reusable named window

```sql
PREDICT SUM(orders.revenue) OVER w - SUM(orders.cost) OVER w
FOR EACH customers.customer_id
WINDOW w AS (30 DAYS FOLLOWING)
```
