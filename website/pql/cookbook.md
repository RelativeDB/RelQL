---
title: Cookbook
description: Real queries from the shared test corpus, by use case.
---

# Cookbook

Copy-paste starting points, drawn from the shared 44-query test corpus.

## Churn (binary classification)

```sql
PREDICT COUNT(transactions.*, 0, 30, days) = 0
FOR EACH customers.customer_id
WHERE COUNT(transactions.*, -90, 0, days) > 0
```

## Spend / LTV slice (regression)

```sql
PREDICT SUM(transactions.price, 0, 30) FOR EACH customers.customer_id
```

## Recommendations (ranking)

```sql
PREDICT LIST_DISTINCT(transactions.article_id, 0, 30) RANK TOP 12
FOR EACH customers.customer_id
```

## Daily demand, 4 weeks out (forecasting)

```sql
PREDICT SUM(usage.count, 0, 1, days) FORECAST 28 TIMEFRAMES
FOR EACH accounts.account_id
```

## Specific entities

```sql
PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR users.user_id IN (42, 123)
```

## Counterfactual

```sql
PREDICT COUNT(orders.*, 0, 90, days) = 0 FOR users.user_id = 42
ASSUMING users.plan = 'premium'
```

## Status prediction (string predicate)

```sql
PREDICT LAST(loan.status, 0, 30) NOT LIKE '%DENIED' FOR EACH loan.id
```

## Missing-attribute prediction (static target)

```sql
PREDICT articles.description IS NULL FOR EACH articles.id
```

## Population carve-outs

```sql
PREDICT SUM(transactions.value, 15, 45, days) > 100
FOR EACH customers.customer_id
WHERE customers.location NOT IN ('ALASKA', 'HAWAII')
```
