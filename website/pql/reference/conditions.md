---
title: Conditions & operators
description: Comparisons, boolean logic, and string predicates.
---

# Conditions and operators

Conditions appear in three places: comparing the target
(`PREDICT COUNT(...) = 0`), filtering entities (`WHERE`), filtering aggregated
rows (inline `WHERE` inside an aggregation), and stating counterfactuals
(`ASSUMING`).

## Comparison operators

`=` `==` `!=` `>` `>=` `<` `<=`

## Boolean composition

`AND`, `OR`, `NOT`, with parentheses.

## Membership and null tests

```sql
customers.location IN ('NY', 'CA')
customers.location NOT IN ('ALASKA', 'HAWAII')
articles.description IS NULL
articles.description IS NOT NULL
```

## String predicates

```sql
loan.status LIKE '%DENIED'        -- SQL % wildcards
movie.title STARTS WITH 'The'
movie.title ENDS WITH 'Returns'
movie.title CONTAINS 'Star'
```

## Examples

```sql
-- entity filter mixing a static attribute and a past-facing aggregation
WHERE customers.age >= 18 AND COUNT(orders.*, -90, 0, days) > 0

-- predicate target: multiclass-style question on a status column
PREDICT LAST(loan.status, 0, 30) NOT LIKE '%DENIED' FOR EACH loan.id
```
