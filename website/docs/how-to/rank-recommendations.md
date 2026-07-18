---
title: Rank recommendations
description: Predict which items each customer will buy next.
---

# Rank recommendations

Goal: for each customer, the top 3 products they are most likely to order in
the next 30 days ("buy it again").

## The query

`LIST_DISTINCT` predicts a set of linked IDs; `RANK TOP K` turns it into a
ranking task:

```sql
PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 3
FOR EACH customers.customer_id
```

## Run it

```python
result = engine.execute(ExecutionInput(query=query, anchor_time=t0))
rankings = {p.id: p.ranked for p in result.predictions}
```

Here `engine` is already wired to your application-owned retrievers. The
result contains a ranked list of product IDs per customer. Note
`orders.product_id` is an FK — the ranking works over graph edges
(`Row.parents`), never over ID feature values.

## Notes

- `K` bounds the returned list, not the candidate set.
- Use `CLASSIFY` instead of `RANK TOP K` for a multilabel-style yes/no per
  item.
- A complete self-checking version (habitual staple ranked #1 per customer)
  lives at `examples/industry/pzn_buy_it_again.py`.
