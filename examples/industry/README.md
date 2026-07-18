# Industry Examples

Runnable, self-checking examples targeted at common industry use cases,
modeled on the Kumo docs' example library. Each generates synthetic data with a **planted
signal**, runs a PQL query through the full pipeline (parse → validate →
retriever hop loop → temporal guard → scoring), and **asserts** the
predictions recover the signal.

The examples import pandas themselves and explicitly declare their schemas.
`pandas_connector.py` is application-side sample code that translates frames
to `Row` objects and wires retrievers; it is not shipped in the Python package.

Run with the Python library's venv:

```bash
cd relativedb/examples/industry
../../python/.venv/bin/python growth_churn.py
```

| Example | Industry | PQL pattern | Checks |
|---|---|---|---|
| `growth_churn.py` | Subscription / streaming | `PREDICT COUNT(events.*, 0, 30, days) = 0 … WHERE COUNT(events.*, -90, 0, days) > 0` | fading users score ≫ engaged; long-inactive users excluded by WHERE |
| `fraud_chargeback.py` | Payments | `PREDICT COUNT(chargebacks.*, 0, 60, days) > 0` | all 8 planted abuser accounts recovered in top-8; clean accounts ≈ 0 |
| `bizops_demand_forecast.py` | Retail | `PREDICT SUM(sales.qty, 0, 7, days) FORECAST 4 TIMEFRAMES` | 4 timeframes/store; flagship ≫ outlet; plausible weekly magnitude |
| `pzn_buy_it_again.py` | Grocery / personalization | `PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 3` | habitual staple ranked #1 per customer (FK ranking via `Row.parents`) |

A Java counterpart of the churn example lives in the test suite:
`java/relativedb-core/src/test/java/com/relativedb/GrowthChurnExampleTest.java`
(runs with `./gradlew test`), demonstrating the retriever SPI + a
context-evidence baseline `ModelBackend`.

Scoring uses the libraries' history-baseline backends — transparent,
model-free stand-ins. Real RT checkpoints plug in through the `ModelBackend`
SPI without touching the examples' data wiring.
