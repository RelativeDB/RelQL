# polymarket-news — can a frozen relational transformer read the tape?

Live prediction-market prices from [pmxt](https://pmxt.dev) and live headlines
from a dozen newsroom RSS feeds, joined into a seven-table database, and one
RelQL query:

```sql
PREDICT SUM(price_ticks.ret) OVER (6 HOURS FOLLOWING) > 0
FROM markets
WHERE markets.market_id IN :ids
RETURN PROBABILITY
```

*"For every market, what is the probability its YES price is higher six hours
from now?"* `ret` is the hourly change in price, so the sum of the next six
hourly returns is the six-hour move, and `> 0` is "it repriced upward".

Nothing is trained. RT-J is frozen, no head is fitted, no features are
engineered — the tables are the raw records from the two sources, the links
say how they relate, and the model discovers the rest.

## Why this task is hard

A prediction market's price *is* a forecast, aggregated over people with money
at stake. Its six-hour direction is close to a coin flip by construction: if
it were predictable from the price history alone, someone would have traded it
away. Anything above chance has to come from information the price has not
absorbed yet — which is what the news tables are there to supply, and what the
ablation measures.

## The database

```
events ──< markets ──< price_ticks
   │           │
   │           └──< market_tags >── tags
   │
   └──< news_mentions >── news_articles
```

| table | rows are | time column |
|---|---|---|
| `events` | a Polymarket event ("Israel x Iran ceasefire continues through…?") | — |
| `markets` | one binary question inside an event | — |
| `price_ticks` | one hourly candle of the YES outcome | `ts` |
| `tags` / `market_tags` | Polymarket's topic labels | — |
| `news_articles` | one headline: text, outlet, publication time | `published_at` |
| `news_mentions` | this headline shares wording with this event | `observed_at` |

Every fact carries the timestamp at which it became true, so the engine's
temporal bound cuts the database at the anchor and a prediction can never read
a candle or a headline from its own future. `markets` deliberately carries no
volume, liquidity or current price: those are read at fetch time, which is
*after* the anchor, and a backtest may not look at them.

## Protocol

- **Universe** — the most-traded active Polymarket markets, priced between
  0.06 and 0.94 (a market pinned at 0.99 has nothing left to predict),
  resolving more than 24h out (so the label is repricing, not resolution),
  with sports and esports excluded (those reprice on a scoreboard, not on the
  newswire).
- **Anchor** — six hours before the last closed hourly candle. Everything
  after it is held out, prices and headlines alike.
- **Label** — the realized sign of the move from the anchor price to the price
  six hours later, computed from candles the model never sees.
- **Baselines** — momentum (the last six hours' move), distance from 0.5 (the
  mean-reversion prior), and a constant, whose accuracy is just the up-rate of
  the window.
- **Ablation** — the identical query against the identical database with the
  two news tables removed. RelQL has an `ABLATE TABLE` clause for exactly
  this; the engine does not implement it yet, so `db.build(with_news=False)`
  drops the tables instead.

## A run

Snapshot of 2026-07-24 20:16 UTC — the afternoon Brent touched $100 and gave
it back on reports of revived US–Iran contacts, which is why the universe is
mostly Iran, Israel, oil, the Fed and crypto. Anchor 14:00 UTC, horizon 6h,
45 markets (22 up / 23 down), 4,201 hourly candles, 141 headlines of which
the 39 published before the anchor are the only ones the model can see.

| signal | accuracy | AUROC | Brier | log loss |
|---|---:|---:|---:|---:|
| momentum (last 6h move) | 0.511 | 0.525 | 0.312 | 1.378 |
| distance from 0.5 | 0.511 | 0.477 | 0.327 | 0.933 |
| always up (p=0.5) | 0.489 | 0.500 | 0.250 | 0.693 |
| **RT-J (markets + news)** | **0.600** | **0.704** | **0.236** | **0.667** |
| RT-J (markets only, ablated) | 0.600 | 0.670 | 0.236 | 0.665 |

The price-only baselines land where theory says they should: on a liquid
market, the recent move tells you almost nothing about the next one. The
frozen model, having never seen a prediction market, ranks the six-hour
direction well above chance.

**The news gap does not survive scrutiny.** Every context in that run hit the
2,048-cell budget, so which rows reached the model depended on the cell
budget. Re-run with `--context-cells 8192` and the two arms converge:

| signal | accuracy | AUROC | Brier | log loss |
|---|---:|---:|---:|---:|
| RT-J (markets + news), 8192 cells | 0.600 | 0.666 | 0.235 | 0.663 |
| RT-J (markets only, ablated), 8192 cells | 0.600 | 0.666 | 0.238 | 0.668 |

So the +0.034 above is a context-budget artifact, not evidence that the
headlines helped. What survives both budgets is the model beating the
price-only baselines by a wide margin — and even that is one snapshot of
**45 markets**, where the standard error on an AUROC near 0.67 is roughly
±0.09 and the markets are not independent (a dozen of them are Iran markets
moving on the same news). Settling either question takes many snapshots
across many days, which is what `fetch.py` is for.

The regression form of the same query — `RETURN EXPECTED VALUE` over the
same window — gets the direction right 27/45 (60%) but its MAE (0.060) is
slightly worse than predicting no move at all (0.058): it ranks well and is
miscalibrated in magnitude, the same pattern the sibling `rel-sentiment`
example shows for uncalibrated regression targets.

The report prints `(contexts truncated)=N` whenever contexts hit the cell
budget — at 2,048 all 45 do, and at 8,192 all 45 still do, because the
per-market history plus peer cohort is simply larger than either budget.
Treat any single-run gap smaller than the spread between budgets as noise.

## Run

```bash
./run.sh                 # venv, deps, fetch a snapshot, predict
./run.sh --refetch       # force new market + news data
```

or step by step:

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python fetch.py --markets 45 --news-hours 12   # -> data/snapshot.json
./.venv/bin/python predict.py
```

Requirements:

- **Python ≥ 3.10** on **macOS arm64 (Apple Silicon)**, where the `relativedb`
  wheel bundles the native inference engine. Elsewhere, build `librt_c` and
  point `RELATIVEDB_RT_LIB` at it.
- First run downloads the RT-J checkpoints (~350MB) and the MiniLM encoder
  into `~/.cache/huggingface`.
- `pmxt` self-hosted needs no API key — the SDK spawns a local `pmxt-core`.
  The RSS feeds need no key either.

## On the news source

The first version queried [GDELT](https://www.gdeltproject.org/), which is the
right shape for this (worldwide, keyword-queryable, ~15-minute granularity)
but throttles an unauthenticated caller into `429`s within a few requests —
enough to stall a fetch for half an hour. It is still available behind
`--gdelt`, with request spacing, backoff and an on-disk cache; the default
path is a dozen newsroom RSS feeds, which are minute-fresh and unmetered.

Reddit and Bluesky would be better still — crowd reaction leads the headline
more often than it follows — but both now return `403` to unauthenticated
clients (and Pushshift redirects to an auth wall), so neither can go in an
example that has to run without credentials. Reddit is reachable with an
OAuth script app; that is a natural next table if you have one.

Article-to-event linking is deliberately simple and local: an article attaches
to an event when their wording shares at least two content words, generic tags
like "Politics" excluded. It lets some noise through, and that is the honest
version of the signal — nobody hands you a labeled article-to-market mapping,
and hand-curating one would quietly solve the interesting part of the problem.

## A bigger version of the question

45 live markets and a six-hour horizon cannot settle anything. [`scale/`](scale/README.md)
runs the same idea on 2,671 settled contracts from two exchanges, with 933k
GDELT headlines and the on-chain trade tape behind them — 13 GB pulled from
Hugging Face and GDELT's bulk archive — and asks whether the model can beat
the market's own price at calling a resolution two to three days out, with a
sealed holdout split.

## Files

- `fetch.py` — snapshots markets, hourly candles and news into `data/snapshot.json`.
- `db.py` — the snapshot as a RelativeDB schema, links, rows and retrievers.
- `predict.py` — anchor, labels, baselines, the two RelQL queries, the report.
- `run.sh` — one-shot: venv, deps, fetch, predict.
