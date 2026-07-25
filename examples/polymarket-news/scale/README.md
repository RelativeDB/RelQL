# Study 2 — beating the crowd, 2,671 contracts at a time

The sibling study asks whether a frozen relational transformer can call a
six-hour price move on 45 live markets. It can't say much: 45 markets is an
anecdote, and a six-hour move is mostly noise.

This one asks a question with a real answer, on data big enough to answer it:

> Two to three days before a prediction market settles, can RT-J beat the
> market's own price at calling the outcome?

```sql
PREDICT markets.resolved_yes
FROM markets
WHERE markets.market_id IN :ids
RETURN PROBABILITY
```

Nothing is trained. The model is the same frozen checkpoint, no head is
fitted, and the only thing that varies between arms is which tables exist.

## The data — 13 GB pulled from four sources

| source | what came out of it |
|---|---|
| [SII-WANGZJ/Polymarket_data](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data) | 1.8M market records; 95M on-chain fills scanned out of a 1.03B-row, 37.5 GB tape |
| [TrevorJS/kalshi-trades](https://huggingface.co/datasets/TrevorJS/kalshi-trades) | Kalshi markets and fills — a second exchange, separate traders, same label |
| [GDELT 2.0 GKG](http://data.gdeltproject.org/gdeltv2/) bulk archive | 933k headlines with named people, organizations and tone, Oct 25 – Dec 31 2025 |
| MiniLM (bundled with `relativedb`) | 304k headline embeddings, for semantic article↔event links |

The fills file is the interesting fetch. It is sorted by time and every
parquet row group carries min/max statistics, so a two-month window is 92 of
1,027 row groups — 3.7 GB on the wire instead of 37.5 GB, and each group is
folded into hourly bars and discarded as it arrives.

GDELT's *API* throttles an unauthenticated caller into `429`s within a few
requests. Its *file archive* does not: 816 zipped CSVs at full speed.

## The database

```
events ──< markets ──< price_ticks
   │
   └──< news_mentions >── news_articles
```

| table | rows | what one row is |
|---|---:|---|
| `events` | 979 | a real-world question both venues might list |
| `markets` | 2,671 | one settled binary contract (1,7xx Polymarket, 1,1xx Kalshi) |
| `price_ticks` | 262,563 | one market-hour: VWAP, high/low, dollars, buy/sell split, distinct takers |
| `news_articles` | 25,094 | one headline: text, outlet, tone, people, organizations |
| `news_mentions` | 56,505 | this headline is about this event (18k lexical, 38k semantic) |

## Two rules that make it a backtest and not a fantasy

**Everything is stamped.** Every row carries the moment it became knowable and
the engine's temporal bound drops the rest. Market rows carry the *anchor*
(midnight, 48–72h before close), so a market's own future is invisible to it.

**Peer outcomes are masked until they settle.** This is the subtle one. A
contract that closes next week already exists in the table this week, with its
outcome sitting in a column. A timestamp filter alone does not catch it — the
row is old, the outcome is not. `build.py`'s retrievers strip `resolved_yes`
from any peer whose close time is after the anchor, so in-context examples are
only ever contracts that had genuinely finished.

Deliberately absent from `markets`: total volume, liquidity, final price.
Those are read from the settled record, which is after the anchor.

## Semantic links

Token overlap misses paraphrase. "Maduro" appears in no Venezuela market
title; "Fed holds rates steady" shares no distinctive word with "Will there be
a rate cut in December?". So candidate headlines are embedded with the same
MiniLM RT-J uses for its own text cells, and the top-40 per event above cosine
0.42 become links — 38,116 of them, on top of the 18k lexical ones.

This stays honest because the encoder is frozen and reads one headline at a
time, and because a link inherits the *article's* publication time. Similarity
decides whether two things are about the same subject; the timestamp still
decides when you were allowed to know it.

Embeddings are cached to `data/scale/embeddings/*.npz` — paid once.

## Splits

Fixed by settlement date before anything ran:

| split | settles | contracts |
|---|---|---:|
| dev | 2025-11-01 .. 2025-12-05 | 1,279 |
| **holdout** | 2025-12-06 .. 2025-12-31 | 1,392 |

Every protocol choice was made on dev — including the baseline. The first
version scored the price as "last VWAP before the anchor", which is weak on
thin contracts where the last fill is a stale dollar. Averaging the last 24
hours of tape lifts the baseline from 0.760 to 0.780 AUROC, so that is the
baseline: the strongest honest version of the thing being beaten.

Holdout was scored once, after all of it was frozen.

## Result: the crowd wins

**Dev** (1,279 contracts, 30.0% YES):

| signal | acc | AUROC | Brier | log loss |
|---|---:|---:|---:|---:|
| market price (24h vwap) | 0.726 | 0.780 | 0.170 | 0.497 |
| always NO (p=0.5) | 0.300 | 0.500 | 0.250 | 0.693 |
| RT-J (markets+tape+news) | 0.759 | 0.716 | 0.211 | 0.613 |
| RT-J (news ablated) | 0.752 | 0.686 | 0.213 | 0.622 |
| price × RT-J (logit mean) | 0.768 | **0.805** | 0.164 | 0.495 |

Blending RT-J into the price gained **+0.025 AUROC [+0.013, +0.036]** on a
paired bootstrap — an interval clear of zero. Better still, the gain looked
*interpretable*: on contracts where the crowd was unsure (price 0.2–0.8, n=773)
RT-J beat the price outright, 0.654 to 0.620, while the price dominated where
it was confident (0.925 to 0.828). A tidy story about a model earning its keep
where the market has least to say.

**Holdout** (1,392 contracts, 24.1% YES, scored once):

| signal | acc | AUROC | Brier | log loss |
|---|---:|---:|---:|---:|
| market price (24h vwap) | 0.728 | **0.748** | 0.171 | 0.503 |
| always NO (p=0.5) | 0.241 | 0.500 | 0.250 | 0.693 |
| RT-J (markets+tape+news) | 0.749 | 0.613 | 0.227 | 0.650 |
| RT-J (news ablated) | 0.723 | 0.561 | 0.230 | 0.661 |
| price × RT-J (logit mean) | 0.742 | 0.749 | 0.173 | 0.517 |

The gain went to **+0.002 [-0.010, +0.013]**. The uncertain-market story
reversed: 0.583 for RT-J against 0.603 for the price. Nothing survived.

### One thing did replicate: the news tables

| | dev | holdout |
|---|---|---|
| RT-J with news | 0.716 | 0.613 |
| RT-J, news ablated | 0.686 | 0.561 |
| **gain** | **+0.030 [-0.000, +0.060]** | **+0.052 [+0.014, +0.090]** |

Deleting `news_articles` and `news_mentions` costs the model real accuracy, in
both splits, and on the held-out one the interval clears zero. The headlines
are doing work — the model reads them and is better for it. It is just not
enough work to catch a market price.

This is worth separating from the headline result, because the two questions
are different. "Does the news help the model?" — yes, replicated. "Does the
model beat the crowd?" — no.

### Where the dev edge actually lived

| stratum | dev n | dev price / RT-J | holdout n | holdout price / RT-J |
|---|---:|---|---:|---|
| crypto strike ("BTC above $90k on Dec 30") | 362 | 0.499 / **0.633** | 244 | 0.479 / **0.463** |
| other | 714 | 0.864 / 0.774 | 892 | 0.808 / 0.675 |
| US politics | 168 | 0.855 / 0.670 | 147 | 0.759 / 0.703 |

Almost all of it was in crypto strike markets — the one stratum where the
price itself ranks outcomes no better than chance, so any noise looks like
skill. RT-J scored 0.633 there on dev and 0.463 on holdout: a coin flip that
landed heads once. In every stratum where the price *is* informative, RT-J
sits below it in both splits.

**Verdict: the news tables help the model (replicated), and the model still does not beat the market price at a two-to-three-day lead (replicated).**
That is the correct prior about liquid prediction markets, and the honest
output of the experiment. Without the holdout this would have been written up
as "+0.025 AUROC over the market" with a plausible mechanism attached.

What the study does establish is that the negative result is trustworthy: the
leakage controls were verified directly (at the 2025-12-10 anchor, 1,440
visible outcome cells, every one settled; 139 open contracts, zero outcomes
exposed; zero rows past the bound), and the baseline was strengthened, not
weakened, before the holdout was opened.

## Running it

```bash
python -m scale.pull_polymarket --start 2025-10-15 --end 2026-01-01
python -m scale.pull_kalshi     --min-volume 25000
python -m scale.pull_gdelt      --start 2025-10-25 --end 2026-01-01 --every 2
python -m scale.semantic        --top-k 40 --min-similarity 0.42
python -m scale.resolve --split dev     --limit 0 --arms news,nonews
python -m scale.analyze --split dev
```

`resolve.py` writes one parquet per arm and reuses it; `analyze.py` never
touches the model, so new baselines, blends and subgroups are free. Scoring
runs at about one contract per second at a 2,048-cell context.
