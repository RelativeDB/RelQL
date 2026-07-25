# Study 3 — does chatter move a prediction market?

A concise experiment built around one ablation. The population is *posts*, not
markets:

```sql
PREDICT comments.price_up
FROM comments
WHERE comments.comment_id IN :ids
RETURN PROBABILITY
```

`price_up` is the sign of the market's move over the six hours after the post
appeared. Both arms of the experiment see the poster, the channel, the
timestamp, the market and its full hourly tape. **Only one of them sees the
words.** Whatever separates them is what the sentence was worth — the cleanest
question you can ask a model that encodes sentences.

## Data — no scraping, ~2.6 GB

| source | what came out of it |
|---|---|
| [mitanshugoel/reddit-2025](https://huggingface.co/datasets/mitanshugoel/reddit-2025) | 120,151 topical comments (Nov 1–3), filtered live out of 21.5M |
| [open-index/hacker-news](https://huggingface.co/datasets/open-index/hacker-news) | 587,509 posts and comments, Nov 1 – Dec 31 |
| trade tapes from `scale/` | 485,513 hourly bars across 1,238 contracts |

A Reddit month is 37 GB of zstd, which the budget could not hold — but the dump
is written in timestamp order and the Hub honours HTTP range requests, so a
2.5 GB prefix yields a contiguous slice from the 1st onward. Filtering to
topical subreddits happens while the stream decodes; 96 MB lands on disk.

Consequence worth stating: **Reddit covers Nov 1–3 only**, so it sits entirely
in dev, while Hacker News spans both splits.

Polymarket's own comments are not on the Hub at all (its public JSON API has
them, and `fetch_comments.py` will pull them if you want that instead).
`ElKulako/stocktwits-crypto` looks ideal until you open it — 1.9M crypto posts
with no timestamps, so nothing can be joined to a price.

## Two graphs

The first version attached a post to a market when they shared two distinctive
words, pointing at the *event* ("What price will Bitcoin hit?") rather than the
market ("Bitcoin above $90,000 on Dec 30"), then let a walk pull whatever fit
in the context. The rich version (`links.py`) changes three things that matter:

1. **Attachment by meaning** — MiniLM cosine against the market question, with
   the similarity carried on the row so the model can tell 0.75 from 0.45.
2. **A subject bridge** — `subjects` are words markets and posts share
   ("hormuz", "khamenei", "shutdown"); a post reaches a market *through the
   thing it is about*, and can reach other posts on that subject two hops out.
3. **Hourly rollups** — `chatter_hours` makes a burst one cell the walk always
   finds, instead of a pattern it must reconstruct from posts it may not sample.

Thread grouping was intended as a fourth change and **did not happen**: the
chatter pull did not keep Reddit's `link_id` or HN's `parent`, so every post
collapsed into one of two "solo" buckets (`threads=2` in the build output).

## Result: the words are worth nothing here

Gain in AUROC from un-muting the text, paired bootstrap over posts:

| | lexical graph | rich graph |
|---|---|---|
| dev | +0.001 [−0.009, +0.011] | +0.007 [−0.006, +0.019] |
| holdout | −0.002 | +0.001 [−0.008, +0.009] |
| well-matched only (cos ≥ 0.5) | — | dev 0.684 vs 0.674, holdout 0.604 vs **0.609** |

Better links did not rescue the text. The subgroup built to give it its best
shot — posts whose embedding matches the market question unambiguously — gains
0.010 on dev and loses 0.005 on holdout. That is noise, not a signal being
throttled by plumbing.

### And the control beats the model

| | lexical dev | lexical holdout | rich dev | rich holdout |
|---|---:|---:|---:|---:|
| always up | 0.500 | 0.500 | 0.500 | 0.500 |
| momentum | 0.356 | 0.307 | — | — |
| **price level (1 − price)** | 0.658 | **0.718** | **0.693** | **0.678** |
| RT-J, text | 0.737 | 0.593 | 0.640 | 0.565 |
| RT-J, muted | 0.736 | 0.595 | 0.633 | 0.564 |

The pilot's headline (0.72 AUROC on six-hour direction) was not the model
reading chatter. It was the price *level*: a contract at 0.03 has almost no
room to fall, so `1 − price` predicts the sign of the next move — and out of
sample it does so **better than the model**. RT-J's dev→holdout collapse
(0.737 → 0.593) while the trivial rule improves (0.658 → 0.718) is the
signature of fitting composition rather than signal.

## Reading it against study 2

Headlines measurably helped predict *resolution* over days
(+0.052 AUROC [+0.014, +0.090] on held-out data). Chatter does nothing for
*price* over hours. The consistent reading is that a market absorbs public
text faster than a six-hour window can measure, and what remains is noise.

The most promising next swing is therefore not more link engineering but a
**longer horizon** — the same ablation at 24–72h, where text was already shown
to carry information.

## Caveats

- Reddit is dev-only (Nov 1–3); Hacker News spans both splits.
- The prefilter requires three shared rare words before a post is embedded, so
  pure paraphrase — the case embeddings exist to catch — never reaches the
  encoder.
- The two graphs are not a clean A/B: semantic attachment is stricter, so the
  rich arm scored 4,784 labelled posts (900 per split) against the control's
  12,000 (1,200 per split).
- Every context was truncated, at 1,024 cells and at 512.

## Running it

```bash
python -m comments.pull_chatter --month 2025-11 --reddit-gb 2.5
python -m comments.chatter_task --horizon 6 --per-split 1200   # lexical links
python -m comments.rich_task    --horizon 6 --per-split 900    # rich links
```

Embeddings cache to `data/scale/embeddings/`, predictions to
`data/chatter/*.parquet`, and the rich arm scores in checkpointed chunks — a
run that dies at 80% keeps its 80%.
