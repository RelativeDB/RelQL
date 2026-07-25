# timing — when should you post tech content?

Short answer, from 21,474 posts across five months of the AI/tech community on
Bluesky:

> **Day of week doesn't matter. Hour of day barely matters. What you post beats
> when you post, by a wide margin.**

The one timing result that survives scrutiny is negative: posting into the
US-afternoon peak (14:00–17:00 Eastern) does *worse*, not better.

## The data

`fetch_history.py` reuses the 250-account community discovered in the parent
example and walks each author's feed back 150 days. It needs no engagement
*edges* — `getAuthorFeed` returns `likeCount` / `repostCount` / `replyCount` /
`quoteCount` for free on every post — so the whole crawl is 492 API calls
instead of tens of thousands.

| | |
|---|---:|
| posts | 21,474 |
| authors with any posts in range | 160 |
| span | 2026-02-25 → 2026-07-25 |
| replies | 65.4% |
| flagged tech | 29.2% |

## Two things that would have produced a wrong answer

**1. Author composition.** "Average likes by hour" mostly measures *which
authors post at which hours*. If the 40k-follower accounts post at 15:00 then
15:00 looks great, and that is useless to you. Every number here is a
**within-author residual**: `log1p(engagement)` minus that author's own mean, so
a positive value at 15:00 means *the same person* does better at 15:00 than they
usually do.

**2. Multiple comparisons.** A 24-cell hour table tested at 5% hands you roughly
one "significant" hour per run out of pure chance. The first version of this
analysis duly reported **Sunday +0.204, bootstrap interval excludes zero** — and
it evaporated on contact: the non-tech slice said Saturday instead, and the
combined slice said neither. So every table now leads with an **omnibus
permutation test** that shuffles residuals *within author*, holding timestamps
fixed, and asks whether the whole table's spread beats chance.

## Results

Omnibus p-values, top-level posts only (replies analysed separately — they are
65% of posts, get far less engagement, and are posted at different hours):

| slice | day of week | hour of day (ET) |
|---|---:|---:|
| tech | p=0.088 | p=0.047 |
| all posts | p=0.338 | p=0.101 |
| non-tech | p=0.345 | p=0.145 |

**Day of week: nothing, in any slice.** The eye-catching Sunday cell was noise.

**Hour of day: nothing that replicates.** Tech alone lands at p=0.047, which is
one test among nine in that table and would not survive any correction for it.
Its "best" cell is 01:00 ET off 27 posts, which is not advice anyone can act on.

### What actually moves engagement

Within-author contrasts, same permutation null (`*` = p<0.05):

| factor | tech | all posts |
|---|---:|---:|
| long text (>200 chars) | **+17%** \* | **+32%** \* |
| very short (<40 chars) | −26% | **−28%** \* |
| has an image | +6% | **+14%** \* |
| **posted 14:00–17:00 ET** | **−12%** \* | **−8%** \* |
| weekend | +9% | +7% \* |
| has a link | −6% | −2% |
| has video | −9% | +7% |

Text length is the dominant effect and it replicates across both slices at
p=0.000. The afternoon-peak penalty replicates too, in the direction opposite to
the folk wisdom — plausibly competition, since that is when everyone posts.

**Actionable version:** write a substantial post, put an image on it, don't
bother scheduling it, and if anything avoid mid-afternoon Eastern.

## The causal question, and the instrument that failed

Everything above is correlational. "Posts made at 15:00 do worse" is consistent
with *posting at 15:00 costs you reach* and with *the posts people dash off at
15:00 are weaker*. A within-author residual controls for **who** posted, not for
**what** they posted.

RelQL has the clause that should settle it:

```sql
PREDICT focal_posts.engagement
FROM focal_posts
WHERE focal_posts.focal_post_id IN :ids
ASSUMING focal_posts.hour_bucket = '20-23'
RETURN EXPECTED VALUE
```

Hold the post — text, author, links, length — completely fixed, intervene on one
column, re-score the same posts, sweep the intervention across every bucket.

The sweep returned a flat curve: 2.5% spread across hour buckets, 1.3% across
days. That looks like a clean confirmation of "timing doesn't matter" — and it
is worth nothing, because the **positive controls are flat too**:

```
intervention   mean pred   vs baseline
 (as posted)        47.0        --
 text_length=20     49.7      +5.7%
 text_length=120    49.7      +5.7%     <- identical
 text_length=300    49.7      +5.7%     <- identical

 has_image=TRUE     46.7      -0.7%
 has_image=FALSE    46.7      -0.7%     <- identical
```

Intervening on columns the descriptive analysis says move engagement by 14–32%
changes the prediction by **exactly zero**. Narrowed down:

- It is not the query. `EXPLAIN CONTEXT` shows the plan carrying
  `focal_posts.is_reply := True` and `:= False` correctly.
- It is not `_apply_assumptions` overwriting sibling evidence rows. That is a
  real hazard — the engine writes an assumed value onto *every* context row of
  the named table, so `ASSUMING posts.x = v` flattens the column across the
  whole context — and `focal_posts` exists in this schema specifically to avoid
  it. Splitting the table did not change the result.
- `ASSUMING is_reply = TRUE` and `= FALSE` return **byte-identical per-post
  predictions**, while both differ from the no-`ASSUMING` baseline. So the
  presence of the clause changes the prediction and its **value is inert**.
- No `AssumptionNotAppliedWarning` is raised. It fails silently.

Reproduce with `python predict_timing.py --sweep length` (and `--no-split` for
the sibling-flattening variant). Until that is fixed, **the counterfactual
numbers in this directory carry no information** and the timing conclusions rest
entirely on the permutation tests.

The positive control is the only reason this is known. A flat timing curve with
no control would have read as a satisfying causal confirmation of the
correlational finding.

## The tech classifier, and how wrong it is

`tech.py` is a lexicon. It was validated by hand-labelling a random sample the
lexicon was *not* tuned on:

- **precision ≈ 79%**, **recall ≈ 61–79%** (the range is how borderline cases
  are judged)

Representative errors — false positives: a Bradbury literary post, a purely
social reply, generic praise for a talk. False negatives: *"Turn this novel into
a game"* (generative AI, no keywords), Gmail sender-impersonation (security),
a *Designing Data-Intensive Applications* podcast plug. Guards were added so
`model` only counts next to `language`/`world`/`reward`/etc., and `security`
only in a computing sense — the White House *physical* security briefing was a
false positive before that.

This matters for the headline: the tech slice is ~20% contaminated and misses a
third of real tech. That is survivable here only because the main conclusions
are *null* results and hold on the unfiltered slice too. Do not quote a tech-vs-
non-tech difference from this classifier.

The better version is `PREDICT posts.topic FROM posts WHERE posts.topic IS NULL`
— the auto-labelling pattern from the top-level README, which reads text instead
of matching it. It is not used here because there is no hand-labelled topic set
to score it against, and an unevaluated classifier is decoration.

## Caveats

- **Saturation.** Counts are read at fetch time, so recent posts are still
  accumulating. Posts newer than 14 days are dropped (4,948 of them), not
  decayed.
- **Timezone.** Reported in US/Eastern and UTC. This community is US-heavy but
  not US-only, and that is an assumption.
- **Survivorship.** Only 31 authors clear "≥25 tech top-level posts in range",
  median 8,880 followers. This is a mid-sized-account result, not advice for a
  new account with 50 followers.
- **Deleted posts** are invisible, so anything that flopped hard and got removed
  is missing from the denominator.
- **The null is not proof of absence.** With this sample, an hour-of-day effect
  smaller than roughly ±15% would not be detected. The claim is "no effect large
  enough to be worth scheduling around", not "exactly zero".

## Running it

```bash
python fetch_history.py --days 150            # ~2 min, ~500 public API calls
python analyze_timing.py --scope tech --kind top
python analyze_timing.py --scope all  --kind top
python predict_timing.py --sweep length       # the failing positive control
```

| file | what it does |
|---|---|
| `fetch_history.py` | 150 days of author feeds, counts only |
| `tech.py` | the tech lexicon, with its measured precision/recall |
| `analyze_timing.py` | within-author residuals + omnibus permutation tests |
| `predict_timing.py` | the RelQL `ASSUMING` counterfactual, and its controls |
