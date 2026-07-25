# ai-discourse — what's worth reading, when you can't trust the likes

The question this started from: *"I want to know which recent posts matter to
the new AI discussions, and I can't just sort by likes and reposts."*

He's right, and the data says why. Across a day of AI Bluesky, the median post
gets **83% of its engagement from accounts that already follow the author**, and
author follower count alone rank-correlates **+0.66** with how much engagement a
post gets. Sorting by likes mostly sorts by audience size. It tells you who has
a big following, not who said something.

So this example predicts something else:

```sql
PREDICT COUNT(engagements.* WHERE engagements.from_follower = FALSE)
        OVER (48 HOURS FOLLOWING) >= 1
FROM posts
WHERE posts.post_id IN :ids
RETURN PROBABILITY
```

*"For every post from the last day and a half, what is the probability it gets
picked up by somebody who wasn't already listening?"*

That is the thing a large account cannot buy with follower count, and it is
only expressible because the database carries the **follow graph** next to the
**engagement graph**. You cannot compute it from a post's own columns, which is
exactly the case relational transformers are for.

Nothing is trained in the headline run: RT-J is frozen, no head is fitted.

### What this example actually shows

Read the [Results](#results) before borrowing anything from here. In one
sentence: **the target is the good idea; the frozen model is not yet the way to
predict it.**

- The premise holds up under measurement — likes really are largely an echo of
  audience size.
- Framing the target in graph terms is worth about **+0.10 AUROC** to the same
  frozen model over asking it the naive "will this get engagement" question.
- Handing RT-J the raw `follows` table on top of that is worth **nothing**
  measurable.
- RT-J (0.68) does **not** beat a one-line heuristic — early engagement per hour
  (0.89) — and light fine-tuning makes it worse, not better.

A negative result on a well-posed question is more useful than a positive one on
a leaky question, and getting the temporal handling right is most of the work
here (one real leak found and closed, one assumption stated). That is what this
example is for.

## Why Bluesky and not Twitter

The task needs to know *who* engaged, not how many. Public Twitter dumps give
you `likes: 43` — a scalar with the identities stripped out. The largest one on
Hugging Face, `enryu43/twitter100m_tweets`, has columns
`user, id, tweet, replies, retweets, likes, quotes, date`: no reply edges, no
retweeter IDs, no follow graph. It hands you precisely the numbers the question
distrusts and nothing to correct them with.

Bluesky's AT Protocol serves the graph, unauthenticated and without an API key:
`getLikes` and `getQuotes` return the actor **and a timestamp per engagement**,
`getPostThread` returns repliers, and `getFollows` returns the follow graph.
That is the whole reason the example lives here. (`searchPosts` is *not* public
— it 403s without a session — so the universe is built from the follow graph
instead, which is more on-theme anyway.)

The AI research community is genuinely on Bluesky: the snapshot's 250 accounts
were discovered by following the graph out from 14 hand-checked seeds, and it
comes back full of working researchers.

## The database

```
accounts ──< posts ──< engagements
    │                       │
    │                       └── (actor_id) ──> accounts
    │
    └──< follows >── accounts
```

| table | rows are | time column |
|---|---|---|
| `accounts` | a Bluesky account — both posters and engagers, in one table | — |
| `posts` | one post: text, language, has-link, has-image | `created_at` |
| `engagements` | *this account* liked / quoted / replied to *this post*, then | `at` |
| `follows` | *this account* follows *this account* | — |

`accounts` deliberately holds both sides of every interaction. That is what
makes the interesting join expressible: from a post you reach its author, from
the author you reach everyone following them, from the post you reach everyone
who engaged — and whether those two sets overlap is the entire question. RT-J
walks it without being told to.

There is **one** engineered column, `engagements.from_follower`. It is derived
from the follow graph rather than from the post, it is only ever set on rows the
temporal bound admits, and the same test applied to the held-out window is the
label. Everything else is raw: no rolling means, no sentiment scores, no TF-IDF
between post and profile. Post text goes in as text, because the model is good
at text.

## Protocol

- **Universe** — the 211 posts made by community members in the 36 hours before
  the anchor. This is "posts from the last day or so", the same thing you would
  be triaging in a feed.
- **Anchor** — 48 hours before the fetch. Everything after it is held out.
- **Label** — engagements landing in the 48 hours *after* the anchor from
  accounts that do not follow the author. Base rate 25.6% (54/211): about a
  quarter of posts reach anybody new at all.
- **Threshold** — `K=1`, chosen as the median of the held-out count plus one. It
  is a single global constant, reported with its base rate, not a per-post
  choice.
- **What the model sees** — the post, its first hours of engagement, every
  author's recent track record with *their* engagement, and the follow graph.
  Every fact is timestamped and the engine's temporal bound cuts the database at
  the anchor, so a prediction can never see the engagement it is predicting.

### One leak found and closed

The engine treats a row with no timestamp as *static* — always admitted, at
every anchor. `getRepostedBy` returns no per-repost time, so reposts would have
been visible to the model at anchors *before they happened*. They are dropped
(5,100 of 28,105 edges). Likes, quotes and replies all carry their own
timestamps and survive.

One assumption remains, and it is stated rather than hidden: `getRelationships`
reports the follow edge as it stands **now**, not as it stood at the anchor.
Follows only accumulate, so somebody who saw a post, liked it, and *then*
followed reads as a pre-existing follower here. That makes "reached a
non-follower" an undercount, never an overcount — the bias runs against the
signal, so a positive result is not an artifact of it.

## The premise, measured

`python analyze.py` — no model involved:

```
== 1. is engagement just an echo of audience size? ==
   spearman(author followers, engagement so far) = +0.660
   spearman(author followers, reach past that audience) = +0.325

== 2. where does engagement come from? ==
   per post, share of engagement from EXISTING followers: median 83.3%, mean 76.5%
   across the whole snapshot: 55.8% of 20711 (author, engager) pairs already followed

== 3. would sorting by likes have found the posts that travelled? ==
   top-20 by early likes vs top-20 by held-out reach: 11/20 overlap
   spearman(early likes, held-out outside reach) = +0.448
```

Engagement tracks audience size at +0.66; reach past that audience tracks it at
only +0.33. The two are different questions, and likes answer the wrong one:
sorting by likes recovers just over half of the posts that actually travelled.

On the AI-only slice (`--ai-only`, 56 of 211 posts) the split is sharper still —
follower correlation rises to **+0.73** and **8 of the top 20 most-liked AI
posts reached nobody new at all**.

## Results

**The frozen model loses to a one-line heuristic.** 211 posts, 54 positives,
8,192-cell contexts:

| signal | acc | AUROC | Brier | log loss |
|---|---:|---:|---:|---:|
| early engagement **rate** (per hour) | 0.763 | **0.894** | 0.179 | 0.541 |
| early NON-follower engagement | 0.791 | 0.786 | 0.165 | 0.510 |
| early likes (engagement so far) | 0.815 | 0.771 | 0.146 | 0.462 |
| author follower count | 0.749 | 0.690 | 0.196 | 0.594 |
| RT-J (ablated: no `follows` table) | 0.735 | 0.688 | 0.209 | 0.611 |
| RT-J (full graph) | 0.744 | 0.676 | 0.211 | 0.614 |
| RT-J (naive `volume` query, no graph) | 0.744 | 0.579 | 0.211 | 0.614 |
| constant (p=0.5) | 0.256 | 0.500 | 0.250 | 0.693 |

Read honestly, three things are true and they point in different directions.

**1. Posing the problem in graph terms is worth ~0.10 AUROC.** The bottom RT-J
row is the same model on a graph-free database asking `COUNT(engagements.*)` —
"will this get engagement" — scored against the same labels. It gets 0.579.
Asking *reach past the author's audience* against a database that carries the
follow relation takes the same frozen model to 0.676.

That arm changes two things at once (the query *and* the database), so the
credit belongs to the pair, not to the wording alone — and point 2 below narrows
it further: since the raw edges add nothing, what is actually earning the 0.10 is
the single `from_follower` column and the question it makes expressible.

**2. The raw `follows` table is not.** Removing it changes nothing
(0.688 vs 0.676 — the ablation arm is nominally *higher*, which at n=211 is
noise). All the graph signal is arriving through the one derived column,
`from_follower`; RT-J is not getting anything extra out of walking 11,689 raw
edges. If you only take one design lesson from this example, that is it.

**3. RT-J does not beat the baselines, and more context does not fix it.**
Every one of the 211 contexts truncated even at 8,192 cells, so the obvious
suspicion is the retrieval budget rather than the model. `sweep.py` tests it on
a fixed 60-post subsample:

> **Correction.** The runs in this section carry a misconfiguration. The
> reference RT-J evaluation (`rt.eval_utils.build_evaluator`) uses
> `local_ctx_size=256, bfs_width=32, num_walks=10_000, walk_length=20` at
> `ctx_size=8192`; this example set `local_context_cells = context_cells // 2`
> = 4,096, sixteen times the reference, spending half the budget on rows
> adjacent to the target before any walk-ranked row was admitted. The graph
> reach itself was never the problem — `ReferenceTraversal` builds its walk
> graph by *unbounded* BFS over the reachable component, so `max_hops` does not
> cap it, and the 10,000 walks of length 20 were running at reference defaults
> throughout. Corrected, on the same 211 posts (`sweep_local.py`):
>
> | `local_context_cells` | AUROC | acc | Brier |
> |---:|---:|---:|---:|
> | 256 (reference) | **0.700** | 0.749 | 0.212 |
> | 1,024 | 0.692 | 0.739 | 0.210 |
> | 4,096 (the runs below) | 0.676 | 0.744 | 0.211 |
>
> Worth +0.024 AUROC and it does not change the conclusion, so the tables below
> are left as they were run rather than silently restated.

```
bar to clear (best baseline): 0.869

  cells   bfs  hops   auroc     acc   trunc
   2048    32     3   0.648   0.783  60/60
   8192    32     3   0.710   0.783  60/60
   8192     8     2   0.697   0.783  60/60
   8192    64     2   0.732   0.783  60/60
  16384    16     3   0.712   0.783  60/60
```

More context helps (0.648 → 0.710) and then plateaus around 0.71–0.73, still
well short of 0.869 — and contexts truncate in *every* configuration, including
16,384 cells. The predicted probabilities bunch into a 0.469–0.508 band: this is
close to a constant predictor with a faint ordering on top, not a model that has
found the signal and been let down by its budget.

### Why the baselines are so strong here

`K=1` asks "did this reach *anybody* new", and that is close to asking "was this
post popular at all" — which early engagement measures directly. Engagement rate
per hour wins because it normalizes by post age, and posts in the universe are
between 0 and 36 hours old at the anchor. The premise ("likes mislead") survives
in the places the analysis measured it — ranking posts by *how far* they
travelled (Spearman +0.448), and the AI-only slice where 8 of the 20 most-liked
posts reached nobody new — but at this threshold, on this snapshot, early
engagement is a genuinely good predictor and the honest result is that the
frozen model does not beat it.

### Fine-tuning

`fit_head` refuses this target — frozen-backbone fitting supports multiclass and
ranking adapters, and this is scalar binary — so `finetune.py` uses
`Engine.finetune`, which differentiates through the whole checkpoint. Training
examples come from the same snapshot strictly earlier in time: each of 216
earlier posts at its *own* anchor (creation + 18h, the median age of a test
post), with every training window required to close before the test anchor.
Both arms run at 2,048 cells so the comparison is like for like.

It makes things worse:

```
448 steps over 224 examples in 2349s, loss 3.8933 -> 0.0000
```

| signal (2,048 cells, same 211 test posts) | acc | AUROC | Brier | log loss |
|---|---:|---:|---:|---:|
| early engagement rate (per hour) | 0.763 | **0.894** | 0.179 | 0.541 |
| RT-J zero-shot | 0.744 | 0.638 | 0.219 | 0.631 |
| RT-J fine-tuned | 0.725 | 0.570 | 0.212 | 0.616 |

A training loss of **exactly 0.0000** is the whole story: 224 examples against a
full checkpoint is memorization, and held-out AUROC drops accordingly (0.638 →
0.570). This is not evidence that fine-tuning cannot work on this task — it is
evidence that 216 posts from one 3-week window of one community is not a
training set. The training posts are also a biased sample: `fetch.py` keeps each
author's few most recent posts before the universe window, so prolific authors
dominate.

What it would take: many anchors across many days, which is what re-running
`fetch.py` on a schedule accumulates. That is the same conclusion the sibling
`polymarket-news` example reaches about its own single snapshot.

### The candidates

Top of RT-J's ranking, against what sorting by likes would have handed you.
`outside` is the held-out truth — how many accounts that don't follow the author
engaged in the next 48 hours:

```
== the candidates: top posts RT-J picked ==
  HIT  p=0.508  outside=7   likes_so_far=156  @minimaxir      gotta go fast
  miss p=0.505  outside=0   likes_so_far=46   @danhon.com     The future of television is trillions of hours of slop microdramas…
  HIT  p=0.491  outside=4   likes_so_far=31   @natolambert    If you're looking for the latest adoption data on open models in US v China…
  HIT  p=0.475  outside=30  likes_so_far=142  @alexhanna      "We found that at least 150 institutions of higher ed... are using Flock's…
  HIT  p=0.470  outside=51  likes_so_far=149  @histoftech     furiously tapping sign
  HIT  p=0.469  outside=118 likes_so_far=101  @timnitgebru    Bernie hasn't tried this on here yet but here he is doing free marketing…

== what sorting by early likes would have given you ==
  HIT  likes_so_far=271  outside=10  @timnitgebru    You gotta hand it to OpenAI, billing this as a *partnership*…
  HIT  likes_so_far=251  outside=4   @alondra        The report "calls for the government to invest in young researchers"…
  HIT  likes_so_far=242  outside=1   @tedunderwood   Just connected Claude Code to Gmail, so that's it for me…
  HIT  likes_so_far=211  outside=8   @simonwillison  I wrote about the completely wild incident where OpenAI were testing…
  miss likes_so_far=149  outside=0   @melaniemitchell The ultimate tldr on the OpenAI "rogue model" hacking incident 😅
```

Note the shape of the difference rather than the hit/miss column. The like-sorted
list is dominated by posts with 200+ early likes that then reached 1–10 new
people — high applause, low travel. `timnitgebru`'s post at 101 early likes went
on to reach 118 non-followers; `tedunderwood`'s at 242 early likes reached 1.
That gap is the thing the friend was asking for, and it is invisible to the
like count. It is also, on this snapshot, better found by early engagement
*rate* than by the model.





## Caveats

- **One snapshot, 211 posts, 54 positives.** The standard error on an AUROC
  near 0.7 at this size is roughly ±0.05, and the posts are not independent —
  a single day's news cycle moves many of them together. One run settles
  nothing; `fetch.py` is what you re-run to get more.
- **The community skews academic NLP**, because the seeds do. Graph expansion
  from different seeds gives a different — and differently biased — universe.
- **The universe is "what the AI community posted", not "AI posts".** A
  complaint about NeurIPS reviewing counts; so does a bear opening a freezer,
  and on this snapshot the bear travelled furthest of anything. `--ai-only`
  narrows by keyword and keeps 27%, at the cost of dropping real AI-community
  content that never says "AI". The better fix is to let the model read the
  text — `PREDICT posts.topic FROM posts WHERE posts.topic IS NULL`, the
  auto-labelling pattern from the top-level README — which is left unmeasured
  here because this snapshot has no hand-labelled topics to score it against.
- **Follower counts are read at fetch time**, i.e. after the anchor. They move
  slowly enough over 48h not to matter much, but they are not a clean
  as-of-anchor feature.

## Running it

```bash
./run.sh                    # venv, deps, snapshot, predict
./run.sh --refetch          # pull a fresh snapshot (moves the anchor)
```

or piecewise:

```bash
python fetch.py             # ~5 min, ~4,700 public API calls, no key
python analyze.py           # the premise, no model
python predict.py --target reach --context-cells 8192
python predict.py --target all --ai-only
python sweep.py --n 60      # context budget vs the baselines
python finetune.py          # train on the past, test on the present (slow)
```

`fetch.py` writes `data/snapshot.json` (~13 MB) holding both sides of the
anchor. The split happens in `db.py`, so you can move the anchor without
re-fetching.

| file | what it does |
|---|---|
| `bsky.py` | ~150 lines of AT Protocol client over `urllib` |
| `fetch.py` | community from the follow graph → posts → engagement edges |
| `db.py` | the snapshot as four tables; `with_follows` / `with_flag` ablations |
| `analyze.py` | the premise, measured, with no model |
| `predict.py` | targets, baselines, ablation, and the ranked candidates |
| `sweep.py` | is RT-J losing to the baselines or to the context budget? |
| `finetune.py` | train on earlier anchors, test on the held-out present |
| `topics.py` | the optional `--ai-only` keyword filter, and why it's crude |
