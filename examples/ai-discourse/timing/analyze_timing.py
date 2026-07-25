"""When should you post tech content?

The naive version of this analysis — average likes by hour — answers a
different question than the one being asked. It mostly measures *which authors
post at which hours*: if the accounts with 40k followers post at 15:00, then
15:00 looks like a great time to post, and that fact is useless to you because
you are not them.

So every number here is a **within-author** comparison. Each post is scored
against its own author's typical post:

    residual = log1p(engagement) - mean(log1p(engagement)) for that author

A positive residual at 15:00 means *the same person* does better at 15:00 than
they do on average. Authors with fewer than ``--min-posts`` posts are dropped,
because a residual against a two-post baseline is noise.

Other things that are controlled or reported rather than ignored:

* **Saturation.** Engagement counts are read at fetch time, so a post from
  yesterday has not finished accumulating. Anything newer than ``--min-age-days``
  is discarded, not decayed.
* **Replies.** ~65% of posts are replies, they get far less engagement, and
  they are posted at different hours. They are analysed separately, never mixed.
* **Timezone.** Reported in US/Eastern as well as UTC. This community is
  US-heavy but not US-only, and that is an assumption, not a measurement.
* **Content covariates.** Links, images, and text length all move engagement,
  and they are not evenly distributed across the day, so their effect is
  reported next to the timing effect for scale.
* **Sample size.** Every cell prints its n and a bootstrap interval. A 7x24
  grid is 168 cells and it is very easy to read noise as a result.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
from tech import is_tech                                       # noqa: E402

DATA = Path(__file__).parent / "data" / "history.json"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def engagement(post: dict) -> int:
    return (post["likes"] + post["reposts"] + post["replies"] + post["quotes"])


def bootstrap_ci(values, rounds=400, seed=0):
    """Percentile bootstrap of the mean. Small n is the norm in a 168-cell
    grid, so the interval is the point of the table, not decoration."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(rounds):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def omnibus(rows, key, n_cells, rounds=2000, seed=0):
    """Is there *any* effect here, before we start reading individual cells?

    Scanning 24 hourly cells at 5% will hand you roughly one "significant" hour
    per run out of pure chance, and a bootstrap interval per cell does nothing
    about that. So test the whole table at once: hold each post's timestamp
    fixed and shuffle the residuals *within each author*, which is exactly the
    null "when this author posts has no bearing on how the post does".

    Statistic is the n-weighted spread of cell means. Returns (observed, p).
    """
    by_author = defaultdict(list)
    for r in rows:
        by_author[r["author_did"]].append(r)

    def spread(assignment):
        total = defaultdict(float)
        count = defaultdict(int)
        for r, resid in assignment:
            cell = key(r)
            total[cell] += resid
            count[cell] += 1
        means = [total[c] / count[c] for c in range(n_cells) if count[c]]
        if len(means) < 2:
            return 0.0
        grand = sum(total.values()) / sum(count.values())
        return sum(count[c] * (total[c] / count[c] - grand) ** 2
                   for c in range(n_cells) if count[c]) / sum(count.values())

    observed = spread([(r, r["resid"]) for r in rows])
    rng = random.Random(seed)
    hits = 0
    for _ in range(rounds):
        shuffled = []
        for author_rows in by_author.values():
            resids = [r["resid"] for r in author_rows]
            rng.shuffle(resids)
            shuffled.extend(zip(author_rows, resids))
        if spread(shuffled) >= observed:
            hits += 1
    return observed, (hits + 1) / (rounds + 1)


def bar(value, scale=0.30, width=22):
    """A signed bar around zero, so the table reads at a glance."""
    if value != value:                                          # NaN
        return " " * (2 * width + 1)
    filled = max(-width, min(width, int(round(value / scale * width))))
    if filled >= 0:
        return " " * width + "|" + "#" * filled + " " * (width - filled)
    return (" " * (width + filled) + "#" * -filled + "|" + " " * width)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--min-age-days", type=int, default=14,
                    help="drop posts younger than this; their counts are still "
                         "accumulating and would look artificially bad")
    ap.add_argument("--min-posts", type=int, default=25,
                    help="drop authors with fewer posts than this")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--scope", default="tech",
                    choices=["tech", "nontech", "all"])
    ap.add_argument("--kind", default="top",
                    choices=["top", "replies", "all"])
    ap.add_argument("--metric", default="engagement",
                    choices=["engagement", "likes", "reposts", "replies"])
    ap.add_argument("--permutations", type=int, default=2000)
    args = ap.parse_args()

    blob = json.loads(Path(args.data).read_text())
    fetched = datetime.fromisoformat(blob["fetched_at"])
    cutoff = fetched - timedelta(days=args.min_age_days)
    zone = ZoneInfo(args.tz)

    def metric(p):
        return engagement(p) if args.metric == "engagement" else p[args.metric]

    # -- filter ----------------------------------------------------------
    rows = []
    dropped_young = 0
    for post in blob["posts"]:
        when = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
        if when > cutoff:
            dropped_young += 1
            continue
        if args.kind == "top" and post["is_reply"]:
            continue
        if args.kind == "replies" and not post["is_reply"]:
            continue
        tech = is_tech(post["text"])
        if args.scope == "tech" and not tech:
            continue
        if args.scope == "nontech" and tech:
            continue
        local = when.astimezone(zone)
        rows.append({**post, "utc": when, "local": local,
                     "value": math.log1p(metric(post))})

    by_author = defaultdict(list)
    for row in rows:
        by_author[row["author_did"]].append(row)
    kept = {a: rs for a, rs in by_author.items() if len(rs) >= args.min_posts}
    rows = [r for a, rs in kept.items() for r in rs]
    if not rows:
        raise SystemExit("no rows survived the filters — loosen --min-posts")

    # -- within-author residuals -----------------------------------------
    for author, author_rows in kept.items():
        mean = sum(r["value"] for r in author_rows) / len(author_rows)
        for r in author_rows:
            r["resid"] = r["value"] - mean

    members = {m["did"]: m for m in blob["members"]}
    print(f"{len(rows)} posts, {len(kept)} authors "
          f"(>= {args.min_posts} posts each)")
    print(f"scope={args.scope}  kind={args.kind}  metric={args.metric}  "
          f"tz={args.tz}")
    print(f"dropped {dropped_young} posts younger than {args.min_age_days}d "
          f"(engagement still accumulating)")
    raw = [math.expm1(r["value"]) for r in rows]
    raw.sort()
    print(f"median {args.metric} per post: {raw[len(raw)//2]:.0f}, "
          f"mean {sum(raw)/len(raw):.1f}\n")
    print("residual = log1p(engagement) - that author's own mean; "
          "0.30 ~ +35%\n")

    def table(title, key, labels):
        _, p = omnibus(rows, key, len(labels), rounds=args.permutations)
        verdict = ("REAL EFFECT" if p < 0.05 else
                   "indistinguishable from noise")
        print(f"== {title} ==")
        print(f"  omnibus permutation test over all {len(labels)} cells: "
              f"p={p:.3f}  ->  {verdict}")
        cells = defaultdict(list)
        authors = defaultdict(set)
        for r in rows:
            cells[key(r)].append(r["resid"])
            authors[key(r)].add(r["author_did"])
        best, worst = None, None
        for label_index, label in enumerate(labels):
            vals = cells.get(label_index, [])
            if not vals:
                print(f"  {label:<5}      n=0")
                continue
            mean = sum(vals) / len(vals)
            lo, hi = bootstrap_ci(vals, seed=label_index)
            solid = "" if (lo < 0 < hi) else "  *"
            print(f"  {label:<5} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"n={len(vals):<5} a={len(authors[label_index]):<3}"
                  f"{bar(mean)}{solid}")
            if best is None or mean > best[1]:
                best = (label, mean)
            if worst is None or mean < worst[1]:
                worst = (label, mean)
        if best and worst:
            print(f"  best {best[0]} ({best[1]:+.3f}), "
                  f"worst {worst[0]} ({worst[1]:+.3f}); "
                  f"spread ~{abs(math.expm1(best[1] - worst[1])):.0%} "
                  f"in engagement")
        print("  n = posts, a = distinct authors contributing to the cell")
        print("  * = per-cell bootstrap excludes zero; with this many cells")
        print("      expect ~1 starred cell by chance. Trust the omnibus.\n")

    table(f"day of week ({args.tz})", lambda r: r["local"].weekday(), DAYS)
    table(f"hour of day ({args.tz})", lambda r: r["local"].hour,
          [f"{h:02d}" for h in range(24)])
    table("hour of day (UTC)", lambda r: r["utc"].hour,
          [f"{h:02d}" for h in range(24)])

    # -- the covariates, for scale ---------------------------------------
    print("== how big is timing next to the things you actually control? ==")
    def contrast(name, predicate):
        """Same within-author permutation null as the tables above — these get
        held to the standard the day/hour cells are held to, not a laxer one."""
        yes = [r["resid"] for r in rows if predicate(r)]
        no = [r["resid"] for r in rows if not predicate(r)]
        if len(yes) < 10 or len(no) < 10:
            return
        delta = sum(yes) / len(yes) - sum(no) / len(no)
        observed, p = omnibus(rows, lambda r: 1 if predicate(r) else 0, 2,
                              rounds=args.permutations, seed=1)
        mark = "  *" if p < 0.05 else ""
        print(f"  {name:<34} {delta:+.3f}  "
              f"({len(yes)} vs {len(no)})   ~{math.expm1(delta):+6.0%}   "
              f"p={p:.3f}{mark}")
    contrast("has a link", lambda r: r["has_link"])
    contrast("has an image", lambda r: r["has_image"])
    contrast("has video", lambda r: r["has_video"])
    contrast("long text (>200 chars)", lambda r: r["text_length"] > 200)
    contrast("very short (<40 chars)", lambda r: r["text_length"] < 40)
    contrast("weekend", lambda r: r["local"].weekday() >= 5)
    best_hours = {14, 15, 16, 17}
    contrast("posted 14:00-17:00 local", lambda r: r["local"].hour in best_hours)

    # -- day x hour, coarse, because 168 cells is too many ---------------
    print("\n== day x 4-hour block (residual, n) ==")
    blocks = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    header = "      " + "".join(f"{lo:02d}-{hi:02d}      " for lo, hi in blocks)
    print(header)
    for day_index, day in enumerate(DAYS):
        line = f"  {day}  "
        for lo, hi in blocks:
            vals = [r["resid"] for r in rows
                    if r["local"].weekday() == day_index
                    and lo <= r["local"].hour < hi]
            if len(vals) < 15:
                line += f"   .({len(vals):>3})  "
            else:
                line += f"{sum(vals)/len(vals):+.2f}({len(vals):>3})  "
        print(line)
    print("  . = fewer than 15 posts, not shown")

    # -- who is in this sample -------------------------------------------
    sizes = sorted((members.get(a, {}) or {}).get("followers") or 0
                   for a in kept)
    print(f"\nauthors: median {sizes[len(sizes)//2]} followers, "
          f"range {sizes[0]}-{sizes[-1]}")


if __name__ == "__main__":
    main()
