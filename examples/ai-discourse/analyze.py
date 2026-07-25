"""The premise, measured: what does a like actually tell you?

No model here. This is the arithmetic that justifies the task — if likes were
already a good ranking of what matters in AI, the rest of the example would be
pointless. Run it before ``predict.py`` to see the problem, and after to see
what the ranking was up against.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import db
import predict
import topics


def spearman(xs, ys) -> float | None:
    """Rank correlation, ties averaged."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else num / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(db.SNAPSHOT))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--ai-only", action="store_true",
                    help="narrow to posts explicitly mentioning AI")
    args = ap.parse_args()

    snapshot = db.load(Path(args.snapshot))
    anchor = db.iso(snapshot["anchor"])
    past, future = predict.split_engagements(snapshot, anchor)

    universe = topics.filter_posts(
        [p for p in snapshot["posts"] if p.get("in_universe")], args.ai_only)
    ids = [p["post_id"] for p in universe]
    members = {m["did"]: m for m in snapshot["members"]}
    author_of = {p["post_id"]: p["author_did"] for p in snapshot["posts"]}
    flags = snapshot.get("follower_flags", {})

    outside = predict.label_counts(snapshot, future, ids,
                                   predict.TARGETS["reach"]["count"])
    total = predict.label_counts(snapshot, future, ids,
                                 predict.TARGETS["volume"]["count"])
    early = {p: len(past.get(p, [])) for p in ids}
    followers = {p: (members.get(author_of[p], {}) or {}).get("followers") or 0
                 for p in ids}

    print(f"{len(ids)} posts, {len({author_of[p] for p in ids})} authors, "
          f"anchor {anchor:%Y-%m-%d %H:%M} UTC\n")

    print("== 1. is engagement just an echo of audience size? ==")
    r = spearman([followers[p] for p in ids], [early[p] for p in ids])
    print(f"   spearman(author followers, engagement so far) = {r:+.3f}")
    r2 = spearman([followers[p] for p in ids], [outside[p] for p in ids])
    print(f"   spearman(author followers, reach past that audience) = {r2:+.3f}")

    print("\n== 2. where does engagement come from? ==")
    share_rows = []
    for p in ids:
        seen = past.get(p, [])
        if not seen:
            continue
        follows = sum(1 for e in seen
                      if flags.get(f'{author_of[p]}|{e["actor_did"]}', False))
        share_rows.append(follows / len(seen))
    if share_rows:
        share_rows.sort()
        print(f"   per post, share of engagement from EXISTING followers: "
              f"median {share_rows[len(share_rows)//2]:.1%}, "
              f"mean {sum(share_rows)/len(share_rows):.1%}")
    if flags:
        print(f"   across the whole snapshot: "
              f"{sum(flags.values())/len(flags):.1%} of "
              f"{len(flags)} (author, engager) pairs already followed")

    print("\n== 3. would sorting by likes have found the posts that travelled? ==")
    by_likes = sorted(ids, key=lambda p: -early[p])[:args.top]
    by_reach = sorted(ids, key=lambda p: -outside[p])[:args.top]
    hit = len(set(by_likes) & set(by_reach))
    print(f"   top-{args.top} by early likes vs top-{args.top} by held-out "
          f"reach: {hit}/{args.top} overlap")
    dead = sum(1 for p in by_likes if outside[p] == 0)
    print(f"   {dead}/{args.top} of the most-liked posts reached NOBODY "
          f"outside the author's audience")
    r3 = spearman([early[p] for p in ids], [outside[p] for p in ids])
    print(f"   spearman(early likes, held-out outside reach) = {r3:+.3f}")

    print(f"\n== 4. the posts likes would have buried ==")
    print("   (high outside reach, low early likes — the ones worth surfacing)")
    text = {p["post_id"]: p["text"] for p in universe}
    handle = {p: (members.get(author_of[p], {}) or {}).get("handle", "?")
              for p in ids}
    buried = sorted((p for p in ids if outside[p] > 0),
                    key=lambda p: (-outside[p] / max(1, early[p])))
    for p in buried[:10]:
        print(f"   outside={outside[p]:<4} early_likes={early[p]:<4} "
              f"total_after={total[p]:<4} @{handle[p][:22]:<22} "
              f"{text[p][:70].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
