"""Synthesize a labelled ladder history for WTI from the futures curve itself.

Kalshi's KXWTI series has 3 settled ladders in the API — far too few to teach
anything. But the *question* the ladder asks ("will the settle be above K?")
can be reconstructed for every day WTI has traded: take the prior close, lay
the same relative strikes across it, and read the answer off the next close.

503 sessions x 30 strikes = ~15,000 labelled rows with exactly the shape of
tomorrow's ladder, and each one carries the market state that preceded it.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import statistics
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "wti"
# the offsets of tomorrow's real ladder, as fractions of spot ($89.31)
LADDER = [84.49, 84.99, 85.49, 85.99, 86.49, 86.99, 87.49, 87.99, 88.49,
          88.99, 89.49, 89.99, 90.49, 90.99, 91.49, 91.99, 92.49, 92.99,
          93.49, 93.99, 94.49, 94.99, 95.49, 95.99, 96.49, 96.99, 97.49,
          97.99, 98.49, 98.99]
SPOT = 89.31


def series(symbol: str, rng: str = "2y"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range={rng}&interval=1d")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = json.loads(response.read())
    result = payload["chart"]["result"][0]
    out = {}
    for stamp, close in zip(result["timestamp"],
                            result["indicators"]["quote"][0]["close"]):
        if close is not None:
            out[dt.datetime.fromtimestamp(stamp, dt.timezone.utc).date()] = close
    return out


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    wti, brent, rbob = series("CL=F"), series("BZ=F"), series("RB=F")
    days = sorted(wti)
    rows, ladders = [], []
    for i in range(31, len(days)):
        day, prev = days[i], days[i - 1]
        window = [math.log(wti[days[j]] / wti[days[j - 1]])
                  for j in range(i - 30, i)]
        vol = statistics.pstdev(window)
        state = {
            "day": str(day), "weekday": day.strftime("%a"),
            "prior_close": round(wti[prev], 2),
            "settle": round(wti[day], 2),
            "ret_1d": round(math.log(wti[prev] / wti[days[i - 2]]) * 100, 3),
            "ret_5d": round(math.log(wti[prev] / wti[days[i - 6]]) * 100, 3),
            "vol_30d": round(vol * 100, 3),
            "brent_spread": (round(brent[prev] - wti[prev], 2)
                             if prev in brent else None),
            "rbob_crack": (round(rbob[prev] * 42 - wti[prev], 2)
                           if prev in rbob else None),
        }
        ladders.append(state)
        for strike in LADDER:
            # same relative distance from the prior close as tomorrow's ladder
            k = round(wti[prev] * strike / SPOT, 2)
            rows.append({
                "strike_id": f"{day}:{strike}",
                "day": str(day),
                "threshold": k,
                "pct_from_prior": round((k / wti[prev] - 1) * 100, 3),
                "above": wti[day] > k,
            })
    (OUT / "days.json").write_text(json.dumps(ladders, indent=1))
    (OUT / "strikes.json").write_text(json.dumps(rows, indent=1))
    yes = sum(r["above"] for r in rows)
    print(f"{len(ladders)} days, {len(rows)} strike-rows, {yes} YES "
          f"({yes/len(rows):.1%})")
    print(f"range {ladders[0]['day']} .. {ladders[-1]['day']}")
    return ladders, rows


if __name__ == "__main__":
    build()
