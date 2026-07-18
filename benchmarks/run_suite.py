"""Generalizability suite across all datasets — the grid view.

    python benchmarks/run_suite.py            # full
    python benchmarks/run_suite.py --quick    # smaller user caps

Prints a datasets x tasks matrix (engine vs. best naive, beats?, per-split
mean ± std) and an overall "beats naive" fraction, and writes
benchmarks/results/suite.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.harness import datasets as D          # noqa: E402
from benchmarks.harness import suite as S             # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def _fmt(x, nd=3):
    return "   n/a" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    args = ap.parse_args()
    cap = 400 if args.quick else 900
    RESULTS.mkdir(exist_ok=True)

    t0 = time.time()
    datasets = [D.online_retail(max_customers=cap), D.movielens(),
                D.brightkite(max_users=cap)]
    print(f"loaded {len(datasets)} datasets in {time.time()-t0:.1f}s "
          f"({', '.join(f'{d.name}={len(d.entity_ids)}' for d in datasets)})")

    res = S.run(datasets, with_stability=not args.no_stability)

    print("\n" + "=" * 92)
    print("GENERALIZABILITY MATRIX  (engine vs. best naive across datasets x tasks x splits)")
    print("=" * 92)
    hdr = f"{'dataset':13} {'task':15} {'metric':9} {'engine':>8} {'naive':>8} {'beats':>6} {'per-split mean±std':>22}  n"
    print(hdr)
    print("-" * len(hdr))
    for c in res["cells"]:
        beats = {True: "  yes", False: "   no", None: "  n/a"}[c["beats_naive"]]
        st = c.get("stability", {})
        stab = (f"{_fmt(st.get('mean'))}±{_fmt(st.get('std'))} ({st.get('n_splits','?')})"
                if st else "")
        print(f"{c['dataset']:13} {c['task']:15} {c['metric']:9} "
              f"{_fmt(c['engine']):>8} {_fmt(c['naive']):>8} {beats:>6} {stab:>22}  {c['n']}")

    frac = res["beats_naive_fraction"]
    print("-" * len(hdr))
    print(f"beats naive in {res['n_won']}/{res['n_beatable']} scorable cells"
          + (f"  ({frac:.0%})" if frac is not None else ""))
    print("per-split std shows split-robustness; a change 'generalizes' only if it")
    print("raises the beats-naive fraction without inflating std across datasets.")

    (RESULTS / "suite.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {RESULTS / 'suite.json'}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
