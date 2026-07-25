#!/usr/bin/env bash
# One-shot: create a venv, install deps, snapshot Bluesky, predict.
# Re-running reuses data/snapshot.json — pass --refetch to pull a fresh one,
# which moves the anchor and therefore changes both universe and labels.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.12}"
REFETCH=0
[ "${1:-}" = "--refetch" ] && { REFETCH=1; shift; }

if [ ! -d .venv ]; then
  echo ">> creating .venv"
  "$PYTHON" -m venv .venv
fi

echo ">> installing deps"
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

if [ "$REFETCH" = 1 ] || [ ! -f data/snapshot.json ]; then
  echo ">> snapshotting the AI corner of Bluesky (10-20 min; thousands of"
  echo "   small public API calls, no key required)"
  ./.venv/bin/python -u fetch.py
fi

echo ">> predicting"
./.venv/bin/python predict.py "$@"
