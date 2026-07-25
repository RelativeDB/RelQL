#!/usr/bin/env bash
# One-shot: create a venv, install deps, snapshot live data, predict.
# Re-running reuses data/snapshot.json — pass --refetch to pull a fresh one,
# which changes both the universe and the labels.
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
  echo ">> fetching live markets + news (a few minutes; GDELT is rate limited)"
  ./.venv/bin/python fetch.py "$@"
fi

echo ">> predicting"
./.venv/bin/python predict.py
