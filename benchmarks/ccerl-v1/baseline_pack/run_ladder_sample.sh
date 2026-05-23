#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

ENGINE="${CCZERO_ENGINE:-$REPO_ROOT/build/cczero}"
OUT_DIR="${1:-$REPO_ROOT/experiments/public_benchmark/baseline_pack_ladder_sample}"
LIMIT="${CCERL_LADDER_LIMIT:-2}"
SIMS="${CCERL_LADDER_SIMS:-64}"
MAX_PLIES="${CCERL_LADDER_MAX_PLIES:-160}"
WORKERS="${CCERL_LADDER_WORKERS:-4}"

if [[ ! -x "$ENGINE" ]]; then
  echo "cczero engine not found or not executable: $ENGINE" >&2
  echo "Build it with: make release" >&2
  exit 1
fi

cd "$REPO_ROOT"

python3 tools/ccbench.py baseline-ladder-native \
  --engine "$ENGINE" \
  --positions benchmarks/ccerl-v1/positions/official_elo_v2.jsonl \
  --out-dir "$OUT_DIR" \
  --pairs all \
  --anchor random \
  --anchor-elo 0 \
  --limit "$LIMIT" \
  --max-plies "$MAX_PLIES" \
  --simulations "$SIMS" \
  --workers "$WORKERS" \
  --label random \
  --label greedy \
  --label converter \
  --label tt-pvs \
  --label fresh029 \
  --label iter057 \
  --label iter060 \
  --force
