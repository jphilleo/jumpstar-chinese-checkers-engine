#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

ENGINE="${CCZERO_ENGINE:-$REPO_ROOT/build/cczero}"
OUT_DIR="${1:-$REPO_ROOT/experiments/public_benchmark/baseline_pack_smoke}"
LIMIT="${CCERL_SMOKE_LIMIT:-1}"
SIMS="${CCERL_SMOKE_SIMS:-16}"
MAX_PLIES="${CCERL_SMOKE_MAX_PLIES:-120}"
WORKERS="${CCERL_SMOKE_WORKERS:-2}"
MOVETIME_MS="${CCERL_SMOKE_MOVETIME_MS:-100}"
TIMEOUT_SECONDS="${CCERL_SMOKE_TIMEOUT_SECONDS:-60}"

if [[ ! -x "$ENGINE" ]]; then
  echo "cczero engine not found or not executable: $ENGINE" >&2
  echo "Build it with: make release" >&2
  exit 1
fi

cd "$REPO_ROOT"

python3 tools/ccbench.py runner-spotcheck \
  --engine "$ENGINE" \
  --positions benchmarks/ccerl-v1/positions/official_elo_v2.jsonl \
  --out-dir "$OUT_DIR" \
  --limit "$LIMIT" \
  --max-plies "$MAX_PLIES" \
  --simulations "$SIMS" \
  --workers "$WORKERS" \
  --movetime-ms "$MOVETIME_MS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --pair iter060:iter057 \
  --pair iter057:fresh046 \
  --pair tt-pvs:converter \
  --pair greedy:random \
  --force
