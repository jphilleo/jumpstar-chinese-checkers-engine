# CCERL Baseline Pack

This pack is the runnable baseline set for `CCERL-2P10-v1`.

It contains:

- `engines.json`: machine-readable baseline definitions
- `MODEL_NOTES.md`: neural checkpoint paths, sizes, and checksums
- `MODEL_RELEASE_TERMS.md`: draft redistribution terms for model artifacts
- `run_smoke.sh`: quick CCP/native contract sanity check
- `run_ladder_sample.sh`: small native all-pairs sample ladder

The source of truth for the benchmark remains:

```text
benchmarks/ccerl-v1/manifest.json
benchmarks/ccerl-v1/baselines.json
benchmarks/ccerl-v1/positions/official_elo_v2.jsonl
```

## Baselines

Algorithmic baselines:

- `random`
- `greedy`
- `traffic-greedy`
- `converter`
- `tt-pvs`

Neural baselines:

- `fresh005`
- `fresh009`
- `fresh029`
- `fresh046`
- `iter057`
- `iter060`

Public baseline Elo is anchored at `random = 0`. Iter60 is the current champion
benchmark, not the origin of the Elo scale. Iter57 remains in the pack as a
historical champion rung.

## Requirements

Build the C++ engine first:

```sh
make release
```

The scripts default to `build/cczero`. Override with:

```sh
CCZERO_ENGINE=/path/to/cczero benchmarks/ccerl-v1/baseline_pack/run_smoke.sh
```

The neural baselines require `.ccpv` checkpoint files at the paths listed in
`MODEL_NOTES.md`. The model files are not stored in Git history; publish or
download them as release assets before running neural-baseline matches.

## Quick Smoke Check

Run:

```sh
benchmarks/ccerl-v1/baseline_pack/run_smoke.sh
```

This performs a small CCP-vs-native spot check over selected pairs and writes
results under:

```text
experiments/public_benchmark/baseline_pack_smoke
```

Use lower settings for a faster local plumbing check:

```sh
CCERL_SMOKE_SIMS=1 \
CCERL_SMOKE_MAX_PLIES=20 \
CCERL_SMOKE_LIMIT=1 \
benchmarks/ccerl-v1/baseline_pack/run_smoke.sh
```

## Sample Ladder

Run:

```sh
benchmarks/ccerl-v1/baseline_pack/run_ladder_sample.sh
```

This runs a small native all-pairs ladder on a subset of baselines and writes
results under:

```text
experiments/public_benchmark/baseline_pack_ladder_sample
```

The sample ladder is not the official public rating. It is a reproducibility and
integration check.

## Official Ratings

Official rating games should be run by the benchmark maintainers or by a signed
official runner. Self-reported logs are useful for debugging, but they should
not be counted in the official list without verification.
