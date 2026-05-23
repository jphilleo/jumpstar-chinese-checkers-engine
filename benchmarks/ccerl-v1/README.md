# CCERL v1 Benchmark

This directory contains the frozen manifest and position schedule for the first
public benchmark track:
`CCERL-2P10-v1`.

The current champion benchmark is Iter60:

```text
experiments/fresh_selfplay_alpha/champions/iter_060_fresh_iter058_1p8m_epoch003/model.ccpv
```

Iter57 remains in the baseline pack as a historical champion rung.

The official Elo schedule is:

```text
benchmarks/ccerl-v1/positions/official_elo_v2.jsonl
```

It has 86 rows selected from the original 128-row generated schedule. Every row
was audited by both Iter57-vs-Fresh046 and Iter57-vs-TT-PVS at 704 simulations,
and rows with max measured side bias above `0.25` were rejected. Official
matches play each row as a paired color swap.

The original generated v1 schedule is still kept at:

```text
benchmarks/ccerl-v1/positions/official_elo.jsonl
```

The v1 baseline definitions are:

```text
benchmarks/ccerl-v1/baselines.json
```

It contains algorithmic baselines (`random`, `greedy`, `traffic-greedy`,
`converter`, `tt-pvs`) and neural rungs (`fresh005`, `fresh009`, `fresh029`,
`fresh046`, `iter057`, `iter060`). Public baseline Elo is anchored at
`random = 0`; Iter60 is the current champion benchmark, not the origin of the
scale.

The runnable baseline pack is:

```text
benchmarks/ccerl-v1/baseline_pack/README.md
benchmarks/ccerl-v1/baseline_pack/PACK_MANIFEST.json
benchmarks/ccerl-v1/baseline_pack/engines.json
```

It includes model checksums plus `run_smoke.sh` and `run_ladder_sample.sh` for
local integration checks.

Use `tools/ccbench.py manifest` for the machine-readable runner defaults.
Use `tools/ccbench.py baseline-pack` for the runnable pack manifest.
Use `docs/benchmark/CCERL_LEADERBOARD.md` for the current public table and
`docs/benchmark/PUBLIC_BENCHMARK.md` for the benchmark structure.

The latest native all-pairs calibration is:

```text
experiments/public_benchmark/native_baseline_ladder_v2_allpairs_s704_limit12_iter060_suite_random0/baseline_ladder.json
```

That run used all 11 baselines, all-pairs scheduling, the first 12 v2
positions, paired color swaps, 704 simulations for neural engines, the native
`match-suite` runner, and `random = 0`.

The native calibration is CCP spot-checked here:

```text
experiments/public_benchmark/runner_spotcheck_v2_iter060_s64_limit2/runner_spotcheck.json
```

The current champion confirmation match is:

```text
experiments/public_benchmark/champion_match_iter060_vs_iter057_s704_c24/champion_match.json
```

Iter60 scored `1920-480-1728` over 4,128 paired games against Iter57, for an
estimated `+16.2` Elo delta with CI95 `+5.6..+26.8`.
