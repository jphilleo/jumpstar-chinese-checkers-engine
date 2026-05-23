# Public Benchmark

This project should expose Aster/CCZero as a benchmark platform, not just as a
strong bot. The current public track is `CCERL-2P10-v2`, with JumpStar_60 as the
public benchmark champion. Published baseline Elo is anchored at `random = 0`;
JumpStar_60 is measured by games, not assigned as the origin of the scale. JumpStar_82 is
the latest promoted training checkpoint, but it did not take the public title in
the v2 direct match.

## Current Champion

- Label: `JumpStar_60`
- Model: `experiments/fresh_selfplay_alpha/champions/iter_060_fresh_iter058_1p8m_epoch003/model.ccpv`
- Latest challenger checked: `JumpStar_63`
- Public champion evidence: JumpStar_63 scored `1416-960-1752` against JumpStar_60 over
  4,128 paired games at 768 simulations, score `0.4593`, Elo delta `-28`,
  CI95 `-39 - -18`, SPRT `accept_h0`.
- Public benchmark command surface: `tools/ccbench.py`.

## Benchmark Shape

The public release has six layers:

1. **Rulebook:** `docs/benchmark/CCERL-2P10-v2.md`.
2. **Frozen position schedule:**
   `benchmarks/ccerl-v1/positions/official_elo_v2.jsonl`.
3. **Interop runner:** `tools/ccp_referee.py`, using CCP subprocess engines and
   the C++ referee for legality.
4. **Rating estimator:** `tools/estimate_elo.py`, a Bradley-Terry Elo fit over
   JSON/JSONL game results.
5. **Submission contract:** `docs/benchmark/CCERL_SUBMISSIONS.md`, covering Docker/process
   submissions and resource classes.
6. **Runner wrapper:** `tools/ccbench.py`, the public command front door.

The current frozen schedule has 86 rows selected from the original 128-row
generated suite. Every selected row has two high-simulation sentinel audits:
JumpStar_57-vs-JumpStar_46 and JumpStar_57-vs-TT-PVS, both at 704 simulations. Rows with max
measured side bias above `0.25` were rejected. A rated match should play each
row twice with colors swapped.

The current v2 core manifest is `experiments/public_benchmark/ccerl_v2_baselines_core.json`.
It includes algorithmic baselines, historical JumpStar checkpoints, and the
latest contender set: `JumpStar_5`, `JumpStar_9`, `JumpStar_29`, `JumpStar_46`, `JumpStar_57`,
`JumpStar_60`, `JumpStar_61`, `JumpStar_62`, `JumpStar_63`, `JumpStar_64`, and `JumpStar_82`. The default public rating anchor
is `random = 0`.

## Rating Lists

Keep these separate:

- **Fixed simulation:** useful for internal `.ccpv` checkpoint history and early
  public challenges. Default: 768 MCTS simulations.
- **Fixed hardware wall clock:** the serious public Elo list once CCP/Docker
  runners exist.
- **Research suites:** tactical/endgame and policy/value position tests, scored
  separately from match Elo.

## Current Calibration

Public benchmark docs:

```text
docs/benchmark/CCERL_LEADERBOARD.md
docs/benchmark/README.md
```

The current native/CCP all-pairs ladder is:

```text
experiments/public_benchmark/ladder_v2_latest_champions_s64/baseline_ladder.json
```

It uses all 86 audited v2 positions, paired color swaps, all 171 displayed-engine pairs, 29,412 games total, 64-simulation neural search, the existing native/CCP logs plus latest-champion incremental runs, and `random = 0`. Open-source adapter rows are fitted in the same all-pairs graph as every other displayed engine.

| Rank | Baseline | Elo | Games | Score |
|---:|---|---:|---:|---:|
| 1 | `JumpStar_60` | 911 | 3,096 | 0.784 |
| 2 | `JumpStar_82` | 900 | 3,096 | 0.775 |
| 3 | `JumpStar_63` | 891 | 3,096 | 0.766 |
| 4 | `JumpStar_64` | 888 | 3,096 | 0.764 |
| 5 | `JumpStar_61` | 879 | 3,096 | 0.756 |
| 6 | `JumpStar_62` | 873 | 3,096 | 0.750 |
| 7 | `JumpStar_57` | 813 | 3,096 | 0.692 |
| 8 | `tt-pvs` | 737 | 3,096 | 0.614 |
| 9 | `JumpStar_46` | 684 | 3,096 | 0.558 |
| 10 | `JumpStar_29` | 654 | 3,096 | 0.526 |
| 11 | `JumpStar_9` | 516 | 3,096 | 0.382 |
| 12 | `converter` | 511 | 3,096 | 0.376 |
| 13 | `JumpStar_5` | 484 | 3,096 | 0.349 |
| 14 | `greedy` | 481 | 3,096 | 0.346 |
| 15 | `marblefish-ab` | 443 | 3,096 | 0.309 |
| 16 | `harryz-rule` | 379 | 3,096 | 0.251 |
| 17 | `zedrichu-minimax` | 370 | 3,096 | 0.244 |
| 18 | `traffic-greedy` | 340 | 3,096 | 0.219 |
| 19 | `random` | 0 | 3,096 | 0.040 |

Because top JumpStar checkpoints can be close in broad ladders, the champion
call uses a promotion-grade direct match:

```text
experiments/public_benchmark/champion_match_iter063_vs_iter060_s768_c24_v2/champion_match.json
```

That run used the full 86-row audited v2 schedule, 24 full cycles, paired color
swaps, 768 simulations, and 4,128 games. JumpStar_63 scored `1416-960-1752` against
JumpStar_60, score `0.4593`, score CI95 `0.4441 - 0.4745`, Elo delta `-28`, Elo
CI95 `-39 - -18`; the SPRT-style `elo0=0`, `elo1=25` check returns
`accept_h0`. The right public claim is that JumpStar_60 remains the CCERL v2 public
benchmark champion.

## Immediate Commands

Regenerate the frozen official Elo position schedule:

```sh
tools/ccbench.py build-suite
```

Challenge a candidate `.ccpv` model against the active champion:

```sh
tools/ccbench.py challenge-ccpv \
  --candidate-model path/to/model.ccpv \
  --out-dir experiments/public_benchmark/candidate_vs_champion
```

Run a small ladder against the active champion:

```sh
tools/ccbench.py ladder-ccpv \
  --model iter057=experiments/fresh_selfplay_alpha/champions/iter_057_fresh_iter056_endgame_deep_epoch004/model.ccpv \
  --out-dir experiments/public_benchmark/iter057_iter060_ladder
```

Print the benchmark manifest:

```sh
tools/ccbench.py manifest
```

Print the baseline definitions:

```sh
tools/ccbench.py baselines
```

Print the runnable baseline pack manifest:

```sh
tools/ccbench.py baseline-pack
```

Run two CCP engines through the trusted referee:

```sh
tools/ccbench.py referee-ccp \
  --engine-a-label JumpStar_60 \
  --engine-a-cmd "python3 tools/ccp_cczero_engine.py --model experiments/fresh_selfplay_alpha/champions/iter_060_fresh_iter058_1p8m_epoch003/model.ccpv --name JumpStar_60" \
  --engine-b-label challenger \
  --engine-b-cmd "./my_engine --ccp" \
  --out experiments/public_benchmark/challenger_vs_iter060/games.jsonl
```

Run the native high-simulation suite audits used to freeze v2:

```sh
tools/ccbench.py audit-positions-native \
  --engine-a-label JumpStar_57 \
  --engine-b-label JumpStar_46 \
  --limit 64 \
  --simulations 704 \
  --workers 4 \
  --out-dir experiments/public_benchmark/native_suite_audit_iter057_fresh046_s704_limit64
```

Freeze an audited suite from audit reports:

```sh
tools/ccbench.py freeze-suite \
  --audit experiments/public_benchmark/native_suite_audit_iter057_fresh046_s704_limit64/suite_audit.json \
  --audit experiments/public_benchmark/native_suite_audit_iter057_ttpvs_s704_limit64/suite_audit.json \
  --audit experiments/public_benchmark/native_suite_audit_iter057_fresh046_s704_offset64_limit64/suite_audit.json \
  --audit experiments/public_benchmark/native_suite_audit_iter057_ttpvs_s704_offset64_limit64/suite_audit.json \
  --out benchmarks/ccerl-v1/positions/official_elo_v2.jsonl \
  --report benchmarks/ccerl-v1/positions/official_elo_v2.report.json \
  --target 96 \
  --min-audits 2 \
  --max-side-bias 0.25
```

Estimate Elo from one or more result logs:

```sh
tools/ccbench.py elo \
  experiments/public_benchmark/challenger_vs_iter060/games.jsonl \
  --anchor random \
  --anchor-elo 0 \
  --out experiments/public_benchmark/challenger_vs_iter060/elo.json
```

Run the baseline pack as a connected native all-pairs ladder:

```sh
tools/ccbench.py baseline-ladder-native \
  --positions benchmarks/ccerl-v1/positions/official_elo_v2.jsonl \
  --out-dir experiments/public_benchmark/native_baseline_ladder_v2_allpairs_s704_limit12_iter060_suite_random0 \
  --pairs all \
  --anchor random \
  --anchor-elo 0 \
  --limit 12 \
  --max-plies 240 \
  --simulations 704 \
  --workers 6
```

Stage neural model artifacts for upload:

```sh
tools/ccbench.py package-artifacts \
  --out-dir experiments/public_benchmark/artifacts/ccerl_baseline_models_20260520 \
  --force
```

Build the GitHub Release package:

```sh
tools/ccbench.py release-package \
  --tag ccerl-v1-rc1 \
  --out-dir experiments/public_benchmark/releases/ccerl-v1-rc1 \
  --force
```

Evaluate a direct improvement claim with the SPRT-style helper:

```sh
tools/ccbench.py sprt path/to/games.jsonl \
  --candidate challenger \
  --opponent iter060 \
  --elo0 0 \
  --elo1 25
```

Run a promotion-grade direct champion match when an engine is in the champion
Elo band:

```sh
tools/ccbench.py champion-match \
  --candidate iter060 \
  --champion iter057 \
  --cycles 12 \
  --simulations 704 \
  --out-dir experiments/public_benchmark/champion_match_iter060_vs_iter057_s704_c12
```

Run the packaged smoke check:

```sh
benchmarks/ccerl-v1/baseline_pack/run_smoke.sh
```

Run the packaged sample ladder:

```sh
benchmarks/ccerl-v1/baseline_pack/run_ladder_sample.sh
```

## Next Engineering Steps

- Publish hosted neural checkpoint downloads from the staged artifact bundle.
- Publish a frozen legality/perft suite beyond the existing core fixtures.
- Wire hosted neural/checkpoint artifact URLs into the `/benchmark` page and
  GitHub release once the release package is uploaded.
