# CCERL Leaderboard Snapshot

Status: current release-candidate snapshot for `CCERL-2P10-v2`.

This is the current local calibration snapshot for the public Chinese Checkers engine benchmark. It is engine-relative, anchored at `random = 0`, and should not be interpreted as human Elo. Source-style open-source adapters are included directly in the all-pairs ladder, not projected from a single match.

## Fixed-Simulation Baseline Ladder

Source:

```text
experiments/public_benchmark/ladder_v2_latest_champions_s64/baseline_ladder.json
```

Protocol:

- ruleset: `CCERL-2P10-v2` / `CCZ-121-Strict-LG-v2`
- runner: existing native/CCP logs plus incremental latest-champion CCP runs
- position suite: `benchmarks/ccerl-v1/positions/official_elo_v2.jsonl`
- position rows: all 86 audited rows
- pairings: all 171 displayed-engine pairs
- games: 29,412
- color swaps: yes
- neural search: 64 simulations in the fixed-simulation ladder
- draw scoring: 0.5
- anchor: `random = 0`
- latest promoted training checkpoint included: `JumpStar_82`

| Rank | Engine | Elo | CI95 | Games | W-D-L | Score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `JumpStar_60` | 911 | 874 - 947 | 3,096 | 2241-371-484 | 0.784 |
| 2 | `JumpStar_82` | 900 | 864 - 937 | 3,096 | 2129-539-428 | 0.775 |
| 3 | `JumpStar_63` | 891 | 855 - 927 | 3,096 | 2134-478-484 | 0.766 |
| 4 | `JumpStar_64` | 888 | 852 - 925 | 3,096 | 2161-409-526 | 0.764 |
| 5 | `JumpStar_61` | 879 | 843 - 916 | 3,096 | 2119-442-535 | 0.756 |
| 6 | `JumpStar_62` | 873 | 837 - 909 | 3,096 | 2075-491-530 | 0.750 |
| 7 | `JumpStar_57` | 813 | 778 - 849 | 3,096 | 1981-324-791 | 0.692 |
| 8 | `tt-pvs` | 737 | 701 - 772 | 3,096 | 1805-189-1102 | 0.614 |
| 9 | `JumpStar_46` | 684 | 649 - 720 | 3,096 | 1562-333-1201 | 0.558 |
| 10 | `JumpStar_29` | 654 | 619 - 689 | 3,096 | 1478-301-1317 | 0.526 |
| 11 | `JumpStar_9` | 516 | 482 - 551 | 3,096 | 1039-285-1772 | 0.382 |
| 12 | `converter` | 511 | 476 - 546 | 3,096 | 1120-88-1888 | 0.376 |
| 13 | `JumpStar_5` | 484 | 449 - 518 | 3,096 | 961-237-1898 | 0.349 |
| 14 | `greedy` | 481 | 446 - 516 | 3,096 | 1039-67-1990 | 0.346 |
| 15 | `marblefish-ab` | 443 | 408 - 478 | 3,096 | 727-460-1909 | 0.309 |
| 16 | `harryz-rule` | 379 | 344 - 414 | 3,096 | 5-1547-1544 | 0.251 |
| 17 | `zedrichu-minimax` | 370 | 335 - 405 | 3,096 | 407-694-1995 | 0.244 |
| 18 | `traffic-greedy` | 340 | 305 - 375 | 3,096 | 607-141-2348 | 0.219 |
| 19 | `random` | 0 | 0 - 0 | 3,096 | 0-248-2848 | 0.040 |

## Champion Direct Match

Because adjacent JumpStar checkpoints are close in broad ladders, the public champion call uses a separate promotion-grade direct match over the full audited v2 suite. The latest completed promotion-grade public direct match remains `JumpStar_63` versus `JumpStar_60`; `JumpStar_82` has been added to the broad ladder but has not yet had a promotion-grade public champion match against `JumpStar_60`.

Champion direct-match artifact:

```text
experiments/public_benchmark/champion_match_iter063_vs_iter060_s768_c24_v2/champion_match.json
```

| Candidate | Opponent | Games | W-D-L | Score | Score CI95 | Elo Delta | Elo CI95 | SPRT |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `JumpStar_63` | `JumpStar_60` | 4,128 | 1416-960-1752 | 0.4593 | 0.4441 - 0.4745 | -28 | -39 - -18 | `accept_h0` |

Interpretation: `JumpStar_60` remains the public benchmark champion for now. `JumpStar_82` is the latest promoted training checkpoint and ranks second in the broad fixed-simulation ladder.

## Open-Source Adapter Check

Source-style adapters for representative public Chinese Checkers bots are included in the main all-pairs ladder above. The retained adapters are `marblefish-ab`, `harryz-rule`, and `zedrichu-minimax`.
