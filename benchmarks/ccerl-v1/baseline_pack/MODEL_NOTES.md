# CCERL Baseline Model Notes

The algorithmic baselines are built into the C++ engine and do not require
external model files.

The neural baselines are `.ccpv` checkpoints referenced by path in
`engines.json` and `benchmarks/ccerl-v1/baselines.json`. In this repository
layout they live under `experiments/`, which is normally ignored by git. A
public release should either publish these exact artifacts alongside the repo or
provide download instructions that restore them to the listed paths.

## Required Checkpoints

| Label | Role | Bytes | SHA256 |
|---|---|---:|---|
| `fresh005` | weak neural | 16354536 | `919345026dc385f66dbb561d766efd3ce4dee5311ead1d9f3233dd505435c13a` |
| `fresh009` | early strong neural | 38950120 | `66cc171c84334d6821607a1706186e3290d8c94fe30162f0b321e5a7dd0ae6c5` |
| `fresh029` | mid neural | 41452776 | `dca792121887df4a3823843262718b1df54384a687f7763b1f83b9bc45e20995` |
| `fresh046` | late-mid neural | 41452776 | `58eb9548b36043c4a0887762c5f95baddf4dda645d8db1012b5a24211d3c1847` |
| `iter057` | former champion | 41452776 | `95c00e9d015d5ef9b6e33843245f97330164f82269f239089a69adcc388d40f4` |
| `iter060` | champion | 41452776 | `91a35c69bfe5c8219c3f5343563d7bc54730e109e700347ca18b06fd28e23b58` |

## Verification

From the repository root:

```sh
shasum -a 256 \
  experiments/fresh_selfplay_alpha/champions/iter_005_effective_goal_v1/model.ccpv \
  experiments/fresh_selfplay_alpha/champions/iter_009_fixed_batched_mcts/model.ccpv \
  experiments/fresh_selfplay_alpha/champions/iter_029_merged768/model.ccpv \
  experiments/fresh_selfplay_alpha/champions/iter_046_value_weight_010/model.ccpv \
  experiments/fresh_selfplay_alpha/champions/iter_057_fresh_iter056_endgame_deep_epoch004/model.ccpv \
  experiments/fresh_selfplay_alpha/champions/iter_060_fresh_iter058_1p8m_epoch003/model.ccpv
```

Before a public release, add explicit redistribution terms for these checkpoint
artifacts.
