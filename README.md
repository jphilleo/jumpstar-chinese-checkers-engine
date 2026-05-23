# JumpStar Chinese Checkers Engine

This is the public engine and benchmark repository for JumpStar, a
self-play-trained Chinese Checkers AI. As of the public launch on **2026-05-23**,
`JumpStar_60` is, to the best of this project's public evidence, the strongest
openly documented Chinese Checkers AI engine under the `CCERL-2P10-v2`
benchmark. That claim is benchmark-scoped: stronger private engines may exist,
and CCERL is designed so public challengers can test the result.

This repository intentionally contains only the public research surface:

- the dependency-light C++20 CCZero engine core
- tests and JSON schemas
- the CCERL benchmark rules, protocol, positions, and baseline definitions
- public leaderboard and direct-match evidence
- a small benchmark runner/tool subset

It does not contain the hosted website, Vercel deployment code, private training
automation, or internal experiment workspaces.

## Build

```sh
cmake -S . -B build -DCCZERO_USE_ACCELERATE=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On macOS, the Makefile also supports:

```sh
make release
make test
```

## Start Here

- [docs/benchmark/README.md](docs/benchmark/README.md): benchmark documentation.
- [docs/benchmark/CCERL_LEADERBOARD.md](docs/benchmark/CCERL_LEADERBOARD.md):
  current public leaderboard snapshot.
- [docs/benchmark/CCERL-2P10-v2.md](docs/benchmark/CCERL-2P10-v2.md):
  current ruleset.
- [docs/benchmark/CCP_PROTOCOL.md](docs/benchmark/CCP_PROTOCOL.md):
  stdin/stdout protocol for external engines.
- [benchmarks/ccerl-v1/positions/official_elo_v2.jsonl](benchmarks/ccerl-v1/positions/official_elo_v2.jsonl):
  frozen v2 position schedule.
- [experiments/public_benchmark/ladder_v2_latest_champions_s64/elo.json](experiments/public_benchmark/ladder_v2_latest_champions_s64/elo.json):
  public Elo summary.
- [experiments/public_benchmark/champion_match_iter063_vs_iter060_s768_c24_v2/champion_match.md](experiments/public_benchmark/champion_match_iter063_vs_iter060_s768_c24_v2/champion_match.md):
  direct champion-defense match summary.

## Quick Commands

Print the benchmark manifest:

```sh
python3 tools/ccbench.py manifest
```

Run a tiny native ladder sample after building `build/cczero`:

```sh
python3 tools/ccbench.py baseline-ladder-native \
  --engine build/cczero \
  --positions benchmarks/ccerl-v1/positions/official_elo_v2.jsonl \
  --out-dir build/public_ladder_smoke \
  --label random \
  --label greedy \
  --limit 2 \
  --simulations 8 \
  --workers 1 \
  --force
```

External engines can implement CCP and be run through:

```sh
python3 tools/ccbench.py referee-ccp --help
```

## Replay Viewer

A small static replay viewer is included at
[viewer/replay.html](viewer/replay.html). It can load the included sample game or
any JSONL produced by:

```sh
./build/cczero match --rules strict --p0 greedy --p1 random --seed 7 --max-plies 80 --log build/game.jsonl
```

## Model Artifacts

The public repository records exact model labels, paths, sizes, and hashes used
for the benchmark. The `.ccpv` model files themselves are not stored in Git
history. If model downloads are published, they should be attached as GitHub
Release assets and restored to the paths documented in
[benchmarks/ccerl-v1/baseline_pack/MODEL_NOTES.md](benchmarks/ccerl-v1/baseline_pack/MODEL_NOTES.md).

## License

No open-source license has been selected yet. Until a license is added, this
source is public for inspection and reproducibility but is not broadly licensed
for reuse or redistribution.
