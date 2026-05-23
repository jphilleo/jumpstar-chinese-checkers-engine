# CCERL Champion Direct Match

Candidate: `iter063`
Champion: `iter060`

## Result

| Games | W-D-L | Score | Score CI95 | Elo Delta | Elo CI95 | SPRT |
|---:|---:|---:|---:|---:|---:|---|
| 4128 | 1416-960-1752 | 0.4593 | 0.4441..0.4745 | -28.3 | -39.0..-17.7 | accept_h0 |

## Protocol

- `runner`: `native_match_suite`
- `positions`: `benchmarks/ccerl-v1/positions/official_elo_v2.jsonl`
- `expanded_positions`: `experiments/public_benchmark/champion_match_iter063_vs_iter060_s768_c24_v2/positions.direct_match.jsonl`
- `source_positions`: `86`
- `paired_starts`: `2064`
- `games`: `4128`
- `cycles`: `24`
- `target_games`: `1000`
- `simulations`: `768`
- `max_plies`: `240`
- `workers`: `8`
- `seed`: `990000`
- `color_swap`: `True`

## Artifacts

- games: `experiments/public_benchmark/champion_match_iter063_vs_iter060_s768_c24_v2/games.jsonl`
- native log: `experiments/public_benchmark/champion_match_iter063_vs_iter060_s768_c24_v2/native/iter063_vs_iter060.jsonl`
- Elo: `experiments/public_benchmark/champion_match_iter063_vs_iter060_s768_c24_v2/elo_delta.json`
- SPRT: `experiments/public_benchmark/champion_match_iter063_vs_iter060_s768_c24_v2/sprt.json`
