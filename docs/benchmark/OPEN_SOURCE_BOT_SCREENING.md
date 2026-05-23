# Open-Source Chinese Checkers Bot Screening

Status: source-style full-suite evaluation, not an official CCERL rating list.

## Candidate Survey

The public GitHub `chinese-checkers` topic page lists several relevant
open-source projects. The most useful candidates for a first comparison were:

| Project | Public description | Fit for CCERL |
|---|---|---|
| `SVJayanthi/ChineseCheckersMCTS` | Java Chinese Checkers player using Monte Carlo Tree Search. | Good algorithmic reference; GUI/original-code integration is brittle, so v1 uses a CCERL MCTS-style adapter. |
| `arthur-x/MarbleFish` | Browser-playable Chinese Checkers AI using alpha-beta search and beam-style pruning. | Good algorithmic reference; v1 ports the published alpha-beta scoring/search style to CCERL. |
| `Zedrichu/Chinese-Checkers-AI` | Python project with random, non-repeating-random, and minimax players. | Useful minimax/heuristic reference, but upstream uses `triangle_size=3`, so direct binary comparison would not be the same game. |
| `HarryZalessky/Chinese-Checkers` | Java graph-based 121-cell, 10-piece game with a rule-based "perfect game" AI. | Good rules/heuristic reference; v1 ports the rule-path style to CCERL. |
| `alexicanesse/ChineseCheckers` | C++ two-player alpha-beta solver with heavier Boost/TensorFlow-era dependencies. | Strong candidate for a later exact integration pass, but not included in this first screening. |
| `henrychess/pygame-chinese-checkers` | Python/PyGame implementation with custom bot support. | Good extensibility reference; default bots are described as random/crude greedy, already covered by CCERL algorithmic baselines. |

Sources:

- GitHub topic page: https://github.com/topics/chinese-checkers
- SVJayanthi MCTS project: https://github.com/SVJayanthi/ChineseCheckersMCTS
- MarbleFish project: https://github.com/arthur-x/MarbleFish
- Zedrichu minimax project: https://github.com/Zedrichu/Chinese-Checkers-AI
- HarryZalessky Java project: https://github.com/HarryZalessky/Chinese-Checkers
- alexicanesse C++ solver: https://github.com/alexicanesse/ChineseCheckers
- henrychess custom-bot project: https://github.com/henrychess/pygame-chinese-checkers

## What Was Implemented

`tools/ccp_open_source_style_bot.py` adds four CCP-compatible source-style
adapters:

| Adapter | Inspired by | Method |
|---|---|---|
| `harryz-rule` | HarryZalessky rule/path AI | deterministic forward-progress, jump, and goal-entry heuristic |
| `zedrichu-minimax` | Zedrichu minimax heuristic players | depth-limited alpha-beta over goal occupancy and distance terms |
| `marblefish-ab` | MarbleFish alpha-beta worker | beam-pruned alpha-beta with vertical-progress and centrality scoring |
| `svjayanthi-mcts-fast` | SVJayanthi MCTS project | small UCT/rollout adapter over CCERL legal moves |

These are not vendored upstream engines. They are source-style CCERL ports:
the trusted CCERL referee owns legality, terminal detection, and the official
121-hole, 10-piece, two-player rules. This avoids an invalid comparison against
projects that use different board sizes, GUI-only loops, or incompatible
coordinate systems.

The adapter's internal move generator was checked against `build/cczero
position-info` on all 86 official `official_elo_v2` rows.

## Full-Suite Result

After optimizing the adapter, the v2 evaluation uses the full audited position
suite rather than the 8-row screening subset.

Protocol:

- Ruleset: `CCERL-2P10-v2`
- Position suite: all 86 rows of `official_elo_v2`
- Pairing: color-swapped games from every row
- Games: 172 per adapter, 688 total
- Anchor model: `iter063`, 64 simulations per move
- Runner: `tools/ccp_referee.py`
- Direct Elo: from the adapter's perspective relative to JumpStar, draws scored as 0.5
- Artifact directory: `experiments/public_benchmark/open_source_v2/`

Evaluation profile:

| Adapter | Profile |
|---|---|
| `harryz-rule` | deterministic rule heuristic |
| `marblefish-ab` | depth 2, beam 4 |
| `zedrichu-minimax` | depth 2, beam 4 |
| `svjayanthi-mcts-fast` | 6 iterations, rollout depth 6 |

| Opponent adapter | Games | Adapter W-D-L | Adapter score | Score CI95 | Elo vs JumpStar | Terminations |
|---|---:|---:|---:|---:|---:|---|
| `harryz-rule` | 172 | 0-97-75 | 0.282 | 0.245 - 0.319 | -161 | all_pieces_in_goal: 7, anti_block_goal_full: 68, max_ply: 97 |
| `marblefish-ab` | 172 | 8-41-123 | 0.166 | 0.124 - 0.208 | -278 | all_pieces_in_goal: 107, anti_block_goal_full: 24, equal_turn_goal_draw: 4, max_ply: 37 |
| `zedrichu-minimax` | 172 | 3-6-163 | 0.035 | 0.011 - 0.059 | -563 | all_pieces_in_goal: 157, anti_block_goal_full: 9, max_ply: 6 |
| `svjayanthi-mcts-fast` | 172 | 0-0-172 | 0.000 | 0.000 - 0.000 | -1015 | anti_block_goal_full: 172 |

The full-suite run preserves the main screening conclusion: even from the adapters' own perspective, JumpStar
reduced to 64 simulations per move is far ahead of the first source-style
open-source adapters. v2 fixed many block conversions, but it did not eliminate
every max-ply behavior; `harryz-rule` still reached max-ply in 97 of 172 games.
That should be treated as a benchmark finding, not hidden from the public report.

## Earlier Screening Result

Protocol:

- Ruleset: `CCERL-2P10-v1`
- Position suite: first 8 rows of `official_elo_v2`
- Pairing: color-swapped games from every row
- Games: 16 per adapter
- Champion: `iter060`, 64 simulations per move
- Runner: `tools/ccp_referee.py`
- Artifact directory: local run output under `experiments/`; not checked into
  the public repository.

| Opponent adapter | Games | Adapter W-D-L | Adapter score | Score CI95 approx | Elo vs Iter60 raw | Elo vs Iter60 adjusted | Terminations |
|---|---:|---:|---:|---:|---:|---:|---|
| `svjayanthi-mcts-fast` | 16 | 0-0-16 | 0.000 | 0.000 - 0.194 | -inf | -607 | anti_block_goal_full:16 |
| `marblefish-ab` | 16 | 1-2-13 | 0.125 | 0.035 - 0.360 | -338 | -305 | all_pieces_in_goal:10, anti_block_goal_full:4, equal_turn_goal_draw:2 |
| `zedrichu-minimax` | 16 | 2-0-14 | 0.125 | 0.035 - 0.360 | -338 | -305 | all_pieces_in_goal:14, anti_block_goal_full:2 |
| `harryz-rule` | 16 | 0-10-6 | 0.312 | 0.142 - 0.556 | -137 | -128 | all_pieces_in_goal:1, anti_block_goal_full:5, max_ply:10 |

The adjusted Elo column uses a half-point prior so all-win rows remain finite.
It should be read as a screening estimate, not as a calibrated rating.

## Interpretation

Even with Iter60 reduced to 64 simulations per move, the champion is far ahead
of the first source-style open-source adapters. This supports the qualitative
public-release claim that JumpStar is not merely beating random or simple
greedy baselines.

The current evidence should remain bounded:

- This is a source-style adapter evaluation, not the official public ladder.
- The adapters are CCERL-rule ports, not exact upstream binaries.
- Stronger exact integrations should target `alexicanesse/ChineseCheckers`,
  `SVJayanthi/ChineseCheckersMCTS`, and `MarbleFish` first.
- Any exact upstream binary benchmark must document rule differences, build
  steps, commit hashes, and coordinate conversion tests before its result is
  mixed into CCERL.
