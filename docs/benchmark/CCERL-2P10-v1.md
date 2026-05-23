# CCERL-2P10-v1 *(superseded — see [CCERL-2P10-v2](CCERL-2P10-v2.md))*

> **Deprecation note (2026-05-21).** This ruleset has been superseded by
> [`CCERL-2P10-v2`](CCERL-2P10-v2.md). A single rule (`anti_block_goal_full`)
> was tightened to close a stall-draw loophole. v1 is preserved verbatim
> below so the historical benchmark, the `ccerl-v1-rc1` release package, and
> any published v1 logs remain reproducible. New evaluations and the public
> leaderboard use v2. See
> [docs/benchmark/CHANGELOG_2P10_V1_TO_V2.md](CHANGELOG_2P10_V1_TO_V2.md) for the full
> rationale.
>
> To replay v1 games, pass `--rules strict-v1` to `cczero match`.

`CCERL-2P10-v1` is the first public Chinese Checkers Engine Rating List ruleset
for this repository. It is intentionally tied to the C++ `CCZ-121-Strict-LG-v1`
rule profile so benchmark games can be reproduced by the checked-in referee.

## Scope

- Two players.
- 10 pieces per player.
- Standard 121-hole star board.
- Side triangles are playable.
- No captures.
- Player 0 starts in the top home triangle and moves toward the bottom goal
  triangle.
- Player 1 starts in the bottom home triangle and moves toward the top goal
  triangle.

## Moves

On each turn, the side to move moves one own piece by either:

- one adjacent step to an empty neighboring hole; or
- a hop chain of one or more hops.

Each hop jumps over one occupied adjacent midpoint, occupied by either player,
and lands on the empty hole directly beyond it. Hop chains may change direction
after each landing. Jumped pieces are not captured. A hop chain may not revisit a
landing hole inside the same move.

The referee stores each move as:

- integer `from` cell id;
- integer `to` cell id;
- witness `path`, including the source, intermediate hop landings, and final
  destination.

## Goal Locking

Goal locking is enabled. Once a piece lands in its own goal triangle, every later
landing for that piece must remain inside that same goal triangle. This applies
both across turns and within a single hop chain.

## Terminal Results

The game ends immediately when one of these conditions is reached:

- `all_pieces_in_goal`: a player has all 10 pieces in their own goal triangle.
- `equal_turn_goal_draw`: both players have all 10 pieces in goal, or player 0
  has just completed and player 1 has an immediate legal completing reply.
- `anti_block_goal_full`: a player's goal triangle is effectively full: every
  goal hole is occupied by either that player or an opponent blocker, at least
  one blocker is present, and the player has vacated their own home triangle.
- `repetition`: the same player-to-move position hash occurs three times.
- `max_ply`: the game reaches the configured maximum ply count.
- `no_legal_moves`: the side to move has no legal move.

Draws score 0.5 for each engine. Wins score 1.0 for the winner and 0.0 for the
loser.

## Benchmark Defaults

The first Iter57-backed benchmark uses:

- engine profile: `strict`;
- champion: `experiments/fresh_selfplay_alpha/champions/iter_057_fresh_iter056_endgame_deep_epoch004/model.ccpv`;
- MCTS simulations: `768`;
- move generator: `bitboard`;
- inference backend: `auto`;
- inference batch size: `64`;
- position schedule: `benchmarks/ccerl-v1/positions/official_elo.jsonl`;
- opening diversification: 102 frozen opening exits sampled from mixed
  classical bot sources at 4, 8, 12, 16, and 24 plies;
- clean-start floor: 26 explicit initial-board rows, roughly `20%` of the 128
  row schedule;
- pairing: each row is played twice with colors swapped;
- default max plies: `240`.

The Elo scale is engine-only. It is not a human rating.
