# CCERL-2P10 v1 → v2 Changelog

**Date:** 2026-05-21
**Scope:** `CCERL-2P10-v1` ruleset, `CCZ-121-Strict-LG-v1` and `CCZ-121-AB-LG-v1`
C++ rule profiles, and the public CCERL benchmark / leaderboard.

## Summary

A single rule was tightened: the **anti-block terminal**
(`anti_block_goal_full`). Under v1, the terminal additionally required the
winning player to have completely vacated their own home triangle
(`home_count[player] == 0`). v2 removes that gate. The terminal now fires as
soon as every cell of the player's goal triangle is occupied and at least one
of those cells contains an opponent blocker.

This is the **only** behavioral change between v1 and v2. Board, moves,
goal-locking, repetition, max-ply, `all_pieces_in_goal`, and
`equal_turn_goal_draw` are all identical.

## Why

The v1 gate created an exploitable stall-draw pattern that we observed at scale
during the open-source bot screening:

| Adapter | Games vs JumpStar60 | W-D-L | `max_ply` terminations | v1 BT Elo | Decisive-only Elo |
|---|---:|---:|---:|---:|---:|
| `harryz-rule` (deterministic 1-ply greedy) | 172 | 0-101-71 | 101 / 172 (**59%**) | 1470.6 | -441.7 |
| `marblefish-ab` | 172 | 4-11-157 | 0 / 172 | 1129.8 | 985.6 |
| `zedrichu-minimax` | 172 | 5-7-160 | 0 / 172 | 1109.5 | 1021.0 |
| `svjayanthi-mcts-fast` | 172 | 0-5-167 | 0 / 172 | 890.6 | -585.1 |

`harryz-rule` never beat the champion, yet earned an Elo only ~150 points
below it because 59% of its games hit the ply cap with a 0.5-point draw. The
decisive-only Elo column (computed with draws discarded) places the same bot
at -441, below the `random` baseline.

**Mechanism.** `harryz-rule`'s one-ply greedy heuristic vacates its home
triangle aggressively but cannot find the multi-jump chains needed to bring
the last 1–2 pieces home. Those stranded pieces sit in their own home, which
geometrically *is* the opponent's goal-blocker territory. Under v1's gate,
their presence prevented `anti_block_goal_full` from firing for the
champion: the champion had 9 pieces in goal, 1 cell blocked, but
`home_count[champion]` was nonzero from those stranded blockers, so the rule
stayed quiet. Both sides then ran out the ply clock and the game was scored
as a draw.

Standard Chinese Checkers play between humans does not produce this failure
mode because humans recognize the won position and resign or accept the
ruling. CCERL needs the rule itself to encode that judgment.

## The change

### Rule text (v2)

> `anti_block_goal_full`: a player's goal triangle is effectively full —
> every goal hole is occupied by either that player or an opponent blocker,
> and at least one blocker is present. *(The v1 requirement that the
> winning player had vacated their own home triangle is removed.)*

### C++ implementation

- New field `bool tighten_anti_block` on `RuleProfile` (defaults to `false`
  for backward compatibility).
- v1 builders (`ccz_121_strict_lg_v1`, `ccz_121_ab_lg_v1`) leave it `false`.
- New v2 builders (`ccz_121_strict_lg_v2`, `ccz_121_ab_lg_v2`) set it
  `true` and use a `-v2` name suffix.
- The terminal check in `terminal_status` gates the `home_count == 0`
  requirement behind `!tighten_anti_block`.

### CLI

The `--rules` parser was updated so that:

- `--rules strict`, `--rules ab`, `--rules strict-v2`, `--rules ab-v2`, and
  the full v2 names all resolve to v2 profiles. **v2 is the new default.**
- `--rules strict-v1`, `--rules ab-v1`, and the explicit `CCZ-121-*-LG-v1`
  names resolve to the legacy v1 profiles. Use these to replay
  pre-2026-05-21 logs.
- Multiplayer profiles (`mp3`, `mp4`, `mp6`) are unaffected.

### Tests

`tests/core_tests.cpp` now covers both v1 and v2 anti-block behavior in a
single test block. v1: `home_count > 0` suppresses the terminal. v2: the
terminal fires regardless of `home_count`. Both: the `own > 0`,
`blockers > 0`, and `own + blockers == 10` conditions are still required.

## Reproducibility

Pre-2026-05-21 benchmark artifacts (the `ccerl-v1-rc1` release package, the
4128-game JumpStar60-vs-JumpStar57 direct match logs, and the open-source
bot screening data) remain bit-for-bit reproducible by passing
`--rules strict-v1` to all `cczero` invocations. Their published numbers
should not be retroactively re-stated under v2 rules; instead they should be
re-stated as having been computed under v1 and superseded.

The first v2 benchmark adds `iter_063` to the leaderboard and re-scores all
prior champions and adapters under the new rule.

## Expected impact on the leaderboard

Engines that primarily produced decisive results under v1 (most of the
leaderboard) should see only small Elo shifts under v2. The notable
exceptions, both inflated by the v1 loophole:

- `harryz-rule`: expected to drop from ~1471 toward the
  decisive-only estimate of ~-441 (still above strict random because some
  of its draws against weaker engines remain legitimate halves).
- `svjayanthi-mcts-fast`: smaller correction, since most of its outcomes
  were already decisive losses. Expected drop from ~891.

Numbers will be updated on the public benchmark page after the v2 re-run
completes.
