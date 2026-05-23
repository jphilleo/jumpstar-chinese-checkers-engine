# Rule Profiles

The engine exposes explicit rule profiles through `--rules`. As of
2026-05-21 the canonical 2-player profiles bumped from `v1` to `v2`, closing
an anti-block stall-draw loophole. The CLI's bare names (`strict`, `ab`)
default to `v2`; pass an explicit `-v1` suffix to opt back into the legacy
profile for replaying historical logs.

## Quick reference

| `--rules` | Full name | Anti-block rule |
|---|---|---|
| `strict`, `strict-v2` | `CCZ-121-Strict-LG-v2` | v2 — tightened, no home-vacate gate |
| `ab`, `ab-v2` | `CCZ-121-AB-LG-v2` | v2 — tightened |
| `strict-v1` | `CCZ-121-Strict-LG-v1` | v1 — requires home_count == 0 |
| `ab-v1` | `CCZ-121-AB-LG-v1` | v1 — requires home_count == 0 |
| `mp3` / `mp4` / `mp6` | `CCZ-121-MPn-v1` | unchanged (multiplayer) |

## v2 profiles (canonical)

### `strict` → `CCZ-121-Strict-LG-v2`

- 121-hole board, two players, 10 pieces each
- side triangles playable
- goal-locking enabled
- repetition and max-ply draws enabled
- wins when all 10 pieces occupy the player's own goal triangle
  (`all_pieces_in_goal`)
- **v2 anti-block:** wins when the player's goal triangle is fully occupied
  (own pieces plus opponent blockers fill every cell) and at least one cell
  is the player's own (`anti_block_goal_full`). The v1 requirement that the
  winning player had vacated their home triangle is **removed**.

Use this profile for self-play, evaluation, and the `CCERL-2P10-v2` public
benchmark rulebook documented in `docs/benchmark/CCERL-2P10-v2.md`.

### `ab` → `CCZ-121-AB-LG-v2`

Same as `strict-v2` above. Historically `ab` was the "research" name for the
anti-block profile and `strict` was the "pre-training" alias; the two are
functionally identical under v2. Either name selects the same profile.

## v1 profiles (legacy, preserved for reproducibility)

`CCZ-121-Strict-LG-v1` and `CCZ-121-AB-LG-v1` retain the original anti-block
gate that additionally required `home_count[player] == 0`. That gate created
a stall-draw exploit (see
[docs/benchmark/CHANGELOG_2P10_V1_TO_V2.md](CHANGELOG_2P10_V1_TO_V2.md)) and is the
sole behavioral difference between v1 and v2.

Use the explicit `-v1` flag when replaying logs collected before 2026-05-21,
the `CCERL-2P10-v1` benchmark, or any artifact stamped with a `-v1` rule
profile name.

## Examples

```sh
# canonical v2 work
./build/cczero match --rules strict --p0 converter --p1 tt-pvs --seed 7 \
    --max-plies 300 --log build/strict_v2_game.jsonl
./build/cczero dataset --rules strict --bot converter --opponent tt-pvs \
    --games 20 --out build/strict_v2_bootstrap.jsonl
./build/cczero tournament --rules strict \
    --bots random,greedy,traffic-greedy,hand-eval --games 2

# replay a v1 log for reproducibility
./build/cczero match --rules strict-v1 --p0 converter --p1 tt-pvs --seed 7 \
    --max-plies 240 --log build/strict_v1_replay.jsonl
```
