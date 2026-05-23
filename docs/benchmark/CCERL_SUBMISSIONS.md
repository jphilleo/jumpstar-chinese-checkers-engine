# CCERL Submission Contract

Status: release-candidate draft for `CCERL-2P10-v2`.

Official ratings are produced by the benchmark runner, not by self-reported
results. Self-reported logs are useful for debugging and reproduction, but they
do not enter the public rating list unless they are rerun or verified by an
official runner.

## Engine Interface

External engines should implement CCP over stdin/stdout:

```text
ccp
isready
position cells <COMPACT121> player <0|1> ply <N>
go movetime <MS>
bestmove <PATH>
quit
```

The referee owns legality, clocks, paired color swaps, termination, and result
logging. Engines should only choose moves.

## Submission Modes

Preferred modes, in order:

1. Docker image implementing CCP.
2. Local executable implementing CCP.
3. HTTP bridge implementing CCP-like requests, for preview runs only.

Official fixed-hardware ratings should use Docker or a local executable run by
the benchmark maintainers. HTTP endpoints are too environment-dependent for the
main list unless a separate open-hardware track is created.

## Docker Contract

A Docker submission should provide:

```json
{
  "schema": "ccerl.submission.v1",
  "engine": {
    "name": "ExampleEngine",
    "version": "2026.05.20",
    "author": "Example Team",
    "license": "proprietary-or-open-source-id",
    "source": "https://example.com/repo-or-release"
  },
  "runtime": {
    "mode": "docker",
    "image": "example/engine:2026.05.20",
    "command": ["/engine", "--ccp"],
    "working_dir": "/work"
  },
  "resources": {
    "track": "fixed-cpu",
    "threads": 8,
    "memory_gb": 16,
    "gpu": false
  }
}
```

## Initial Resource Classes

| Track | CPU | Memory | GPU | Notes |
|---|---:|---:|---|---|
| `fixed-cpu` | 8 vCPU | 16 GB | no | Serious public rating track. |
| `fixed-gpu` | 8 vCPU | 32 GB | one declared GPU | Neural-engine track once hosted GPU runners exist. |
| `open-hardware` | declared by submitter | declared by submitter | declared by submitter | Fun/comparison track only, not the main scientific list. |
| `fixed-simulation` | local/native | local/native | optional | Internal `.ccpv` checkpoint and baseline calibration track. |

Each published rating must include the track, hardware class, runner version,
ruleset, position suite, time/simulation control, game count, score, Elo, and
confidence interval.

## Process Limits

The official runner should enforce:

- wall-clock move timeout plus grace margin
- total process timeout per game
- memory limit
- CPU/thread limit
- no network for fixed-hardware official runs
- read-only benchmark inputs
- writable scratch directory only
- crash/illegal-move losses recorded with reason codes

## Rating Policy

The default public scale is anchored at `random = 0`. The champion is measured
by games and is not assigned a fixed rating by hand.

New submissions should first run a small smoke gauntlet against `random`,
`greedy`, `tt-pvs`, and the current champion. Official placement should then
use nearby-rated engines and paired positions until the confidence interval is
reasonable.

For direct improvement claims such as "Engine X is stronger than Iter60", use
`tools/ccbench.py sprt` on official logs with a declared null and alternative
Elo delta.

## Champion-Contender Matches

If a submitted engine lands near the champion on the broad ladder, switch from
rating-list mode to direct-match mode.

Trigger a champion-contender match when:

- the challenger is within roughly 150 Elo of the champion, or
- the challenger confidence interval overlaps the champion confidence interval,
  or
- the claim is specifically "this engine is stronger than the champion."

Default direct-match protocol:

- use the audited `official_elo_v2` schedule
- cycle the full schedule enough times to reach the target game count
- play every start as a paired color swap
- use the declared track's time/simulation control
- publish W-D-L, score CI, Elo delta CI, and SPRT decision

Recommended minimums:

- 500 games for a first promotion-grade check
- 1,000 games when the first check is close
- 2,000+ games when the expected edge is small or the champion title is at stake

Local command shape:

```sh
tools/ccbench.py champion-match \
  --candidate challenger \
  --champion iter060 \
  --cycles 12 \
  --simulations 704 \
  --out-dir experiments/public_benchmark/champion_match_challenger_vs_iter060
```
