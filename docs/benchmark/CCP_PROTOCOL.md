# CCP Protocol

`CCP` is the Chinese Checkers Protocol for external engines. The public rating
runner uses this protocol so C++, Python, Rust, JavaScript, neural, and
remote-wrapped engines can all be tested by the same referee.

The protocol is text over stdin/stdout. The referee owns legality, clocks,
terminal detection, logging, and adjudication.

Current tools:

- `tools/ccp_referee.py`: trusted subprocess referee for CCP engines.
- `tools/ccp_cczero_engine.py`: adapter that exposes a `.ccpv` model as a CCP
  engine, used for Iter57 and checkpoint baselines.
- `tools/ccp_cczero_bot.py`: adapter that exposes built-in algorithmic bots as
  CCP engines, used for `random`, `greedy`, `traffic-greedy`, `converter`, and
  `tt-pvs` baselines.
- `./build/cczero position-info`: C++ legality/terminal oracle used by the
  referee for every move.

## Minimum Engine

An engine must support:

```text
ccp
isready
position cells <COMPACT121> player <0|1> ply <N>
go movetime <MS>
bestmove <FROM>-<TO>
quit
```

`COMPACT121` is the repository's compact board encoding: one character per cell
id, using `.` for empty, `0` for player 0, and `1` for player 1.

For hop chains, an engine should return the full witness path when it knows it:

```text
bestmove 4-15-28-41
```

The referee may also accept an endpoint-only move when the endpoint is
unambiguous among legal moves from the current position:

```text
bestmove 4-41
```

## Recommended Commands

```text
ccp
protocol
isready
setoption name Threads value 8
setoption name GPU value 0
position startpos
position cells <COMPACT121> player 1 ply 73
go movetime 1000
go nodes 100000
go depth 12
go wtime 600000 btime 600000
stop
quit
```

## Recommended Responses

```text
id name MyEngine
id author Example
option name Threads type spin default 1 min 1 max 128
readyok
info depth 8 nodes 24109 score cp 42 pv 4-15-28
bestmove 4-15-28
```

## Analysis Extension

Benchmark position analysis should use:

```text
analyze topk 10
```

and return JSON on one line:

```text
analysis {"value":0.17,"moves":[{"move":"4-15-28","policy":0.31},{"move":"5-16","policy":0.22}]}
```

This extension is optional for match play, but useful for tactical suites and
research leaderboards.

## Runner Contract

The official runner:

- launch engines as subprocesses or Docker containers;
- send only legal positions;
- validate every returned move against the C++ referee;
- forfeit illegal moves, protocol timeouts, crashes, and stderr floods according
  to the benchmark rules;
- record all games as JSONL with engine metadata, ruleset, hardware, clocks,
  seeds or position ids, and termination reason.
