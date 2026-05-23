#!/usr/bin/env python3
"""Run CCP engines under the trusted CCZero referee."""

from __future__ import annotations

import argparse
import json
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EngineProc:
    label: str
    cmd: list[str]
    proc: subprocess.Popen[str]
    output: queue.Queue[str | None]

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefixes: tuple[str, ...], timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                line = self.output.get(timeout=remaining)
            except queue.Empty:
                break
            if line is None:
                raise RuntimeError(f"{self.label} exited")
            text = line.strip()
            if any(text.startswith(prefix) for prefix in prefixes):
                return text
        raise TimeoutError(f"{self.label} timed out waiting for {prefixes}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=1.0)
            except Exception:
                self.proc.kill()


def read_stdout(proc: subprocess.Popen[str], output: queue.Queue[str | None]) -> None:
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            output.put(line)
    finally:
        output.put(None)


def launch(label: str, command: str, timeout: float) -> EngineProc:
    proc = subprocess.Popen(
        shlex.split(command),
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    output: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=read_stdout, args=(proc, output), daemon=True).start()
    engine = EngineProc(label=label, cmd=shlex.split(command), proc=proc, output=output)
    engine.send("ccp")
    engine.read_until(("ccpok", "id ", "protocol "), timeout)
    engine.send("isready")
    engine.read_until(("readyok",), timeout)
    return engine


def run_json(cmd: list[str]) -> dict[str, Any]:
    last_error: FileNotFoundError | None = None
    for attempt in range(3):
        try:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            break
        except FileNotFoundError as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(0.1 * (attempt + 1))
    else:
        raise last_error or RuntimeError("failed to run JSON command")
    return json.loads(completed.stdout)


def inspect(engine: Path, rules: str, movegen: str, cells: str, player: int, ply: int) -> dict[str, Any]:
    return run_json(
        [
            str(engine),
            "position-info",
            "--rules",
            rules,
            "--movegen",
            movegen,
            "--cells",
            cells,
            "--player",
            str(player),
            "--ply",
            str(ply),
        ]
    )


def parse_move(text: str) -> list[int]:
    if text.startswith("bestmove "):
        text = text.split(None, 1)[1]
    text = text.strip()
    if not text or text == "0000":
        return []
    return [int(item) for item in text.replace(",", "-").split("-") if item]


def match_legal(path: list[int], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(path) < 2:
        return None
    exact = [move for move in legal if [int(item) for item in move.get("path", [])] == path]
    if exact:
        return exact[0]
    endpoint = [move for move in legal if int(move["from"]) == path[0] and int(move["to"]) == path[-1]]
    if len(endpoint) == 1:
        return endpoint[0]
    return None


def apply_move(cells: str, player: int, move: dict[str, Any]) -> str:
    source = int(move["from"])
    target = int(move["to"])
    next_cells = list(cells)
    if next_cells[source] != str(player):
        raise RuntimeError("validated move source mismatch")
    if next_cells[target] != ".":
        raise RuntimeError("validated move target mismatch")
    next_cells[source] = "."
    next_cells[target] = str(player)
    return "".join(next_cells)


def load_positions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    positions = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            positions.append(json.loads(line))
            if limit is not None and len(positions) >= limit:
                break
    return positions


def play_game(
    *,
    args: argparse.Namespace,
    game_id: int,
    position: dict[str, Any],
    p0: EngineProc,
    p1: EngineProc,
    out,
) -> dict[str, Any]:
    cells = str(position["cells"])
    player = int(position["player"])
    ply = int(position.get("ply", 0))
    max_ply = max(args.max_plies, ply)
    engines = {0: p0, 1: p1}
    start = {
        "type": "ccp_game_start",
        "game_id": game_id,
        "position_id": position.get("id"),
        "ruleset": position.get("ruleset", "CCERL-2P10-v1"),
        "rule_profile": args.rules,
        "initial_cells": cells,
        "initial_player": player,
        "initial_ply": ply,
        "p0": p0.label,
        "p1": p1.label,
        "max_plies": max_ply,
    }
    out.write(json.dumps(start, sort_keys=True) + "\n")
    while ply < max_ply:
        info = inspect(args.engine, args.rules, args.movegen, cells, player, ply)
        if info["terminal"]["terminal"]:
            break
        engine = engines[player]
        engine.send(f"position cells {cells} player {player} ply {ply}")
        engine.send(f"go movetime {args.movetime_ms}")
        try:
            response = engine.read_until(("bestmove",), args.timeout_seconds)
            requested = parse_move(response)
            legal = match_legal(requested, info["legal"])
        except Exception as exc:
            winner = 1 - player
            end = {
                "type": "ccp_game_end",
                "game_id": game_id,
                "draw": False,
                "winner": winner,
                "reason": "protocol_error",
                "detail": str(exc),
                "plies": ply,
            }
            out.write(json.dumps(end, sort_keys=True) + "\n")
            return end
        if legal is None:
            end = {
                "type": "ccp_game_end",
                "game_id": game_id,
                "draw": False,
                "winner": 1 - player,
                "reason": "illegal_move",
                "detail": response,
                "plies": ply,
            }
            out.write(json.dumps(end, sort_keys=True) + "\n")
            return end
        cells = apply_move(cells, player, legal)
        move_record = {
            "type": "ccp_move",
            "game_id": game_id,
            "ply": ply,
            "player": player,
            "engine": engine.label,
            "from": int(legal["from"]),
            "to": int(legal["to"]),
            "path": legal.get("path", []),
            "cells_after": cells,
        }
        out.write(json.dumps(move_record, sort_keys=True) + "\n")
        player = 1 - player
        ply += 1

    info = inspect(args.engine, args.rules, args.movegen, cells, player, ply)
    terminal = info["terminal"]
    if terminal["terminal"]:
        end = {
            "type": "ccp_game_end",
            "game_id": game_id,
            "draw": bool(terminal["draw"]),
            "winner": terminal["winner"],
            "reason": terminal["reason"],
            "plies": ply,
        }
    else:
        end = {
            "type": "ccp_game_end",
            "game_id": game_id,
            "draw": True,
            "winner": None,
            "reason": "max_ply",
            "plies": ply,
        }
    out.write(json.dumps(end, sort_keys=True) + "\n")
    return end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=Path("build/cczero"))
    parser.add_argument("--engine-a-cmd", required=True)
    parser.add_argument("--engine-b-cmd", required=True)
    parser.add_argument("--engine-a-label", default="engine_a")
    parser.add_argument("--engine-b-label", default="engine_b")
    parser.add_argument("--positions", type=Path, default=Path("benchmarks/ccerl-v1/positions/official_elo.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rules", default="strict")
    parser.add_argument("--movegen", default="bitboard")
    parser.add_argument("--movetime-ms", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--no-swap", action="store_true")
    args = parser.parse_args()
    args.engine = args.engine if args.engine.is_absolute() else ROOT / args.engine
    positions_path = args.positions if args.positions.is_absolute() else ROOT / args.positions
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    positions = load_positions(positions_path, args.limit)

    engine_a = launch(args.engine_a_label, args.engine_a_cmd, args.timeout_seconds)
    engine_b = launch(args.engine_b_label, args.engine_b_cmd, args.timeout_seconds)
    results = []
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as out:
            game_id = 0
            for position in positions:
                results.append(
                    play_game(args=args, game_id=game_id, position=position, p0=engine_a, p1=engine_b, out=out)
                )
                game_id += 1
                if not args.no_swap:
                    results.append(
                        play_game(args=args, game_id=game_id, position=position, p0=engine_b, p1=engine_a, out=out)
                    )
                    game_id += 1
    finally:
        engine_a.close()
        engine_b.close()

    print(json.dumps({"out": str(out_path), "games": len(results)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
