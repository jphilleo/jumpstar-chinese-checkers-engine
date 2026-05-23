#!/usr/bin/env python3
"""Expose a .ccpv CCZero model as a minimal CCP engine."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_best_move(args: argparse.Namespace, cells: str, player: int, ply: int, movetime_ms: int | None) -> str:
    simulations = args.simulations
    if movetime_ms is not None and args.movetime_simulations_scale > 0:
        simulations = max(1, min(args.max_simulations, movetime_ms // args.movetime_simulations_scale))
    cmd = [
        str(args.engine),
        "best-move",
        "--model",
        str(args.model),
        "--cells",
        cells,
        "--player",
        str(player),
        "--ply",
        str(ply),
        "--rules",
        args.rules,
        "--simulations",
        str(simulations),
        "--movegen",
        args.movegen,
        "--inference-backend",
        args.inference_backend,
        "--inference-batch-size",
        str(args.inference_batch_size),
    ]
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
        except FileNotFoundError:
            if attempt == 2:
                raise
            time.sleep(0.1 * (attempt + 1))
    payload = json.loads(completed.stdout)
    path = payload["move"]["path"]
    if not path:
        path = [payload["move"]["from"], payload["move"]["to"]]
    return "-".join(str(item) for item in path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=Path("build/cczero"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", default="cczero-ccpv")
    parser.add_argument("--rules", default="strict")
    parser.add_argument("--simulations", type=int, default=768)
    parser.add_argument("--max-simulations", type=int, default=1536)
    parser.add_argument("--movetime-simulations-scale", type=int, default=0)
    parser.add_argument("--movegen", default="bitboard")
    parser.add_argument("--inference-backend", default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    args = parser.parse_args()
    args.engine = args.engine if args.engine.is_absolute() else ROOT / args.engine
    args.model = args.model if args.model.is_absolute() else ROOT / args.model

    cells = ""
    player = 0
    ply = 0
    print(f"id name {args.name}", flush=True)
    print("id author CCZero", flush=True)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        command = parts[0]
        try:
            if command == "ccp":
                print("ccpok", flush=True)
            elif command == "protocol":
                print("protocol ccp 1", flush=True)
            elif command == "isready":
                print("readyok", flush=True)
            elif command == "position":
                if len(parts) >= 2 and parts[1] == "cells":
                    cells = parts[2]
                    if "player" in parts:
                        player = int(parts[parts.index("player") + 1])
                    if "ply" in parts:
                        ply = int(parts[parts.index("ply") + 1])
                else:
                    raise ValueError("expected: position cells <COMPACT121> player <0|1> ply <N>")
            elif command == "go":
                movetime_ms = None
                if "movetime" in parts:
                    movetime_ms = int(parts[parts.index("movetime") + 1])
                move = run_best_move(args, cells, player, ply, movetime_ms)
                print(f"bestmove {move}", flush=True)
            elif command == "quit":
                return 0
            else:
                print(f"info string ignored unknown command {command}", flush=True)
        except Exception as exc:  # Keep the protocol process alive for debuggability.
            print(f"error {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
