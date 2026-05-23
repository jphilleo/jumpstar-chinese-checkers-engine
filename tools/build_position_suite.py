#!/usr/bin/env python3
"""Build a frozen CCERL official Elo position suite.

The generator intentionally samples from several cheap policy families instead
of pure random plies: random, distance-greedy, traffic/conversion heuristics,
and shallow search baselines. The default path stays lightweight enough to rerun
in CI/local smoke.

The emitted JSONL is a schedule, not merely a set of unique positions. Clean
initial starts are repeated explicitly so the referee can consume the file in
order and preserve the intended weighting without a separate scheduler.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOT_SOURCES = (
    "random:random",
    "greedy:traffic-greedy",
    "traffic-greedy:converter",
    "converter:tt-pvs",
    "beam:pvs",
    "pvs:converter",
)


@dataclass(frozen=True)
class Candidate:
    record: dict[str, Any]
    diversity_key: tuple[Any, ...]
    balance: float


def run_json(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def run_match(cmd: list[str], log_path: Path) -> None:
    subprocess.run(
        [*cmd, "--log", str(log_path)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def initial_cells(engine: Path, rules: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "initial.jsonl"
        run_match(
            [
                str(engine),
                "match",
                "--rules",
                rules,
                "--p0",
                "random",
                "--p1",
                "random",
                "--seed",
                "1",
                "--max-plies",
                "1",
            ],
            log_path,
        )
        start = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        return str(start["initial_cells"])


def apply_move(cells: str, move: dict[str, Any]) -> str:
    player = str(move["player"])
    source = int(move["from"])
    target = int(move["to"])
    next_cells = list(cells)
    if next_cells[source] != player:
        raise ValueError(f"source {source} does not contain player {player}")
    if next_cells[target] != ".":
        raise ValueError(f"target {target} is occupied")
    next_cells[source] = "."
    next_cells[target] = player
    return "".join(next_cells)


def position_after_log(path: Path) -> tuple[str, int, int, bool]:
    cells = ""
    player = 0
    ply = 0
    terminal = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "game_start":
                cells = str(record["initial_cells"])
                player = int(record.get("initial_player", 0))
                ply = int(record.get("initial_ply", 0))
            elif record.get("type") == "move":
                cells = apply_move(cells, record)
                ply = int(record["ply"]) + 1
                player = 1 - int(record["player"])
            elif record.get("type") == "game_end":
                terminal = bool(record.get("draw")) or record.get("winner") is not None
    if not cells:
        raise ValueError(f"missing game_start in {path}")
    return cells, player, ply, terminal


def inspect_position(engine: Path, rules: str, movegen: str, cells: str, player: int, ply: int) -> dict[str, Any]:
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


def phase_for_ply(ply: int) -> str:
    if ply == 0:
        return "initial"
    if ply <= 6:
        return "shallow"
    if ply <= 14:
        return "normal"
    return "early_midgame"


def candidate_from_position(
    *,
    engine: Path,
    rules: str,
    movegen: str,
    cells: str,
    player: int,
    ply: int,
    source: dict[str, Any],
    suite: str,
    index_hint: int,
) -> Candidate | None:
    info = inspect_position(engine, rules, movegen, cells, player, ply)
    if info["terminal"]["terminal"] or int(info["legal_count"]) <= 0:
        return None
    features = info["features"]
    p0 = features["p0"]
    p1 = features["p1"]
    distance_delta = int(p0["goal_distance"]) - int(p1["goal_distance"])
    goal_delta = int(p0["goal_count"]) - int(p1["goal_count"])
    home_delta = int(p0["home_count"]) - int(p1["home_count"])
    balance = abs(distance_delta) + 5.0 * abs(goal_delta) + 1.5 * abs(home_delta)
    phase = phase_for_ply(ply)
    record = {
        "type": "ccerl_position",
        "schema": "ccerl.position.v1",
        "suite": suite,
        "ruleset": "CCERL-2P10-v1",
        "rule_profile": info["rule_profile"],
        "id": f"{suite}_{index_hint:04d}",
        "cells": cells,
        "player": player,
        "ply": ply,
        "hash": info["hash"],
        "phase": phase,
        "source": source,
        "features": {
            "legal_count": int(info["legal_count"]),
            "p0_goal_distance": int(p0["goal_distance"]),
            "p1_goal_distance": int(p1["goal_distance"]),
            "distance_delta": distance_delta,
            "p0_goal_count": int(p0["goal_count"]),
            "p1_goal_count": int(p1["goal_count"]),
            "goal_delta": goal_delta,
            "p0_home_count": int(p0["home_count"]),
            "p1_home_count": int(p1["home_count"]),
            "home_delta": home_delta,
            "balance_score": balance,
        },
    }
    diversity_key = (
        phase,
        int(info["legal_count"]) // 8,
        round(distance_delta / 4),
        goal_delta,
        home_delta,
        source.get("family"),
    )
    return Candidate(record=record, diversity_key=diversity_key, balance=balance)


def generate_bot_candidates(args: argparse.Namespace, engine: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    bot_sources = [item for item in args.bot_source if item]
    depths = [int(item) for item in args.depths.split(",") if item.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        game_index = 0
        for source_index, spec in enumerate(bot_sources):
            p0, p1 = spec.split(":", 1)
            for depth in depths:
                for seed_offset in range(args.seeds_per_source):
                    seed = args.seed + source_index * 100003 + depth * 1009 + seed_offset * 7919
                    log_path = tmp_path / f"candidate_{game_index:05d}.jsonl"
                    cmd = [
                        str(engine),
                        "match",
                        "--rules",
                        args.rules,
                        "--p0",
                        p0,
                        "--p1",
                        p1,
                        "--seed",
                        str(seed),
                        "--max-plies",
                        str(depth),
                    ]
                    run_match(cmd, log_path)
                    cells, player, ply, terminal = position_after_log(log_path)
                    if terminal and ply < depth:
                        continue
                    candidate = candidate_from_position(
                        engine=engine,
                        rules=args.rules,
                        movegen=args.movegen,
                        cells=cells,
                        player=player,
                        ply=ply,
                        source={
                            "family": "bot",
                            "p0": p0,
                            "p1": p1,
                            "depth": depth,
                            "seed": seed,
                        },
                        suite=args.suite,
                        index_hint=game_index,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                    game_index += 1
    return candidates


def select_diverse(candidates: list[Candidate], target: int) -> list[Candidate]:
    if target <= 0:
        return []
    selected: list[Candidate] = []
    seen_positions: set[tuple[str, int]] = set()
    seen_keys: set[tuple[Any, ...]] = set()
    by_phase = {"initial": 0, "shallow": 0, "normal": 0, "early_midgame": 0}
    phase_caps = {
        "initial": max(1, math.ceil(target * 0.05)),
        "shallow": max(1, math.ceil(target * 0.25)),
        "normal": max(1, math.ceil(target * 0.50)),
        "early_midgame": max(1, target),
    }
    ordered = sorted(candidates, key=lambda item: (item.balance, item.record["ply"], item.record["hash"]))
    for prefer_new_key in (True, False):
        for candidate in ordered:
            key = (candidate.record["cells"], int(candidate.record["player"]))
            if key in seen_positions:
                continue
            phase = candidate.record["phase"]
            if by_phase.get(phase, 0) >= phase_caps.get(phase, target):
                continue
            if prefer_new_key and candidate.diversity_key in seen_keys:
                continue
            selected.append(candidate)
            seen_positions.add(key)
            seen_keys.add(candidate.diversity_key)
            by_phase[phase] = by_phase.get(phase, 0) + 1
            if len(selected) >= target:
                return selected
    return selected


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def clone_record(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=Path("build/cczero"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/ccerl-v1/positions/official_elo.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/ccerl-v1/positions/official_elo.report.json"))
    parser.add_argument("--suite", default="official_elo")
    parser.add_argument("--rules", default="strict")
    parser.add_argument("--movegen", default="bitboard")
    parser.add_argument("--target", type=int, default=128)
    parser.add_argument("--initial-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=960000)
    parser.add_argument("--depths", default="4,8,12,16,24")
    parser.add_argument("--seeds-per-source", type=int, default=6)
    parser.add_argument("--bot-source", action="append", default=list(DEFAULT_BOT_SOURCES))
    args = parser.parse_args()

    engine = args.engine if args.engine.is_absolute() else ROOT / args.engine
    if not engine.exists():
        raise SystemExit(f"engine does not exist: {engine}")
    if args.target <= 0:
        raise SystemExit("--target must be positive")
    if args.initial_fraction < 0.0 or args.initial_fraction > 1.0:
        raise SystemExit("--initial-fraction must be between 0 and 1")

    cells = initial_cells(engine, args.rules)
    initial = candidate_from_position(
        engine=engine,
        rules=args.rules,
        movegen=args.movegen,
        cells=cells,
        player=0,
        ply=0,
        source={"family": "initial"},
        suite=args.suite,
        index_hint=0,
    )

    initial_count = math.ceil(args.target * args.initial_fraction)
    initial_count = max(0, min(args.target, initial_count))
    if initial_count > 0 and initial is None:
        raise SystemExit("failed to inspect the initial position")

    opening_target = args.target - initial_count
    candidates = generate_bot_candidates(args, engine)
    selected = select_diverse(candidates, opening_target)
    if len(selected) < opening_target:
        raise SystemExit(f"selected {len(selected)} opening positions, need {opening_target}")

    initial_records = []
    if initial is not None:
        for repeat in range(initial_count):
            record = clone_record(initial.record)
            record["source"] = {
                "family": "initial",
                "repeat": repeat,
                "reason": "clean_start_floor",
            }
            initial_records.append(record)
    opening_records = []
    for candidate in selected:
        record = clone_record(candidate.record)
        opening_records.append(record)

    records = []
    initial_index = 0
    opening_index = 0
    for slot in range(args.target):
        expected_initial = round((slot + 1) * args.initial_fraction)
        should_take_initial = (
            initial_index < len(initial_records)
            and (initial_index < expected_initial or opening_index >= len(opening_records))
        )
        if should_take_initial:
            records.append(initial_records[initial_index])
            initial_index += 1
        elif opening_index < len(opening_records):
            records.append(opening_records[opening_index])
            opening_index += 1
    for index, record in enumerate(records):
        record["id"] = f"{args.suite}_{index:04d}"
    out = args.out if args.out.is_absolute() else ROOT / args.out
    report = args.report if args.report.is_absolute() else ROOT / args.report
    write_jsonl(out, records)
    phase_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for record in records:
        phase_counts[record["phase"]] = phase_counts.get(record["phase"], 0) + 1
        family = record["source"].get("family", "unknown")
        source_counts[family] = source_counts.get(family, 0) + 1
    report_payload = {
        "suite": args.suite,
        "rules": args.rules,
        "movegen": args.movegen,
        "target": args.target,
        "initial_fraction": args.initial_fraction,
        "initial_records": initial_count,
        "opening_records": len(selected),
        "candidates": len(candidates) + (1 if initial is not None else 0),
        "bot_candidates": len(candidates),
        "selected": len(records),
        "out": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
        "phase_counts": phase_counts,
        "source_counts": source_counts,
        "bot_sources": args.bot_source,
        "depths": [int(item) for item in args.depths.split(",") if item.strip()],
        "seed": args.seed,
    }
    report.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
