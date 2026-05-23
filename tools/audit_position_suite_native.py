#!/usr/bin/env python3
"""Audit a CCERL suite with native cczero match games in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any

from audit_position_suite import (
    aggregate,
    load_positions,
    parse_games,
    rel,
    repo_path,
    safe_name,
    summarize_positions,
    write_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = Path("benchmarks/ccerl-v1/baselines.json")
DEFAULT_POSITIONS = Path("benchmarks/ccerl-v1/positions/official_elo.jsonl")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def native_spec(baselines: dict[str, Any], label: str) -> dict[str, Any]:
    for entry in baselines.get("baselines", []):
        if entry.get("label") != label:
            continue
        if entry.get("family") == "neural":
            return {"label": label, "bot": "mcts", "model": entry["model"]}
        if entry.get("bot"):
            return {"label": label, "bot": entry["bot"], "model": None}
    raise SystemExit(f"unknown baseline label: {label}")


def append_player(cmd: list[str], side: str, spec: dict[str, Any]) -> None:
    cmd.extend([f"--{side}", str(spec["bot"])])
    if spec.get("model"):
        cmd.extend([f"--{side}-model", str(repo_path(spec["model"]))])


def parse_native_end(path: Path) -> dict[str, Any]:
    end = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "game_end":
                end = record
    if end is None:
        raise RuntimeError(f"native match log has no game_end: {path}")
    return end


def run_game(
    *,
    args: argparse.Namespace,
    game_id: int,
    position: dict[str, Any],
    p0: dict[str, Any],
    p1: dict[str, Any],
    native_log: Path,
) -> list[dict[str, Any]]:
    start = {
        "type": "ccp_game_start",
        "game_id": game_id,
        "position_id": position.get("id"),
        "ruleset": position.get("ruleset", "CCERL-2P10-v1"),
        "rule_profile": args.rules,
        "initial_cells": position["cells"],
        "initial_player": int(position["player"]),
        "initial_ply": int(position.get("ply", 0)),
        "p0": p0["label"],
        "p1": p1["label"],
        "max_plies": args.max_plies,
        "runner": "native_match",
        "native_log": rel(native_log),
    }
    cmd = [
        str(repo_path(args.engine)),
        "match",
        "--rules",
        args.rules,
        "--seed",
        str(args.seed + game_id),
        "--max-plies",
        str(args.max_plies),
        "--initial-cells",
        str(position["cells"]),
        "--initial-player",
        str(int(position["player"])),
        "--initial-ply",
        str(int(position.get("ply", 0))),
        "--mcts-simulations",
        str(args.simulations),
        "--mcts-movegen",
        args.movegen,
        "--mcts-inference-backend",
        args.inference_backend,
        "--mcts-inference-batch-size",
        str(args.inference_batch_size),
        "--log",
        str(native_log),
    ]
    append_player(cmd, "p0", p0)
    append_player(cmd, "p1", p1)
    native_log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    native_end = parse_native_end(native_log)
    end = {
        "type": "ccp_game_end",
        "game_id": game_id,
        "draw": bool(native_end.get("draw")),
        "winner": native_end.get("winner"),
        "reason": native_end.get("reason"),
        "plies": native_end.get("plies"),
        "runner": "native_match",
        "native_log": rel(native_log),
    }
    return [start, end]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cczero"))
    parser.add_argument("--engine-a-label", default="iter057")
    parser.add_argument("--engine-b-label", default="fresh046")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--simulations", type=int, default=704)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--rules", default="strict")
    parser.add_argument("--movegen", default="bitboard")
    parser.add_argument("--inference-backend", default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--target-score", type=float, default=0.65)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / f"{safe_name(args.engine_a_label)}_vs_{safe_name(args.engine_b_label)}.jsonl"
    native_log_dir = out_dir / "native_logs"
    if args.force and games_path.exists():
        games_path.unlink()
    baselines = load_json(repo_path(args.baselines))
    spec_a = native_spec(baselines, args.engine_a_label)
    spec_b = native_spec(baselines, args.engine_b_label)
    all_positions = load_positions(repo_path(args.positions), None)
    end = args.offset + args.limit if args.limit is not None else None
    positions = all_positions[args.offset : end]

    if not games_path.exists():
        futures = []
        records_by_game: dict[int, list[dict[str, Any]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            game_id = 0
            for position in positions:
                futures.append(
                    pool.submit(
                        run_game,
                        args=args,
                        game_id=game_id,
                        position=position,
                        p0=spec_a,
                        p1=spec_b,
                        native_log=native_log_dir / f"{game_id:05d}.jsonl",
                    )
                )
                game_id += 1
                futures.append(
                    pool.submit(
                        run_game,
                        args=args,
                        game_id=game_id,
                        position=position,
                        p0=spec_b,
                        p1=spec_a,
                        native_log=native_log_dir / f"{game_id:05d}.jsonl",
                    )
                )
                game_id += 1
            total = len(futures)
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                records = future.result()
                records_by_game[int(records[0]["game_id"])] = records
                if completed % 16 == 0 or completed == total:
                    print(json.dumps({"completed_games": completed, "total_games": total}), flush=True)
        with games_path.open("w", encoding="utf-8") as out:
            for game_id in sorted(records_by_game):
                for record in records_by_game[game_id]:
                    out.write(json.dumps(record, sort_keys=True) + "\n")

    games_by_position = parse_games(games_path, args.engine_a_label)
    rows = summarize_positions(
        positions=positions,
        games_by_position=games_by_position,
        engine_a_label=args.engine_a_label,
        target_score=args.target_score,
    )
    report = {
        "schema": "ccerl.position_audit.v1",
        "protocol": {
            "positions": rel(repo_path(args.positions)),
            "position_limit": args.limit,
            "position_offset": args.offset,
            "engine": rel(repo_path(args.engine)),
            "runner": "native_match",
            "engine_a_label": args.engine_a_label,
            "engine_b_label": args.engine_b_label,
            "engine_a_cmd": f"native:{spec_a['bot']}",
            "engine_b_cmd": f"native:{spec_b['bot']}",
            "neural_simulations": args.simulations,
            "max_plies": args.max_plies,
            "workers": args.workers,
            "seed": args.seed,
            "target_score": args.target_score,
        },
        "games_log": rel(games_path),
        "balanced_out": None,
        "summary": aggregate(rows),
        "balanced_positions": sorted(rows, key=lambda row: (float(row["side_bias"]), int(row["order"]))),
        "rows": rows,
    }
    report_path = out_dir / "suite_audit.json"
    markdown_path = out_dir / "suite_audit.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(markdown_path, report)
    print(
        json.dumps(
            {
                "report": rel(report_path),
                "markdown": rel(markdown_path),
                "games": rel(games_path),
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
