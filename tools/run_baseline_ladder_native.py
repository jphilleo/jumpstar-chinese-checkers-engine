#!/usr/bin/env python3
"""Run a baseline ladder with native cczero match games."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from audit_position_suite import load_positions, rel, repo_path, safe_name
from audit_position_suite_native import append_player, load_json, native_spec
from run_baseline_ladder import pair_specs, select_baselines, summarize_pair, write_markdown


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = Path("benchmarks/ccerl-v1/baselines.json")


def run_pair_native(
    *,
    args: argparse.Namespace,
    pair_index: int,
    label_a: str,
    label_b: str,
    spec_a: dict[str, Any],
    spec_b: dict[str, Any],
    positions: list[dict[str, Any]],
    games_path: Path,
) -> None:
    native_log_dir = games_path.parent / "native_logs" / f"{pair_index:03d}_{safe_name(label_a)}_vs_{safe_name(label_b)}"
    native_log = native_log_dir / "suite.jsonl"
    cmd = [
        str(repo_path(args.engine)),
        "match-suite",
        "--rules",
        args.rules,
        "--seed",
        str(args.seed),
        "--max-plies",
        str(args.max_plies),
        "--positions",
        str(repo_path(args.positions)),
        "--mcts-simulations",
        str(args.simulations),
        "--mcts-movegen",
        args.movegen,
        "--mcts-inference-backend",
        args.inference_backend,
        "--mcts-inference-batch-size",
        str(args.inference_batch_size),
        "--workers",
        str(args.workers),
        "--log",
        str(native_log),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.no_swap:
        cmd.append("--no-swap")
    append_player(cmd, "p0", spec_a)
    append_player(cmd, "p1", spec_b)
    native_log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

    native_ends: list[dict[str, Any]] = []
    with native_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "game_end":
                native_ends.append(record)

    expected_games = len(positions) * (1 if args.no_swap else 2)
    if len(native_ends) != expected_games:
        raise RuntimeError(f"expected {expected_games} native game_end records, found {len(native_ends)} in {native_log}")

    records_by_game: dict[int, list[dict[str, Any]]] = {}
    game_id = 0
    for position in positions:
        for p0, p1 in ((spec_a, spec_b),) if args.no_swap else ((spec_a, spec_b), (spec_b, spec_a)):
            native_end = native_ends[game_id]
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
                "runner": "native_match_suite",
                "native_log": rel(native_log),
            }
            end = {
                "type": "ccp_game_end",
                "game_id": game_id,
                "draw": bool(native_end.get("draw")),
                "winner": native_end.get("winner"),
                "reason": native_end.get("reason"),
                "plies": native_end.get("plies"),
                "runner": "native_match_suite",
                "native_log": rel(native_log),
            }
            records_by_game[game_id] = [start, end]
            game_id += 1
    print(
        json.dumps(
            {
                "pair": pair_index,
                "a": label_a,
                "b": label_b,
                "completed_games": expected_games,
                "total_games": expected_games,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    games_path.parent.mkdir(parents=True, exist_ok=True)
    with games_path.open("w", encoding="utf-8") as out:
        for game_id in sorted(records_by_game):
            for record in records_by_game[game_id]:
                out.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cczero"))
    parser.add_argument("--label", action="append")
    parser.add_argument("--pairs", choices=("adjacent", "anchor", "all"), default="all")
    parser.add_argument("--anchor")
    parser.add_argument("--anchor-elo", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--simulations", type=int, default=704)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=980000)
    parser.add_argument("--rules", default="strict")
    parser.add_argument("--movegen", default="bitboard")
    parser.add_argument("--inference-backend", default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--draws", choices=("half", "ignore"), default="half")
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    baselines_path = repo_path(args.baselines)
    payload = load_json(baselines_path)
    args.positions = args.positions or Path(payload.get("position_schedule", "benchmarks/ccerl-v1/positions/official_elo_v2.jsonl"))
    args.engine = repo_path(args.engine)
    anchor = args.anchor or payload.get("anchor", {}).get("label", "random")
    anchor_elo = args.anchor_elo if args.anchor_elo is not None else float(payload.get("anchor", {}).get("elo", 0.0))

    selected = select_baselines(payload, args.label)
    labels = [str(entry["label"]) for entry in selected]
    by_label = {str(entry["label"]): entry for entry in selected}
    specs = {label: native_spec(payload, label) for label in labels}
    positions = load_positions(repo_path(args.positions), args.limit)
    out_dir = repo_path(args.out_dir)
    games_dir = out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    log_paths = []
    for index, (label_a, label_b) in enumerate(pair_specs(labels, args.pairs, anchor)):
        games_path = games_dir / f"{index:03d}_{safe_name(label_a)}_vs_{safe_name(label_b)}.jsonl"
        if args.force or not games_path.exists():
            print(json.dumps({"pair": index, "a": label_a, "b": label_b, "out": rel(games_path)}, sort_keys=True), flush=True)
            run_pair_native(
                args=args,
                pair_index=index,
                label_a=label_a,
                label_b=label_b,
                spec_a=specs[label_a],
                spec_b=specs[label_b],
                positions=positions,
                games_path=games_path,
            )
        pair_rows.append(summarize_pair(games_path, label_a, label_b))
        log_paths.append(games_path)

    elo_path = out_dir / "elo.json"
    cmd = [
        "python3",
        "tools/estimate_elo.py",
        *[str(path) for path in log_paths],
        "--anchor",
        anchor,
        "--anchor-elo",
        str(anchor_elo),
        "--draws",
        args.draws,
        "--out",
        str(elo_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    elo = load_json(elo_path)
    ladder = {
        "schema": "ccerl.baseline_ladder.v1",
        "baselines": {label: by_label[label] for label in labels},
        "protocol": {
            "runner": "native_match_suite",
            "baselines": rel(baselines_path),
            "positions": rel(repo_path(args.positions)),
            "position_limit": args.limit,
            "pairs": args.pairs,
            "anchor": anchor,
            "anchor_elo": anchor_elo,
            "max_plies": args.max_plies,
            "simulations": args.simulations,
            "neural_simulations": args.simulations,
            "workers": args.workers,
            "seed": args.seed,
            "draws": args.draws,
            "color_swap": not args.no_swap,
        },
        "pairs": pair_rows,
        "elo": elo,
    }
    report_path = out_dir / "baseline_ladder.json"
    markdown_path = out_dir / "baseline_ladder.md"
    report_path.write_text(json.dumps(ladder, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(markdown_path, ladder)
    print(json.dumps({"report": rel(report_path), "markdown": rel(markdown_path), "pairs": len(pair_rows), "games": elo["games"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
