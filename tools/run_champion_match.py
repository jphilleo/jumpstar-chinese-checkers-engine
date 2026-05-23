#!/usr/bin/env python3
"""Run a high-confidence direct champion/contender match on the CCERL suite."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from audit_position_suite import load_positions, rel, repo_path, safe_name
from audit_position_suite_native import append_player, load_json, native_spec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = Path("benchmarks/ccerl-v1/baselines.json")
DEFAULT_POSITIONS = Path("benchmarks/ccerl-v1/positions/official_elo_v2.jsonl")


def wilson_interval(points: float, games: int, z: float = 1.96) -> list[float]:
    if games <= 0:
        return [0.0, 0.0]
    p = points / games
    denom = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def expand_positions(source: list[dict[str, Any]], *, cycles: int | None, target_games: int | None) -> list[dict[str, Any]]:
    if not source:
        raise SystemExit("no positions loaded")
    if cycles is None:
        requested_games = target_games if target_games is not None else 1000
        starts = max(1, math.ceil(requested_games / 2))
    else:
        starts = max(1, cycles * len(source))
    rows = []
    for index in range(starts):
        original = dict(source[index % len(source)])
        repeat = index // len(source)
        original["source_position_id"] = original.get("source_position_id", original.get("id"))
        original["id"] = f"{original.get('id', 'position')}_direct_{repeat:03d}"
        original["direct_match_repeat"] = repeat
        rows.append(original)
    return rows


def write_positions(path: Path, positions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in positions:
            out.write(json.dumps(row, sort_keys=True) + "\n")


def native_ends(path: Path) -> list[dict[str, Any]]:
    ends = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "game_end":
                ends.append(record)
    return ends


def write_games(
    *,
    games_path: Path,
    native_log: Path,
    positions: list[dict[str, Any]],
    candidate: dict[str, Any],
    champion: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    ends = native_ends(native_log)
    expected = len(positions) * 2
    if len(ends) != expected:
        raise RuntimeError(f"expected {expected} native game_end records, found {len(ends)}")
    games_path.parent.mkdir(parents=True, exist_ok=True)
    with games_path.open("w", encoding="utf-8") as out:
        game_id = 0
        for position in positions:
            for p0, p1 in ((candidate, champion), (champion, candidate)):
                start = {
                    "type": "ccp_game_start",
                    "game_id": game_id,
                    "position_id": position.get("id"),
                    "source_position_id": position.get("source_position_id"),
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
                native_end = ends[game_id]
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
                out.write(json.dumps(start, sort_keys=True) + "\n")
                out.write(json.dumps(end, sort_keys=True) + "\n")
                game_id += 1


def candidate_score(games_path: Path, candidate: str) -> tuple[float, int, int, int, int]:
    current_start: dict[str, Any] | None = None
    wins = draws = losses = games = 0
    points = 0.0
    with games_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "ccp_game_start":
                current_start = record
            elif record.get("type") == "ccp_game_end" and current_start is not None:
                if bool(record.get("draw")) or record.get("winner") is None:
                    score = 0.5
                else:
                    winner_label = current_start["p0"] if int(record["winner"]) == 0 else current_start["p1"]
                    score = 1.0 if winner_label == candidate else 0.0
                games += 1
                points += score
                if score == 1.0:
                    wins += 1
                elif score == 0.5:
                    draws += 1
                else:
                    losses += 1
                current_start = None
    return points, games, wins, draws, losses


def write_champion_markdown(path: Path, report: dict[str, Any]) -> None:
    direct = report["direct"]
    elo = report["elo_delta"]
    sprt = report["sprt"]
    lines = [
        "# CCERL Champion Direct Match",
        "",
        f"Candidate: `{report['candidate']}`",
        f"Champion: `{report['champion']}`",
        "",
        "## Result",
        "",
        "| Games | W-D-L | Score | Score CI95 | Elo Delta | Elo CI95 | SPRT |",
        "|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {direct['games']} | {direct['wins']}-{direct['draws']}-{direct['losses']} | "
            f"{direct['score']:.4f} | {direct['score_ci95'][0]:.4f}..{direct['score_ci95'][1]:.4f} | "
            f"{elo['elo']:.1f} | {elo['ci95'][0]:.1f}..{elo['ci95'][1]:.1f} | {sprt['decision']} |"
        ),
        "",
        "## Protocol",
        "",
    ]
    for key, value in report["protocol"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- games: `{report['artifacts']['games']}`",
            f"- native log: `{report['artifacts']['native_log']}`",
            f"- Elo: `{report['artifacts']['elo']}`",
            f"- SPRT: `{report['artifacts']['sprt']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cczero"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--target-games", type=int, default=1000)
    parser.add_argument("--cycles", type=int, help="Full cycles through the loaded position suite. Overrides --target-games.")
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--simulations", type=int, default=704)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=990000)
    parser.add_argument("--rules", default="strict")
    parser.add_argument("--movegen", default="bitboard")
    parser.add_argument("--inference-backend", default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--elo1", type=float, default=25.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = load_json(repo_path(args.baselines))
    candidate = native_spec(payload, args.candidate)
    champion = native_spec(payload, args.champion)
    source_positions = load_positions(repo_path(args.positions), None)
    positions = expand_positions(source_positions, cycles=args.cycles, target_games=args.target_games)

    out_dir = repo_path(args.out_dir)
    expanded_positions = out_dir / "positions.direct_match.jsonl"
    native_log = out_dir / "native" / f"{safe_name(args.candidate)}_vs_{safe_name(args.champion)}.jsonl"
    games_path = out_dir / "games.jsonl"
    elo_path = out_dir / "elo_delta.json"
    sprt_path = out_dir / "sprt.json"
    report_path = out_dir / "champion_match.json"
    markdown_path = out_dir / "champion_match.md"

    if report_path.exists() and not args.force:
        raise SystemExit(f"report already exists, pass --force: {report_path}")

    write_positions(expanded_positions, positions)
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
        str(expanded_positions),
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
    append_player(cmd, "p0", candidate)
    append_player(cmd, "p1", champion)
    native_log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

    write_games(
        games_path=games_path,
        native_log=native_log,
        positions=positions,
        candidate=candidate,
        champion=champion,
        args=args,
    )

    subprocess.run(
        [
            "python3",
            "tools/estimate_elo.py",
            str(games_path),
            "--anchor",
            args.champion,
            "--anchor-elo",
            "0",
            "--out",
            str(elo_path),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "python3",
            "tools/sprt_decision.py",
            str(games_path),
            "--candidate",
            args.candidate,
            "--opponent",
            args.champion,
            "--elo0",
            "0",
            "--elo1",
            str(args.elo1),
            "--alpha",
            str(args.alpha),
            "--beta",
            str(args.beta),
            "--out",
            str(sprt_path),
        ],
        cwd=ROOT,
        check=True,
    )

    elo_report = load_json(elo_path)
    sprt_report = load_json(sprt_path)
    points, games, wins, draws, losses = candidate_score(games_path, args.candidate)
    candidate_row = next(row for row in elo_report["ratings"] if row["player"] == args.candidate)
    report = {
        "schema": "ccerl.champion_match.v1",
        "candidate": args.candidate,
        "champion": args.champion,
        "protocol": {
            "runner": "native_match_suite",
            "positions": rel(repo_path(args.positions)),
            "expanded_positions": rel(expanded_positions),
            "source_positions": len(source_positions),
            "paired_starts": len(positions),
            "games": games,
            "cycles": args.cycles,
            "target_games": args.target_games,
            "simulations": args.simulations,
            "max_plies": args.max_plies,
            "workers": args.workers,
            "seed": args.seed,
            "color_swap": True,
        },
        "direct": {
            "games": games,
            "points": points,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": points / games,
            "score_ci95": wilson_interval(points, games),
        },
        "elo_delta": {
            "anchor": args.champion,
            "elo": candidate_row["elo"],
            "ci95": candidate_row["ci95"],
            "standard_error": candidate_row["standard_error"],
        },
        "sprt": sprt_report,
        "artifacts": {
            "games": rel(games_path),
            "native_log": rel(native_log),
            "elo": rel(elo_path),
            "sprt": rel(sprt_path),
            "markdown": rel(markdown_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_champion_markdown(markdown_path, report)
    print(json.dumps({"report": rel(report_path), "markdown": rel(markdown_path), "games": games}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
