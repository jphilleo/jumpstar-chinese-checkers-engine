#!/usr/bin/env python3
"""Audit a frozen CCERL position suite for side bias and rating usefulness."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = Path("benchmarks/ccerl-v1/baselines.json")
DEFAULT_POSITIONS = Path("benchmarks/ccerl-v1/positions/official_elo.jsonl")


def repo_path(path: Path | str) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else ROOT / parsed


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_positions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def baseline_command(entry: dict[str, Any], neural_simulations: int | None, engine: Path) -> str:
    label = str(entry["label"])
    family = str(entry.get("family", ""))
    engine_arg = f"--engine {engine}"
    if family == "neural" and entry.get("model"):
        simulations = neural_simulations if neural_simulations is not None else int(entry.get("simulations", 768))
        return (
            f"python3 tools/ccp_cczero_engine.py {engine_arg} --model {entry['model']} "
            f"--name {label} --simulations {simulations}"
        )
    if family == "algorithmic" and entry.get("bot"):
        return f"python3 tools/ccp_cczero_bot.py {engine_arg} --bot {entry['bot']} --name {label}"
    return str(entry["ccp_command"])


def command_for_label(
    *,
    baselines: dict[str, Any],
    label: str,
    override: str | None,
    neural_simulations: int | None,
    engine: Path,
) -> str:
    if override:
        return override
    for entry in baselines.get("baselines", []):
        if entry.get("label") == label:
            return baseline_command(entry, neural_simulations, engine)
    raise SystemExit(f"unknown baseline label and no command provided: {label}")


def run_referee(args: argparse.Namespace, games_path: Path, engine_a_cmd: str, engine_b_cmd: str) -> None:
    cmd = [
        "python3",
        "tools/ccp_referee.py",
        "--engine",
        str(repo_path(args.engine)),
        "--engine-a-label",
        args.engine_a_label,
        "--engine-b-label",
        args.engine_b_label,
        "--engine-a-cmd",
        engine_a_cmd,
        "--engine-b-cmd",
        engine_b_cmd,
        "--positions",
        str(repo_path(args.positions)),
        "--out",
        str(games_path),
        "--rules",
        args.rules,
        "--movegen",
        args.movegen,
        "--movetime-ms",
        str(args.movetime_ms),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-plies",
        str(args.max_plies),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    subprocess.run(cmd, cwd=ROOT, check=True)


def score_label(record: dict[str, Any], start: dict[str, Any], label: str) -> float:
    if bool(record.get("draw")) or record.get("winner") is None:
        return 0.5
    winner_label = start["p0"] if int(record["winner"]) == 0 else start["p1"]
    return 1.0 if winner_label == label else 0.0


def score_player0(record: dict[str, Any]) -> float:
    if bool(record.get("draw")) or record.get("winner") is None:
        return 0.5
    return 1.0 if int(record["winner"]) == 0 else 0.0


def parse_games(path: Path, engine_a_label: str) -> dict[str, list[dict[str, Any]]]:
    by_position: dict[str, list[dict[str, Any]]] = {}
    current_start: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "ccp_game_start":
                current_start = record
            elif record.get("type") == "ccp_game_end" and current_start is not None:
                position_id = str(current_start.get("position_id"))
                row = {
                    "game_id": current_start.get("game_id"),
                    "position_id": position_id,
                    "p0": current_start.get("p0"),
                    "p1": current_start.get("p1"),
                    "draw": bool(record.get("draw")),
                    "winner": record.get("winner"),
                    "reason": record.get("reason", "unknown"),
                    "plies": record.get("plies"),
                    "engine_a_score": score_label(record, current_start, engine_a_label),
                    "player0_score": score_player0(record),
                }
                by_position.setdefault(position_id, []).append(row)
                current_start = None
    return by_position


def summarize_positions(
    *,
    positions: list[dict[str, Any]],
    games_by_position: dict[str, list[dict[str, Any]]],
    engine_a_label: str,
    target_score: float,
) -> list[dict[str, Any]]:
    rows = []
    for order, position in enumerate(positions):
        position_id = str(position["id"])
        games = games_by_position.get(position_id, [])
        reasons: dict[str, int] = {}
        for game in games:
            reason = str(game.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        game_count = len(games)
        engine_a_score = sum(float(game["engine_a_score"]) for game in games) / game_count if game_count else 0.0
        player0_score = sum(float(game["player0_score"]) for game in games) / game_count if game_count else 0.0
        draw_rate = sum(1 for game in games if game["draw"]) / game_count if game_count else 0.0
        side_bias = abs(player0_score - 0.5) if game_count else 0.5
        bad_endings = sum(
            count
            for reason, count in reasons.items()
            if reason in {"illegal_move", "protocol_error", "timeout", "crash"}
        )
        rows.append(
            {
                "id": position_id,
                "order": order,
                "phase": position.get("phase", "unknown"),
                "source": position.get("source", "unknown"),
                "initial_player": position.get("player"),
                "initial_ply": position.get("ply", 0),
                "games": game_count,
                "engine_a": engine_a_label,
                "engine_a_score": engine_a_score,
                "player0_score": player0_score,
                "side_bias": side_bias,
                "draw_rate": draw_rate,
                "bad_endings": bad_endings,
                "target_distance": abs(engine_a_score - target_score),
                "reasons": reasons,
                "games_detail": games,
            }
        )
    return rows


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    incomplete = 0 if int(row["games"]) >= 2 else 1
    return (
        float(incomplete),
        float(row["bad_endings"]),
        float(row["side_bias"]),
        float(row["target_distance"]),
        int(row["order"]),
    )


def selectable_rows(rows: list[dict[str, Any]], exclude_ids: set[str], max_side_bias: float | None) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row["id"] in exclude_ids:
            continue
        if max_side_bias is not None and float(row["side_bias"]) > max_side_bias:
            continue
        selected.append(row)
    return selected


def write_balanced_positions(path: Path, positions: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    by_id = {str(position["id"]): position for position in positions}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(by_id[row["id"]], sort_keys=True) + "\n")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audited = [row for row in rows if int(row["games"]) > 0]
    if not audited:
        return {"positions": len(rows), "audited": 0}
    side_bias_values = [float(row["side_bias"]) for row in audited]
    engine_scores = [float(row["engine_a_score"]) for row in audited]
    draw_rates = [float(row["draw_rate"]) for row in audited]
    return {
        "positions": len(rows),
        "audited": len(audited),
        "mean_side_bias": sum(side_bias_values) / len(side_bias_values),
        "zero_side_bias_positions": sum(1 for value in side_bias_values if value == 0.0),
        "high_side_bias_positions": sum(1 for value in side_bias_values if value >= 0.5),
        "mean_engine_a_score": sum(engine_scores) / len(engine_scores),
        "mean_draw_rate": sum(draw_rates) / len(draw_rates),
        "bad_endings": sum(int(row["bad_endings"]) for row in audited),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# CCERL Position Suite Audit", ""]
    protocol = report["protocol"]
    summary = report["summary"]
    limit = protocol.get("position_limit")
    limit_text = str(limit) if limit is not None else "full"
    lines.append(
        f"Protocol: `{protocol['engine_a_label']}` vs `{protocol['engine_b_label']}`, "
        f"positions=`{limit_text}`, max plies=`{protocol['max_plies']}`, "
        f"neural sims=`{protocol['neural_simulations']}`."
    )
    lines.append("")
    lines.append(
        f"Audited {summary.get('audited', 0)} of {summary.get('positions', 0)} positions. "
        f"Mean side bias: {summary.get('mean_side_bias', 0.0):.3f}; "
        f"zero-bias positions: {summary.get('zero_side_bias_positions', 0)}; "
        f"high-bias positions: {summary.get('high_side_bias_positions', 0)}."
    )
    if report.get("balanced_positions"):
        lines.extend(["", "## Candidate Balanced Rows", ""])
        lines.extend(["| ID | Phase | A Score | P0 Score | Side Bias | Draw Rate | Reasons |", "|---|---|---:|---:|---:|---:|---|"])
        for row in report["balanced_positions"][:32]:
            reasons = ", ".join(f"{key}:{value}" for key, value in sorted(row["reasons"].items()))
            lines.append(
                f"| `{row['id']}` | {row['phase']} | {row['engine_a_score']:.3f} | "
                f"{row['player0_score']:.3f} | {row['side_bias']:.3f} | "
                f"{row['draw_rate']:.3f} | {reasons} |"
            )
    lines.extend(["", "## Highest Side Bias", ""])
    lines.extend(["| ID | Phase | A Score | P0 Score | Side Bias | Reasons |", "|---|---|---:|---:|---:|---|"])
    for row in sorted(report["rows"], key=lambda item: (-float(item["side_bias"]), int(item["order"])))[:32]:
        reasons = ", ".join(f"{key}:{value}" for key, value in sorted(row["reasons"].items()))
        lines.append(
            f"| `{row['id']}` | {row['phase']} | {row['engine_a_score']:.3f} | "
            f"{row['player0_score']:.3f} | {row['side_bias']:.3f} | {reasons} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cczero"))
    parser.add_argument("--engine-a-label", default="iter057")
    parser.add_argument("--engine-b-label", default="fresh046")
    parser.add_argument("--engine-a-cmd")
    parser.add_argument("--engine-b-cmd")
    parser.add_argument("--neural-simulations", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--movetime-ms", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--rules", default="strict")
    parser.add_argument("--movegen", default="bitboard")
    parser.add_argument("--target-score", type=float, default=0.65)
    parser.add_argument("--balanced-out", type=Path)
    parser.add_argument("--balanced-target", type=int)
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--max-side-bias", type=float)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    games_path = out_dir / f"{safe_name(args.engine_a_label)}_vs_{safe_name(args.engine_b_label)}.jsonl"
    baselines = load_json(repo_path(args.baselines))
    engine_path = repo_path(args.engine)
    engine_a_cmd = command_for_label(
        baselines=baselines,
        label=args.engine_a_label,
        override=args.engine_a_cmd,
        neural_simulations=args.neural_simulations,
        engine=engine_path,
    )
    engine_b_cmd = command_for_label(
        baselines=baselines,
        label=args.engine_b_label,
        override=args.engine_b_cmd,
        neural_simulations=args.neural_simulations,
        engine=engine_path,
    )
    if args.force or not games_path.exists():
        run_referee(args, games_path, engine_a_cmd, engine_b_cmd)

    positions = load_positions(repo_path(args.positions), args.limit)
    games_by_position = parse_games(games_path, args.engine_a_label)
    rows = summarize_positions(
        positions=positions,
        games_by_position=games_by_position,
        engine_a_label=args.engine_a_label,
        target_score=args.target_score,
    )
    balanced_rows = sorted(selectable_rows(rows, set(args.exclude_id), args.max_side_bias), key=selection_key)
    if args.balanced_target is not None:
        balanced_rows = balanced_rows[: args.balanced_target]
    if args.balanced_out is not None:
        write_balanced_positions(repo_path(args.balanced_out), positions, balanced_rows)

    report = {
        "schema": "ccerl.position_audit.v1",
        "protocol": {
            "positions": rel(repo_path(args.positions)),
            "position_limit": args.limit,
            "engine": rel(engine_path),
            "engine_a_label": args.engine_a_label,
            "engine_b_label": args.engine_b_label,
            "engine_a_cmd": engine_a_cmd,
            "engine_b_cmd": engine_b_cmd,
            "neural_simulations": args.neural_simulations,
            "max_plies": args.max_plies,
            "movetime_ms": args.movetime_ms,
            "timeout_seconds": args.timeout_seconds,
            "target_score": args.target_score,
            "exclude_ids": sorted(set(args.exclude_id)),
            "max_side_bias": args.max_side_bias,
        },
        "games_log": rel(games_path),
        "balanced_out": rel(repo_path(args.balanced_out)) if args.balanced_out else None,
        "summary": aggregate(rows),
        "balanced_positions": balanced_rows,
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
                "balanced_out": report["balanced_out"],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
