#!/usr/bin/env python3
"""Run CCP-vs-native spot checks for selected baseline pairs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from audit_position_suite import load_positions, rel, repo_path, safe_name
from run_baseline_ladder import baseline_command, load_json, summarize_pair


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = Path("benchmarks/ccerl-v1/baselines.json")
DEFAULT_POSITIONS = Path("benchmarks/ccerl-v1/positions/official_elo_v2.jsonl")
DEFAULT_PAIRS = [
    "iter057:fresh046",
    "iter057:tt-pvs",
    "fresh029:fresh046",
    "tt-pvs:converter",
    "greedy:random",
]
BAD_REASONS = {"illegal_move", "protocol_error", "timeout", "crash"}


def parse_pair(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("pair must be LABEL_A:LABEL_B")
    left, right = text.split(":", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise argparse.ArgumentTypeError("pair labels cannot be empty")
    return left, right


def selected_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entry["label"]): entry for entry in payload.get("baselines", [])}


def run_ccp_pair(
    *,
    args: argparse.Namespace,
    payload: dict[str, Any],
    entries: dict[str, dict[str, Any]],
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    cmd = [
        "python3",
        "tools/ccp_referee.py",
        "--engine",
        str(repo_path(args.engine)),
        "--engine-a-label",
        label_a,
        "--engine-b-label",
        label_b,
        "--engine-a-cmd",
        baseline_command(entries[label_a], args.simulations, repo_path(args.engine)),
        "--engine-b-cmd",
        baseline_command(entries[label_b], args.simulations, repo_path(args.engine)),
        "--positions",
        str(repo_path(args.positions)),
        "--out",
        str(out_path),
        "--limit",
        str(args.limit),
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
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_native_pair(
    *,
    args: argparse.Namespace,
    label_a: str,
    label_b: str,
    out_dir: Path,
) -> Path:
    cmd = [
        "python3",
        "tools/audit_position_suite_native.py",
        "--baselines",
        str(repo_path(args.baselines)),
        "--positions",
        str(repo_path(args.positions)),
        "--out-dir",
        str(out_dir),
        "--engine",
        str(repo_path(args.engine)),
        "--engine-a-label",
        label_a,
        "--engine-b-label",
        label_b,
        "--limit",
        str(args.limit),
        "--max-plies",
        str(args.max_plies),
        "--simulations",
        str(args.simulations),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--rules",
        args.rules,
        "--movegen",
        args.movegen,
        "--force",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out_dir / f"{safe_name(label_a)}_vs_{safe_name(label_b)}.jsonl"


def read_game_records(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    games = []
    current_start: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "ccp_game_start":
                current_start = record
            elif record.get("type") == "ccp_game_end" and current_start is not None:
                games.append((current_start, record))
                current_start = None
    return games


def validate_log(
    *,
    path: Path,
    positions: list[dict[str, Any]],
    label_a: str,
    label_b: str,
    max_plies: int,
) -> dict[str, Any]:
    expected_by_id = {str(position["id"]): position for position in positions}
    games = read_game_records(path)
    errors: list[str] = []
    reasons: dict[str, int] = {}
    by_position: dict[str, list[dict[str, Any]]] = {}
    for start, end in games:
        position_id = str(start.get("position_id"))
        position = expected_by_id.get(position_id)
        if position is None:
            errors.append(f"unknown position id {position_id}")
        else:
            if start.get("initial_cells") != position.get("cells"):
                errors.append(f"{position_id}: initial_cells mismatch")
            if int(start.get("initial_player", -1)) != int(position.get("player", -2)):
                errors.append(f"{position_id}: initial_player mismatch")
            if int(start.get("initial_ply", -1)) != int(position.get("ply", -2)):
                errors.append(f"{position_id}: initial_ply mismatch")
        reason = str(end.get("reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
        if reason in BAD_REASONS:
            errors.append(f"{position_id}: bad termination reason {reason}")
        if int(end.get("plies", 0)) > max_plies:
            errors.append(f"{position_id}: plies exceed max_plies")
        if end.get("winner") is not None and int(end["winner"]) not in (0, 1):
            errors.append(f"{position_id}: invalid winner {end.get('winner')}")
        by_position.setdefault(position_id, []).append(start)

    expected_games = len(positions) * 2
    if len(games) != expected_games:
        errors.append(f"expected {expected_games} games, found {len(games)}")
    for position in positions:
        starts = by_position.get(str(position["id"]), [])
        seen = {(start.get("p0"), start.get("p1")) for start in starts}
        expected = {(label_a, label_b), (label_b, label_a)}
        if seen != expected:
            errors.append(f"{position['id']}: expected color swap {expected}, saw {seen}")

    pair_summary = summarize_pair(path, label_a, label_b)
    return {
        "path": rel(path),
        "games": len(games),
        "pass": not errors,
        "errors": errors,
        "reasons": reasons,
        "score_for_a": pair_summary["score"],
        "wins": pair_summary["wins"],
        "draws": pair_summary["draws"],
        "losses": pair_summary["losses"],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Runner Spot Checks", ""]
    protocol = report["protocol"]
    lines.append(
        f"Positions: `{protocol['positions']}`, limit `{protocol['limit']}`, "
        f"simulations `{protocol['simulations']}`, max plies `{protocol['max_plies']}`."
    )
    lines.extend(["", "| Pair | CCP | Native | CCP Score | Native Score | Notes |", "|---|---|---|---:|---:|---|"])
    for row in report["pairs"]:
        ccp = row["ccp"]
        native = row["native"]
        notes = []
        if row["score_delta"] is not None:
            notes.append(f"score delta {row['score_delta']:.3f}")
        if ccp["errors"]:
            notes.append("ccp errors")
        if native["errors"]:
            notes.append("native errors")
        lines.append(
            f"| `{row['a']}` vs `{row['b']}` | {'PASS' if ccp['pass'] else 'FAIL'} | "
            f"{'PASS' if native['pass'] else 'FAIL'} | {ccp['score_for_a']:.3f} | "
            f"{native['score_for_a']:.3f} | {', '.join(notes)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cczero"))
    parser.add_argument("--pair", type=parse_pair, action="append")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=990000)
    parser.add_argument("--rules", default="strict")
    parser.add_argument("--movegen", default="bitboard")
    parser.add_argument("--movetime-ms", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = load_json(repo_path(args.baselines))
    entries = selected_entries(payload)
    pairs = args.pair or [parse_pair(text) for text in DEFAULT_PAIRS]
    for left, right in pairs:
        if left not in entries:
            raise SystemExit(f"unknown baseline label: {left}")
        if right not in entries:
            raise SystemExit(f"unknown baseline label: {right}")

    positions = load_positions(repo_path(args.positions), args.limit)
    out_dir = repo_path(args.out_dir)
    ccp_dir = out_dir / "ccp"
    native_dir = out_dir / "native"
    ccp_dir.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)

    pair_reports = []
    for index, (label_a, label_b) in enumerate(pairs):
        pair_name = f"{index:02d}_{safe_name(label_a)}_vs_{safe_name(label_b)}"
        ccp_path = ccp_dir / f"{pair_name}.jsonl"
        if args.force or not ccp_path.exists():
            run_ccp_pair(
                args=args,
                payload=payload,
                entries=entries,
                label_a=label_a,
                label_b=label_b,
                out_path=ccp_path,
            )
        native_pair_dir = native_dir / pair_name
        native_path = native_pair_dir / f"{safe_name(label_a)}_vs_{safe_name(label_b)}.jsonl"
        if args.force or not native_path.exists():
            native_path = run_native_pair(args=args, label_a=label_a, label_b=label_b, out_dir=native_pair_dir)
        ccp_report = validate_log(
            path=ccp_path,
            positions=positions,
            label_a=label_a,
            label_b=label_b,
            max_plies=args.max_plies,
        )
        native_report = validate_log(
            path=native_path,
            positions=positions,
            label_a=label_a,
            label_b=label_b,
            max_plies=args.max_plies,
        )
        pair_reports.append(
            {
                "a": label_a,
                "b": label_b,
                "ccp": ccp_report,
                "native": native_report,
                "score_delta": abs(ccp_report["score_for_a"] - native_report["score_for_a"]),
            }
        )
        print(
            json.dumps(
                {
                    "pair": pair_name,
                    "ccp_pass": ccp_report["pass"],
                    "native_pass": native_report["pass"],
                    "ccp_score": ccp_report["score_for_a"],
                    "native_score": native_report["score_for_a"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report = {
        "schema": "ccerl.runner_spotcheck.v1",
        "protocol": {
            "baselines": rel(repo_path(args.baselines)),
            "positions": rel(repo_path(args.positions)),
            "limit": args.limit,
            "max_plies": args.max_plies,
            "simulations": args.simulations,
            "rules": args.rules,
            "movegen": args.movegen,
            "seed": args.seed,
            "note": "Exact results may differ because CCP adapters call best-move independently while native match carries in-game RNG/search context.",
        },
        "pairs": pair_reports,
        "pass": all(row["ccp"]["pass"] and row["native"]["pass"] for row in pair_reports),
    }
    report_path = out_dir / "runner_spotcheck.json"
    markdown_path = out_dir / "runner_spotcheck.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(markdown_path, report)
    print(json.dumps({"report": rel(report_path), "markdown": rel(markdown_path), "pass": report["pass"]}, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
