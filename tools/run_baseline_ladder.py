#!/usr/bin/env python3
"""Run a CCP baseline ladder from benchmarks/ccerl-v1/baselines.json."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = Path("benchmarks/ccerl-v1/baselines.json")


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


def select_baselines(payload: dict[str, Any], labels: list[str] | None) -> list[dict[str, Any]]:
    baselines = list(payload.get("baselines", []))
    if labels is None:
        return baselines
    wanted = set(labels)
    selected = [entry for entry in baselines if entry.get("label") in wanted]
    missing = sorted(wanted - {entry.get("label") for entry in selected})
    if missing:
        raise SystemExit(f"unknown baseline label(s): {', '.join(missing)}")
    return selected


def pair_specs(labels: list[str], mode: str, anchor: str) -> list[tuple[str, str]]:
    if mode == "adjacent":
        return [(labels[index], labels[index - 1]) for index in range(1, len(labels))]
    if mode == "anchor":
        if anchor not in labels:
            raise SystemExit(f"anchor {anchor!r} is not in selected baselines")
        return [(label, anchor) for label in labels if label != anchor]
    if mode == "all":
        return [(labels[right], labels[left]) for right in range(1, len(labels)) for left in range(right)]
    raise ValueError(f"unknown pair mode: {mode}")


def run_pair(
    *,
    args: argparse.Namespace,
    index: int,
    a: dict[str, Any],
    b: dict[str, Any],
    games_path: Path,
) -> None:
    cmd = [
        "python3",
        "tools/ccp_referee.py",
        "--engine",
        str(repo_path(args.engine)),
        "--engine-a-label",
        str(a["label"]),
        "--engine-b-label",
        str(b["label"]),
        "--engine-a-cmd",
        baseline_command(a, args.neural_simulations, repo_path(args.engine)),
        "--engine-b-cmd",
        baseline_command(b, args.neural_simulations, repo_path(args.engine)),
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
    if args.no_swap:
        cmd.append("--no-swap")
    print(json.dumps({"pair": index, "a": a["label"], "b": b["label"], "out": rel(games_path)}, sort_keys=True))
    subprocess.run(cmd, cwd=ROOT, check=True)


def score_for(record: dict[str, Any], label: str, start: dict[str, Any]) -> float:
    if bool(record.get("draw")) or record.get("winner") is None:
        return 0.5
    winner_label = start["p0"] if int(record["winner"]) == 0 else start["p1"]
    return 1.0 if winner_label == label else 0.0


def summarize_pair(path: Path, a: str, b: str) -> dict[str, Any]:
    current_start: dict[str, Any] | None = None
    wins = draws = losses = 0
    reasons: dict[str, int] = {}
    games = 0
    points = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "ccp_game_start":
                current_start = record
            elif record.get("type") == "ccp_game_end" and current_start is not None:
                score = score_for(record, a, current_start)
                games += 1
                points += score
                if score == 1.0:
                    wins += 1
                elif score == 0.5:
                    draws += 1
                else:
                    losses += 1
                reason = str(record.get("reason", "unknown"))
                reasons[reason] = reasons.get(reason, 0) + 1
                current_start = None
    return {
        "a": a,
        "b": b,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": points / games if games else 0.0,
        "reasons": reasons,
        "log": rel(path),
    }


def write_markdown(path: Path, ladder: dict[str, Any]) -> None:
    lines = ["# CCERL Baseline Ladder", ""]
    protocol = ladder["protocol"]
    limit = protocol.get("position_limit")
    limit_text = str(limit) if limit is not None else "full"
    lines.append(
        f"Protocol: pairs=`{protocol['pairs']}`, positions=`{limit_text}`, "
        f"max plies=`{protocol['max_plies']}`, neural sims=`{protocol['neural_simulations']}`, "
        f"draws=`{protocol.get('draws', 'half')}`."
    )
    lines.extend(["", "## Elo", "", "| Rank | Baseline | Elo | CI95 | Games | W-D-L | Score |", "|---:|---|---:|---:|---:|---:|---:|"])
    for rank, row in enumerate(ladder["elo"]["ratings"], 1):
        ci = row["ci95"]
        wdl = f"{row['wins']}-{row['draws']}-{row['losses']}"
        lines.append(
            f"| {rank} | `{row['player']}` | {row['elo']:.1f} | "
            f"{ci[0]:.1f}..{ci[1]:.1f} | {row['games']} | {wdl} | {row['score_rate']:.3f} |"
        )
    lines.extend(["", "## Pairs", "", "| A | B | Score A | W-D-L | Games | Reasons |", "|---|---|---:|---:|---:|---|"])
    for row in ladder["pairs"]:
        wdl = f"{row['wins']}-{row['draws']}-{row['losses']}"
        reasons = ", ".join(f"{key}:{value}" for key, value in sorted(row["reasons"].items()))
        lines.append(f"| `{row['a']}` | `{row['b']}` | {row['score']:.3f} | {wdl} | {row['games']} | {reasons} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("build/cczero"))
    parser.add_argument("--label", action="append", help="Restrict to a baseline label; repeatable.")
    parser.add_argument("--pairs", choices=("adjacent", "anchor", "all"), default="adjacent")
    parser.add_argument("--anchor")
    parser.add_argument("--anchor-elo", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--movetime-ms", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--neural-simulations", type=int)
    parser.add_argument("--draws", choices=("half", "ignore"), default="half")
    parser.add_argument("--rules", default="strict")
    parser.add_argument("--movegen", default="bitboard")
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    baselines_path = repo_path(args.baselines)
    payload = load_json(baselines_path)
    args.positions = args.positions or Path(payload.get("position_schedule", "benchmarks/ccerl-v1/positions/official_elo.jsonl"))
    anchor = args.anchor or payload.get("anchor", {}).get("label", "iter057")
    anchor_elo = args.anchor_elo if args.anchor_elo is not None else float(payload.get("anchor", {}).get("elo", 3000.0))

    selected = select_baselines(payload, args.label)
    by_label = {str(entry["label"]): entry for entry in selected}
    labels = [str(entry["label"]) for entry in selected]
    if len(labels) != len(set(labels)):
        raise SystemExit("baseline labels must be unique")

    out_dir = repo_path(args.out_dir)
    games_dir = out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    log_paths = []
    for index, (a_label, b_label) in enumerate(pair_specs(labels, args.pairs, anchor)):
        games_path = games_dir / f"{index:03d}_{safe_name(a_label)}_vs_{safe_name(b_label)}.jsonl"
        if args.force or not games_path.exists():
            run_pair(args=args, index=index, a=by_label[a_label], b=by_label[b_label], games_path=games_path)
        pair_rows.append(summarize_pair(games_path, a_label, b_label))
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
            "baselines": rel(baselines_path),
            "positions": rel(repo_path(args.positions)),
            "position_limit": args.limit,
            "pairs": args.pairs,
            "anchor": anchor,
            "anchor_elo": anchor_elo,
            "max_plies": args.max_plies,
            "movetime_ms": args.movetime_ms,
            "timeout_seconds": args.timeout_seconds,
            "neural_simulations": args.neural_simulations or payload.get("default_neural_simulations"),
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
