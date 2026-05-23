#!/usr/bin/env python3
"""Small public entry point for the CCERL benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path("benchmarks/ccerl-v1/manifest.json")
BASELINES_PATH = Path("benchmarks/ccerl-v1/baselines.json")
BASELINE_PACK_PATH = Path("benchmarks/ccerl-v1/baseline_pack/PACK_MANIFEST.json")
DEFAULT_POSITIONS = Path("benchmarks/ccerl-v1/positions/official_elo_v2.jsonl")
DEFAULT_ENGINE = Path("build/cczero")
ENGINE_RULE_PROFILE = "strict"


def repo_path(path: Path | str) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else ROOT / parsed


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def print_json(path: Path) -> int:
    print(json.dumps(read_json(repo_path(path)), indent=2, sort_keys=True))
    return 0


def command_manifest(_args: argparse.Namespace) -> int:
    return print_json(MANIFEST_PATH)


def command_baselines(_args: argparse.Namespace) -> int:
    return print_json(BASELINES_PATH)


def command_baseline_pack(_args: argparse.Namespace) -> int:
    return print_json(BASELINE_PACK_PATH)


def command_referee_ccp(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "tools/ccp_referee.py"),
        "--engine",
        str(repo_path(args.engine)),
        "--engine-a-label",
        args.engine_a_label,
        "--engine-b-label",
        args.engine_b_label,
        "--engine-a-cmd",
        args.engine_a_cmd,
        "--engine-b-cmd",
        args.engine_b_cmd,
        "--positions",
        str(repo_path(args.positions)),
        "--out",
        str(repo_path(args.out)),
        "--rules",
        ENGINE_RULE_PROFILE,
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
    run(cmd)
    return 0


def command_elo(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "tools/estimate_elo.py"),
        *[str(repo_path(path)) for path in args.inputs],
        "--anchor",
        args.anchor,
        "--anchor-elo",
        str(args.anchor_elo),
        "--draws",
        args.draws,
    ]
    if args.out is not None:
        cmd.extend(["--out", str(repo_path(args.out))])
    run(cmd)
    return 0


def command_baseline_ladder_native(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "tools/run_baseline_ladder_native.py"),
        "--baselines",
        str(repo_path(args.baselines)),
        "--positions",
        str(repo_path(args.positions)),
        "--out-dir",
        str(repo_path(args.out_dir)),
        "--engine",
        str(repo_path(args.engine)),
        "--pairs",
        args.pairs,
        "--max-plies",
        str(args.max_plies),
        "--simulations",
        str(args.simulations),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--rules",
        ENGINE_RULE_PROFILE,
        "--movegen",
        args.movegen,
        "--inference-backend",
        args.inference_backend,
        "--inference-batch-size",
        str(args.inference_batch_size),
        "--draws",
        args.draws,
    ]
    if args.anchor is not None:
        cmd.extend(["--anchor", args.anchor])
    if args.anchor_elo is not None:
        cmd.extend(["--anchor-elo", str(args.anchor_elo)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.no_swap:
        cmd.append("--no-swap")
    if args.force:
        cmd.append("--force")
    for label in args.label or []:
        cmd.extend(["--label", label])
    run(cmd)
    return 0


def command_runner_spotcheck(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "tools/validate_runner_spotchecks.py"),
        "--baselines",
        str(repo_path(args.baselines)),
        "--positions",
        str(repo_path(args.positions)),
        "--out-dir",
        str(repo_path(args.out_dir)),
        "--engine",
        str(repo_path(args.engine)),
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
        ENGINE_RULE_PROFILE,
        "--movegen",
        args.movegen,
        "--movetime-ms",
        str(args.movetime_ms),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    for pair in args.pair or []:
        cmd.extend(["--pair", pair])
    if args.force:
        cmd.append("--force")
    run(cmd)
    return 0


def add_common_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--baselines", type=Path, default=BASELINES_PATH)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--movegen", default="bitboard")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="Print the benchmark manifest.")
    manifest.set_defaults(func=command_manifest)

    baselines = sub.add_parser("baselines", help="Print baseline definitions.")
    baselines.set_defaults(func=command_baselines)

    pack = sub.add_parser("baseline-pack", help="Print the baseline-pack manifest.")
    pack.set_defaults(func=command_baseline_pack)

    referee = sub.add_parser("referee-ccp", help="Run two CCP engines through the referee.")
    referee.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    referee.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    referee.add_argument("--engine-a-label", required=True)
    referee.add_argument("--engine-b-label", required=True)
    referee.add_argument("--engine-a-cmd", required=True)
    referee.add_argument("--engine-b-cmd", required=True)
    referee.add_argument("--out", type=Path, required=True)
    referee.add_argument("--limit", type=int)
    referee.add_argument("--max-plies", type=int, default=240)
    referee.add_argument("--movetime-ms", type=int, default=1000)
    referee.add_argument("--timeout-seconds", type=float, default=60.0)
    referee.add_argument("--movegen", default="bitboard")
    referee.add_argument("--no-swap", action="store_true")
    referee.set_defaults(func=command_referee_ccp)

    elo = sub.add_parser("elo", help="Estimate Bradley-Terry Elo from JSON/JSONL results.")
    elo.add_argument("inputs", nargs="+", type=Path)
    elo.add_argument("--anchor", default="random")
    elo.add_argument("--anchor-elo", type=float, default=0.0)
    elo.add_argument("--draws", choices=["half", "decisive-only"], default="half")
    elo.add_argument("--out", type=Path)
    elo.set_defaults(func=command_elo)

    native = sub.add_parser("baseline-ladder-native", help="Run native all-pairs baseline games.")
    add_common_runner_args(native)
    native.add_argument("--pairs", choices=["all", "adjacent"], default="all")
    native.add_argument("--simulations", type=int, default=64)
    native.add_argument("--workers", type=int, default=1)
    native.add_argument("--seed", type=int, default=986000)
    native.add_argument("--inference-backend", default="auto")
    native.add_argument("--inference-batch-size", type=int, default=64)
    native.add_argument("--draws", choices=["half", "decisive-only"], default="half")
    native.add_argument("--anchor")
    native.add_argument("--anchor-elo", type=float)
    native.add_argument("--limit", type=int)
    native.add_argument("--label", action="append")
    native.add_argument("--no-swap", action="store_true")
    native.add_argument("--force", action="store_true")
    native.set_defaults(func=command_baseline_ladder_native)

    spotcheck = sub.add_parser("runner-spotcheck", help="Compare CCP and native runner invariants.")
    add_common_runner_args(spotcheck)
    spotcheck.add_argument("--limit", type=int, default=1)
    spotcheck.add_argument("--simulations", type=int, default=16)
    spotcheck.add_argument("--workers", type=int, default=1)
    spotcheck.add_argument("--seed", type=int, default=990000)
    spotcheck.add_argument("--movetime-ms", type=int, default=100)
    spotcheck.add_argument("--timeout-seconds", type=float, default=60.0)
    spotcheck.add_argument("--pair", action="append", default=[])
    spotcheck.add_argument("--force", action="store_true")
    spotcheck.set_defaults(func=command_runner_spotcheck)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
