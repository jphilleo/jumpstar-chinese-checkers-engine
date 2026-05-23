#!/usr/bin/env python3
"""Compute an SPRT-style decision from CCERL game logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from estimate_elo import games_from_records, read_json_or_jsonl


def expected_score(elo_delta: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_delta / 400.0))


def log_likelihood(score: float, expected: float) -> float:
    expected = min(1.0 - 1.0e-12, max(1.0e-12, expected))
    return score * math.log(expected) + (1.0 - score) * math.log(1.0 - expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponent")
    parser.add_argument("--elo0", type=float, default=0.0, help="Null hypothesis candidate Elo delta.")
    parser.add_argument("--elo1", type=float, default=25.0, help="Alternative hypothesis candidate Elo delta.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    scores = []
    wins = draws = losses = 0
    for path in args.logs:
        for game in games_from_records(read_json_or_jsonl(path), path.stem):
            if args.candidate not in {game.a, game.b}:
                continue
            opponent = game.b if game.a == args.candidate else game.a
            if args.opponent is not None and opponent != args.opponent:
                continue
            score = game.score_a if game.a == args.candidate else 1.0 - game.score_a
            scores.append(score)
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1

    if not scores:
        raise SystemExit("no matching games found")

    p0 = expected_score(args.elo0)
    p1 = expected_score(args.elo1)
    llr = sum(log_likelihood(score, p1) - log_likelihood(score, p0) for score in scores)
    upper = math.log((1.0 - args.beta) / args.alpha)
    lower = math.log(args.beta / (1.0 - args.alpha))
    if llr >= upper:
        decision = "accept_h1"
    elif llr <= lower:
        decision = "accept_h0"
    else:
        decision = "continue"

    result: dict[str, Any] = {
        "schema": "ccerl.sprt_decision.v1",
        "candidate": args.candidate,
        "opponent": args.opponent,
        "elo0": args.elo0,
        "elo1": args.elo1,
        "alpha": args.alpha,
        "beta": args.beta,
        "llr": llr,
        "lower_bound": lower,
        "upper_bound": upper,
        "decision": decision,
        "games": len(scores),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": sum(scores) / len(scores),
        "expected_score_h0": p0,
        "expected_score_h1": p1,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
