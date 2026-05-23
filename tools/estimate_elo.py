#!/usr/bin/env python3
"""Estimate Elo ratings from CCZero/CCP results with a Bradley-Terry model."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOG10_OVER_400 = math.log(10.0) / 400.0


@dataclass(frozen=True)
class Game:
    a: str
    b: str
    score_a: float


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, list) else [payload]
        except json.JSONDecodeError:
            pass
    records = []
    for line in text.splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def games_from_records(records: list[dict[str, Any]], fallback_name: str) -> list[Game]:
    games: list[Game] = []
    current_start: dict[str, Any] | None = None
    for record in records:
        rtype = record.get("type")
        if rtype in {"ccp_game_start", "game_start"}:
            current_start = record
        elif rtype in {"ccp_game_end", "game_end"} and current_start is not None:
            p0 = str(current_start.get("p0", "p0"))
            p1 = str(current_start.get("p1", "p1"))
            if bool(record.get("draw")) or record.get("winner") is None:
                score = 0.5
            else:
                score = 1.0 if int(record["winner"]) == 0 else 0.0
            games.append(Game(p0, p1, score))
            current_start = None
        elif "aggregate" in record and "rows" in record:
            candidate = str(record.get("candidate_label") or Path(record.get("candidate_model", fallback_name)).stem)
            for row in record.get("rows", []):
                opponent = str(row.get("opponent_label") or row.get("opponent") or row.get("opponent_model") or "opponent")
                wins = int(row.get("wins", 0) or 0)
                draws = int(row.get("draws", 0) or 0)
                losses = int(row.get("losses", 0) or 0)
                games.extend(Game(candidate, opponent, 1.0) for _ in range(wins))
                games.extend(Game(candidate, opponent, 0.5) for _ in range(draws))
                games.extend(Game(candidate, opponent, 0.0) for _ in range(losses))
        elif "rows" in record and "ladder" in record:
            for row in record.get("rows", []):
                a = str(row["candidate"])
                b = str(row["champion"])
                wins = int(row.get("wins", 0) or 0)
                draws = int(row.get("draws", 0) or 0)
                losses = int(row.get("losses", 0) or 0)
                games.extend(Game(a, b, 1.0) for _ in range(wins))
                games.extend(Game(a, b, 0.5) for _ in range(draws))
                games.extend(Game(a, b, 0.0) for _ in range(losses))
    return games


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    mat = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(mat[row][col]))
        if abs(mat[pivot][col]) < 1.0e-12:
            mat[pivot][col] = 1.0e-12
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
        denom = mat[col][col]
        for item in range(col, n + 1):
            mat[col][item] /= denom
        for row in range(n):
            if row == col:
                continue
            factor = mat[row][col]
            if factor == 0.0:
                continue
            for item in range(col, n + 1):
                mat[row][item] -= factor * mat[col][item]
    return [mat[row][n] for row in range(n)]


def invert_matrix(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    inv = []
    for col in range(n):
        rhs = [0.0] * n
        rhs[col] = 1.0
        inv.append(solve_linear(a, rhs))
    return [[inv[col][row] for col in range(n)] for row in range(n)]


def fit_elo(games: list[Game], anchor: str, anchor_elo: float, draw_policy: str) -> dict[str, Any]:
    players = sorted({game.a for game in games} | {game.b for game in games})
    if anchor not in players:
        players.append(anchor)
        players.sort()
    index = {player: i for i, player in enumerate(players)}
    free_players = [player for player in players if player != anchor]
    free_index = {player: i for i, player in enumerate(free_players)}
    ratings = {player: anchor_elo for player in players}
    for player in free_players:
        ratings[player] = anchor_elo

    for _ in range(80):
        n = len(free_players)
        grad = [0.0] * n
        info = [[0.0 for _ in range(n)] for _ in range(n)]
        for game in games:
            diff = ratings[game.a] - ratings[game.b]
            p = 1.0 / (1.0 + math.exp(-LOG10_OVER_400 * diff))
            residual = game.score_a - p
            weight = (LOG10_OVER_400 ** 2) * p * (1.0 - p)
            for player, sign in ((game.a, 1.0), (game.b, -1.0)):
                if player in free_index:
                    grad[free_index[player]] += sign * LOG10_OVER_400 * residual
            if game.a in free_index:
                ia = free_index[game.a]
                info[ia][ia] += weight
            if game.b in free_index:
                ib = free_index[game.b]
                info[ib][ib] += weight
            if game.a in free_index and game.b in free_index:
                ia = free_index[game.a]
                ib = free_index[game.b]
                info[ia][ib] -= weight
                info[ib][ia] -= weight
        for i in range(n):
            info[i][i] += 1.0e-6
        if n == 0:
            break
        step = solve_linear(info, grad)
        max_step = 0.0
        for player, delta in zip(free_players, step):
            clipped = max(-80.0, min(80.0, delta))
            ratings[player] += clipped
            max_step = max(max_step, abs(clipped))
        if max_step < 1.0e-4:
            break

    n = len(free_players)
    info = [[0.0 for _ in range(n)] for _ in range(n)]
    scores = {player: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0} for player in players}
    log_likelihood = 0.0
    for game in games:
        diff = ratings[game.a] - ratings[game.b]
        p = 1.0 / (1.0 + math.exp(-LOG10_OVER_400 * diff))
        p = min(1.0 - 1.0e-12, max(1.0e-12, p))
        log_likelihood += game.score_a * math.log(p) + (1.0 - game.score_a) * math.log(1.0 - p)
        weight = (LOG10_OVER_400 ** 2) * p * (1.0 - p)
        if game.a in free_index:
            ia = free_index[game.a]
            info[ia][ia] += weight
        if game.b in free_index:
            ib = free_index[game.b]
            info[ib][ib] += weight
        if game.a in free_index and game.b in free_index:
            ia = free_index[game.a]
            ib = free_index[game.b]
            info[ia][ib] -= weight
            info[ib][ia] -= weight
        for player, score in ((game.a, game.score_a), (game.b, 1.0 - game.score_a)):
            row = scores[player]
            row["games"] += 1
            row["points"] += score
            if score == 1.0:
                row["wins"] += 1
            elif score == 0.5:
                row["draws"] += 1
            else:
                row["losses"] += 1
    for i in range(n):
        info[i][i] += 1.0e-6
    covariance = invert_matrix(info) if n else []
    rows = []
    for player in players:
        se = 0.0
        if player in free_index:
            var = max(0.0, covariance[free_index[player]][free_index[player]])
            se = math.sqrt(var)
        score_rate = scores[player]["points"] / scores[player]["games"] if scores[player]["games"] else 0.0
        rows.append(
            {
                "player": player,
                "elo": ratings[player],
                "ci95": [ratings[player] - 1.96 * se, ratings[player] + 1.96 * se],
                "standard_error": se,
                "games": scores[player]["games"],
                "wins": scores[player]["wins"],
                "draws": scores[player]["draws"],
                "losses": scores[player]["losses"],
                "score_rate": score_rate,
            }
        )
    rows.sort(key=lambda row: row["elo"], reverse=True)
    return {
        "anchor": anchor,
        "anchor_elo": anchor_elo,
        "draw_policy": draw_policy,
        "games": len(games),
        "players": len(players),
        "log_likelihood": log_likelihood,
        "ratings": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--anchor", default="iter57")
    parser.add_argument("--anchor-elo", type=float, default=3000.0)
    parser.add_argument(
        "--draws",
        choices=("half", "ignore"),
        default="half",
        help="Use draws as 0.5 results, or ignore drawn games for decisive-only sensitivity.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    games: list[Game] = []
    for path in args.inputs:
        records = read_json_or_jsonl(path)
        games.extend(games_from_records(records, path.stem))
    if args.draws == "ignore":
        games = [game for game in games if game.score_a != 0.5]
    if not games:
        raise SystemExit("no games found")
    report = fit_elo(games, args.anchor, args.anchor_elo, args.draws)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("rank,player,elo,ci95_low,ci95_high,games,wins,draws,losses,score_rate")
    for rank, row in enumerate(report["ratings"], 1):
        print(
            f"{rank},{row['player']},{row['elo']:.1f},{row['ci95'][0]:.1f},{row['ci95'][1]:.1f},"
            f"{row['games']},{row['wins']},{row['draws']},{row['losses']},{row['score_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
