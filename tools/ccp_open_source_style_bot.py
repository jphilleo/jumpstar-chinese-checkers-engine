#!/usr/bin/env python3
"""CCP adapters for published open-source Chinese Checkers bot styles.

These are rule-compatible ports, not vendored upstream engines. They use the
CCERL board, legal-move rules, and CCP protocol while matching the public
algorithmic ideas from the referenced projects closely enough to measure them
under the same referee as JumpStar.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
WIN_SCORE = 1_000_000.0


@dataclass(frozen=True)
class Hole:
    id: int
    q: int
    r: int
    s: int


@dataclass(frozen=True)
class Move:
    source: int
    target: int
    path: tuple[int, ...]


def build_board() -> tuple[list[Hole], dict[tuple[int, int], int]]:
    coords: set[tuple[int, int]] = set()

    def add(q: int, r: int) -> None:
        coords.add((q, r))

    for q in range(-4, 5):
        for r in range(-4, 5):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= 4:
                add(q, r)
    for d in range(1, 5):
        for r in range(-4, -d + 1):
            add(4 + d, r)
        for r in range(d, 5):
            add(-4 - d, r)
        for q in range(-4, -d + 1):
            add(q, 4 + d)
        for q in range(d, 5):
            add(q, -4 - d)
        for q in range(-4, -d + 1):
            add(q, -4 - d - q)
        for q in range(d, 5):
            add(q, 4 + d - q)

    holes = [
        Hole(id=index, q=q, r=r, s=-q - r)
        for index, (q, r) in enumerate(sorted(coords, key=lambda item: (item[1], item[0])))
    ]
    index = {(hole.q, hole.r): hole.id for hole in holes}
    return holes, index


HOLES, HOLE_BY_COORD = build_board()
HOLE_Q = tuple(hole.q for hole in HOLES)
HOLE_R = tuple(hole.r for hole in HOLES)
HOLE_S = tuple(hole.s for hole in HOLES)
GOALS = {
    0: tuple(hole.id for hole in HOLES if hole.r > 4),
    1: tuple(hole.id for hole in HOLES if hole.r < -4),
}
HOMES = {
    0: tuple(hole.id for hole in HOLES if hole.r < -4),
    1: tuple(hole.id for hole in HOLES if hole.r > 4),
}
STARTPOS = "".join(
    "0" if hole.id in HOMES[0] else "1" if hole.id in HOMES[1] else "."
    for hole in HOLES
)
GOAL_SET = {player: frozenset(goals) for player, goals in GOALS.items()}
HOME_SET = {player: frozenset(homes) for player, homes in HOMES.items()}
IN_GOAL = {
    player: tuple(hole_id in GOAL_SET[player] for hole_id in range(len(HOLES)))
    for player in (0, 1)
}
IN_HOME = {
    player: tuple(hole_id in HOME_SET[player] for hole_id in range(len(HOLES)))
    for player in (0, 1)
}
NEIGHBORS = tuple(
    tuple(
        target
        for dq, dr in DIRECTIONS
        for target in [HOLE_BY_COORD.get((hole.q + dq, hole.r + dr))]
        if target is not None
    )
    for hole in HOLES
)
JUMPS = tuple(
    tuple(
        (mid, landing)
        for dq, dr in DIRECTIONS
        for mid in [HOLE_BY_COORD.get((hole.q + dq, hole.r + dr))]
        for landing in [HOLE_BY_COORD.get((hole.q + 2 * dq, hole.r + 2 * dr))]
        if mid is not None and landing is not None
    )
    for hole in HOLES
)
PROGRESS = {
    0: tuple(HOLE_R),
    1: tuple(-value for value in HOLE_R),
}
CENTRALITY = tuple(abs(HOLE_Q[hole_id]) + abs(HOLE_S[hole_id]) * 0.35 for hole_id in range(len(HOLES)))


def hex_distance(a: Hole, b: Hole) -> int:
    return max(abs(a.q - b.q), abs(a.r - b.r), abs(a.s - b.s))


@lru_cache(maxsize=None)
def distance_to_goal(player: int, hole_id: int) -> int:
    hole = HOLES[hole_id]
    return min(hex_distance(hole, HOLES[goal]) for goal in GOALS[player])


DISTANCE_TO_GOAL = {
    player: tuple(distance_to_goal(player, hole_id) for hole_id in range(len(HOLES)))
    for player in (0, 1)
}
EUCLID_TO_GOAL = {
    player: tuple(
        min(math.hypot(HOLE_Q[hole_id] - HOLE_Q[goal], HOLE_R[hole_id] - HOLE_R[goal]) for goal in GOALS[player])
        for hole_id in range(len(HOLES))
    )
    for player in (0, 1)
}


@lru_cache(maxsize=200_000)
def is_terminal(cells: str) -> int | None:
    for player in (0, 1):
        if all(cells[goal] == str(player) for goal in GOALS[player]):
            return player
    return None


@lru_cache(maxsize=500_000)
def apply_move(cells: str, move: Move, player: int) -> str:
    source = move.source
    target = move.target
    piece = str(player)
    if source < target:
        return cells[:source] + "." + cells[source + 1 : target] + piece + cells[target + 1 :]
    return cells[:target] + piece + cells[target + 1 : source] + "." + cells[source + 1 :]


@lru_cache(maxsize=200_000)
def legal_moves(cells: str, player: int) -> tuple[Move, ...]:
    occupied = {index for index, value in enumerate(cells) if value != "."}
    moves: list[Move] = []
    seen_endpoints: set[tuple[int, int]] = set()

    def locked_target_allowed(source: int, target: int) -> bool:
        return not IN_GOAL[player][source] or IN_GOAL[player][target]

    for source, value in enumerate(cells):
        if value != str(player):
            continue
        for target in NEIGHBORS[source]:
            if cells[target] == "." and locked_target_allowed(source, target):
                key = (source, target)
                if key not in seen_endpoints:
                    seen_endpoints.add(key)
                    moves.append(Move(source, target, (source, target)))

        base_occupied = occupied - {source}
        jump_seen = {source}
        stack: list[tuple[int, tuple[int, ...]]] = [(source, (source,))]
        while stack:
            current, path = stack.pop()
            chain_occupied = base_occupied | {current}
            for mid, landing in JUMPS[current]:
                if mid not in chain_occupied or landing in chain_occupied or landing in jump_seen:
                    continue
                if IN_GOAL[player][current] and not IN_GOAL[player][landing]:
                    continue
                if not locked_target_allowed(source, landing):
                    continue
                jump_seen.add(landing)
                next_path = path + (landing,)
                key = (source, landing)
                if key not in seen_endpoints:
                    seen_endpoints.add(key)
                    moves.append(Move(source, landing, next_path))
                stack.append((landing, next_path))

    return tuple(moves)


def pieces(cells: str, player: int) -> list[int]:
    return [index for index, value in enumerate(cells) if value == str(player)]


def progress(player: int, hole_id: int) -> int:
    return PROGRESS[player][hole_id]


def centrality(hole_id: int) -> float:
    return CENTRALITY[hole_id]


@lru_cache(maxsize=500_000)
def conversion_score(cells: str, player: int) -> float:
    winner = is_terminal(cells)
    if winner == player:
        return WIN_SCORE
    if winner == 1 - player:
        return -WIN_SCORE

    mine = pieces(cells, player)
    theirs = pieces(cells, 1 - player)
    my_dist = sum(DISTANCE_TO_GOAL[player][hole] for hole in mine)
    their_dist = sum(DISTANCE_TO_GOAL[1 - player][hole] for hole in theirs)
    my_goal = sum(1 for hole in mine if IN_GOAL[player][hole])
    their_goal = sum(1 for hole in theirs if IN_GOAL[1 - player][hole])
    my_home = sum(1 for hole in mine if IN_HOME[player][hole])
    their_home = sum(1 for hole in theirs if IN_HOME[1 - player][hole])
    my_center = sum(CENTRALITY[hole] for hole in mine)
    their_center = sum(CENTRALITY[hole] for hole in theirs)
    blockers = sum(1 for hole in GOALS[player] if cells[hole] == str(1 - player))
    return (
        18.0 * (their_dist - my_dist)
        + 85.0 * (my_goal - their_goal)
        - 18.0 * (my_home - their_home)
        + 1.4 * (their_center - my_center)
        - 24.0 * blockers
    )


@lru_cache(maxsize=500_000)
def zedrichu_score(cells: str, player: int) -> float:
    winner = is_terminal(cells)
    if winner == player:
        return WIN_SCORE
    if winner == 1 - player:
        return -WIN_SCORE

    def terms(side: int) -> tuple[float, float, float, float]:
        side_pieces = pieces(cells, side)
        distances = [DISTANCE_TO_GOAL[side][hole] for hole in side_pieces]
        euclid = [EUCLID_TO_GOAL[side][hole] for hole in side_pieces]
        in_goal = sum(1 for hole in side_pieces if IN_GOAL[side][hole])
        return (
            float(in_goal),
            sum(distances) / max(1, len(distances)),
            sum(euclid) / max(1, len(euclid)),
            max(distances) if distances else 0.0,
        )

    my_goal, my_man, my_euc, my_max = terms(player)
    op_goal, op_man, op_euc, op_max = terms(1 - player)
    # Mirrors the published weighted heuristic: goal occupancy plus average
    # Manhattan/Euclidean and max-distance terms.
    return (
        35.0 * (my_goal - op_goal)
        + 10.0 * (op_man - my_man) * 0.3
        + 10.0 * (op_euc - my_euc) * 0.4
        + 10.0 * (op_max - my_max) * 0.2
    )


@lru_cache(maxsize=500_000)
def marblefish_score(cells: str, player: int) -> float:
    winner = is_terminal(cells)
    if winner == player:
        return WIN_SCORE
    if winner == 1 - player:
        return -WIN_SCORE

    def raw(side: int) -> float:
        side_pieces = pieces(cells, side)
        vertical = sum(PROGRESS[side][hole] for hole in side_pieces) - 20.0
        horizontal = sum(abs(abs(HOLE_Q[hole] + HOLE_R[hole] / 2.0) - 1.0) for hole in side_pieces)
        goal = 20.0 * sum(1 for hole in side_pieces if IN_GOAL[side][hole])
        return vertical - 0.5 * horizontal + goal

    return raw(player) - raw(1 - player)


def move_order(cells: str, player: int, moves: Iterable[Move], score_fn) -> list[Move]:
    return sorted(
        moves,
        key=lambda move: (
            score_fn(apply_move(cells, move, player), player),
            len(move.path),
            -move.source,
            -move.target,
        ),
        reverse=True,
    )


def alphabeta(
    cells: str,
    player_to_move: int,
    root_player: int,
    depth: int,
    alpha: float,
    beta: float,
    score_fn,
    beam: int | None = None,
) -> float:
    winner = is_terminal(cells)
    if winner is not None or depth <= 0:
        return score_fn(cells, root_player)
    moves = legal_moves(cells, player_to_move)
    if not moves:
        return score_fn(cells, root_player)
    ordered = move_order(cells, player_to_move, moves, score_fn)
    if beam is not None:
        ordered = ordered[:beam]
    if player_to_move == root_player:
        value = -float("inf")
        for move in ordered:
            value = max(
                value,
                alphabeta(
                    apply_move(cells, move, player_to_move),
                    1 - player_to_move,
                    root_player,
                    depth - 1,
                    alpha,
                    beta,
                    score_fn,
                    beam,
                ),
            )
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = float("inf")
    for move in ordered:
        value = min(
            value,
            alphabeta(
                apply_move(cells, move, player_to_move),
                1 - player_to_move,
                root_player,
                depth - 1,
                alpha,
                beta,
                score_fn,
                beam,
            ),
        )
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def choose_alphabeta(cells: str, player: int, depth: int, score_fn, beam: int | None) -> Move:
    moves = legal_moves(cells, player)
    if not moves:
        raise RuntimeError("no legal moves")
    best_score = -float("inf")
    best = moves[0]
    for move in move_order(cells, player, moves, score_fn):
        score = alphabeta(
            apply_move(cells, move, player),
            1 - player,
            player,
            depth - 1,
            -float("inf"),
            float("inf"),
            score_fn,
            beam,
        )
        if score > best_score:
            best_score = score
            best = move
    return best


def choose_rule_path(cells: str, player: int) -> Move:
    moves = legal_moves(cells, player)
    if not moves:
        raise RuntimeError("no legal moves")

    def key(move: Move) -> tuple[float, int, int]:
        before = DISTANCE_TO_GOAL[player][move.source]
        after = DISTANCE_TO_GOAL[player][move.target]
        jump_bonus = max(0, len(move.path) - 2)
        goal_bonus = 10 if IN_GOAL[player][move.target] else 0
        home_penalty = 3 if IN_HOME[player][move.source] else 0
        return (before - after + jump_bonus + goal_bonus + home_penalty, progress(player, move.target), -after)

    return max(moves, key=key)


def rollout_move_key(move: Move, player: int) -> tuple[int, int, int, int]:
    before = DISTANCE_TO_GOAL[player][move.source]
    after = DISTANCE_TO_GOAL[player][move.target]
    return (
        before - after,
        1 if IN_GOAL[player][move.target] else 0,
        len(move.path),
        PROGRESS[player][move.target],
    )


def random_playout(cells: str, player: int, rng: random.Random, depth: int) -> float:
    current = cells
    to_move = player
    for _ in range(depth):
        winner = is_terminal(current)
        if winner == player:
            return 1.0
        if winner == 1 - player:
            return 0.0
        moves = legal_moves(current, to_move)
        if not moves:
            break
        if rng.random() < 0.75:
            move = max(moves, key=lambda candidate: rollout_move_key(candidate, to_move))
        else:
            move = rng.choice(moves)
        current = apply_move(current, move, to_move)
        to_move = 1 - to_move
    score = conversion_score(current, player)
    return 1.0 / (1.0 + math.exp(-score / 140.0))


def choose_mcts(cells: str, player: int, rng: random.Random, iterations: int, rollout_depth: int) -> Move:
    moves = legal_moves(cells, player)
    if not moves:
        raise RuntimeError("no legal moves")
    wins = [0.0 for _ in moves]
    visits = [0 for _ in moves]
    for index, move in enumerate(moves):
        wins[index] += random_playout(apply_move(cells, move, player), 1 - player, rng, rollout_depth)
        visits[index] += 1
    for total in range(len(moves), max(len(moves), iterations)):
        log_total = math.log(total + 1)
        index = max(
            range(len(moves)),
            key=lambda i: wins[i] / visits[i] + 1.35 * math.sqrt(log_total / visits[i]),
        )
        move = moves[index]
        wins[index] += random_playout(apply_move(cells, move, player), 1 - player, rng, rollout_depth)
        visits[index] += 1
    best_index = max(range(len(moves)), key=lambda i: (wins[i] / visits[i], visits[i]))
    return moves[best_index]


def choose_move(args: argparse.Namespace, cells: str, player: int, rng: random.Random) -> Move:
    if args.style == "zedrichu-minimax":
        return choose_alphabeta(cells, player, args.depth, zedrichu_score, beam=args.beam)
    if args.style == "marblefish-ab":
        return choose_alphabeta(cells, player, args.depth, marblefish_score, beam=args.beam)
    if args.style == "harryz-rule":
        return choose_rule_path(cells, player)
    if args.style in {"svjayanthi-mcts", "svjayanthi-mcts-fast"}:
        return choose_mcts(cells, player, rng, args.iterations, args.rollout_depth)
    raise RuntimeError(f"unknown style: {args.style}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--style",
        choices=("zedrichu-minimax", "marblefish-ab", "harryz-rule", "svjayanthi-mcts", "svjayanthi-mcts-fast"),
        required=True,
    )
    parser.add_argument("--name")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--beam", type=int)
    parser.add_argument("--iterations", type=int, default=96)
    parser.add_argument("--rollout-depth", type=int, default=28)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    name = args.name or args.style
    cells = ""
    player = 0
    request_index = 0
    print(f"id name {name}", flush=True)
    print("id author CCERL source-style adapter", flush=True)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        command = parts[0]
        try:
            if command == "ccp":
                print("ccpok", flush=True)
            elif command == "protocol":
                print("protocol ccp 1", flush=True)
            elif command == "isready":
                print("readyok", flush=True)
            elif command == "position":
                if len(parts) >= 2 and parts[1] == "startpos":
                    cells = STARTPOS
                    if "player" in parts:
                        player = int(parts[parts.index("player") + 1])
                elif len(parts) >= 3 and parts[1] == "cells":
                    cells = parts[2]
                    if "player" in parts:
                        player = int(parts[parts.index("player") + 1])
                else:
                    raise ValueError("expected: position cells <COMPACT121> player <0|1> ply <N>")
            elif command == "go":
                rng = random.Random(args.seed + request_index * 1_000_003)
                start = time.perf_counter()
                move = choose_move(args, cells, player, rng)
                request_index += 1
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                print(f"info string style={args.style} elapsed_ms={elapsed_ms}", flush=True)
                print("bestmove " + "-".join(str(item) for item in move.path), flush=True)
            elif command == "quit":
                return 0
            else:
                print(f"info string ignored unknown command {command}", flush=True)
        except Exception as exc:
            print(f"error {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
