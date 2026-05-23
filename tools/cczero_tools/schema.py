"""Versioned JSONL contract checks for CCZero datasets."""

from __future__ import annotations

import math
import re
from typing import Any

from cczero_tools.compact import (
    BOARD_SIZE,
    SELFPLAY_COMPACT_MP_SCHEMA,
    SELFPLAY_COMPACT_SCHEMA,
    TRAINING_MP_SCHEMA,
    TRAINING_COMPACT_SCHEMA,
    chosen_action,
    is_compact_record,
    legal_actions,
    minimal_legal,
    prior_weights,
    visit_counts,
)

PLAYER_VALUES = (0, 1)
RESULT_VALUES = (-1, 0, 1)
RECORD_TYPES = ("training_position", "selfplay_position")

RICH_SELFPLAY_METRICS = (
    "distance_before",
    "opponent_distance_before",
    "distance_advantage",
    "goal_count_before",
    "opponent_goal_count_before",
    "goal_advantage",
    "home_count_before",
    "opponent_home_count_before",
    "home_advantage",
    "goal_blockers_before",
    "opponent_goal_blockers_before",
)

SEARCH_CONFIG_FIELDS = (
    "simulations",
    "cpuct",
    "root_noise",
    "root_dirichlet_alpha",
    "root_noise_fraction",
    "temperature",
    "draw_leaf_value",
    "anti_draw_logit_scale",
    "transpositions",
    "reuse_tree",
    "adaptive_simulations",
    "min_simulations",
    "adaptive_confidence",
    "movegen",
    "inference_backend",
    "inference_batch_size",
    "max_plies",
)

SEARCH_STATS_FIELDS = (
    "nodes",
    "evals",
    "simulations",
    "elapsed_ms",
    "root_legal_moves",
    "transposition_hits",
    "inference_batches",
    "adaptive_stopped",
    "movegen_ms",
    "eval_ms",
    "policy_ms",
    "select_ms",
    "backup_ms",
)

PROVENANCE_INT_FIELDS = (
    "source_iter",
    "target_age",
    "requested_simulations",
    "actual_simulations",
)

PROVENANCE_STRING_FIELDS = (
    "target_model_id",
    "search_config_hash",
    "target_model_strength_tier",
)

TIMING_STAT_FIELDS = {
    "elapsed_ms",
    "movegen_ms",
    "eval_ms",
    "policy_ms",
    "select_ms",
    "backup_ms",
}

BASE_REQUIRED_FIELDS = (
    "type",
    "game_id",
    "ply",
    "player",
    "hash",
    "cells",
    "rule_profile",
    "result",
    "chosen",
    "legal_count",
    "legal",
)

TRAINING_REQUIRED_FIELDS = BASE_REQUIRED_FIELDS + ("bot",)
SELFPLAY_REQUIRED_FIELDS = BASE_REQUIRED_FIELDS + (
    "schema",
    "seed",
    "model_id",
    "visit_sum",
    "search",
    "stats",
)
COMPACT_BASE_REQUIRED_FIELDS = (
    "type",
    "game_id",
    "ply",
    "player",
    "hash",
    "cells",
    "rule_profile",
    "result",
    "chosen_action",
    "actions",
)
COMPACT_TRAINING_REQUIRED_FIELDS = COMPACT_BASE_REQUIRED_FIELDS + ("schema", "bot")
COMPACT_SELFPLAY_REQUIRED_FIELDS = COMPACT_BASE_REQUIRED_FIELDS + (
    "schema",
    "seed",
    "model_id",
    "visits",
    "visit_sum",
)
MP_RULE_PROFILES = {
    3: "CCZ-121-MP3-v1",
    4: "CCZ-121-MP4-v1",
    6: "CCZ-121-MP6-v1",
}
MP_COMPACT_SELFPLAY_REQUIRED_FIELDS = (
    "type",
    "schema",
    "game_id",
    "seed",
    "ply",
    "player_count",
    "player",
    "seat",
    "seat_order",
    "hash",
    "cells",
    "model_id",
    "rule_profile",
    "placements",
    "score_vector",
    "result_vector",
    "chosen_action",
    "actions",
    "visits",
    "visit_sum",
)
MP_TRAINING_REQUIRED_FIELDS = (
    "type",
    "schema",
    "game_id",
    "ply",
    "player_count",
    "player",
    "seat",
    "seat_order",
    "hash",
    "cells",
    "bot",
    "rule_profile",
    "placements",
    "score_vector",
    "result_vector",
    "chosen",
    "legal_count",
    "legal",
)

HASH_RE = re.compile(r"^0x[0-9a-fA-F]{16}(:[A-Za-z0-9_.-]+)?$")


def move_key(move: dict) -> tuple[int | None, int | None]:
    return move.get("from"), move.get("to")


def _prefix(line_no: int | None) -> str:
    return f"line {line_no}: " if line_no is not None else ""


def _check_int(record: dict, key: str, errors: list[str], line_no: int | None) -> None:
    if not isinstance(record.get(key), int):
        errors.append(f"{_prefix(line_no)}{key} must be an integer")


def _check_required(
    record: dict, fields: tuple[str, ...], errors: list[str], line_no: int | None
) -> None:
    for field in fields:
        if field not in record:
            errors.append(f"{_prefix(line_no)}{field} is missing")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_finite_range(
    value: Any,
    errors: list[str],
    line_no: int | None,
    context: str,
    low: float | None = None,
    high: float | None = None,
) -> None:
    if not _is_finite_number(value):
        errors.append(f"{_prefix(line_no)}{context} must be a finite number")
        return
    if low is not None and float(value) < low:
        errors.append(f"{_prefix(line_no)}{context} must be >= {low:g}")
    if high is not None and float(value) > high:
        errors.append(f"{_prefix(line_no)}{context} must be <= {high:g}")


def _check_nonnegative_int(value: Any, errors: list[str], line_no: int | None, context: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{_prefix(line_no)}{context} must be a non-negative integer")


def validate_move(move: Any, errors: list[str], line_no: int | None, context: str) -> None:
    if not isinstance(move, dict):
        errors.append(f"{_prefix(line_no)}{context} must be an object")
        return
    for key in ("from", "to"):
        value = move.get(key)
        if not isinstance(value, int) or value < 0 or value >= BOARD_SIZE:
            errors.append(f"{_prefix(line_no)}{context}.{key} must be a board id")
    path = move.get("path")
    if path is not None:
        if not isinstance(path, list) or len(path) < 2:
            errors.append(f"{_prefix(line_no)}{context}.path must contain at least two ids")
        elif any(not isinstance(cell, int) or cell < 0 or cell >= BOARD_SIZE for cell in path):
            errors.append(f"{_prefix(line_no)}{context}.path contains an invalid board id")
        elif isinstance(move.get("from"), int) and isinstance(move.get("to"), int):
            if path[0] != move["from"] or path[-1] != move["to"]:
                errors.append(f"{_prefix(line_no)}{context}.path endpoints do not match from/to")


def validate_search(search: Any, errors: list[str], line_no: int | None) -> None:
    if not isinstance(search, dict):
        errors.append(f"{_prefix(line_no)}search must be an object")
        return
    for field in SEARCH_CONFIG_FIELDS:
        if field not in search:
            errors.append(f"{_prefix(line_no)}search.{field} is missing")
    for field in (
        "simulations",
        "min_simulations",
        "adaptive_check_interval",
        "inference_batch_size",
        "max_plies",
    ):
        if field in search:
            _check_nonnegative_int(search[field], errors, line_no, f"search.{field}")
    for field in (
        "cpuct",
        "root_dirichlet_alpha",
        "root_noise_fraction",
        "temperature",
        "draw_leaf_value",
        "anti_draw_logit_scale",
        "progress_prior_scale",
        "home_pressure_scale",
        "adaptive_confidence",
    ):
        if field in search:
            low = 0.0 if field not in ("draw_leaf_value",) else -1.0
            high = 1.0 if field in ("root_noise_fraction", "draw_leaf_value", "adaptive_confidence") else None
            _check_finite_range(search[field], errors, line_no, f"search.{field}", low, high)
    for field in ("root_noise", "transpositions", "reuse_tree", "adaptive_simulations"):
        if field in search and not isinstance(search[field], bool):
            errors.append(f"{_prefix(line_no)}search.{field} must be a boolean")
    if "movegen" in search and search["movegen"] not in ("reference", "fast", "bitboard"):
        errors.append(f"{_prefix(line_no)}search.movegen has unknown backend")
    if "inference_backend" in search and search["inference_backend"] not in (
        "auto",
        "portable",
        "accelerate",
    ):
        errors.append(f"{_prefix(line_no)}search.inference_backend has unknown backend")
    if "inference_resolved_backend" in search and search["inference_resolved_backend"] not in (
        "portable",
        "accelerate",
    ):
        errors.append(f"{_prefix(line_no)}search.inference_resolved_backend has unknown backend")


def validate_stats(stats: Any, errors: list[str], line_no: int | None) -> None:
    if not isinstance(stats, dict):
        errors.append(f"{_prefix(line_no)}stats must be an object")
        return
    for field in SEARCH_STATS_FIELDS:
        if field not in stats:
            errors.append(f"{_prefix(line_no)}stats.{field} is missing")
    for field in (
        "nodes",
        "evals",
        "simulations",
        "root_legal_moves",
        "transposition_hits",
        "inference_batches",
    ):
        if field in stats:
            _check_nonnegative_int(stats[field], errors, line_no, f"stats.{field}")
    for field in TIMING_STAT_FIELDS:
        if field in stats:
            _check_finite_range(stats[field], errors, line_no, f"stats.{field}", 0.0)
    if "adaptive_stopped" in stats and not isinstance(stats["adaptive_stopped"], bool):
        errors.append(f"{_prefix(line_no)}stats.adaptive_stopped must be a boolean")


def _check_mp_number_vector(
    value: Any,
    player_count: int,
    errors: list[str],
    line_no: int | None,
    context: str,
) -> None:
    if not isinstance(value, list) or len(value) != player_count:
        errors.append(f"{_prefix(line_no)}{context} must contain {player_count} values")
        return
    for index, item in enumerate(value):
        _check_finite_range(item, errors, line_no, f"{context}[{index}]", 0.0, 1.0)


def _check_mp_int_vector(
    value: Any,
    player_count: int,
    errors: list[str],
    line_no: int | None,
    context: str,
) -> None:
    if not isinstance(value, list) or len(value) != player_count:
        errors.append(f"{_prefix(line_no)}{context} must contain {player_count} values")
        return
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 1 or item > player_count:
            errors.append(f"{_prefix(line_no)}{context}[{index}] must be a placement")


def validate_multiplayer_position_record(
    record: dict, line_no: int | None = None
) -> list[str]:
    errors: list[str] = []
    schema = record.get("schema")
    compact = schema == SELFPLAY_COMPACT_MP_SCHEMA
    training = schema == TRAINING_MP_SCHEMA
    if not compact and not training:
        return errors
    _check_required(
        record,
        MP_COMPACT_SELFPLAY_REQUIRED_FIELDS if compact else MP_TRAINING_REQUIRED_FIELDS,
        errors,
        line_no,
    )

    player_count = record.get("player_count")
    if player_count not in MP_RULE_PROFILES:
        errors.append(f"{_prefix(line_no)}player_count must be one of 3, 4, or 6")
        player_count = 0
    player = record.get("player")
    if not isinstance(player, int) or isinstance(player, bool) or player < 0 or player >= player_count:
        errors.append(f"{_prefix(line_no)}player must be a valid multiplayer seat")
    seat = record.get("seat")
    if not isinstance(seat, int) or isinstance(seat, bool) or seat < 0 or seat >= player_count:
        errors.append(f"{_prefix(line_no)}seat must be a valid multiplayer seat")
    seat_order = record.get("seat_order")
    if (
        not isinstance(seat_order, list)
        or len(seat_order) != player_count
        or sorted(seat_order) != list(range(player_count))
    ):
        errors.append(f"{_prefix(line_no)}seat_order must contain each active seat exactly once")

    cells = record.get("cells")
    valid_digits = {str(index) for index in range(player_count)}
    if not isinstance(cells, str) or len(cells) != BOARD_SIZE:
        errors.append(f"{_prefix(line_no)}cells must be a {BOARD_SIZE}-character string")
    elif any(ch != "." and ch not in valid_digits for ch in cells):
        errors.append(f"{_prefix(line_no)}cells contains inactive player ids")
    else:
        for seat_index in range(player_count):
            if cells.count(str(seat_index)) != 10:
                errors.append(
                    f"{_prefix(line_no)}cells must contain exactly 10 pieces for player {seat_index}"
                )

    _check_nonnegative_int(record.get("game_id"), errors, line_no, "game_id")
    _check_nonnegative_int(record.get("ply"), errors, line_no, "ply")
    hash_value = record.get("hash")
    if not isinstance(hash_value, str) or HASH_RE.match(hash_value) is None:
        errors.append(f"{_prefix(line_no)}hash must be a 0x-prefixed 64-bit hex string")
    rule_profile = record.get("rule_profile")
    if player_count in MP_RULE_PROFILES and rule_profile != MP_RULE_PROFILES[player_count]:
        errors.append(
            f"{_prefix(line_no)}rule_profile must be {MP_RULE_PROFILES[player_count]}"
        )
    _check_mp_int_vector(record.get("placements"), player_count, errors, line_no, "placements")
    _check_mp_number_vector(record.get("score_vector"), player_count, errors, line_no, "score_vector")
    _check_mp_number_vector(record.get("result_vector"), player_count, errors, line_no, "result_vector")
    if "winner_seats" in record:
        winner_seats = record["winner_seats"]
        if not isinstance(winner_seats, list) or any(
            not isinstance(seat_value, int)
            or isinstance(seat_value, bool)
            or seat_value < 0
            or seat_value >= player_count
            for seat_value in winner_seats
        ):
            errors.append(f"{_prefix(line_no)}winner_seats must contain valid seats")
    if "phase" in record:
        _check_nonnegative_int(record["phase"], errors, line_no, "phase")

    if compact:
        if record.get("type") != "selfplay_position":
            errors.append(f"{_prefix(line_no)}compact multiplayer records must be selfplay_position")
        if not isinstance(record.get("model_id"), str) or not record.get("model_id"):
            errors.append(f"{_prefix(line_no)}model_id must be a non-empty string")
        try:
            chosen_id = chosen_action(record)
        except (KeyError, TypeError, ValueError):
            chosen_id = -1
            errors.append(f"{_prefix(line_no)}chosen_action must be an action id")
        actions = legal_actions(record)
        if not actions:
            errors.append(f"{_prefix(line_no)}actions must be a non-empty list")
        for index, action in enumerate(actions):
            if not isinstance(action, int) or action < 0 or action >= BOARD_SIZE * BOARD_SIZE:
                errors.append(f"{_prefix(line_no)}actions[{index}] must be an action id")
        if actions and chosen_id not in actions:
            errors.append(f"{_prefix(line_no)}chosen_action is not in actions")
        legal_count = record.get("legal_count", len(actions))
        if not isinstance(legal_count, int) or legal_count <= 0:
            errors.append(f"{_prefix(line_no)}legal_count must be a positive integer")
        elif legal_count != len(actions):
            errors.append(
                f"{_prefix(line_no)}legal_count={legal_count} but actions has {len(actions)}"
            )
        visits = visit_counts(record)
        if len(visits) != len(actions):
            errors.append(f"{_prefix(line_no)}visits length must match actions length")
        elif any(not isinstance(visit, int) or visit < 0 for visit in visits):
            errors.append(f"{_prefix(line_no)}visits must contain non-negative integers")
        visit_sum = record.get("visit_sum")
        if not isinstance(visit_sum, int) or visit_sum < 0:
            errors.append(f"{_prefix(line_no)}visit_sum must be a non-negative integer")
        elif visits and visit_sum != sum(visits):
            errors.append(
                f"{_prefix(line_no)}visit_sum={visit_sum} but visits sum to {sum(visits)}"
            )
        priors = prior_weights(record)
        if priors and len(priors) != len(actions):
            errors.append(f"{_prefix(line_no)}priors length must match actions length")
        if priors:
            for index, prior in enumerate(priors):
                _check_finite_range(prior, errors, line_no, f"priors[{index}]", 0.0, 1.0)
            if abs(sum(priors) - 1.0) > 1.0e-3:
                errors.append(f"{_prefix(line_no)}priors must sum to 1")
    else:
        if record.get("type") != "training_position":
            errors.append(f"{_prefix(line_no)}multiplayer training records must be training_position")
        if not isinstance(record.get("bot"), str) or not record.get("bot"):
            errors.append(f"{_prefix(line_no)}bot must be a non-empty string")
        chosen = record.get("chosen")
        legal = record.get("legal")
        validate_move(chosen, errors, line_no, "chosen")
        if not isinstance(legal, list) or not legal:
            errors.append(f"{_prefix(line_no)}legal must be a non-empty list")
            legal = []
        else:
            legal_keys: set[tuple[int | None, int | None]] = set()
            for index, move in enumerate(legal):
                validate_move(move, errors, line_no, f"legal[{index}]")
                if isinstance(move, dict):
                    legal_keys.add(move_key(move))
            if isinstance(chosen, dict) and move_key(chosen) not in legal_keys:
                errors.append(f"{_prefix(line_no)}chosen move is not in legal move list")
        legal_count = record.get("legal_count")
        if not isinstance(legal_count, int) or legal_count <= 0:
            errors.append(f"{_prefix(line_no)}legal_count must be a positive integer")
        elif legal_count != len(legal):
            errors.append(
                f"{_prefix(line_no)}legal_count={legal_count} but legal has {len(legal)}"
            )

    return errors


def validate_position_record(
    record: dict, line_no: int | None = None, require_rich_selfplay: bool = False
) -> list[str]:
    errors: list[str] = []
    record_type = record.get("type")
    if record_type not in RECORD_TYPES:
        return errors
    if record.get("schema") in (SELFPLAY_COMPACT_MP_SCHEMA, TRAINING_MP_SCHEMA):
        return validate_multiplayer_position_record(record, line_no)
    compact = is_compact_record(record)
    _check_required(
        record,
        (
            COMPACT_SELFPLAY_REQUIRED_FIELDS
            if compact and record_type == "selfplay_position"
            else COMPACT_TRAINING_REQUIRED_FIELDS
            if compact
            else SELFPLAY_REQUIRED_FIELDS
            if record_type == "selfplay_position"
            else TRAINING_REQUIRED_FIELDS
        ),
        errors,
        line_no,
    )

    cells = record.get("cells")
    if not isinstance(cells, str) or len(cells) != BOARD_SIZE:
        errors.append(f"{_prefix(line_no)}cells must be a {BOARD_SIZE}-character string")
    elif any(ch not in ".01" for ch in cells):
        errors.append(f"{_prefix(line_no)}cells contains characters outside .01")
    elif cells.count("0") != 10 or cells.count("1") != 10:
        errors.append(f"{_prefix(line_no)}cells must contain exactly 10 pieces per player")

    if record.get("player") not in PLAYER_VALUES:
        errors.append(f"{_prefix(line_no)}player must be 0 or 1")
    if record.get("result") not in RESULT_VALUES:
        errors.append(f"{_prefix(line_no)}result must be -1, 0, or 1")
    _check_nonnegative_int(record.get("game_id"), errors, line_no, "game_id")
    _check_nonnegative_int(record.get("ply"), errors, line_no, "ply")
    hash_value = record.get("hash")
    if not isinstance(hash_value, str) or HASH_RE.match(hash_value) is None:
        errors.append(f"{_prefix(line_no)}hash must be a 0x-prefixed 64-bit hex string")
    if not isinstance(record.get("rule_profile"), str) or not record.get("rule_profile"):
        errors.append(f"{_prefix(line_no)}rule_profile must be a non-empty string")
    if record_type == "training_position" and (
        not isinstance(record.get("bot"), str) or not record.get("bot")
    ):
        errors.append(f"{_prefix(line_no)}bot must be a non-empty string")
    if compact and record_type == "training_position" and record.get("schema") != TRAINING_COMPACT_SCHEMA:
        errors.append(
            f"{_prefix(line_no)}compact training schema must be {TRAINING_COMPACT_SCHEMA}"
        )
    if record_type == "selfplay_position" and (
        not isinstance(record.get("model_id"), str) or not record.get("model_id")
    ):
        errors.append(f"{_prefix(line_no)}model_id must be a non-empty string")

    if compact:
        try:
            chosen_id = chosen_action(record)
        except (KeyError, TypeError, ValueError):
            errors.append(f"{_prefix(line_no)}chosen_action must be an action id")
            chosen_id = -1
        if not isinstance(chosen_id, int) or chosen_id < 0 or chosen_id >= BOARD_SIZE * BOARD_SIZE:
            errors.append(f"{_prefix(line_no)}chosen_action must be an action id")
        actions = legal_actions(record)
        if not actions:
            errors.append(f"{_prefix(line_no)}actions must be a non-empty list")
        for index, action in enumerate(actions):
            if not isinstance(action, int) or action < 0 or action >= BOARD_SIZE * BOARD_SIZE:
                errors.append(f"{_prefix(line_no)}actions[{index}] must be an action id")
        if actions and chosen_id not in actions:
            errors.append(f"{_prefix(line_no)}chosen_action is not in actions")
        legal = minimal_legal(record)
        declared_legal_count = record.get("legal_count", len(actions))
        if not isinstance(declared_legal_count, int) or declared_legal_count <= 0:
            errors.append(f"{_prefix(line_no)}legal_count must be a positive integer")
        elif declared_legal_count != len(actions):
            errors.append(
                f"{_prefix(line_no)}legal_count={declared_legal_count} but actions has {len(actions)}"
            )
    else:
        chosen = record.get("chosen")
        legal = record.get("legal")
        validate_move(chosen, errors, line_no, "chosen")
        if not isinstance(legal, list) or not legal:
            errors.append(f"{_prefix(line_no)}legal must be a non-empty list")
            legal = []
        else:
            legal_keys: set[tuple[int | None, int | None]] = set()
            for index, move in enumerate(legal):
                validate_move(move, errors, line_no, f"legal[{index}]")
                if isinstance(move, dict):
                    legal_keys.add(move_key(move))
            if isinstance(chosen, dict) and move_key(chosen) not in legal_keys:
                errors.append(f"{_prefix(line_no)}chosen move is not in legal move list")

        declared_legal_count = record.get("legal_count")
        if not isinstance(declared_legal_count, int) or declared_legal_count <= 0:
            errors.append(f"{_prefix(line_no)}legal_count must be a positive integer")
        elif declared_legal_count != len(legal):
            errors.append(
                f"{_prefix(line_no)}legal_count={declared_legal_count} but legal has {len(legal)}"
            )

    if "phase" in record:
        _check_nonnegative_int(record["phase"], errors, line_no, "phase")
    if "finish_margin_moves" in record:
        _check_nonnegative_int(record["finish_margin_moves"], errors, line_no, "finish_margin_moves")
    if "finish_margin_max_moves" in record:
        _check_nonnegative_int(
            record["finish_margin_max_moves"], errors, line_no, "finish_margin_max_moves"
        )
    if "finish_margin_capped" in record and not isinstance(record["finish_margin_capped"], bool):
        errors.append(f"{_prefix(line_no)}finish_margin_capped must be a boolean")
    if "score_margin" in record:
        _check_finite_range(record["score_margin"], errors, line_no, "score_margin", -1.0, 1.0)
    if "score_margin_source" in record and not isinstance(record["score_margin_source"], str):
        errors.append(f"{_prefix(line_no)}score_margin_source must be a string")
    if "root_q" in record:
        _check_finite_range(record["root_q"], errors, line_no, "root_q", -1.0, 1.0)
    if "root_q_source" in record and not isinstance(record["root_q_source"], str):
        errors.append(f"{_prefix(line_no)}root_q_source must be a string")

    if record_type == "selfplay_position":
        for field in PROVENANCE_INT_FIELDS:
            if field in record:
                _check_nonnegative_int(record.get(field), errors, line_no, field)
        for field in PROVENANCE_STRING_FIELDS:
            if field in record and not isinstance(record.get(field), str):
                errors.append(f"{_prefix(line_no)}{field} must be a string")
        if "trusted_target" in record and not isinstance(record.get("trusted_target"), bool):
            errors.append(f"{_prefix(line_no)}trusted_target must be a boolean")

        if compact:
            if record.get("schema") != SELFPLAY_COMPACT_SCHEMA:
                errors.append(
                    f"{_prefix(line_no)}compact selfplay schema must be {SELFPLAY_COMPACT_SCHEMA}"
                )
        elif record.get("schema") != "cczero.selfplay.v1":
            errors.append(f"{_prefix(line_no)}selfplay schema must be cczero.selfplay.v1")
        has_any_rich_metric = any(metric in record for metric in RICH_SELFPLAY_METRICS)
        if require_rich_selfplay and not has_any_rich_metric:
            errors.append(f"{_prefix(line_no)}rich selfplay metrics are required")
        if has_any_rich_metric:
            for metric in RICH_SELFPLAY_METRICS:
                _check_int(record, metric, errors, line_no)
            if (
                isinstance(record.get("distance_before"), int)
                and isinstance(record.get("opponent_distance_before"), int)
                and isinstance(record.get("distance_advantage"), int)
                and record["distance_advantage"]
                != record["opponent_distance_before"] - record["distance_before"]
            ):
                errors.append(f"{_prefix(line_no)}distance_advantage is inconsistent")
            if (
                isinstance(record.get("goal_count_before"), int)
                and isinstance(record.get("opponent_goal_count_before"), int)
                and isinstance(record.get("goal_advantage"), int)
                and record["goal_advantage"]
                != record["goal_count_before"] - record["opponent_goal_count_before"]
            ):
                errors.append(f"{_prefix(line_no)}goal_advantage is inconsistent")
            if (
                isinstance(record.get("home_count_before"), int)
                and isinstance(record.get("opponent_home_count_before"), int)
                and isinstance(record.get("home_advantage"), int)
                and record["home_advantage"]
                != record["opponent_home_count_before"] - record["home_count_before"]
            ):
                errors.append(f"{_prefix(line_no)}home_advantage is inconsistent")

        visit_sum = record.get("visit_sum")
        compact_visits = visit_counts(record) if compact else []
        legal_visit_sum = 0
        policy_sum = 0.0
        prior_sum = 0.0
        for move in legal:
            if not isinstance(move, dict):
                continue
            visits = move.get("visits")
            if not isinstance(visits, int) or visits < 0:
                errors.append(f"{_prefix(line_no)}selfplay legal move has invalid visits")
            else:
                legal_visit_sum += visits
            _check_finite_range(move.get("policy_target"), errors, line_no, "policy_target", 0.0, 1.0)
            if not compact:
                _check_finite_range(move.get("prior"), errors, line_no, "prior", 0.0, 1.0)
                _check_finite_range(move.get("q"), errors, line_no, "q", -1.0, 1.0)
            if _is_finite_number(move.get("policy_target")):
                policy_sum += float(move.get("policy_target", 0.0))
            if _is_finite_number(move.get("prior")):
                prior_sum += float(move.get("prior", 0.0))
        if compact and len(compact_visits) != len(legal_actions(record)):
            errors.append(f"{_prefix(line_no)}visits length must match actions length")
        compact_priors = prior_weights(record) if compact else []
        if compact_priors and len(compact_priors) != len(legal_actions(record)):
            errors.append(f"{_prefix(line_no)}priors length must match actions length")
        if not isinstance(visit_sum, int) or visit_sum < 0:
            errors.append(f"{_prefix(line_no)}visit_sum must be a non-negative integer")
        if isinstance(visit_sum, int) and visit_sum != legal_visit_sum:
            errors.append(
                f"{_prefix(line_no)}visit_sum={visit_sum} but legal visits sum to {legal_visit_sum}"
            )
        if legal and isinstance(visit_sum, int) and visit_sum > 0 and abs(policy_sum - 1.0) > 1.0e-3:
            errors.append(f"{_prefix(line_no)}policy_target values must sum to 1")
        if legal and prior_sum > 0.0 and abs(prior_sum - 1.0) > 1.0e-3:
            errors.append(f"{_prefix(line_no)}prior values must sum to 1")

        if not compact or "search" in record:
            validate_search(record.get("search"), errors, line_no)
        if not compact or "stats" in record:
            validate_stats(record.get("stats"), errors, line_no)

    return errors


def semantic_record(record: dict) -> dict:
    """Return a deterministic-comparison view with timing-only fields removed."""
    clean = dict(record)
    stats = clean.get("stats")
    if isinstance(stats, dict):
        clean["stats"] = {k: v for k, v in stats.items() if k not in TIMING_STAT_FIELDS}
    return clean
