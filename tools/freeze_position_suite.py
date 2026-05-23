#!/usr/bin/env python3
"""Freeze an audited CCERL position suite from one or more audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSITIONS = Path("benchmarks/ccerl-v1/positions/official_elo.jsonl")


def repo_path(path: Path | str) -> Path:
    parsed = Path(path)
    return parsed if parsed.is_absolute() else ROOT / parsed


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_positions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_phase_quotas(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    quotas: dict[str, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        quotas[key.strip()] = int(value)
    return quotas


def proportional_quotas(positions: list[dict[str, Any]], target: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for position in positions:
        phase = str(position.get("phase", "unknown"))
        counts[phase] = counts.get(phase, 0) + 1
    raw = {phase: target * count / len(positions) for phase, count in counts.items()}
    quotas = {phase: int(value) for phase, value in raw.items()}
    remaining = target - sum(quotas.values())
    for phase, _value in sorted(raw.items(), key=lambda item: (-(item[1] - int(item[1])), item[0])):
        if remaining <= 0:
            break
        quotas[phase] += 1
        remaining -= 1
    return quotas


def load_audits(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_position: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        report = json.loads(repo_path(path).read_text(encoding="utf-8"))
        audit_label = f"{report['protocol']['engine_a_label']}_vs_{report['protocol']['engine_b_label']}"
        for row in report.get("rows", []):
            if int(row.get("games", 0)) <= 0:
                continue
            item = dict(row)
            item["audit"] = audit_label
            item["audit_report"] = rel(repo_path(path))
            by_position.setdefault(str(row["id"]), []).append(item)
    return by_position


def summarize_position(position: dict[str, Any], audits: list[dict[str, Any]], target_score: float) -> dict[str, Any]:
    side_biases = [float(row["side_bias"]) for row in audits]
    bad_endings = [int(row.get("bad_endings", 0)) for row in audits]
    target_distances = [abs(float(row["engine_a_score"]) - target_score) for row in audits]
    draw_rates = [float(row.get("draw_rate", 0.0)) for row in audits]
    return {
        "id": position["id"],
        "order": int(position.get("id", "0").rsplit("_", 1)[-1]) if "_" in str(position.get("id")) else 0,
        "phase": position.get("phase", "unknown"),
        "source": position.get("source", {}),
        "audit_count": len(audits),
        "max_side_bias": max(side_biases) if side_biases else 1.0,
        "mean_side_bias": sum(side_biases) / len(side_biases) if side_biases else 1.0,
        "bad_endings": sum(bad_endings),
        "mean_target_distance": sum(target_distances) / len(target_distances) if target_distances else 1.0,
        "mean_draw_rate": sum(draw_rates) / len(draw_rates) if draw_rates else 0.0,
        "audits": audits,
    }


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        float(row["bad_endings"]),
        float(row["max_side_bias"]),
        float(row["mean_side_bias"]),
        float(row["mean_target_distance"]),
        int(row["order"]),
    )


def select_rows(
    *,
    rows: list[dict[str, Any]],
    target: int,
    quotas: dict[str, int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    rows_by_quality = sorted(rows, key=selection_key)
    for phase, quota in quotas.items():
        phase_rows = [row for row in rows_by_quality if row["phase"] == phase]
        for row in phase_rows[:quota]:
            if row["id"] not in used:
                selected.append(row)
                used.add(row["id"])
    for row in rows_by_quality:
        if len(selected) >= target:
            break
        if row["id"] in used:
            continue
        selected.append(row)
        used.add(row["id"])
    return sorted(selected, key=lambda row: int(row["order"]))


def write_positions(path: Path, positions: list[dict[str, Any]], selected: list[dict[str, Any]], suite: str) -> None:
    by_id = {str(position["id"]): dict(position) for position in positions}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            position = by_id[row["id"]]
            position["suite"] = suite
            position["id"] = f"{suite}_{index:04d}"
            position["source_position_id"] = row["id"]
            position["audit_summary"] = {
                "audit_count": row["audit_count"],
                "max_side_bias": row["max_side_bias"],
                "mean_side_bias": row["mean_side_bias"],
                "bad_endings": row["bad_endings"],
                "mean_draw_rate": row["mean_draw_rate"],
            }
            handle.write(json.dumps(position, sort_keys=True) + "\n")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Frozen Position Suite", ""]
    protocol = report["protocol"]
    lines.append(
        f"Selected {report['selected_count']} of {report['eligible_count']} eligible audited rows "
        f"from `{protocol['positions']}`."
    )
    lines.append("")
    lines.append(
        f"Filters: min audits `{protocol['min_audits']}`, max side bias `{protocol['max_side_bias']}`, "
        f"target `{protocol['target']}`."
    )
    lines.extend(["", "## Phase Counts", ""])
    lines.extend(["| Phase | Selected | Quota |", "|---|---:|---:|"])
    selected_counts: dict[str, int] = report["selected_phase_counts"]
    quotas: dict[str, int] = report["phase_quotas"]
    for phase in sorted(set(selected_counts) | set(quotas)):
        lines.append(f"| {phase} | {selected_counts.get(phase, 0)} | {quotas.get(phase, 0)} |")
    lines.extend(["", "## Selected Rows", ""])
    lines.extend(["| New ID | Source ID | Phase | Audits | Max Bias | Mean Bias | Draw Rate |", "|---|---|---|---:|---:|---:|---:|"])
    for row in report["selected_rows"]:
        lines.append(
            f"| `{row['new_id']}` | `{row['id']}` | {row['phase']} | {row['audit_count']} | "
            f"{row['max_side_bias']:.3f} | {row['mean_side_bias']:.3f} | {row['mean_draw_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--suite", default="official_elo_v2")
    parser.add_argument("--target", type=int, default=64)
    parser.add_argument("--phase-quotas")
    parser.add_argument("--min-audits", type=int, default=1)
    parser.add_argument("--max-side-bias", type=float, default=0.25)
    parser.add_argument("--target-score", type=float, default=0.65)
    parser.add_argument("--exclude-id", action="append", default=[])
    args = parser.parse_args()

    positions = load_positions(repo_path(args.positions))
    audits_by_position = load_audits(args.audit)
    excluded = set(args.exclude_id)
    summaries = []
    rejected = []
    for position in positions:
        position_id = str(position["id"])
        row = summarize_position(position, audits_by_position.get(position_id, []), args.target_score)
        reason = None
        if position_id in excluded:
            reason = "explicit_exclude"
        elif row["audit_count"] < args.min_audits:
            reason = "too_few_audits"
        elif row["bad_endings"] > 0:
            reason = "bad_endings"
        elif row["max_side_bias"] > args.max_side_bias:
            reason = "side_bias"
        if reason is None:
            summaries.append(row)
        else:
            row["reject_reason"] = reason
            rejected.append(row)

    quotas = parse_phase_quotas(args.phase_quotas) or proportional_quotas(positions, args.target)
    selected = select_rows(rows=summaries, target=min(args.target, len(summaries)), quotas=quotas)
    selected_for_report = []
    for index, row in enumerate(selected):
        selected_for_report.append({"new_id": f"{args.suite}_{index:04d}", **row})
    selected_phase_counts: dict[str, int] = {}
    for row in selected:
        phase = str(row["phase"])
        selected_phase_counts[phase] = selected_phase_counts.get(phase, 0) + 1

    out_path = repo_path(args.out)
    write_positions(out_path, positions, selected, args.suite)
    report = {
        "schema": "ccerl.position_freeze.v1",
        "protocol": {
            "positions": rel(repo_path(args.positions)),
            "audits": [rel(repo_path(path)) for path in args.audit],
            "suite": args.suite,
            "target": args.target,
            "phase_quotas": quotas,
            "min_audits": args.min_audits,
            "max_side_bias": args.max_side_bias,
            "target_score": args.target_score,
            "exclude_ids": sorted(excluded),
        },
        "out": rel(out_path),
        "selected_count": len(selected),
        "eligible_count": len(summaries),
        "rejected_count": len(rejected),
        "selected_phase_counts": selected_phase_counts,
        "phase_quotas": quotas,
        "selected_rows": selected_for_report,
        "rejected_rows": rejected,
    }
    report_path = repo_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = repo_path(args.markdown) if args.markdown else report_path.with_suffix(".md")
    write_markdown(markdown_path, report)
    print(
        json.dumps(
            {
                "out": rel(out_path),
                "report": rel(report_path),
                "markdown": rel(markdown_path),
                "selected": len(selected),
                "eligible": len(summaries),
                "rejected": len(rejected),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
