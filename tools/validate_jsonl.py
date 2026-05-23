#!/usr/bin/env python3
"""Validate CCZero training/self-play JSONL records against the shared contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cczero_tools.jsonl import open_text
from cczero_tools.schema import RECORD_TYPES, validate_position_record


def validate_file(
    path: Path, require_rich_selfplay: bool = False, allow_other_types: bool = False
) -> dict:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    position_records = 0
    with open_text(path, "rt") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            counts[str(record.get("type", "<missing>"))] += 1
            if record.get("type") in RECORD_TYPES:
                position_records += 1
            elif not allow_other_types:
                errors.append(f"line {line_no}: unsupported record type {record.get('type')!r}")
            errors.extend(validate_position_record(record, line_no, require_rich_selfplay))
    return {
        "path": str(path),
        "records": sum(counts.values()),
        "position_records": position_records,
        "types": dict(sorted(counts.items())),
        "errors": errors,
        "ok": not errors and position_records > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--require-rich-selfplay", action="store_true")
    parser.add_argument("--allow-other-types", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-errors", type=int, default=25)
    args = parser.parse_args()

    report = validate_file(args.path, args.require_rich_selfplay, args.allow_other_types)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Dataset: {report['path']}")
        print(f"Records: {report['records']}")
        print(f"Types: {report['types']}")
        if report["errors"]:
            print("Errors:")
            for error in report["errors"][: args.max_errors]:
                print(f"  - {error}")
            if len(report["errors"]) > args.max_errors:
                print(f"  ... {len(report['errors']) - args.max_errors} more")
        print(f"Status: {'ok' if report['ok'] else 'failed'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
