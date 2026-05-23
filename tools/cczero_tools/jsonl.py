"""JSONL IO helpers."""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


def gzip_compresslevel() -> int:
    value = os.environ.get("CCZERO_GZIP_LEVEL", "1")
    try:
        return max(1, min(9, int(value)))
    except ValueError:
        return 1


@contextmanager
def open_text(path: Path, mode: str = "rt") -> Iterator[TextIO]:
    """Open plain or gzip-compressed text.

    Reads sniff gzip magic bytes because some compact artifacts may keep a .gz
    suffix after being rewritten as plain JSONL by storage cleanup tools.
    """
    if "b" in mode:
        raise ValueError("open_text only supports text modes")
    compressed = False
    if "r" in mode and path.exists():
        with path.open("rb") as probe:
            compressed = probe.read(2) == b"\x1f\x8b"
    elif path.suffix == ".gz":
        compressed = True
    if compressed:
        kwargs = {"compresslevel": gzip_compresslevel()} if any(flag in mode for flag in "wax") else {}
        with gzip.open(path, mode, encoding="utf-8", **kwargs) as handle:
            yield handle
    else:
        with path.open(mode, encoding="utf-8") as handle:
            yield handle


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with open_text(path, "rt") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_jsonl_records(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def copy_jsonl_lines(inputs: list[Path], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text(out, "wt") as target:
        for path in inputs:
            with open_text(path, "rt") as source:
                for line in source:
                    if not line.strip():
                        continue
                    target.write(line)
                    if not line.endswith("\n"):
                        target.write("\n")
                    count += 1
    return count


def write_jsonl(path: Path, records: list[dict]) -> None:
    write_jsonl_records(path, records)
