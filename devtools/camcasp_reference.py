#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

COMPONENTS_BY_RANK = {
    0: ("00",),
    1: ("10", "11c", "11s"),
    2: ("20", "21c", "21s", "22c", "22s"),
    3: ("30", "31c", "31s", "32c", "32s", "33c", "33s"),
}
COMPONENTS_L3 = tuple(
    component
    for rank in range(4)
    for component in COMPONENTS_BY_RANK[rank]
)


class ReferenceFormatError(ValueError):
    pass


@dataclass(frozen=True)
class FrequencyPoint:
    index: int
    squared_source_text: str
    squared_frequency: float
    omega: float


@dataclass(frozen=True)
class SphericalModel:
    components: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class FrequencyBlock:
    index: int
    atoms: dict[str, SphericalModel]


FREQ2_RE = re.compile(
    r"\bFREQ2\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EDed][-+]?\d+)?)"
)


def _float(text: str, context: str) -> float:
    try:
        value = float(text.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ReferenceFormatError(f"{context}: invalid float {text!r}") from exc
    if not math.isfinite(value):
        raise ReferenceFormatError(f"{context}: non-finite value {text!r}")
    return value


def parse_frequencies(path: Path) -> list[FrequencyPoint]:
    unique: list[tuple[str, float]] = []
    for match in FREQ2_RE.finditer(path.read_text()):
        raw = match.group(1)
        value = _float(raw, f"{path}: FREQ2")
        if not unique or value != unique[-1][1]:
            unique.append((raw, value))

    if len(unique) != 11:
        raise ReferenceFormatError(
            f"{path}: expected 11 unique frequency blocks, found {len(unique)}"
        )
    if unique[0][1] != 0.0:
        raise ReferenceFormatError(f"{path}: first frequency is not static zero")

    points = []
    for index, (raw, squared) in enumerate(unique):
        if index and squared >= 0.0:
            raise ReferenceFormatError(
                f"{path}: dynamic FREQ2 at index {index} is not negative"
            )
        omega = 0.0 if index == 0 else math.sqrt(-squared)
        points.append(FrequencyPoint(index, raw, squared, omega))

    if any(points[index].omega >= points[index + 1].omega for index in range(10)):
        raise ReferenceFormatError(f"{path}: imaginary frequencies are not increasing")
    return points


INDEX_RE = re.compile(r"^\s*#\s*INDEX\s+(\d{3})\s*$")
ATOM_RE = re.compile(r"^\s*(\S+)\s+\1\s*$")


def parse_refined_polarizabilities(
    path: Path,
    atom_labels: Sequence[str],
    limit: int,
) -> list[FrequencyBlock]:
    if limit != 3:
        raise ReferenceFormatError(f"accepted model requires limit 3, got {limit}")
    lines = path.read_text().splitlines()
    blocks: list[FrequencyBlock] = []
    position = 0

    while position < len(lines):
        match = INDEX_RE.match(lines[position])
        if not match:
            position += 1
            continue
        index = int(match.group(1))
        if index != len(blocks):
            raise ReferenceFormatError(
                f"{path}: expected frequency index {len(blocks):03d}, found {index:03d}"
            )
        position += 1
        atoms: dict[str, SphericalModel] = {}

        for expected_atom in atom_labels:
            while position < len(lines) and not lines[position].strip():
                position += 1
            if position >= len(lines) or lines[position].split() != [
                expected_atom,
                expected_atom,
            ]:
                found = "<end>" if position >= len(lines) else lines[position].strip()
                raise ReferenceFormatError(
                    f"{path}: frequency {index:03d} expected atom "
                    f"{expected_atom}, found {found!r}"
                )
            position += 1
            matrix = []
            for row_index in range(16):
                if position >= len(lines):
                    raise ReferenceFormatError(
                        f"{path}: frequency {index:03d} atom {expected_atom} "
                        "requires 16 rows"
                    )
                fields = lines[position].split()
                if len(fields) != 16:
                    raise ReferenceFormatError(
                        f"{path}: frequency {index:03d} atom {expected_atom} "
                        f"row {row_index} requires 16 values, found {len(fields)}"
                    )
                matrix.append(
                    tuple(
                        _float(
                            field,
                            f"{path}: frequency {index:03d} atom "
                            f"{expected_atom} row {row_index}",
                        )
                        for field in fields
                    )
                )
                position += 1
            atoms[expected_atom] = SphericalModel(COMPONENTS_L3, tuple(matrix))
        blocks.append(FrequencyBlock(index, atoms))

    if len(blocks) != 11:
        raise ReferenceFormatError(
            f"{path}: expected 11 refined blocks, found {len(blocks)}"
        )
    return blocks
