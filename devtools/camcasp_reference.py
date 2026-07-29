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


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    length = _norm(vector)
    if length <= 1.0e-14:
        raise ReferenceFormatError("axis direction has zero length")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(zip(*matrix))  # type: ignore[return-value]


def _matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    right_t = _transpose(right)
    return tuple(
        tuple(_dot(row, column) for column in right_t)
        for row in left
    )  # type: ignore[return-value]


def _determinant(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def validate_rotation_matrix(rotation: Matrix3, tolerance: float = 1.0e-10) -> None:
    for row in rotation:
        for value in row:
            if not math.isfinite(value):
                raise ReferenceFormatError("local frame contains non-finite value")
    product = _matmul(rotation, _transpose(rotation))
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            if abs(product[row][column] - expected) > tolerance:
                raise ReferenceFormatError("local frame is not orthonormal")
    determinant = _determinant(rotation)
    if determinant < 0.0:
        raise ReferenceFormatError("local frame is left-handed")
    if abs(determinant - 1.0) > tolerance:
        raise ReferenceFormatError(
            f"local frame determinant is {determinant}, expected +1"
        )


def build_local_frames(
    geometry: Mapping[str, Vector3],
    axes_text: str,
) -> dict[str, Matrix3]:
    frames: dict[str, Matrix3] = {
        label: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        for label in geometry
    }
    rule = re.compile(
        r"^\s*(\S+)\s+z\s+global\s+Z\s+x\s+from\s+(\S+)\s+to\s+(\S+)\s*$",
        re.IGNORECASE,
    )
    required_sites = ("H1", "H2")
    parsed_sites: set[str] = set()
    for line in axes_text.splitlines():
        match = rule.match(line)
        if not match:
            continue
        site, origin, target = match.groups()
        if site not in geometry or origin not in geometry or target not in geometry:
            raise ReferenceFormatError(f"unknown site in axis rule: {line.strip()}")
        if site not in required_sites:
            raise ReferenceFormatError(
                f"axis rule must define H1 or H2, found {site}"
            )
        if site in parsed_sites:
            raise ReferenceFormatError(f"duplicate axis rule for {site}")
        local_z = (0.0, 0.0, 1.0)
        direction = tuple(
            geometry[target][index] - geometry[origin][index]
            for index in range(3)
        )
        projected = tuple(
            direction[index] - _dot(direction, local_z) * local_z[index]
            for index in range(3)
        )
        local_x = _normalize(projected)  # type: ignore[arg-type]
        local_y = _normalize(_cross(local_z, local_x))
        rotation = tuple(zip(local_x, local_y, local_z))  # type: ignore[assignment]
        validate_rotation_matrix(rotation)
        frames[site] = rotation
        parsed_sites.add(site)

    missing_sites = [site for site in required_sites if site not in parsed_sites]
    if missing_sites:
        raise ReferenceFormatError(
            f"missing axis rules for {', '.join(missing_sites)}"
        )
    return frames


# CamCASP real spherical dipoles: 10 -> z, 11c -> x, 11s -> y.
CARTESIAN_TO_SPHERICAL_DIPOLE = ("11c", "11s", "10")


def _validate_symmetric(
    matrix: Matrix3,
    context: str,
    tolerance: float = 1.0e-8,
) -> None:
    for row in range(3):
        for column in range(3):
            if abs(matrix[row][column] - matrix[column][row]) > tolerance:
                raise ReferenceFormatError(f"{context} is not symmetric")


def dipole_local_cartesian(model: SphericalModel) -> Matrix3:
    index = {
        label: model.components.index(label)
        for label in CARTESIAN_TO_SPHERICAL_DIPOLE
    }
    result = tuple(
        tuple(
            model.matrix[index[left]][index[right]]
            for right in CARTESIAN_TO_SPHERICAL_DIPOLE
        )
        for left in CARTESIAN_TO_SPHERICAL_DIPOLE
    )
    for row in result:
        for value in row:
            if not math.isfinite(value):
                raise ReferenceFormatError(
                    "local Cartesian dipole tensor contains non-finite value"
                )
    _validate_symmetric(result, "local Cartesian dipole tensor")
    return result  # type: ignore[return-value]


def rotate_tensor(local: Matrix3, rotation: Matrix3) -> Matrix3:
    validate_rotation_matrix(rotation)
    global_tensor = _matmul(_matmul(rotation, local), _transpose(rotation))
    _validate_symmetric(global_tensor, "global Cartesian dipole tensor")
    for row in global_tensor:
        for value in row:
            if not math.isfinite(value):
                raise ReferenceFormatError(
                    "global Cartesian tensor contains non-finite value"
                )
    return global_tensor


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
ACCEPTED_ATOM_LABELS = ("O", "H1", "H2")


def parse_refined_polarizabilities(
    path: Path,
    atom_labels: Sequence[str],
    limit: int,
) -> list[FrequencyBlock]:
    if limit != 3:
        raise ReferenceFormatError(f"accepted model requires limit 3, got {limit}")
    provided_atom_labels = tuple(atom_labels)
    if provided_atom_labels != ACCEPTED_ATOM_LABELS:
        raise ReferenceFormatError(
            f"accepted model requires atom labels {ACCEPTED_ATOM_LABELS!r}, "
            f"got {provided_atom_labels!r}"
        )
    lines = path.read_text().splitlines()
    blocks: list[FrequencyBlock] = []
    position = 0

    while position < len(lines):
        while position < len(lines) and (
            not lines[position].strip()
            or (
                lines[position].lstrip().startswith("#")
                and not INDEX_RE.match(lines[position])
            )
        ):
            position += 1
        if position >= len(lines):
            break
        match = INDEX_RE.match(lines[position])
        if not match:
            raise ReferenceFormatError(
                f"{path}: expected frequency index, found "
                f"{lines[position].strip()!r}"
            )
        index = int(match.group(1))
        if index != len(blocks):
            raise ReferenceFormatError(
                f"{path}: expected frequency index {len(blocks):03d}, found {index:03d}"
            )
        position += 1
        atoms: dict[str, SphericalModel] = {}

        for expected_atom in ACCEPTED_ATOM_LABELS:
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

        while position < len(lines) and not INDEX_RE.match(lines[position]):
            if lines[position].strip() and not lines[position].lstrip().startswith("#"):
                raise ReferenceFormatError(
                    f"{path}: frequency {index:03d} unexpected content after atom "
                    f"H2: {lines[position].strip()!r}"
                )
            position += 1

    if len(blocks) != 11:
        raise ReferenceFormatError(
            f"{path}: expected 11 refined blocks, found {len(blocks)}"
        )
    return blocks
