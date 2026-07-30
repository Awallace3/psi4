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
        stripped = line.strip()
        if (
            not stripped
            or stripped in {"Axes", "End"}
            or stripped.startswith(("!", "#"))
        ):
            continue
        match = rule.match(line)
        if not match:
            raise ReferenceFormatError(f"invalid axes line: {stripped}")
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


CN_ORDERS = ("C6", "C8", "C10", "C12")
PAIR_HEADER_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(C\d+\b.*)$", re.IGNORECASE)


def parse_isotropic_cn(
    path: Path,
    atom_labels: Sequence[str],
    atom_types: Mapping[str, str],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    lines = path.read_text().splitlines()
    by_types: dict[tuple[str, str], dict[str, float]] = {}
    index = 0

    while index < len(lines):
        header = PAIR_HEADER_RE.match(lines[index])
        if not header:
            index += 1
            continue
        left_type, right_type, columns_text = header.groups()
        pair_key = tuple(sorted((left_type, right_type)))
        columns = columns_text.upper().split()
        for order in CN_ORDERS:
            if order not in columns:
                raise ReferenceFormatError(
                    f"{path}: {pair_key} missing required {order} column"
                )
        column_index = {name: columns.index(name) for name in CN_ORDERS}
        if pair_key in by_types:
            raise ReferenceFormatError(f"{path}: duplicate pair block {pair_key}")
        index += 1
        isotropic = None

        while index < len(lines) and lines[index].strip().lower() != "end":
            if PAIR_HEADER_RE.match(lines[index]):
                raise ReferenceFormatError(
                    f"{path}: pair block {pair_key} missing explicit End "
                    "terminator before the next pair block"
                )
            fields = lines[index].split()
            if len(fields) >= 3 and fields[:3] == ["00", "00", "0"]:
                if isotropic is not None:
                    raise ReferenceFormatError(
                        f"{path}: duplicate 00 00 0 row for {pair_key}"
                    )
                numeric = fields[3:]
                required_width = max(column_index.values()) + 1
                if len(numeric) < required_width:
                    raise ReferenceFormatError(
                        f"{path}: {pair_key} 00 00 0 row requires "
                        f"{required_width} numeric values, found {len(numeric)}"
                    )
                isotropic = {
                    order: _float(
                        numeric[column_index[order]],
                        f"{path}: {pair_key} {order}",
                    )
                    for order in CN_ORDERS
                }
            index += 1
        if index >= len(lines):
            raise ReferenceFormatError(
                f"{path}: pair block {pair_key} missing explicit End terminator"
            )
        if isotropic is None:
            raise ReferenceFormatError(f"{path}: missing 00 00 0 row for {pair_key}")
        by_types[pair_key] = isotropic
        index += 1

    required_pairs = {
        tuple(sorted((atom_types[left], atom_types[right])))
        for left in atom_labels
        for right in atom_labels
    }
    missing = required_pairs - set(by_types)
    if missing:
        raise ReferenceFormatError(f"{path}: missing atom-type pairs {sorted(missing)}")

    matrices = {}
    for order in CN_ORDERS:
        matrix = tuple(
            tuple(
                by_types[tuple(sorted((atom_types[left], atom_types[right])))][order]
                for right in atom_labels
            )
            for left in atom_labels
        )
        for row in range(len(atom_labels)):
            for column in range(len(atom_labels)):
                if not math.isfinite(matrix[row][column]):
                    raise ReferenceFormatError(f"{path}: non-finite {order} value")
                if abs(matrix[row][column] - matrix[column][row]) > 1.0e-8:
                    raise ReferenceFormatError(f"{path}: {order} matrix is not symmetric")
        matrices[order] = matrix
    return matrices


def _active_lines(text: str) -> list[tuple[int, str]]:
    active = []
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("!", "#")):
            continue
        active.append((line_number, stripped))
    return active


def _require_exact_directive(
    observed: list[str], expected: str, context: str, name: str
) -> None:
    if not observed:
        raise ReferenceFormatError(f"{context}: missing active {name} directive")
    if len(observed) != 1:
        raise ReferenceFormatError(
            f"{context}: active {name} directive occurs {len(observed)} times"
        )
    if observed[0].casefold() != expected.casefold():
        raise ReferenceFormatError(
            f"{context}: conflicting {name} directive {observed[0]!r}; "
            f"expected {expected!r}"
        )


def _validate_clt_protocol(text: str) -> None:
    expected = {
        "basis": "Basis aVTZ",
        "scfcode": "SCFcode psi4",
        "method": "Method DFT",
        "functional": "Functional PBE0",
        "kernel": "Kernel ALDA+CHF",
        "options": "Options Tests",
        "localization": "Localization",
    }
    observed = {name: [] for name in expected}
    section = None
    properties_sections = 0
    for line_number, line in _active_lines(text):
        normalized = " ".join(line.split())
        lower = normalized.casefold()
        if lower == "end":
            section = None
            continue
        if lower.startswith("run-type "):
            section = lower
            if lower == "run-type properties":
                properties_sections += 1
            continue
        if section is None and (lower == "global" or lower.startswith("molecule ")):
            section = lower
            continue
        if lower == "finish":
            section = None
            continue
        fields = normalized.split()
        key = fields[0].casefold()
        if len(fields) >= 2 and key == "no" and fields[1].casefold() == "localization":
            raise ReferenceFormatError(
                f"H2O.clt:{line_number}: conflicting Localization directive {line!r}"
            )
        if key not in observed:
            continue
        if section != "run-type properties":
            raise ReferenceFormatError(
                f"H2O.clt:{line_number}: {fields[0]} directive is outside "
                "Run-type properties"
            )
        observed[key].append(normalized)
    if properties_sections != 1:
        raise ReferenceFormatError(
            "H2O.clt: expected exactly one active Run-type properties section"
        )
    for name, directive in expected.items():
        _require_exact_directive(
            observed[name], directive, "H2O.clt", directive.split()[0]
        )


def _parse_cks_blocks(text: str) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    block_name = None
    block_lines: list[str] = []
    for _line_number, line in _active_lines(text):
        normalized = " ".join(line.split())
        lower = normalized.casefold()
        if lower.startswith(("set ", "begin ")):
            if block_name is not None:
                raise ReferenceFormatError(
                    f"H2O.cks: nested section {normalized!r} in {block_name!r}"
                )
            block_name = lower
            block_lines = []
        elif lower == "end":
            if block_name is not None:
                blocks.append((block_name, block_lines))
                block_name = None
                block_lines = []
        elif block_name is not None:
            block_lines.append(normalized)
    if block_name is not None:
        raise ReferenceFormatError(f"H2O.cks: unterminated section {block_name!r}")
    return blocks


def _validate_cks_protocol(text: str) -> None:
    blocks = _parse_cks_blocks(text)
    quad_blocks = [lines for name, lines in blocks if name == "set quad"]
    if len(quad_blocks) != 1:
        raise ReferenceFormatError(
            "H2O.cks: expected exactly one active SET QUAD section"
        )
    quad_lines = quad_blocks[0]
    for key, expected in (("type", "Type Gauss-Legendre"), ("beta", "Beta 0.5")):
        observed = [
            line for line in quad_lines if line.split()[0].casefold() == key
        ]
        _require_exact_directive(
            observed, expected, "H2O.cks SET QUAD", expected.split()[0]
        )

    polar_blocks = [
        lines for name, lines in blocks if name == "begin polarizability"
    ]
    orient_blocks = [
        lines
        for lines in polar_blocks
        if any(line.casefold().startswith("print pols for") for line in lines)
    ]
    if len(orient_blocks) != 1:
        raise ReferenceFormatError(
            "H2O.cks: expected exactly one Polarizability section with "
            "an active Print pols for directive"
        )
    orient_lines = orient_blocks[0]
    for key, expected in (("quad", "Quad 10"), ("rank", "Rank 4")):
        observed = [line for line in orient_lines if line.split()[0].casefold() == key]
        _require_exact_directive(
            observed, expected, "H2O.cks Polarizability", expected.split()[0]
        )
    prints = [
        line for line in orient_lines if line.casefold().startswith("print pols for")
    ]
    _require_exact_directive(
        prints,
        "Print pols for Orient",
        "H2O.cks Polarizability",
        "Print pols for Orient",
    )


def _report_value(line: str, label_pattern: str) -> str | None:
    match = re.match(
        rf"{label_pattern}\s*(?:=|:)?\s*(\S+)\s*$", line, flags=re.IGNORECASE
    )
    return None if match is None else match.group(1)


def _require_report_value(
    lines: Sequence[str], label_pattern: str, expected: str, name: str
) -> None:
    candidates = [
        value
        for line in lines
        if (value := _report_value(line, label_pattern)) is not None
    ]
    if len(candidates) != 1:
        raise ReferenceFormatError(
            f"cluster output: expected exactly one active {name} report, "
            f"found {len(candidates)}"
        )
    if candidates[0].casefold() != expected.casefold():
        raise ReferenceFormatError(
            f"cluster output: conflicting {name} value {candidates[0]!r}; "
            f"expected {expected!r}"
        )


def _validate_cluster_report(text: str) -> None:
    lines = [line for _number, line in _active_lines(text)]
    _require_report_value(lines, r"AC\s+options\s*:\s*type", "GRAC", "AC options")
    _require_report_value(lines, r"functional", "PBE0", "functional")
    _require_report_value(lines, r"kernel", "ALDA+CHF", "kernel")


def _validate_psi4_orientation(text: str) -> None:
    lines = [" ".join(line.split()) for _number, line in _active_lines(text)]
    symmetry = [line for line in lines if line.split()[0].casefold() == "symmetry"]
    _require_exact_directive(
        symmetry, "symmetry c1", "generated Psi4 input", "symmetry"
    )
    for canonical, contradictory in (("no_com", "com"), ("no_reorient", "reorient")):
        observed = [line for line in lines if line.split()[0].casefold() == canonical]
        if any(line.split()[0].casefold() == contradictory for line in lines):
            raise ReferenceFormatError(
                "generated Psi4 input: contradictory "
                f"{contradictory} and {canonical} controls"
            )
        _require_exact_directive(observed, canonical, "generated Psi4 input", canonical)


def validate_generated_protocol(
    clt_text: str,
    cks_text: str,
    cluster_log_text: str,
    psi4_input_text: str,
) -> None:
    _validate_clt_protocol(clt_text)
    _validate_cks_protocol(cks_text)
    _validate_cluster_report(cluster_log_text)
    _validate_psi4_orientation(psi4_input_text)


def _require_terminal_finished(path: Path) -> None:
    lines = [
        line.strip()
        for line in path.read_text(errors="replace").splitlines()
        if line.strip()
    ]
    completions = [line for line in lines if line.casefold() == "finished"]
    if len(completions) != 1 or not lines or lines[-1].casefold() != "finished":
        raise ReferenceFormatError(
            f"{path}: requires one unambiguous terminal Finished completion"
        )


def _normalized_individual_pol(path: Path, expected_index: int) -> tuple[str, ...]:
    normalized = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        index_match = INDEX_RE.match(line)
        if index_match:
            if normalized or int(index_match.group(1)) != expected_index:
                raise ReferenceFormatError(
                    f"{path}: unexpected frequency index {index_match.group(1)}"
                )
            continue
        if stripped.startswith(("#", "!")):
            continue
        normalized.append(" ".join(stripped.split()))
    return tuple(normalized)


def _combined_pol_sections(path: Path) -> dict[int, tuple[str, ...]]:
    sections: dict[int, list[str]] = {}
    current = None
    for line in path.read_text(errors="replace").splitlines():
        match = INDEX_RE.match(line)
        if match:
            current = int(match.group(1))
            sections[current] = []
            continue
        stripped = line.strip()
        if current is None or not stripped or stripped.startswith(("#", "!")):
            continue
        sections[current].append(" ".join(stripped.split()))
    return {index: tuple(content) for index, content in sections.items()}


@dataclass(frozen=True)
class _InlineTextArtifact:
    label: str
    text: str

    def read_text(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.label


CASIMIR_DISPERSION_MARKER_RE = re.compile(
    r"^\s*Dispersion\s+coefficients\b.*$", flags=re.IGNORECASE | re.MULTILINE
)


def _parse_casimir_output(
    path: Path,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    text = path.read_text(errors="replace")
    markers = list(CASIMIR_DISPERSION_MARKER_RE.finditer(text))
    if len(markers) != 1:
        raise ReferenceFormatError(
            f"{path}: expected exactly one Dispersion coefficients marker, "
            f"found {len(markers)}"
        )
    body = text[markers[0].end():]
    if not body.strip():
        raise ReferenceFormatError(
            f"{path}: Dispersion coefficients marker has no following body"
        )
    return parse_isotropic_cn(
        _InlineTextArtifact(f"{path} body", body),  # type: ignore[arg-type]
        ACCEPTED_ATOM_LABELS,
        {"O": "O", "H1": "H", "H2": "H"},
    )


STAGE_ERROR_MARKER_RE = re.compile(
    r"segmentation\s+fault|fatal(?:\s+error)?|error\s+stop|"
    r"pfit[._ -]?error|orient[._ -]?error|traceback|truncat(?:ed|ion)|"
    r"unexpected\s+(?:end|eof)|premature\s+(?:end|eof)",
    flags=re.IGNORECASE,
)


def validate_stage_artifacts(work_dir: Path, job: str) -> dict[str, Path]:
    work_dir = Path(work_dir)
    orient_outputs = [
        work_dir / f"{job}_L3_{index:03d}.out" for index in range(11)
    ]
    pfit_outputs = [
        work_dir / f"{job}_ref_wt4_L3_{index:03d}.out" for index in range(11)
    ]
    pfit_pols = [
        work_dir / f"{job}_ref_wt4_L3_{index:03d}.pol" for index in range(11)
    ]
    combined = work_dir / f"{job}_ref_wt4_L3_0f10.pol"
    casimir_output = work_dir / f"{job}_ref_wt4_L3_casimir.out"
    casimir_pot = work_dir / f"{job}_ref_wt4_L3_C12.pot"
    pdef = work_dir / f"{job}.pdef"
    required = [
        *orient_outputs,
        *pfit_outputs,
        *pfit_pols,
        combined,
        casimir_output,
        casimir_pot,
        pdef,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise ReferenceFormatError(f"missing or empty stage artifact: {path}")

    cardinalities = (
        (f"{job}_L3_[0-9][0-9][0-9].out", 11, "ORIENT"),
        (f"{job}_ref_wt4_L3_[0-9][0-9][0-9].out", 11, "PFIT output"),
        (f"{job}_ref_wt4_L3_[0-9][0-9][0-9].pol", 11, "PFIT polarizability"),
    )
    for pattern, expected, context in cardinalities:
        observed = list(work_dir.glob(pattern))
        if len(observed) != expected:
            raise ReferenceFormatError(
                f"expected exactly {expected} {context} artifacts, "
                f"found {len(observed)}"
            )

    for path in (*orient_outputs, *pfit_outputs):
        _require_terminal_finished(path)

    relevant_suffixes = {".out", ".pol", ".pot", ".pdef"}
    for path in work_dir.rglob("*"):
        if path.is_file() and path.suffix.casefold() in relevant_suffixes:
            if STAGE_ERROR_MARKER_RE.search(path.read_text(errors="replace")):
                raise ReferenceFormatError(
                    f"fatal or truncation marker in stage artifact: {path}"
                )

    parse_refined_polarizabilities(combined, ACCEPTED_ATOM_LABELS, limit=3)
    combined_sections = _combined_pol_sections(combined)
    if set(combined_sections) != set(range(11)):
        raise ReferenceFormatError(f"{combined}: incomplete indexed frequency blocks")
    for index, path in enumerate(pfit_pols):
        individual = _normalized_individual_pol(path, index)
        if not individual or individual != combined_sections[index]:
            raise ReferenceFormatError(
                f"{path}: content does not match combined # INDEX {index:03d} block"
            )

    potential_cn = parse_isotropic_cn(
        casimir_pot,
        ACCEPTED_ATOM_LABELS,
        {"O": "O", "H1": "H", "H2": "H"},
    )
    output_cn = _parse_casimir_output(casimir_output)
    if output_cn != potential_cn:
        raise ReferenceFormatError(
            f"{casimir_output}: parsed dispersion body does not match {casimir_pot}"
        )
    return {path.name: path for path in required}


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


REQUIRED_TOP_LEVEL = (
    "schema_version",
    "generated_at_utc",
    "generator",
    "repository",
    "tools",
    "scientific_protocol",
    "frequencies",
    "polarizabilities",
    "dispersion",
    "inputs",
    "sources",
)

CANONICAL_ATOMS = (
    {"label": "O", "element": "O", "xyz": [0.0, 0.0, 0.0]},
    {
        "label": "H1",
        "element": "H",
        "xyz": [-1.4536519600, 0.0, -1.1216873200],
    },
    {
        "label": "H2",
        "element": "H",
        "xyz": [1.4536519600, 0.0, -1.1216873200],
    },
)

EXPECTED_ELECTRONIC_STRUCTURE = {
    "method": "PBE0",
    "basis": "aug-cc-pVTZ",
    "camcasp_basis": "aVTZ",
    "asymptotic_correction": "Psi4 GRAC",
    "ionization_potential_ev": 12.62063,
    "homo_hartree": -0.3989,
    "kernel": "ALDA+CHF",
    "grid": "Options Tests",
}

EXPECTED_FREQUENCY_GRID = {
    "kind": "Gauss-Legendre",
    "nonzero_count": 10,
    "scale_au": 0.5,
}

CANONICAL_FREQUENCIES = (
    0.0,
    0.0066096015960872435,
    0.036174811998630957,
    0.095447363690348272,
    0.1976442118453127,
    0.3704172128053672,
    0.6749146404580301,
    1.264899172436498,
    2.619244684547324,
    6.910885950408292,
    37.82376235021415,
)
# Accommodates only the final-decimal roundoff in CamCASP's emitted FREQ2 text.
FREQUENCY_REL_TOLERANCE = 2.0e-14
FREQUENCY_ABS_TOLERANCE = 2.0e-15
CAMCASP_EXECUTABLE_NAMES = ("camcasp", "cluster", "process", "pfit", "casimir")

EXPECTED_MODEL = {
    "nonlocal_rank": 4,
    "localization_method": "LW",
    "localization_limit": 3,
    "wsm_limit": 3,
    "hydrogen_limit": 3,
    "pfit_weight": 4,
    "pfit_weight_coefficient": 0.001,
    "pfit_cutoff": 0.0001,
}

EXPECTED_DISPERSION_UNITS = {
    "C6": "hartree * bohr^6",
    "C8": "hartree * bohr^8",
    "C10": "hartree * bohr^10",
    "C12": "hartree * bohr^12",
}

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
UTC_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z"
)


def _as_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReferenceFormatError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ReferenceFormatError(f"{context} keys must be strings")
    return value


def _as_sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ReferenceFormatError(f"{context} must be an array")
    return value


def _require(mapping: Mapping[str, object], key: str, context: str) -> object:
    if key not in mapping:
        raise ReferenceFormatError(f"{context}: missing required field {key}")
    return mapping[key]


def _require_fields(
    value: object,
    required: Sequence[str],
    context: str,
    optional: Sequence[str] = (),
) -> Mapping[str, object]:
    mapping = _as_mapping(value, context)
    for key in required:
        _require(mapping, key, context)
    unexpected = set(mapping) - set(required) - set(optional)
    if unexpected:
        raise ReferenceFormatError(
            f"{context}: unexpected field {sorted(unexpected)[0]}"
        )
    return mapping


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReferenceFormatError(f"{context} must be a non-empty string")
    return value


def _require_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ReferenceFormatError(f"{context} must be a boolean")
    return value


def _require_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceFormatError(f"{context} must be a number")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ReferenceFormatError(
            f"{context} contains an out-of-range number"
        ) from exc
    if not math.isfinite(converted):
        raise ReferenceFormatError(f"{context} contains a non-finite number")
    return converted


def _validate_finite_numbers(value: object, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReferenceFormatError(f"{context} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_numbers(child, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_numbers(child, f"{context}[{index}]")


def _validate_sha256(value: object, context: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ReferenceFormatError(f"{context} is invalid")


def _validate_commit(value: object, context: str) -> None:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise ReferenceFormatError(f"{context} is invalid")


def _validate_expected_object(
    value: object,
    expected: Mapping[str, object],
    context: str,
) -> Mapping[str, object]:
    mapping = _require_fields(value, tuple(expected), context)
    for key, expected_value in expected.items():
        if mapping[key] != expected_value or type(mapping[key]) is not type(expected_value):
            raise ReferenceFormatError(
                f"{context}.{key} must be {expected_value!r}"
            )
    return mapping


def _validated_matrix(
    value: object,
    size: int,
    context: str,
) -> tuple[tuple[float, ...], ...]:
    rows = _as_sequence(value, context)
    if len(rows) != size:
        raise ReferenceFormatError(f"{context} must be {size}x{size}")
    result = []
    for row_index, row_value in enumerate(rows):
        row = _as_sequence(row_value, f"{context}[{row_index}]")
        if len(row) != size:
            raise ReferenceFormatError(f"{context} must be {size}x{size}")
        result.append(
            tuple(
                _require_number(candidate, f"{context}[{row_index}][{column}]")
                for column, candidate in enumerate(row)
            )
        )
    return tuple(result)


def _validate_matrix_symmetric(
    matrix: Sequence[Sequence[float]],
    context: str,
    tolerance: float = 1.0e-8,
) -> None:
    for row in range(len(matrix)):
        for column in range(row):
            if abs(matrix[row][column] - matrix[column][row]) > tolerance:
                raise ReferenceFormatError(f"{context} is not symmetric")


def _validate_checksum_entry(
    value: object,
    context: str,
    require_path: bool,
) -> None:
    required = ("path", "sha256") if require_path else ("sha256",)
    entry = _require_fields(value, required, context)
    if require_path:
        _require_string(entry["path"], f"{context}.path")
    _validate_sha256(entry["sha256"], f"{context}.sha256")


def _validate_executable(value: object, context: str) -> None:
    _validate_checksum_entry(value, context, require_path=True)


def _validate_tools(value: object) -> None:
    tools = _require_fields(value, ("camcasp", "orient", "psi4"), "tools")

    camcasp = _require_fields(
        tools["camcasp"],
        ("version", "commit", "executables"),
        "tools.camcasp",
    )
    _require_string(camcasp["version"], "tools.camcasp.version")
    _validate_commit(camcasp["commit"], "tools.camcasp.commit")
    executables = _require_fields(
        camcasp["executables"],
        CAMCASP_EXECUTABLE_NAMES,
        "tools.camcasp.executables",
    )
    for name in CAMCASP_EXECUTABLE_NAMES:
        _validate_executable(
            executables[name], f"tools.camcasp.executables.{name}"
        )

    orient = _require_fields(
        tools["orient"],
        ("version", "commit", "executable"),
        "tools.orient",
    )
    _require_string(orient["version"], "tools.orient.version")
    _validate_commit(orient["commit"], "tools.orient.commit")
    _validate_executable(orient["executable"], "tools.orient.executable")

    psi4 = _require_fields(
        tools["psi4"],
        ("version", "commit", "dirty", "executable"),
        "tools.psi4",
    )
    _require_string(psi4["version"], "tools.psi4.version")
    _validate_commit(psi4["commit"], "tools.psi4.commit")
    _require_bool(psi4["dirty"], "tools.psi4.dirty")
    _validate_executable(psi4["executable"], "tools.psi4.executable")


def _validate_protocol(value: object) -> None:
    protocol = _require_fields(
        value,
        ("geometry", "electronic_structure", "frequency_grid", "model"),
        "scientific_protocol",
    )
    geometry = _require_fields(
        protocol["geometry"],
        ("units", "charge", "multiplicity", "atom_order", "atoms", "orientation"),
        "scientific_protocol.geometry",
    )
    if geometry["units"] != "bohr":
        raise ReferenceFormatError("scientific_protocol.geometry.units must be 'bohr'")
    if type(geometry["charge"]) is not int or geometry["charge"] != 0:
        raise ReferenceFormatError("scientific_protocol.geometry.charge must be 0")
    if type(geometry["multiplicity"]) is not int or geometry["multiplicity"] != 1:
        raise ReferenceFormatError("scientific_protocol.geometry.multiplicity must be 1")
    if geometry["atom_order"] != ["O", "H1", "H2"]:
        raise ReferenceFormatError(
            "scientific_protocol.geometry.atom_order must be O,H1,H2"
        )
    atoms = _as_sequence(
        geometry["atoms"], "scientific_protocol.geometry.atoms"
    )
    if len(atoms) != len(CANONICAL_ATOMS):
        raise ReferenceFormatError(
            "scientific_protocol.geometry.atoms must be the canonical O,H1,H2 geometry"
        )
    for index, expected_atom in enumerate(CANONICAL_ATOMS):
        atom_context = f"scientific_protocol.geometry.atoms[{index}]"
        atom = _require_fields(atoms[index], ("label", "element", "xyz"), atom_context)
        if atom["label"] != expected_atom["label"] or atom["element"] != expected_atom["element"]:
            raise ReferenceFormatError(
                "scientific_protocol.geometry.atoms must be the canonical O,H1,H2 geometry"
            )
        coordinates = _as_sequence(atom["xyz"], f"{atom_context}.xyz")
        if len(coordinates) != 3:
            raise ReferenceFormatError(f"{atom_context}.xyz must contain three coordinates")
        observed_xyz = [
            _require_number(coordinate, f"{atom_context}.xyz[{axis}]")
            for axis, coordinate in enumerate(coordinates)
        ]
        if observed_xyz != expected_atom["xyz"]:
            raise ReferenceFormatError(
                "scientific_protocol.geometry.atoms must be the canonical O,H1,H2 geometry"
            )
    if geometry["orientation"] != ["symmetry c1", "no_com", "no_reorient"]:
        raise ReferenceFormatError(
            "scientific_protocol.geometry.orientation does not match the approved protocol"
        )

    _validate_expected_object(
        protocol["electronic_structure"],
        EXPECTED_ELECTRONIC_STRUCTURE,
        "scientific_protocol.electronic_structure",
    )
    _validate_expected_object(
        protocol["frequency_grid"],
        EXPECTED_FREQUENCY_GRID,
        "scientific_protocol.frequency_grid",
    )
    _validate_expected_object(
        protocol["model"], EXPECTED_MODEL, "scientific_protocol.model"
    )


def _validate_frequencies(value: object) -> list[float]:
    frequencies = _require_fields(
        value,
        ("units", "values", "squared_source_values"),
        "frequencies",
    )
    if frequencies["units"] != "hartree":
        raise ReferenceFormatError("frequencies.units must be 'hartree'")
    values_raw = _as_sequence(frequencies["values"], "frequencies.values")
    squared_raw = _as_sequence(
        frequencies["squared_source_values"],
        "frequencies.squared_source_values",
    )
    if len(values_raw) != 11 or len(squared_raw) != 11:
        raise ReferenceFormatError("frequencies must contain eleven hartree values")
    values = [
        _require_number(candidate, f"frequencies.values[{index}]")
        for index, candidate in enumerate(values_raw)
    ]
    if values[0] != 0.0 or any(
        values[index] >= values[index + 1] for index in range(10)
    ):
        raise ReferenceFormatError(
            "frequencies must be static zero plus ten increasing values"
        )
    if any(
        not math.isclose(
            value,
            expected,
            rel_tol=FREQUENCY_REL_TOLERANCE,
            abs_tol=FREQUENCY_ABS_TOLERANCE,
        )
        for value, expected in zip(values, CANONICAL_FREQUENCIES)
    ):
        raise ReferenceFormatError(
            "frequencies do not match the canonical Gauss-Legendre Beta=0.5 grid"
        )
    for index, source_value in enumerate(squared_raw):
        source_text = _require_string(
            source_value, f"frequencies.squared_source_values[{index}]"
        )
        squared = _float(
            source_text, f"frequencies.squared_source_values[{index}]"
        )
        if index == 0:
            if squared != 0.0:
                raise ReferenceFormatError("static squared source frequency must be zero")
        elif squared >= 0.0:
            raise ReferenceFormatError("dynamic squared source frequencies must be negative")
        if not math.isclose(
            squared,
            -(values[index] ** 2),
            rel_tol=FREQUENCY_REL_TOLERANCE,
            abs_tol=FREQUENCY_ABS_TOLERANCE,
        ):
            raise ReferenceFormatError(
                f"frequencies.squared_source_values[{index}] does not match omega"
            )
    return values


def _validate_polarizabilities(value: object, frequencies: Sequence[float]) -> None:
    polar = _require_fields(
        value,
        ("units", "spherical_frame", "cartesian_frame", "frequency_blocks"),
        "polarizabilities",
    )
    expected_metadata = {
        "units": "atomic units",
        "spherical_frame": "atom-local real spherical",
        "cartesian_frame": "global Cartesian",
    }
    for key, expected in expected_metadata.items():
        if polar[key] != expected:
            raise ReferenceFormatError(f"polarizabilities.{key} must be {expected!r}")

    blocks = _as_sequence(
        polar["frequency_blocks"], "polarizabilities.frequency_blocks"
    )
    if len(blocks) != 11:
        raise ReferenceFormatError(
            "polarizabilities requires eleven frequency_blocks"
        )
    for expected_index, block_value in enumerate(blocks):
        block_context = f"polarizabilities.frequency_blocks[{expected_index}]"
        block = _require_fields(
            block_value, ("index", "omega", "atoms"), block_context
        )
        if type(block["index"]) is not int or block["index"] != expected_index:
            raise ReferenceFormatError(
                "polarizability blocks are not frequency-major"
            )
        omega = _require_number(block["omega"], f"{block_context}.omega")
        if not math.isclose(omega, frequencies[expected_index], abs_tol=1.0e-14):
            raise ReferenceFormatError(
                f"{block_context}.omega does not match frequencies.values"
            )
        atoms = _as_mapping(block["atoms"], f"{block_context}.atoms")
        if tuple(atoms) != ACCEPTED_ATOM_LABELS:
            raise ReferenceFormatError(
                "polarizability atom order must be O,H1,H2"
            )
        for label in ACCEPTED_ATOM_LABELS:
            atom_context = f"{block_context}.atoms.{label}"
            atom = _require_fields(
                atoms[label],
                (
                    "spherical",
                    "local_cartesian",
                    "local_to_global",
                    "global_cartesian",
                ),
                atom_context,
            )
            spherical = _require_fields(
                atom["spherical"],
                ("components", "matrix"),
                f"{atom_context}.spherical",
            )
            if spherical["components"] != list(COMPONENTS_L3):
                raise ReferenceFormatError(
                    f"{label}: incomplete L3 component ordering"
                )
            _validated_matrix(
                spherical["matrix"], 16, f"{atom_context}.spherical.matrix"
            )
            local = _validated_matrix(
                atom["local_cartesian"], 3, f"{atom_context}.local_cartesian"
            )
            rotation = _validated_matrix(
                atom["local_to_global"], 3, f"{atom_context}.local_to_global"
            )
            global_tensor = _validated_matrix(
                atom["global_cartesian"], 3, f"{atom_context}.global_cartesian"
            )
            validate_rotation_matrix(rotation)  # type: ignore[arg-type]
            _validate_matrix_symmetric(local, f"{atom_context}.local_cartesian")
            _validate_matrix_symmetric(
                global_tensor, f"{atom_context}.global_cartesian"
            )
            expected_global = _matmul(
                _matmul(rotation, local), _transpose(rotation)  # type: ignore[arg-type]
            )
            for row in range(3):
                for column in range(3):
                    if abs(expected_global[row][column] - global_tensor[row][column]) > 1.0e-8:
                        raise ReferenceFormatError(
                            f"{atom_context}.global_cartesian is inconsistent with its local frame"
                        )


def _validate_dispersion(value: object) -> None:
    dispersion = _require_fields(
        value,
        ("component", "atom_order", "units", "matrices"),
        "dispersion",
    )
    if dispersion["component"] != "00 00 0":
        raise ReferenceFormatError("dispersion component must be 00 00 0")
    if dispersion["atom_order"] != ["O", "H1", "H2"]:
        raise ReferenceFormatError("dispersion atom order must be O,H1,H2")
    _validate_expected_object(
        dispersion["units"], EXPECTED_DISPERSION_UNITS, "dispersion.units"
    )
    matrices = _require_fields(
        dispersion["matrices"], CN_ORDERS, "dispersion.matrices"
    )
    for order in CN_ORDERS:
        matrix = _validated_matrix(matrices[order], 3, f"dispersion.matrices.{order}")
        _validate_matrix_symmetric(matrix, f"{order} matrix")


def validate_reference_document(document: Mapping[str, object]) -> None:
    root = _require_fields(document, REQUIRED_TOP_LEVEL, "document")
    _validate_finite_numbers(root, "document")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ReferenceFormatError("schema_version must be 1")

    generated_at = root["generated_at_utc"]
    if not isinstance(generated_at, str) or not UTC_TIMESTAMP_RE.fullmatch(generated_at):
        raise ReferenceFormatError("generated_at_utc must be an RFC 3339 UTC timestamp")
    try:
        datetime.fromisoformat(generated_at[:-1] + "+00:00")
    except ValueError as exc:
        raise ReferenceFormatError(
            "generated_at_utc must be an RFC 3339 UTC timestamp"
        ) from exc

    generator = _require_fields(
        root["generator"], ("path", "sha256"), "generator"
    )
    _require_string(generator["path"], "generator.path")
    _validate_sha256(generator["sha256"], "generator.sha256")

    repository = _require_fields(
        root["repository"], ("commit", "dirty"), "repository"
    )
    _validate_commit(repository["commit"], "repository.commit")
    _require_bool(repository["dirty"], "repository.dirty")

    _validate_tools(root["tools"])
    _validate_protocol(root["scientific_protocol"])
    frequencies = _validate_frequencies(root["frequencies"])
    _validate_polarizabilities(root["polarizabilities"], frequencies)
    _validate_dispersion(root["dispersion"])

    inputs = _as_mapping(root["inputs"], "inputs")
    if not inputs:
        raise ReferenceFormatError("inputs must not be empty")
    for name, entry in inputs.items():
        _validate_checksum_entry(entry, f"inputs.{name}", require_path=False)

    sources = _as_mapping(root["sources"], "sources")
    if not sources:
        raise ReferenceFormatError("sources must not be empty")
    for name, entry in sources.items():
        _validate_checksum_entry(entry, f"sources.{name}", require_path=True)


def write_atomic_json(path: Path, document: Mapping[str, object]) -> None:
    validate_reference_document(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                document,
                handle,
                indent=2,
                sort_keys=False,
                allow_nan=False,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reference_document(
    *,
    frequency_path: Path,
    refined_path: Path,
    casimir_path: Path,
    axes_path: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    document = json.loads(json.dumps(metadata, allow_nan=False))
    frequencies = parse_frequencies(Path(frequency_path))
    models = parse_refined_polarizabilities(
        Path(refined_path), ACCEPTED_ATOM_LABELS, limit=3
    )
    geometry = {
        atom["label"]: tuple(atom["xyz"])
        for atom in CANONICAL_ATOMS
    }
    frames = build_local_frames(geometry, Path(axes_path).read_text())
    cn = parse_isotropic_cn(
        Path(casimir_path),
        ACCEPTED_ATOM_LABELS,
        {"O": "O", "H1": "H", "H2": "H"},
    )

    frequency_blocks = []
    for point, block in zip(frequencies, models):
        atoms = {}
        for label in ACCEPTED_ATOM_LABELS:
            spherical = block.atoms[label]
            local = dipole_local_cartesian(spherical)
            global_tensor = rotate_tensor(local, frames[label])
            atoms[label] = {
                "spherical": {
                    "components": list(spherical.components),
                    "matrix": [list(row) for row in spherical.matrix],
                },
                "local_cartesian": [list(row) for row in local],
                "local_to_global": [list(row) for row in frames[label]],
                "global_cartesian": [list(row) for row in global_tensor],
            }
        frequency_blocks.append(
            {"index": point.index, "omega": point.omega, "atoms": atoms}
        )

    document["frequencies"] = {
        "units": "hartree",
        "values": [point.omega for point in frequencies],
        "squared_source_values": [
            point.squared_source_text for point in frequencies
        ],
    }
    polar = _as_mapping(document.get("polarizabilities"), "polarizabilities")
    document["polarizabilities"] = {
        **polar,
        "frequency_blocks": frequency_blocks,
    }
    dispersion = _as_mapping(document.get("dispersion"), "dispersion")
    document["dispersion"] = {
        **dispersion,
        "matrices": {
            order: [list(row) for row in cn[order]] for order in CN_ORDERS
        },
    }
    sources = dict(_as_mapping(document.get("sources"), "sources"))
    sources.update(
        {
            "nonlocal_pol": {
                "path": str(Path(frequency_path).resolve()),
                "sha256": _sha256(Path(frequency_path)),
            },
            "refined_pol": {
                "path": str(Path(refined_path).resolve()),
                "sha256": _sha256(Path(refined_path)),
            },
            "casimir_pot": {
                "path": str(Path(casimir_path).resolve()),
                "sha256": _sha256(Path(casimir_path)),
            },
            "axes": {
                "path": str(Path(axes_path).resolve()),
                "sha256": _sha256(Path(axes_path)),
            },
        }
    )
    document["sources"] = sources
    validate_reference_document(document)
    return document


def _python_array(name: str, values: object) -> str:
    rendered = json.dumps(values, indent=4, allow_nan=False, ensure_ascii=False)
    return f"{name} = np.array({rendered}, dtype=float)\n"


def render_python_literals(document: Mapping[str, object]) -> str:
    validate_reference_document(document)
    polar = _as_mapping(document["polarizabilities"], "polarizabilities")
    blocks = _as_sequence(
        polar["frequency_blocks"], "polarizabilities.frequency_blocks"
    )
    packed = []
    for block_value in blocks:
        block = _as_mapping(block_value, "polarizability block")
        atoms = _as_mapping(block["atoms"], "polarizability atoms")
        for label in ACCEPTED_ATOM_LABELS:
            atom = _as_mapping(atoms[label], f"polarizability atom {label}")
            tensor = _as_sequence(atom["global_cartesian"], "global_cartesian")
            packed.append(
                [
                    tensor[0][0],
                    tensor[0][1],
                    tensor[0][2],
                    tensor[1][1],
                    tensor[1][2],
                    tensor[2][2],
                ]
            )
    frequencies = _as_mapping(document["frequencies"], "frequencies")
    dispersion = _as_mapping(document["dispersion"], "dispersion")
    matrices = _as_mapping(dispersion["matrices"], "dispersion.matrices")
    output = [
        "# Generated by: bash devtools/regenerate-camcasp.sh\n",
        _python_array(
            "REFERENCE_FREQUENCIES",
            [[value] for value in frequencies["values"]],
        ),
        _python_array("REFERENCE_STATIC_ATOMIC_POLARIZABILITIES", packed[:3]),
        _python_array("REFERENCE_DYNAMIC_ATOMIC_POLARIZABILITIES", packed),
    ]
    for order in CN_ORDERS:
        output.append(
            _python_array(f"REFERENCE_ATOMIC_{order}", matrices[order])
        )
    return "\n".join(output)


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceFormatError(f"could not load JSON {path}: {exc}") from exc
    return _as_mapping(value, str(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and build the canonical CamCASP reference document."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("file", type=Path)

    artifacts = subparsers.add_parser("validate-artifacts")
    artifacts.add_argument("--work-dir", type=Path, required=True)
    artifacts.add_argument("--job", required=True)

    attest = subparsers.add_parser("attest-protocol")
    attest.add_argument("--clt", type=Path, required=True)
    attest.add_argument("--cks", type=Path, required=True)
    attest.add_argument("--cluster-log", type=Path, required=True)
    attest.add_argument("--psi4-input", type=Path, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--frequency-file", type=Path, required=True)
    build.add_argument("--refined-file", type=Path, required=True)
    build.add_argument("--casimir-file", type=Path, required=True)
    build.add_argument("--axes-file", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "validate":
            validate_reference_document(_load_json(arguments.file))
        elif arguments.command == "validate-artifacts":
            validate_stage_artifacts(arguments.work_dir, arguments.job)
        elif arguments.command == "attest-protocol":
            validate_generated_protocol(
                arguments.clt.read_text(),
                arguments.cks.read_text(),
                arguments.cluster_log.read_text(errors="replace"),
                arguments.psi4_input.read_text(),
            )
        elif arguments.command == "build":
            document = build_reference_document(
                frequency_path=arguments.frequency_file,
                refined_path=arguments.refined_file,
                casimir_path=arguments.casimir_file,
                axes_path=arguments.axes_file,
                metadata=_load_json(arguments.manifest),
            )
            write_atomic_json(arguments.output, document)
            print(render_python_literals(document), end="")
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(arguments.command)
    except (OSError, ReferenceFormatError) as exc:
        print(f"camcasp-reference: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
