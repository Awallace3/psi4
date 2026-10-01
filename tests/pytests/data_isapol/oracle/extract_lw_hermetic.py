#!/usr/bin/env python3
# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Opt-in, bounded numerical extraction; no executables, imports of Psi4, or fitting.

CLI requires an explicitly authorized source directory containing precisely the two
named inputs. Only those paths are opened, never enumerated. Pytest uses the static
JSON, not this CLI. Decimal strings preserve every printed token including -0.
Hashes establish identity, not correctness of the external numerical producer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

LABELS = ("O", "H1", "H2")
SOURCE_HASHES = {
    "H2O_NL4_000.pol": "9b6130f42fc50b50b80d5860f13cc6b5b198002c4c74a1b3500893481502c166",
    "H2O_L3_000.pol": "9c027937ffe3eb8c80c67015dc927983ca70464bd7e265e0f417eb59a5114172",
}
NUMBER = re.compile(r"[+-]?\d+\.\d+(?:E[+-]\d+)?")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_pol(text, distributed):
    """Accept only the reviewed static dense dialects; consume the entire file.

    NL4 declares spherical S and explicit 1-based indices. L3 has neither an
    index-pair nor a CARTSPHER field: SITE-NAMES fixes diagonal identity, and
    real Racah spherical representation is an explicit provenance declaration,
    not an imaginary header field. Its sections have no individual END marker.
    """
    lines = text.splitlines()
    pairs = [(a, b) for a in range(3) for b in range(3)] if distributed else [(a, a) for a in range(3)]
    n = 25 if distributed else 15
    cursor = 0
    sections = []

    def take():
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError(f"premature EOF at line {cursor + 1}")
        value = lines[cursor]
        cursor += 1
        return value

    for a, b in pairs:
        line_number = cursor + 1
        header = take()
        if distributed:
            expected = (f"ALPHA INDEX 001 SITE-LABELS {LABELS[a]} {LABELS[b]} "
                        f"SITE-INDICES {a + 1} {b + 1} RANK 0 : 4 BY 0 : 4 "
                        "FREQ2 0.0000000E+00 CARTSPHER S")
        else:
            expected = (f"ALPHA H2O SITE-NAMES {LABELS[a]} {LABELS[b]} "
                        "RANK 1 TO 3 INDEX 1 FREQSQ 0.0000000")
        if header.split() != expected.split():
            raise ValueError(f"invalid header/identity at line {line_number}: {header!r}")
        rows = []
        for _ in range(n):
            tokens = take().split()
            if len(tokens) != n or any(NUMBER.fullmatch(x) is None for x in tokens):
                raise ValueError(f"invalid dense numerical row at line {cursor}")
            # Preserve all tokens: no float round trip, clipping or symmetrization.
            rows.append(tokens)
        if distributed and take() != "END":
            raise ValueError(f"missing END at line {cursor}")
        sections.append({"labels": [LABELS[a], LABELS[b]], "site_indices": [a + 1, b + 1],
                         "header_line": line_number, "header": header, "values": rows})
    if take() != "ENDFILE" or cursor != len(lines):
        raise ValueError("missing ENDFILE or trailing content")
    return sections


def extract(source_directory, destination):
    records = {}
    for name, distributed in [("H2O_NL4_000.pol", True), ("H2O_L3_000.pol", False)]:
        raw = (source_directory / name).read_bytes()
        if sha256(raw) != SOURCE_HASHES[name]:
            raise ValueError(f"unreviewed source identity: {name}")
        records["distributed" if distributed else "expected_local"] = {
            "filename": name, "sha256": sha256(raw),
            "rank_min": 0 if distributed else 1, "rank_max": 4 if distributed else 3,
            "representation": "real_Racah_spherical", "raw_frequency_index": 1,
            "frequency_squared": "0.0000000E+00" if distributed else "0.0000000",
            "sections": parse_pol(raw.decode("ascii"), distributed),
        }
    authority_names = ["manifest.json", "H2O.sites", "H2O.axes"]
    authority = {name: sha256((destination / name).read_bytes()) for name in authority_names}
    manifest = json.loads((destination / "manifest.json").read_text())
    result = {
        "schema_version": 1, "molecule": "H2O", "frequency": 0.0, "units": "atomic",
        "geometry_units": "bohr", "frame_convention": "local_to_global_columns",
        "input_frame": "global", "expected_frame": "site_local",
        "component_order": "00,10,11c,11s,20,21c,21s,22c,22s,30,31c,31s,32c,32s,33c,33s,40,41c,41s,42c,42s,43c,43s,44c,44s",
        "bonds_zero_based": [[0, 1], [0, 2]], "sites": manifest["sites"],
        "authority_sha256": authority,
        "provenance": {
            "producer": "external ORIENT 5.0.10 (d8d8610), supplied historical H2O track",
            "source_directory": str(source_directory),
            "extractor": "tests/pytests/data_isapol/oracle/extract_lw_hermetic.py",
            "extractor_sha256": sha256(Path(__file__).read_bytes()),
            "method": "strict full-file parsing; literal decimal tokens; no tensor transformations",
            "origin_frame_authority": "existing manifest sites only, checked against H2O.sites/H2O.axes; O identity explicitly declared",
            "l3_representation_authority": "NEW-format L3 real Racah convention declared in orient_local/README.md; no CARTSPHER field in source",
            "claim": "native LW processing of supplied nonlocal tensors only; no wavefunction generation or native PFIT",
            "identity_limit": "hashes pin identity, not producer correctness; expected literals are external unrefined L3, not candidate outputs",
        },
        **records,
    }
    path = destination / "lw-hermetic-water.json"
    payload = (json.dumps(result, indent=2) + "\n").encode()
    # Exclusive creation protects existing fixtures during opt-in re-extraction.
    with path.open("xb") as stream:
        stream.write(payload)
    with path.with_suffix(".json.sha256").open("x") as stream:
        stream.write(f"{sha256(payload)}  {path.name}\n")
    print(f"{path}: {sha256(payload)}; 5625 distributed + 675 expected literals")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    extract(args.source_directory, args.destination)
