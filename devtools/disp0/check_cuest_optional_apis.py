#!/usr/bin/env python3
"""Compile the actual CMake optional-API probes against two SDK header trees.

No linking, native loading or GPU execution. Both SDK directories must be
provided explicitly; this is a migration check, not a skipped host unit test.
"""
import argparse
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-include", type=Path, required=True)
    parser.add_argument("--new-include", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / "psi4/src/psi4/CMakeLists.txt"
    text = source.read_text()
    probes = re.findall(
        r'check_cxx_source_compiles\("((?:\\.|[^"\\])*)"\s+'
        r'(PSI4_CUEST_HAS_PREVIEW_(?:GRIDK|DIRECTJ))\)', text, re.DOTALL
    )
    if len(probes) != 2:
        raise RuntimeError("Expected exactly the two production optional-API probes")
    for include, expected in ((args.legacy_include, True), (args.new_include, False)):
        if not (include / "cuest.h").is_file():
            raise RuntimeError(f"Missing explicit SDK headers: {include}")
        for body, name in probes:
            body = body.replace('\\"', '"')
            result = subprocess.run(
                [args.cxx, "-std=c++17", "-fsyntax-only", "-x", "c++", "-I", str(include), "-"],
                input=body, text=True, capture_output=True,
            )
            observed = result.returncode == 0
            if observed != expected:
                raise RuntimeError(f"{include}: {name}: expected {expected}\n{result.stderr}")
            print(f"PASS {include}: {name} supported={observed}")


if __name__ == "__main__":
    main()
