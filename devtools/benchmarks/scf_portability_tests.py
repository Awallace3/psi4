#!/usr/bin/env python3
"""Run SCF regressions against the native staged build, not a conda Psi4."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=("x64", "arm64"))
    args = parser.parse_args()
    machine = platform.machine().lower()
    expected = {"x64": {"x86_64", "amd64"}, "arm64": {"arm64", "aarch64"}}
    assert machine in expected[args.architecture], (machine, args.architecture)
    import psi4
    import pytest
    stage = (Path.cwd() / "stage/lib/psi4").resolve()
    assert Path(psi4.__file__).resolve().is_relative_to(stage), psi4.__file__
    assert Path(psi4.core.__file__).resolve().is_relative_to(stage), psi4.core.__file__
    psi4.set_num_threads(2)
    assert psi4.core.get_num_threads() == 2, "OpenMP unavailable"
    psi4.set_num_threads(1)
    source = Path(__file__).resolve().parents[2]
    provenance = dict(
        machine=machine, expected_architecture=args.architecture,
        platform=platform.platform(), python=sys.version,
        psi4=psi4.__file__, core=psi4.core.__file__, version=psi4.__version__,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        run_id=os.environ.get("GITHUB_RUN_ID"), runner_arch=os.environ.get("RUNNER_ARCH"),
        threads=psi4.core.get_num_threads(),
    )
    Path("scf-portability-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)
    tests = [
        "test_scf_optimization.py", "test_scf_options.py", "test_dfjk.py",
        "test_jkmemory.py", "test_dft_blocking.py", "test_matrix.py",
        "test_tdscf_products.py", "test_tdscf_excitations.py",
    ]
    return pytest.main([
        *(str(source / "tests/pytests" / name) for name in tests),
        "-v", "-ra", "--durations=20", "--junitxml=scf-portability.xml",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
