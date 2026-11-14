#!/usr/bin/env python3
"""Run SCF regressions against a verified native staged build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=("x64", "arm64"))
    parser.add_argument("--openmp", choices=("on", "off"), default="on")
    parser.add_argument("--extended", action="store_true",
                        help="Include the large methyl-oxirane TDDFT cases (slow on one thread).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent pytest processes; each starts with one OpenMP/BLAS thread.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    machine = platform.machine().lower()
    expected = {"x64": {"x86_64", "amd64"}, "arm64": {"arm64", "aarch64"}}
    if machine not in expected[args.architecture]:
        raise RuntimeError(f"Expected {args.architecture}; Python is running on {machine}")
    import psi4
    import pytest
    stage = (Path.cwd() / "stage/lib/psi4").resolve()
    for module in (psi4, psi4.core):
        if not Path(module.__file__).resolve().is_relative_to(stage):
            raise RuntimeError(f"Not testing the staged build: {module.__file__}")
    if args.openmp == "on":
        psi4.set_num_threads(2)
        if psi4.core.get_num_threads() != 2:
            raise RuntimeError("Expected an OpenMP build")
    psi4.set_num_threads(1)
    source = Path(__file__).resolve().parents[2]
    diff = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=source)
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only", "-z"], cwd=source)
    changed += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=source)
    manifest = {}
    for name in sorted(set(changed.decode().split("\0")) - {""}):
        path = source / name
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "deleted"
    provenance = dict(
        machine=machine, expected_architecture=args.architecture,
        platform=platform.platform(), python=sys.version,
        psi4=psi4.__file__, core=psi4.core.__file__, version=psi4.__version__,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
        changed_source_files=manifest,
        core_sha256=hashlib.sha256(Path(psi4.core.__file__).read_bytes()).hexdigest(),
        run_id=os.environ.get("GITHUB_RUN_ID"), runner_arch=os.environ.get("RUNNER_ARCH"),
        requested_openmp=args.openmp, threads=psi4.core.get_num_threads(),
        pytest_workers=args.workers,
    )
    Path("scf-portability-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)
    names = [
        "test_scf_portability.py", "test_scf_switching.py", "test_scf_optimization.py", "test_scf_options.py",
        "test_dfjk.py", "test_jkmemory.py", "test_dft_blocking.py", "test_matrix.py",
        "test_diis.py", "test_tdscf_products.py", "test_tdscf_excitations.py",
    ]
    tests = [source / "tests/pytests" / name for name in names]
    return pytest.main([
        *(str(path) for path in tests if path.exists()),
        *([] if args.extended else ["-k", "not METHYLOXIRANE"]),
        *([] if args.workers == 1 else ["-n", str(args.workers)]),
        "-v", "-ra", "--durations=20", "--junitxml=scf-portability.xml",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
