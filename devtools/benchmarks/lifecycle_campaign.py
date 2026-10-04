#!/usr/bin/env python3
"""Pinned, serial old/new CPU/GPU comparison on a single H200 allocation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from saptdft_cuest_grac import atomic_json
from build_receipt import verify

CASES = [
    ("water", "cc-pvdz"), ("water", "aug-cc-pvdz"),
    ("benzene", "cc-pvdz"), ("benzene", "aug-cc-pvdz"),
    ("peptide", "6-31+g**"), ("nanotube", "6-31+g**"),
    ("protein83", "6-31+g**"),
]
ARMS = [
    ("old-cpu", "old", "cpu", None),
    ("old-gpu", "old", "gpu", None),
    ("new-cpu", "new", "cpu", "cpu"),
    ("new-gpu-cpu-sad", "new", "gpu", "cpu"),
    ("new-gpu-gpu-sad", "new", "gpu", "gpu"),
]

def compare_arms(paired):
    """Gate regressions within a backend; report baseline CPU/GPU disagreement."""
    comparisons = []
    for arm, reference_arm, kind in [
        ("old-gpu", "old-cpu", "baseline-cross-backend"),
        ("new-cpu", "old-cpu", "regression"),
        ("new-gpu-cpu-sad", "old-gpu", "regression"),
        ("new-gpu-gpu-sad", "old-gpu", "regression"),
    ]:
        result, reference = paired[arm], paired[reference_arm]
        errors = {key: abs(result["components_hartree"][key] - value)
                  for key, value in reference["components_hartree"].items()}
        shifts = {key: abs(result["grac_shifts_hartree"][key] - value)
                  for key, value in reference["grac_shifts_hartree"].items()}
        if not all(math.isfinite(v) for v in [*errors.values(), *shifts.values()]):
            raise ValueError(f"Nonfinite accuracy comparison: {arm} vs {reference_arm}")
        comparisons.append({
            "arm": arm, "reference_arm": reference_arm, "kind": kind,
            "component_errors_hartree": errors, "shift_errors_hartree": shifts,
            "within_tolerance": max(errors.values()) <= 1.e-6 and max(shifts.values()) <= 1.e-6,
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-package", type=Path, required=True)
    parser.add_argument("--new-package", type=Path, required=True)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, required=True)
    parser.add_argument("--old-commit", required=True)
    parser.add_argument("--new-commit", required=True)
    parser.add_argument("--old-receipt", type=Path, required=True)
    parser.add_argument("--new-receipt", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory", default="112 GiB")
    parser.add_argument("--protein157", action="store_true",
                        help="Explicitly run only the separately resourced Protein157 case")
    parser.add_argument("--case-timeout", type=int, default=2400)
    parser.add_argument("--require-in-core", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
    if args.case_timeout < 1:
        parser.error("case timeout must be positive")
    cases = [("protein157", "6-31+g**")] if args.protein157 else CASES
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("saptdft_cuest_grac.py").resolve()
    versions = {}
    for version in ("old", "new"):
        source = getattr(args, f"{version}_source").resolve()
        expected = getattr(args, f"{version}_commit")
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        assert actual == expected, (version, actual, expected)
        assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--",
                                            "psi4", "tests"], text=True).strip(), source
        package = getattr(args, f"{version}_package").resolve()
        receipt = verify(json.loads(getattr(args, f"{version}_receipt").read_text()))
        assert receipt["package"] == str(package) and receipt["source"] == str(source)
        core, = package.glob("core*.so")
        versions[version] = {"source": str(source), "commit": actual,
                             "package": str(package),
                             "build_receipt": receipt,
                             "binary_sha256": hashlib.sha256(core.read_bytes()).hexdigest()}
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                    "--format=csv,noheader"], text=True).strip()
    assert "H200" in gpu, gpu
    manifest = {
        "versions": versions, "gpu": gpu,
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "job_id": os.environ.get("SLURM_JOB_ID"), "cases": cases, "arms": ARMS,
        "harness_commit": subprocess.check_output(
            ["git", "-C", str(script.parent), "rev-parse", "HEAD"], text=True).strip(),
        "case_timeout_s": args.case_timeout, "require_in_core": args.require_in_core,
        "repeats": args.repeats, "threads": args.threads, "memory": args.memory,
        "timing": "fresh-process energy() wall; inclusive backend/SCF setup, excludes import and pre-count bases",
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "geometry_sha256": hashlib.sha256(script.with_name("saptdft_suite_geometries.json").read_bytes()).hexdigest(),
        "accuracy_policy": "1e-6 Eh component and GRAC-shift regression limits within each backend; "
                           "pre-existing old CPU/GPU differences are reported, not certified equivalent",
        "records": [], "accuracy": [],
    }
    atomic_json(args.output / "campaign.json", manifest)
    for system, basis in cases:
        for repeat in range(args.repeats):
            # Rotate/reverse arm order to reduce systematic warm-node bias.
            order = ARMS[repeat % len(ARMS):] + ARMS[:repeat % len(ARMS)]
            if repeat % 2:
                order = list(reversed(order))
            paired = {}
            for arm, version, mode, sad in order:
                name = f"{system}-{basis}-{arm}-{repeat + 1}"
                directory = args.output / name
                directory.mkdir()
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path(versions[version]["package"]).parent)
                env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(args.threads)
                env["PYTHONNOUSERSITE"] = "1"
                scratch = Path(os.environ["TMPDIR"]) / name
                scratch.mkdir()
                env["PSI_SCRATCH"] = env["SCRATCH"] = str(scratch)
                cmd = [sys.executable, str(script), "--case", "--system", system, "--basis", basis,
                       "--mode", mode, "--output", str(directory), "--threads", str(args.threads),
                       "--memory", args.memory, "--grac-compute", "ITERATIVE",
                       "--source-commit", versions[version]["commit"],
                       "--expected-package", versions[version]["package"]]
                if sad:
                    cmd += ["--sad-route", sad]
                if system == "protein157":
                    cmd += ["--allow-protein157-cpu"]
                if args.require_in_core:
                    cmd += ["--require-in-core"]
                print("START", name, flush=True)
                start = time.perf_counter()
                with (directory / "console.log").open("w") as stream:
                    try:
                        rc = subprocess.run(cmd, cwd=directory, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=args.case_timeout).returncode
                    except subprocess.TimeoutExpired:
                        rc = 124
                item = {"name": name, "arm": arm, "system": system, "basis": basis,
                        "repeat": repeat + 1, "returncode": rc, "process_wall_s": time.perf_counter() - start}
                manifest["records"].append(item)
                atomic_json(args.output / "campaign.json", manifest)
                print("END", name, rc, f"{item['process_wall_s']:.2f}s", flush=True)
                if rc:
                    return rc
                result = json.loads((directory / "result.json").read_text())
                assert result["ok"]
                paired[arm] = result
            for comparison in compare_arms(paired):
                comparison.update(system=system, basis=basis, repeat=repeat + 1)
                manifest["accuracy"].append(comparison)
                atomic_json(args.output / "campaign.json", manifest)
                if not comparison["within_tolerance"] and comparison["kind"] == "regression":
                    atomic_json(args.output / "ACCURACY_FAILED.json", comparison)
                    return 2
    expected = len(cases) * len(ARMS) * args.repeats
    assert len(manifest["records"]) == expected
    warnings = [x for x in manifest["accuracy"]
                if x["kind"] == "baseline-cross-backend" and not x["within_tolerance"]]
    atomic_json(args.output / "COMPLETE.json", {
        "ok": True, "count": expected, "same_backend_regressions_passed": True,
        "baseline_cross_backend_discrepancies": warnings})
    return 0


if __name__ == "__main__":
    sys.exit(main())
