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
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error("repeats and threads must be positive")
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
        "job_id": os.environ.get("SLURM_JOB_ID"), "cases": CASES, "arms": ARMS,
        "repeats": args.repeats, "threads": args.threads, "memory": args.memory,
        "timing": "fresh-process energy() wall; inclusive backend/SCF setup, excludes import and pre-count bases",
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "geometry_sha256": hashlib.sha256(script.with_name("saptdft_suite_geometries.json").read_bytes()).hexdigest(),
        "records": [],
    }
    atomic_json(args.output / "campaign.json", manifest)
    for system, basis in CASES:
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
                print("START", name, flush=True)
                start = time.perf_counter()
                with (directory / "console.log").open("w") as stream:
                    try:
                        rc = subprocess.run(cmd, cwd=directory, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=2400).returncode
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
            reference = paired["old-cpu"]
            for arm, result in paired.items():
                errors = {key: abs(result["components_hartree"][key] - value)
                          for key, value in reference["components_hartree"].items()}
                shift_errors = [abs(result["grac_shifts_hartree"][key] - value)
                                for key, value in reference["grac_shifts_hartree"].items()]
                shift_error = max(shift_errors)
                # Explicit scientific acceptance gate, not np.allclose defaults.
                # Archive exact errors; stop instead of timing repeated wrong answers.
                if (not all(math.isfinite(v) for v in [*errors.values(), *shift_errors])
                        or max(errors.values()) > 1.e-6 or shift_error > 1.e-6):
                    atomic_json(args.output / "ACCURACY_FAILED.json",
                                {"system": system, "basis": basis, "repeat": repeat + 1,
                                 "arm": arm, "component_errors_hartree": errors,
                                 "max_shift_error_hartree": shift_error})
                    return 2
    expected = len(CASES) * len(ARMS) * args.repeats
    assert len(manifest["records"]) == expected
    atomic_json(args.output / "COMPLETE.json", {"ok": True, "count": expected})
    return 0


if __name__ == "__main__":
    sys.exit(main())
