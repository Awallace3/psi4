"""Two additional protein controls; same frozen core, SVD opt-out then opt-in.

Separate from the accepted five-control campaign. No build, new allocation,
scientific setting changes, or confidence-interval/speedup extrapolation.
Invoke only through the exclusive shared-step helper after owner admission.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


PIN = "1d1c593db90e7deb54d74c99f95d915d021acbd1"
CORE = "79f0153405716a26aa3dcbf30d1e4c5eba0e9b8ff21c62b41486fcd336afbc4e"
RUNNER = "d599720a06b25a3d813f8b448c1ebb858a7477d4"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def source(root, pin):
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip() == pin
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()


def validate(sample, reference, enabled):
    assert sample["ok"] and sample["arm"] == "gpu"
    for key in ("system", "natom", "nbf", "naux", "threads", "memory", "options",
                "cache_sha256", "metadata_sha256", "runner_sha256",
                "numpy_version", "numpy_configuration", "job_id", "hostname",
                "occupied", "virtual", "binary_pin", "core_sha256", "libraries_sha256"):
        assert sample[key] == reference[key], key
    assert sample["gpu_pinv_requested"] is enabled
    assert tuple(sample[k] for k in ("gpu_timer_calls", "gpu_exchange_timer_calls",
                                    "gpu_metric_timer_calls", "gpu_response_timer_calls",
                                    "gpu_pinv_timer_calls")) == (20, 4, 12, 60, 22 if enabled else 0)
    assert len(sample["pinv_calls"]) == 22
    assert math.isfinite(sample["fdds_seconds"]) and sample["fdds_seconds"] > 0
    assert set(sample["energies_Eh"]) == set(reference["energies_Eh"])
    errors = {}
    for key, value in sample["energies_Eh"].items():
        baseline = reference["energies_Eh"][key]
        assert math.isfinite(value) and math.isfinite(baseline)
        errors[key] = abs(value - baseline)
        assert errors[key] <= 1.e-5
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "runner", "reference", "accepted", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    assert os.environ["SLURM_JOB_ID"] == "14250361"
    assert os.environ["SLURM_CPUS_PER_TASK"] == "8"
    source(a.stage, PIN)
    source(a.runner, RUNNER)
    reference = read(a.reference)
    assert reference["binary_pin"] == PIN and reference["core_sha256"] == CORE
    assert reference["system"] == "protein83" and reference["natom"] == 83
    assert reference["threads"] == 8 and reference["memory"] == "96 GiB"
    assert str(reference["job_id"]) == "14250361"
    accepted = read(a.accepted)
    assert accepted["candidate_pin"] == PIN
    assert len(accepted["accepted_controls"]) == 5
    assert accepted["gpu"]["unique_full_suite_tests"] == 112
    assert accepted["gpu"]["executed_tests_including_repeats"] == 176
    package = a.stage / "build_fdds_validation/stage/lib/psi4"
    assert sha(next(package.glob("core.*.so"))) == CORE
    benchmark = a.runner / "devtools/benchmarks/fdds_benchmark.py"
    assert sha(benchmark) == reference["runner_sha256"]
    envroot = Path("/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt")
    scratch = Path("/var/tmp/awallace43-fdds-14250361-cycle1")
    assert read(scratch / "STORAGE_VERIFIED.json")["job_id"] == 14250361
    env = dict(os.environ, PATH=f"{envroot}/bin:{os.environ['PATH']}",
               LD_LIBRARY_PATH=f"{envroot}/lib:{envroot}/targets/x86_64-linux/lib:"
                               + os.environ.get("LD_LIBRARY_PATH", ""),
               PYTHONPATH=str(package.parent),
               PSIDATADIR=str(a.stage / "build_fdds_validation/stage/share/psi4"),
               TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               CUEST_FDDS_READONLY_BUILD_ROOT=str(a.stage), CUEST_FDDS_BINARY_PIN=PIN,
               CUEST_FDDS_EXPECTED_CORE_SHA256=CORE)
    a.output.mkdir(mode=0o700)
    samples, errors = {}, {}
    try:
        for label, enabled in (("svd-off", False), ("svd-on", True)):
            command = ["timeout", "--signal=TERM", "--kill-after=30s", "1050",
                       str(envroot / "bin/python"), str(benchmark),
                       "--system", "protein83", "--cache",
                       "/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-optimization-01/caches/protein83",
                       "--output", str(a.output / label), "--arm", "gpu",
                       "--threads", "8", "--memory", "96 GiB", "--binary-pin", PIN,
                       "--expected-core-sha256", CORE, "--expected-package", str(package),
                       "--require-gpu-exchange", "--require-gpu-metric",
                       "--require-gpu-response-products"]
            if enabled:
                command.append("--gpu-pinv")
            with (a.output / (label + ".log")).open("x") as log:
                subprocess.run(command, env=env, cwd=a.output, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            samples[label] = read(a.output / label / "result.json")
            errors[label] = validate(samples[label], reference, enabled)
        source(a.stage, PIN)
        assert sha(next(package.glob("core.*.so"))) == CORE
        result = dict(scope="additional_same_binary_two_control_diagnostic",
                      source_pin=PIN, core_sha256=CORE, runner_pin=RUNNER,
                      controller_sha256=sha(Path(__file__)), errors_Eh=errors,
                      seconds={k: v["fdds_seconds"] for k, v in samples.items()},
                      off_over_on=samples["svd-off"]["fdds_seconds"] /
                                  samples["svd-on"]["fdds_seconds"],
                      note="One sample per setting; separate from prior five controls. "
                           "No new CPU comparison or confidence interval.")
        pending = a.output / "VERIFIED.pending.json"
        pending.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        os.rename(pending, a.output / "VERIFIED.json")
    except BaseException as error:
        (a.output / "FAILED.json").write_text(json.dumps({"error": repr(error)}) + "\n")
        raise


if __name__ == "__main__":
    main()
