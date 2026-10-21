"""Bounded, exclusive-step SVD gate and declared protein83 controls.

This is invoked only through cuest_shared_step.sh. No allocation/job mutation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from xml.etree import ElementTree as ET


ORDER = ("cpu0", "base-gpu0", "candidate-gpu0", "candidate-gpu1", "base-gpu1")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def source(root, pin):
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip() == pin
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()


def dependencies(d):
    return {p: h for p, h in d["libraries_sha256"].items()
            if not Path(p).name.startswith("core.")}


def energy_errors(sample, reference):
    assert set(sample) == set(reference) and reference
    assert all(math.isfinite(v) for v in reference.values())
    assert all(math.isfinite(v) for v in sample.values())
    errors = {k: abs(sample[k] - value) for k, value in reference.items()}
    assert all(math.isfinite(v) and v <= 1.e-5 for v in errors.values()), errors
    return errors


def worker_environment(env, stage, pin, core_hash):
    return dict(env, PYTHONPATH=str(stage / "build_fdds_validation/stage/lib"),
                PSIDATADIR=str(stage / "build_fdds_validation/stage/share/psi4"),
                CUEST_FDDS_READONLY_BUILD_ROOT=str(stage), CUEST_FDDS_BINARY_PIN=pin,
                CUEST_FDDS_EXPECTED_CORE_SHA256=core_hash)


def publish(pending, verified, result):
    # Printing may fail (e.g. transport closes). VERIFIED must not survive a
    # later failure: the atomic publication is the final operation.
    print(json.dumps(result, indent=2), flush=True)
    os.rename(pending, verified)


def summarize(samples, expected_solver):
    assert set(samples) == set(ORDER), "All five declared controls are mandatory"
    reference = samples["cpu0"]
    errors = {}
    for label in ORDER:
        d = samples[label]
        assert d["ok"] and d["system"] == "protein83" and d["natom"] == 83
        assert d["arm"] == ("cpu" if label == "cpu0" else "gpu")
        assert d["threads"] == 8 and d["memory"] == "96 GiB"
        assert d["nbf"] == 922 and d["naux"] == 3820 and len(d["pinv_calls"]) == 22
        for key in ("options", "cache_sha256", "metadata_sha256", "runner_sha256",
                    "numpy_version", "numpy_configuration", "job_id", "hostname",
                    "occupied", "virtual"):
            assert d[key] == reference[key], (label, key)
        assert math.isfinite(d["fdds_seconds"]) and d["fdds_seconds"] > 0
        candidate = label.startswith("candidate")
        assert d["gpu_pinv_requested"] == candidate
        routes = tuple(d[k] for k in ("gpu_timer_calls", "gpu_exchange_timer_calls",
                                      "gpu_metric_timer_calls", "gpu_response_timer_calls",
                                      "gpu_pinv_timer_calls"))
        assert routes == ((0, 0, 0, 0, 0) if label == "cpu0"
                          else (20, 4, 12, 60, 22 if candidate else 0)), (label, routes)
        expected = dict(dependencies(reference))
        if candidate:
            expected.update(expected_solver)
        assert dependencies(d) == expected, (label, "exact dependency mismatch")
        errors[label] = energy_errors(d["energies_Eh"], reference["energies_Eh"])
    base = statistics.median(samples[k]["fdds_seconds"] for k in ("base-gpu0", "base-gpu1"))
    new = statistics.median(samples[k]["fdds_seconds"] for k in ("candidate-gpu0", "candidate-gpu1"))
    return dict(system="protein83", fdds_only_not_whole_sapt=True,
                accepted_controls=list(ORDER), samples={k: samples[k]["fdds_seconds"] for k in ORDER},
                errors_Eh=errors, base_gpu_over_candidate_gpu=base / new,
                native_cpu_over_candidate_gpu=reference["fdds_seconds"] / new,
                note="One CPU/two GPU per build; no confidence interval or cross-allocation ratio")


def check_suite(path, count):
    s = ET.parse(path).find("testsuite")
    assert int(s.get("tests")) == count, s.attrib
    assert all(int(s.get(k, "0")) == 0 for k in ("failures", "errors", "skipped")), s.attrib
    return dict(s.attrib)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "compile", "controls", "runner", "base-source", "native-evidence", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--pin", required=True)
    p.add_argument("--base-pin", required=True)
    p.add_argument("--runner-pin", required=True)
    p.add_argument("--job-id", required=True)
    args = p.parse_args()
    assert os.environ["SLURM_JOB_ID"] == args.job_id
    assert os.environ["SLURM_CPUS_PER_TASK"] == "8"
    source(args.source, args.pin)
    source(args.base_source, args.base_pin)
    source(args.runner, args.runner_pin)
    # The same external benchmark bytes run every arm despite differing cores.
    benchmark = args.runner / "devtools/benchmarks/fdds_benchmark.py"
    assert sha(benchmark) == sha(args.source / "devtools/benchmarks/fdds_benchmark.py")
    assert read(args.controls / "plan.json")["controls"] == list(ORDER)
    assert read(args.native_evidence / "provenance.json")["scientific_pin"] == args.pin
    native_counts = check_suite(args.native_evidence / "native.xml", 183)
    compiled = read(args.compile / "import.json")
    assert (args.compile / "BUILD_ONLY.txt").exists()
    package = args.source / "build_fdds_validation/stage/lib/psi4"
    assert compiled["package"] == str(package)
    core = Path(compiled["core"])
    assert sha(core) == compiled["core_sha256"]
    expected_solver = {path: compiled["linked_library_sha256"][name]
                       for name, path in compiled["linked_libraries"].items()
                       if name.startswith("libcusolver")}
    assert expected_solver
    env_root = Path("/storage/project/r-cs207-0/awallace43/miniconda/envs/p4cuest_sapt")
    scratch = Path(f"/var/tmp/awallace43-fdds-{args.job_id}-cycle1")
    assert read(scratch / "STORAGE_VERIFIED.json")["job_id"] == int(args.job_id)
    args.output.mkdir(mode=0o700)
    validation = args.output / "validation"
    validation.mkdir()
    env = dict(os.environ, PATH=f"{env_root}/bin:{os.environ['PATH']}",
               LD_LIBRARY_PATH=f"{env_root}/lib:{env_root}/targets/x86_64-linux/lib:"
                               + os.environ.get("LD_LIBRARY_PATH", ""),
               PYTHONPATH=str(package.parent),
               PSIDATADIR=str(args.source / "build_fdds_validation/stage/share/psi4"),
               TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               CUEST_FDDS_READONLY_BUILD_ROOT=str(args.source), CUEST_FDDS_BINARY_PIN=args.pin,
               CUEST_FDDS_EXPECTED_CORE_SHA256=compiled["core_sha256"],
               CUEST_FDDS_COMPILE_ONLY="0",
               CUEST_FDDS_FOCUSED="1", CUEST_FDDS_FULL_AFTER_FOCUSED="1")
    samples = {k: read(args.controls / k / "result.json") for k in ORDER[:2]}
    base_compiled = read(args.controls.parent / "20261021-density-cycle-compile-01/import.json")
    for d in samples.values():
        assert d["core_sha256"] == base_compiled["core_sha256"]
        assert d["binary_pin"] == args.base_pin and d["runner_sha256"] == sha(benchmark)
        assert str(d["job_id"]) == args.job_id and d["ok"]
    try:
        with (args.output / "validation-driver.log").open("x") as log:
            subprocess.run(["timeout", "--signal=TERM", "--kill-after=30s", "720",
                            "bash", str(args.source / "devtools/benchmarks/fdds_validate.sh"),
                            str(args.source), str(env_root), args.pin, str(validation)],
                           env=env, cwd=args.output, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read(validation / "VERIFIED.json")
        assert report["unique_full_suite_tests"] == 112
        assert report["executed_tests_including_repeats"] == 176
        for file, count in (("fdds.xml", 63), ("sapt.xml", 1), ("full-fdds.xml", 112)):
            check_suite(validation / file, count)
        for label in ORDER[2:]:
            candidate = label.startswith("candidate")
            stage = args.source if candidate else args.base_source
            meta = compiled if candidate else base_compiled
            target = args.controls / label
            command = ["timeout", "--signal=TERM", "--kill-after=30s", "1050",
                       str(env_root / "bin/python"), str(benchmark),
                       "--system", "protein83", "--cache",
                       "/storage/project/r-cs207-0/awallace43/runs/psi4-fdds/20261020-optimization-01/caches/protein83",
                       "--output", str(target), "--arm", "gpu", "--threads", "8",
                       "--memory", "96 GiB", "--binary-pin", args.pin if candidate else args.base_pin,
                       "--expected-core-sha256", meta["core_sha256"], "--expected-package",
                       str(stage / "build_fdds_validation/stage/lib/psi4"),
                       "--require-gpu-exchange", "--require-gpu-metric", "--require-gpu-response-products"]
            if candidate:
                command.append("--gpu-pinv")
            worker_env = worker_environment(env, stage, args.pin if candidate else args.base_pin,
                                            meta["core_sha256"])
            with (args.controls / (label + ".log")).open("x") as log:
                subprocess.run(command, env=worker_env, cwd=args.output,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            samples[label] = read(target / "result.json")
            assert samples[label]["core_sha256"] == meta["core_sha256"]
            assert samples[label]["binary_pin"] == (args.pin if candidate else args.base_pin)
            assert samples[label]["runner_sha256"] == sha(benchmark)
            assert Path(samples[label]["psi4_package"]).parent == \
                stage / "build_fdds_validation/stage/lib/psi4"
            for name, path in meta["linked_libraries"].items():
                assert samples[label]["libraries_sha256"][path] == meta["linked_library_sha256"][name]
            # The first physical candidate is also a numerical gate before
            # spending the second candidate timing window.
            energy_errors(samples[label]["energies_Eh"], samples["cpu0"]["energies_Eh"])
            assert samples[label]["gpu_pinv_timer_calls"] == (22 if candidate else 0)
        result = summarize(samples, expected_solver)
        assert str(samples["cpu0"]["job_id"]) == args.job_id
        result.update(candidate_pin=args.pin, base_pin=args.base_pin, runner_pin=args.runner_pin,
                      native=native_counts, gpu=report)
        pending = args.output / "AUDIT_PENDING.json"
        pending.write_text(json.dumps(result, indent=2) + "\n")
        source(args.source, args.pin)
        source(args.base_source, args.base_pin)
        source(args.runner, args.runner_pin)
        assert sha(core) == compiled["core_sha256"]
        publish(pending, args.output / "VERIFIED.json", result)
    except BaseException as error:
        (args.output / "FAILED.json").write_text(json.dumps(
            dict(error=repr(error), completed_controls=list(samples), all_controls_required=list(ORDER)),
            indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
