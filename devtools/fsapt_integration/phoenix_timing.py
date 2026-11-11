"""Matched eight-thread HF F-SAPT peptide timings on one H200 embers allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def clean(source, pin):
    if (subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                text=True).strip() != pin
            or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"],
                                       text=True).strip()):
        raise RuntimeError("Pinned clean checkout required")


def summary(values):
    if len(values) != 3 or any(not 0 < value < float("inf") for value in values):
        raise ValueError("Three positive finite timings required; no dropped samples")
    return {"samples_seconds": values, "median_seconds": statistics.median(values),
            "min_seconds": min(values), "max_seconds": max(values)}


def ratios(cpu, gpu):
    result = {}
    for key in sorted((set(cpu[0]) & set(gpu[0])) - {"other", "gpu_fragment_overhead"}):
        left, right = summary([row[key] for row in cpu]), summary([row[key] for row in gpu])
        result[key] = {"cpu": left, "gpu": right,
                       "speedup": left["median_seconds"] / right["median_seconds"]}
    return result


def execute(manifest_path, root):
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != os.environ["FSAPT_MANIFEST_SHA256"]:
        raise RuntimeError("Submission manifest changed")
    source = Path(manifest["science_source"])
    outer = Path(manifest["helper_source"])
    clean(source, manifest["science_pin"])
    clean(outer, manifest["helper_pin"])
    subprocess.run(["git", "-C", str(outer), "diff", "--exit-code",
                    manifest["science_pin"], manifest["helper_pin"], "--",
                    "psi4", "psi4_runtime_support"], check=True)
    helper = source / "devtools/disp0/phoenix_validation"
    sys.path.insert(0, str(helper))
    from gpu_runtime import read_config, retirement_channel
    from stage_scaling import require_cpu_runtime_admission
    from run_gate import environment, compare
    from run_full import execute_owned, compare_energy
    from repeat_gauge import compare_repeat
    from full_inputs import load_point
    from prepare_inputs import input_basis_profile
    from sealed_h200 import step_scratch
    from split_jobs import validate_allocation
    from validate_cpu_allocation import fields

    os.environ.update(DISP0_SOURCE=str(source), DISP0_PIN=manifest["science_pin"])
    report = {"status": "STARTED", "accepted_speedup_count": 0, "processes": [],
              "manifest_sha256": digest(manifest_path), "manifest": manifest,
              "sanitizer_qualification": "NOT_PASSED_LOADER_INCOMPATIBILITY",
              "scope": "HF_FSAPT_NOT_KS_XC_CPKS; NUMERICAL_GATES_NOT_MEMORY_CERTIFICATION"}
    checkpoint = lambda: save(root / "timing-result.json", report)
    groups, owner = set(), None
    checkpoint()
    try:
        raw = subprocess.check_output(
            ["/opt/slurm/current/bin/scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
            text=True, timeout=30)
        validate_allocation(fields(raw), manifest)
        report["scheduler_job"] = raw
        report["node"] = os.uname().nodename
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        if len(report["cpu_affinity"]) < 8:
            raise RuntimeError("Eight allocated CPUs required")
        report["cpu_hardware"] = subprocess.check_output(["lscpu"], text=True, timeout=30)
        report["gpu_hardware"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader"], text=True, timeout=30)
        if "H200" not in report["gpu_hardware"]:
            raise RuntimeError("H200 required")
        if digest(Path(manifest["build_root"]) / "manifest.json") != manifest["build_manifest_sha256"]:
            raise RuntimeError("Prerequisite build manifest changed")
        build = Path(manifest["build_root"]) / ("job-" + manifest["build_job"])
        config_path = root / "runtime-config.json"
        base_path = build / "cpu-sealed-admission/runtime-config.json"
        base, proof = read_config(base_path, digest(base_path))
        if proof["pin"] != manifest["science_pin"] or base["threads"] != 8:
            raise RuntimeError("Wrong build pin or thread policy")
        report["cpu_runtime_admission"] = require_cpu_runtime_admission(base)
        aliases = root / "sealed-workers"
        aliases.mkdir(mode=0o700)
        save(config_path, {**base, "sealed_alias_parent": str(aliases)})
        config_sha = digest(config_path)
        config, _ = read_config(config_path, config_sha)
        require_cpu_runtime_admission(config)
        inputs = Path(manifest["inputs"])
        _, dimensions = load_point(inputs, manifest["inputs_sha256"], "peptide")
        if input_basis_profile(json.loads(inputs.read_text())) != "jun-cc-pvdz":
            raise RuntimeError("Qualified jun-cc-pVDZ inputs required")
        owner = step_scratch(root)
        scratch = Path(os.environ["TMPDIR"])
        report["scratch_mount"] = fields(subprocess.check_output(
            ["findmnt", "-T", str(scratch), "-n", "-P", "-o", "SOURCE,FSTYPE,TARGET"],
            text=True, timeout=30))
        disk = os.statvfs(scratch)
        report["scratch_free_bytes"] = disk.f_bavail * disk.f_frsize
        if report["scratch_free_bytes"] < 32 * 1024**3:
            raise RuntimeError("Peptide scratch headroom below 32 GiB")
        report["storage_scope"] = (
            "SAME_SHARED_SCRATCH_BOTH_ARMS; PRACTICAL_WORKFLOW_WALL_TIME_"
            "NOT_STORAGE_INDEPENDENT_COMPUTE_SPEEDUP")
        deadline = time.monotonic() + manifest["allocation_wall_seconds"] - 240

        def process(label, argv, env):
            row = {"label": label, "argv": argv}
            report["processes"].append(row)
            checkpoint()
            remaining = deadline - time.monotonic()
            if remaining < 60:
                raise TimeoutError("Allocation deadline exhausted")
            print("START", label, flush=True)
            with (root / (label + ".stdout")).open("x") as out, (
                    root / (label + ".stderr")).open("x") as err, retirement_channel(env) as channel:
                code = execute_owned(argv, env, out, err, min(3500, remaining),
                                     groups.add, reaped=groups.discard, pass_fds=channel.pass_fds)
                row.update(returncode=code, retirement_reason=channel.reason())
            checkpoint()
            if code or row["retirement_reason"] is not None:
                raise RuntimeError("Failed child: " + label)
            print("DONE", label, flush=True)
            return row

        gate = root / "small-gate"
        gate.mkdir()
        process("small-gate", [sys.executable, "-B", str(helper / "run_gate.py"),
                              "--config", str(config_path), "--config-sha256", config_sha,
                              "--output", str(gate)], dict(os.environ))
        release = json.loads((gate / "gate-result.json").read_text())
        if (release["status"] != "PASS_SMALL_H200_ONLY"
                or not release["gpu_small_case_correctness_verified"]):
            raise RuntimeError("Small H200 correctness gate failed")
        report["small_gate_sha256"] = digest(gate / "gate-result.json")

        def worker(label, arm, instrumented=False, oracle=None):
            output = root / label
            work = scratch / label
            work.mkdir()
            alias = aliases / label
            alias.mkdir(mode=0o700)
            env = environment(config, gpu=arm == "gpu", sealed_alias_directory=alias)
            env.update(TMPDIR=str(work), SCRATCH=str(work), PSI_SCRATCH=str(work))
            bootstrap = helper / "sealed_bootstrap.py"
            entry = ([str(outer / "devtools/fsapt_integration/timing_worker.py"),
                      "--bootstrap", str(bootstrap), "--receipt", str(root / (label + "-timers.json"))]
                     if instrumented else [str(bootstrap)])
            payload = [config["dependency_environment"] + "/bin/python", "-B", *entry,
                       "--config", str(config_path), "--config-sha256", config_sha,
                       "--arm", arm, "--scaled", "--benchmark", "--output", str(output),
                       "--inputs", str(inputs), "--inputs-sha256", manifest["inputs_sha256"],
                       "--system", "peptide", "--block-size", "24", "--factor-cache-bytes", "0",
                       "--scf-convergence", "10", "--scf-density-convergence", "12"]
            if oracle is not None:
                payload += ["--oracle-from", str(oracle)]
            argv = ["/bin/bash", "-c",
                    'set -euo pipefail; export PYTHONPATH="${PYTHONPATH-}"; '
                    'eval "$("$1" --psiapi)"; shift; exec "$@"',
                    "fsapt-timing", config["stage"] + "/bin/psi4", *payload]
            row = process(label, argv, env)
            doc = json.loads((output / "result.json").read_text())
            if (doc["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                    or doc["source_pin"] != manifest["science_pin"]
                    or doc["runtime_config_sha256"] != config_sha
                    or doc["arm"] != arm or doc["system"] != "peptide"
                    or not doc["benchmark"] or doc["orbital_dimensions"] != dimensions):
                raise RuntimeError("Worker provenance mismatch")
            for name, checksum in doc["artifact_sha256"].items():
                if digest(output / name) != checksum:
                    raise RuntimeError("Worker artifact changed: " + name)
            row.update(result_sha256=digest(output / "result.json"), path=str(output),
                       energy_seconds=doc["benchmark_whole_energy_seconds"])
            seconds = None
            if instrumented:
                receipt_path = root / (label + "-timers.json")
                receipt = json.loads(receipt_path.read_text())
                if receipt["status"] != "PASS_OBSERVED_TIMERS" or receipt["arm"] != arm:
                    raise RuntimeError("Incomplete timer receipt")
                seconds = receipt["seconds"]
                row["timers_sha256"] = digest(receipt_path)
            checkpoint()
            return output, doc, seconds

        samples = {mode: {arm: [] for arm in ("cpu", "gpu")}
                   for mode in ("baseline", "instrumented")}
        checks, references = [], {}
        for mode in samples:
            for repeat in range(3):
                pair = {}
                for arm in (("cpu", "gpu") if repeat % 2 == 0 else ("gpu", "cpu")):
                    path, doc, seconds = worker(f"{mode}-{arm}-{repeat}", arm,
                                               instrumented=mode == "instrumented")
                    if arm in references:
                        checks.append(compare_repeat(references[arm], path,
                                                     dimensions=dimensions, arm=arm))
                    else:
                        references[arm] = path
                    pair[arm] = doc
                    samples[mode][arm].append(
                        seconds if seconds is not None else {"total": doc["benchmark_whole_energy_seconds"]})
                checks.append({"energy": compare_energy(pair["cpu"], pair["gpu"]),
                               "signed_terms": compare(pair["cpu"], pair["gpu"])})
                report["numerical_checks"] = checks
                checkpoint()
        _, oracle, _ = worker("same-gauge-oracle", "cpu", oracle=references["gpu"])
        report["same_gauge_fields"] = oracle["same_gauge_compared_fields"]
        report["same_gauge_errors"] = oracle["same_gauge_errors"]
        report["baseline"] = ratios(samples["baseline"]["cpu"], samples["baseline"]["gpu"])
        report["instrumented_steps"] = ratios(samples["instrumented"]["cpu"],
                                               samples["instrumented"]["gpu"])
        report["instrumented_raw_samples"] = samples["instrumented"]
        report["timing_notes"] = [
            "Baseline: no source tracer, but normal qualification/audit work remains charged.",
            "Step table: separately instrumented workers; never reconstruct baseline from traced percentages.",
            "GPU term windows synchronize at boundaries; these are host wall times, not kernel times.",
            "GPU term windows exclude later scaling/downloads; those remain GPU fragment overhead.",
            "CPU/GPU localization implements corresponding stages, not identical algorithms.",
            "Both arms use the same persistent shared scratch (mount recorded), not local NVMe. "
            "IO and shared-filesystem contention remain charged; ratios are workflow timings, "
            "not storage-independent compute speedups.",
            "Other residuals have different contents; their ratio is not an algorithmic speedup.",
            "Delta-HF assembly excludes dimer SCF, already charged once under scf_dimer.",
            "Fragment dispersion supplies canonical dispersion; no duplicate canonical dispersion row.",
            "No sanitizer pass; numerical qualification only, pending loader-compatible sanitizer workflow."]
        clean(source, manifest["science_pin"])
        clean(outer, manifest["helper_pin"])
        report.update(status="PASS_NUMERICAL_TIMING_SANITIZER_OUTSTANDING",
                      accepted_speedup_count=len(report["instrumented_steps"]) + 1)
    except BaseException as error:
        report.update(status="FAILED", accepted_speedup_count=0,
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if owner is not None:
                report["scratch_cleanup"] = owner.cleanup(groups)
        except BaseException as error:
            report.update(status="FAILED", accepted_speedup_count=0,
                          cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The owned-process helper translates TERM into failure and reaps descendants.
    manifest = json.loads(args.manifest.read_text())
    sys.path.insert(0, str(Path(manifest["science_source"]) / "devtools/disp0/phoenix_validation"))
    from owned_process import termination_signals
    with termination_signals():
        execute(args.manifest, args.output)
