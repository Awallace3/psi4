"""Owned-process launcher for a standalone dimer energy diagnostic."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from phoenix_timing import clean, digest, save


def resources(backend):
    if backend not in ("cpu", "gpu"):
        raise ValueError("Explicit cpu/gpu required")
    gpu = backend == "gpu"
    return dict(threads=8 if gpu else 48, memory_GiB=160 if gpu else 360,
                wall_seconds=1800 if gpu else 3600,
                partition="gpu-h200" if gpu else "cpu-gnr",
                constraint="H200" if gpu else "graniterapids")


def native_headers(text):
    normalized = " ".join(text.split())
    return {
        "native_fp64_header_observed": "Math mode: Native FP64" in normalized,
        "cuest_header_observed": "cuESTJK: GPU-Accelerated" in normalized,
    }


def execute(path, root, worker=False):
    manifest = json.loads(path.read_text())
    if digest(path) != os.environ["FSAPT_MANIFEST_SHA256"]:
        raise RuntimeError("Frozen diagnostic manifest changed")
    source, helper = Path(manifest["science_source"]), Path(manifest["helper_source"])
    clean(source, manifest["science_pin"])
    clean(helper, manifest["helper_pin"])
    if Path(__file__).resolve().parents[2] != helper.resolve():
        raise RuntimeError("Wrong helper checkout")
    # Python helper changes do not overlay the frozen native stage.
    sys.path.insert(0, str(source / "devtools/disp0/phoenix_validation"))
    from gpu_runtime import read_config, retirement_channel
    from stage_scaling import require_cpu_runtime_admission
    from run_full import execute_owned
    from run_gate import environment
    from split_scratch import OwnedScratch
    from validate_cpu_allocation import fields, seconds, memory_bytes
    os.environ.update(DISP0_SOURCE=str(source), DISP0_PIN=manifest["science_pin"])
    config_path = Path(manifest["config"])
    config, build = read_config(config_path, manifest["config_sha256"])
    policy = resources(manifest["backend"])
    if (config["source_pin"] != manifest["science_pin"]
            or config["threads"] != policy["threads"]
            or require_cpu_runtime_admission(config) != manifest["cpu_admission"]):
        raise RuntimeError("Frozen native admission/configuration changed")
    if digest(Path(manifest["inputs"])) != manifest["inputs_sha256"]:
        raise RuntimeError("Input manifest changed")
    if worker:
        from sealed_bootstrap import bootstrap
        from gpu_runtime import AuditedRuntime
        owner = bootstrap(config, build)
        runtime = AuditedRuntime(config_path, manifest["config_sha256"])
        os.chdir(root)
        if manifest.get("mode", "energy") == "fixed-jk":
            from dimer_fixed_jk import run
            runtime.guarded_call(run, runtime.psi4, Path(manifest["inputs"]),
                                 root / "energy", manifest["backend"], policy["threads"],
                                 Path(manifest["reference"]), manifest["reference_sha256"])
            return
        from dimer_energy_probe import run
        runtime.guarded_call(run, runtime.psi4, Path(manifest["inputs"]),
                             root / "energy", manifest["backend"], policy["threads"])
        return
    gpu = manifest["backend"] == "gpu"
    groups, scratch_owner = set(), None
    report = dict(status="STARTED", manifest=manifest, manifest_sha256=digest(path),
                  accepted_speedup_count=0,
                  scope=("FIXED_INPUT_JK_NUMERICAL_DIAGNOSTIC"
                         if manifest.get("mode", "energy") == "fixed-jk"
                         else "STANDALONE_HF_NUMERICAL_DIAGNOSTIC"))
    try:
        raw = subprocess.check_output(
            ["/opt/slurm/current/bin/scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
            text=True, timeout=30)
        job = fields(raw)
        report["scheduler_job"] = raw
        if (job.get("Account") != "gts-cs207-chemx" or job.get("QOS") != "embers"
                or job.get("Partition") != policy["partition"]
                or job.get("NumNodes") != "1" or job.get("NumTasks") != "1"
                or job.get("NumCPUs") != str(policy["threads"])
                or seconds(job["TimeLimit"]) != policy["wall_seconds"]
                or memory_bytes(job["MinMemoryNode"]) != policy["memory_GiB"] * 1024**3):
            raise RuntimeError("Allocation differs from explicit diagnostic resources")
        allocated_gpu = [x for x in job.get("AllocTRES", "").split(",") if x.startswith("gres/gpu")]
        if gpu:
            if "gres/gpu:h200=1" not in allocated_gpu:
                raise RuntimeError("One H200 required")
            report["gpu_hardware"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
                 "--format=csv,noheader"], text=True, timeout=30)
            if "H200" not in report["gpu_hardware"] or "9.0" not in report["gpu_hardware"]:
                raise RuntimeError("Actual H200/sm90 required")
        elif allocated_gpu or os.environ.get("SLURM_JOB_GPUS"):
            raise RuntimeError("CPU control must allocate no GPU")
        report.update(node=os.uname().nodename, affinity=sorted(os.sched_getaffinity(0)))
        if len(report["affinity"]) < policy["threads"]:
            raise RuntimeError("Insufficient CPU affinity")
        parent = Path("/storage/scratch1/3/awallace43") if gpu else Path(
            os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/scratch")
        parent = parent.resolve(strict=True)
        scratch = parent / ("dimer-energy-" + os.environ["SLURM_JOB_ID"])
        scratch.mkdir(mode=0o700)
        scratch_owner = OwnedScratch(scratch, str(scratch), root)
        mount = fields(subprocess.check_output(
            ["findmnt", "-T", str(scratch), "-n", "-P", "-o", "SOURCE,FSTYPE,TARGET"],
            text=True, timeout=30))
        report["scratch_mount"] = mount
        if not gpu and mount.get("FSTYPE") not in ("xfs", "ext4", "ext3", "btrfs"):
            raise RuntimeError("CPU control requires node-local scratch")
        stat = os.statvfs(scratch)
        if stat.f_bavail * stat.f_frsize < 64 * 1024**3:
            raise RuntimeError("Insufficient dimer scratch headroom")
        aliases = Path(config["sealed_alias_parent"]) / ("worker-" + os.environ["SLURM_JOB_ID"])
        aliases.mkdir(mode=0o700)
        env = environment(config, gpu=gpu, sealed_alias_directory=aliases)
        env.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
                   FSAPT_MANIFEST_SHA256=digest(path))
        argv = ["/bin/bash", "-c",
                'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B "$3" "${@:4}"',
                "dimer-energy", str(Path(config["stage"]) / "bin/psi4"),
                config["dependency_environment"] + "/bin/python", str(Path(__file__).resolve()),
                "--manifest", str(path), "--output", str(root), "--worker"]
        with (root / "stdout").open("x") as stdout, (root / "stderr").open("x") as stderr, \
                retirement_channel(env) as channel:
            code = execute_owned(argv, env, stdout, stderr, policy["wall_seconds"] - 180,
                                 groups.add, reaped=groups.discard, pass_fds=channel.pass_fds)
            report.update(returncode=code, retirement_reason=channel.reason(),
                          owned_groups_reaped=not groups)
        result = root / "energy/result.json"
        if result.is_file():
            report["energy_result_sha256"] = digest(result)
            report["energy_status"] = json.loads(result.read_text())["status"]
        log = root / "energy/psi4.out"
        text = log.read_text() if log.is_file() else ""
        report.update(native_headers(text))
        if code or report["retirement_reason"] is not None or groups:
            raise RuntimeError("Energy worker failed; inspect preserved energy log/result")
        if (gpu and not (report["native_fp64_header_observed"] and report["cuest_header_observed"])) or (
                not gpu and report["cuest_header_observed"]):
            raise RuntimeError("Unexpected native backend/math mode")
        expected = ("EXECUTED_FIXED_INPUT_DIAGNOSTIC_ONLY"
                    if manifest.get("mode", "energy") == "fixed-jk"
                    else "CONVERGED_DIAGNOSTIC_ONLY")
        if report.get("energy_status") != expected:
            raise RuntimeError("Missing complete diagnostic result")
        report["status"] = expected
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if scratch_owner is not None:
                report["scratch_cleanup"] = scratch_owner.cleanup(groups)
        finally:
            save(root / "diagnostic-result.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    execute(args.manifest.resolve(strict=True), args.output.resolve(strict=True), args.worker)
