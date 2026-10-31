"""One embers H200 peptide pilot after a separate exact whole CPU build.

The small ten-process physical gate runs first. Three fresh peptide GPU samples
then retain benchmark_suite's work, repeat, memory and cleanup checks. No large
CPU reference/oracle runs while holding the GPU; no accepted CPU/GPU ratio is
reported until independent CPU qualification. No automatic resubmission.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from gpu_runtime import digest, read_config, require_cuda_manifest
from owned_process import termination_signals
from submit_split import save_new, source_pin

CONTRACT = "process-owned-kernel-sealed-snapshots-v1"
PRELOAD = ("libcudart.so.13", "libnvJitLink.so.13", "libcublasLt.so.13",
           "libcublas.so.13", "libcusparse.so.12", "libcusolver.so.12",
           "libnvrtc-builtins.so.13.3", "libnvrtc.so.13", "libcuest.so.0.2.50")


def command(source, root, parent):
    if not str(parent).isdecimal():
        raise ValueError("Exact CPU build job ID required")
    return ["sbatch", "--parsable", "--job-name=disp0-sealed-h200",
            "--account=gts-cs207-chemx", "--qos=embers", "--partition=gpu-h200",
            "--constraint=H200", "--gres=gpu:h200:1", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--mem=160G", "--time=08:00:00",
            "--no-requeue", "--signal=TERM@120",
            "--dependency=afterany:"+str(parent), "--kill-on-invalid-dep=yes",
            "--chdir="+str(root), "--output="+str(root/"logs/%x-%j.out"),
            "--error="+str(root/"logs/%x-%j.err"), "--export=ALL",
            str(source/"devtools/disp0/phoenix_validation/sealed_h200.sbatch")]


def configuration(base, build, root):
    """Explicit opt-in, without changing any science/cache/workspace policy."""
    if ((base["schema"], base["device_family"]) != (1, "H200")
            or build["status"] != "PASS" or build["pin"] != base["source_pin"]):
        raise RuntimeError("Exact whole Phoenix CPU build required")
    rows = require_cuda_manifest(base)["components"]
    paths = [str(Path(row["path"]).resolve(strict=True)) for row in rows]
    native = str((Path(base["cuest_root"])/
                  "lib/13/libcuest.so.0.2.50").resolve(strict=True))
    paths.append(native)
    preloads = []
    for name in PRELOAD:
        matches = [p for p in paths if Path(p).name == name
                   or Path(p).name.startswith(name+".")]
        if len(matches) != 1:
            raise RuntimeError("Ambiguous/missing pinned preload: "+name)
        preloads.append(matches[0])
    aliases = root/"sealed-workers"
    aliases.mkdir(mode=0o700)
    return {**base, "identity_contract": CONTRACT,
            "startup_library_path": str(Path(base["dependency_environment"])/"lib"),
            "sealed_alias_parent": str(aliases),
            "native_preload_paths": preloads,
            "threads": 8, "host_memory": "96 GiB",
            "device_memory_limit_bytes": 120*1024**3,
            "native_persistent_bytes": 12*1024**3}


def execute(manifest_path):
    from split_scratch import OwnedScratch
    from run_full import save
    root = Path(os.environ["DISP0_JOB_RUN"]).resolve(strict=True)
    owner = OwnedScratch(Path(os.environ["TMPDIR"]).resolve(strict=True),
                         os.environ["DISP0_SPLIT_SCRATCH_CREATED"], root)
    groups = set()
    path = root/"pilot-result.json"
    try:
        _execute(manifest_path, root, groups)
    except BaseException as error:
        save(path, {"status": "FAILED", "error": f"{type(error).__name__}: {error}",
                    "accepted_speedup_count": 0})
        raise
    finally:
        doc = json.loads(path.read_text()) if path.exists() else {"status": "FAILED"}
        try:
            doc["scratch_cleanup"] = owner.cleanup(groups)
        except BaseException as error:
            doc.update(status="FAILED", accepted_speedup_count=0,
                       scratch_cleanup={"complete": False,
                                        "error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            save(path, doc)


def _execute(manifest_path, root, groups):
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest["source"])
    if source_pin(source) != manifest["helper_pin"]:
        raise RuntimeError("Frozen H200 source changed")
    build_root = Path(manifest["build_root"])
    status = (build_root/"job-status.env").read_text().splitlines()
    if "exit_code=0" not in status or "term_received=0" not in status:
        raise RuntimeError("CPU prerequisite failed/interrupted; no GPU science")
    base_path = build_root/"gpu-runtime/runtime-config.json"
    base, build = read_config(base_path, digest(base_path))
    if base["source_pin"] != manifest["helper_pin"]:
        raise RuntimeError("Whole stage and helper pin differ")
    from split_jobs import validate_allocation
    from validate_cpu_allocation import fields
    raw = subprocess.check_output(
        ["scontrol","show","job","-o",os.environ["SLURM_JOB_ID"]], text=True, timeout=30)
    validate_allocation(fields(raw), manifest)
    if len(os.sched_getaffinity(0)) < 8:
        raise RuntimeError("Insufficient allocated CPU affinity")
    save_new(root/"allocation.json", {"scheduler_job": raw,
                                     "kernel": os.uname().release})
    config = configuration(base, build, root)
    path = root/"runtime-config.json"
    save_new(path, config)
    checksum = digest(path)
    read_config(path, checksum)
    helper = source/"devtools/disp0/phoenix_validation"
    gate = root/"small-gate"
    gate.mkdir()
    # Reuse the unchanged small scientific/independent CPU oracle gate.
    from run_full import execute_owned
    with (root/"small-gate.stdout").open("x") as out, (root/"small-gate.stderr").open("x") as err:
        code = execute_owned(
            [sys.executable, "-B", str(helper/"run_gate.py"),
             "--config", str(path), "--config-sha256", checksum, "--output", str(gate)],
            dict(os.environ), out, err, 3500, groups.add, reaped=groups.discard)
    if code:
        raise RuntimeError("Small H200 numerical gate failed; no peptide benchmark")
    proof = json.loads((gate/"gate-result.json").read_text())
    if (proof["status"] != "PASS_SMALL_H200_ONLY"
            or not proof["gpu_small_case_correctness_verified"]):
        raise RuntimeError("No benchmark without the full small H200 gate")
    # benchmark_suite checks the scheduler, actual device, artifacts, repeats,
    # full continuation and teardown. Its GPU role cannot compute a large CPU
    # reference or oracle and never publishes an accepted speedup by itself.
    bench = root/"benchmark"
    bench.mkdir()
    worker = bench/"execution"
    worker.mkdir()
    cpu = build_root/"cpu-fragment-scaled/result.json"
    draft = {**manifest, "config": str(path), "config_sha256": checksum,
             "science_pin": config["source_pin"], "systems": ["peptide"],
             "cpu_water_result": str(cpu), "cpu_water_sha256": digest(cpu)}
    save_new(bench/"manifest.json", draft)
    from benchmark_suite import execute as benchmark
    benchmark(worker)
    save_new(root/"pilot-result.json", {
        "status": "PASS_GPU_PILOT_AWAITING_INDEPENDENT_CPU_QUALIFICATION",
        "source_pin": config["source_pin"], "job": os.environ["SLURM_JOB_ID"],
        "config_sha256": checksum, "small_gate_sha256": digest(gate/"gate-result.json"),
        "benchmark_result": str(worker/"benchmark-result.json"),
        "benchmark_sha256": digest(worker/"benchmark-result.json"),
        "accepted_speedup_count": 0, "automatic_retry": False})


def submit(args):
    source = args.source.resolve(strict=True)
    pin = source_pin(source)
    if pin != args.pin or not str(args.build_job).isdecimal():
        raise RuntimeError("Clean exact source/build prerequisite required")
    from full_inputs import load_point
    load_point(args.inputs, args.inputs_sha256, "peptide")
    root = args.root
    if not root.is_absolute():
        raise ValueError("Fresh absolute run root required")
    root.mkdir(mode=0o700)
    (root/"logs").mkdir()
    build_root = args.build_root.resolve()
    if build_root.name != "job-"+str(args.build_job) or not build_root.parent.is_dir():
        raise RuntimeError("Explicit submitted CPU build root/job locator required")
    manifest = {
        "source": str(source), "helper_pin": pin, "science_pin": pin,
        "build_root": str(build_root),
        "build_job": str(args.build_job), "kind": "gpu",
        "inputs": str(args.inputs.resolve(strict=True)), "inputs_sha256": args.inputs_sha256,
        "systems": ["peptide"], "qos": "embers", "account": "gts-cs207-chemx",
        "allocation_wall_seconds": 28800, "automatic_retry": False,
        "scope": "H200_PEPTIDE_PILOT_REQUIRES_INDEPENDENT_CPU_FOR_SPEEDUP"}
    save_new(root/"manifest.json", manifest)
    env = {k:v for k,v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
    env.update(DISP0_SOURCE=str(source), DISP0_PIN=pin, DISP0_RUN_ROOT=str(root),
               DISP0_MANIFEST_SHA256=digest(root/"manifest.json"))
    argv = command(source, root, args.build_job)
    for trial, output in ((True, "test-only.json"), (False, "submission.json")):
        if not trial:
            save_new(root/"SUBMISSION_ATTEMPTED.json", {"command": argv})
        response = subprocess.run([argv[0], "--test-only", *argv[1:]] if trial else argv,
                                  env=env, capture_output=True, text=True, timeout=60)
        save_new(root/output, {"returncode": response.returncode,
                             "stdout": response.stdout, "stderr": response.stderr})
        print(response.stdout+response.stderr, flush=True)
        if response.returncode:
            raise RuntimeError("Scheduler failure/ambiguity; no automatic retry")
    job = response.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError("Ambiguous job ID; inspect saved response before recovery")
    with (root/"job-id.txt").open("x") as stream:
        stream.write(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(root), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--pin")
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--build-job")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inputs-sha256")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.execute:
        if digest(args.execute) != os.environ["DISP0_MANIFEST_SHA256"]:
            raise RuntimeError("Submitted H200 manifest changed")
        execute(args.execute.resolve(strict=True))
    else:
        if not all((args.source,args.pin,args.build_root,args.build_job,
                    args.inputs,args.inputs_sha256,args.root)):
            parser.error("Explicit source, build prerequisite, inputs and output required")
        submit(args)


if __name__ == "__main__":
    with termination_signals():
        main()
