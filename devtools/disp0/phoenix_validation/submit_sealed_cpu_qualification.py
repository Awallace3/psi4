"""CPU-only qualification of a finalized sealed H200 peptide producer.

Submission recipe only: the existing immutable whole stage, benchmark driver,
scientific workers and comparison helpers execute from --source unchanged.
No GPU allocation, rebuild, dependency wait, retry or scientific overlay.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from full_inputs import load_point
from gpu_runtime import digest
from submit_split import save_new, source_pin


def command(source, root):
    return ["sbatch", "--parsable", "--job-name=disp0-sealed-cpu-qual",
            "--account=gts-cs207-chemx", "--qos=embers", "--partition=cpu-small",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=24", "--mem=160G",
            "--time=02:00:00", "--exclusive",
            "--constraint=core24&gold6226&localNVMe",
            "--no-requeue", "--signal=TERM@120", "--export=ALL",
            "--chdir="+str(root), "--output="+str(root/"logs/%x-%j.out"),
            "--error="+str(root/"logs/%x-%j.err"),
            str(source/"devtools/disp0/phoenix_validation/split_job.sbatch")]


def cpu_configuration(gpu):
    if ((gpu["schema"],gpu["device_family"]) != (1,"H200")
            or gpu["threads"] != 8 or gpu.get("identity_contract") !=
            "process-owned-kernel-sealed-snapshots-v1"):
        raise RuntimeError("Exact sealed H200 eight-thread peer configuration required")
    return {**gpu,"threads":24}  # Oracle contract permits this difference only.


def producer(root, pin):
    if (root/"job-status.env").read_text().splitlines() != ["exit_code=0","term_received=0"]:
        raise RuntimeError("Finalized exit0/no-TERM GPU job required")
    pilot = json.loads((root/"pilot-result.json").read_text())
    benchmark_path = root/"benchmark/execution/benchmark-result.json"
    gate_path = root/"small-gate/gate-result.json"
    config_path = root/"runtime-config.json"
    manifest_path = root/"benchmark/manifest.json"
    if (pilot["status"] != "PASS_GPU_PILOT_AWAITING_INDEPENDENT_CPU_QUALIFICATION"
            or pilot["source_pin"] != pin or not pilot["scratch_cleanup"]["complete"]
            or Path(pilot["benchmark_result"]).resolve() != benchmark_path.resolve()
            or pilot["benchmark_sha256"] != digest(benchmark_path)
            or pilot["small_gate_sha256"] != digest(gate_path)
            or pilot["config_sha256"] != digest(config_path)):
        raise RuntimeError("Incomplete/changed final H200 pilot")
    gate = json.loads(gate_path.read_text())
    benchmark = json.loads(benchmark_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    cpu_configuration(config)
    if (gate["status"] != "PASS_SMALL_H200_ONLY"
            or gate["gpu_small_case_correctness_verified"] is not True
            or gate["source_pin"] != pin or config["source_pin"] != pin
            or benchmark["source_pin"] != pin or benchmark["helper_pin"] != pin
            or not benchmark["scratch_cleanup"]["complete"]
            or manifest["science_pin"] != pin or manifest["helper_pin"] != pin
            or manifest["config_sha256"] != digest(config_path)):
        raise RuntimeError("Matching completed physical gate/producer/source required")
    case = benchmark["cases"]["peptide"]
    if (case["status"] != "PASS_REPEATS_AWAITING_OTHER_RESOURCE_CLASS"
            or case.get("qualification_invalidated_by_cleanup") or len(case["samples"]) != 3):
        raise RuntimeError("Three complete qualified GPU repeats required")
    _, dimensions = load_point(Path(manifest["inputs"]),manifest["inputs_sha256"],"peptide")
    pids, values = set(), []
    for index, row in enumerate(case["samples"]):
        path = Path(row["path"])
        if (path.resolve() != (benchmark_path.parent/f"peptide-r{index}").resolve()
                or digest(path/"result.json") != row["result_sha256"]):
            raise RuntimeError("GPU sample result seal/path changed")
        sample = json.loads((path/"result.json").read_text())
        if (sample["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                or sample["arm"] != "gpu" or sample["system"] != "peptide"
                or sample["source_pin"] != pin or not sample["benchmark"]
                or sample["scaled"] is not True or sample["options"]["use_cuest"] is not True
                or sample["orbital_dimensions"] != dimensions or sample["pid"] in pids
                or sample["runtime_config_sha256"] != digest(config_path)
                or sample["benchmark_whole_energy_seconds"] != row["seconds"]):
            raise RuntimeError("GPU sample provenance/timing mismatch")
        pids.add(sample["pid"])
        value = float(row["seconds"])
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("Finite positive GPU timings required")
        values.append(value)
    expected = {"n":3,"median_seconds":statistics.median(values),
                "minimum_seconds":min(values),"maximum_seconds":max(values)}
    if case["timing"] != expected:
        raise RuntimeError("GPU summary does not match its three actual energy timers")
    # Large NPZ payloads are verified by the CPU worker before each oracle;
    # this login-node preflight hashes only small finalized proof documents.
    return config, manifest, {
        "gpu_pilot_result_sha256":digest(root/"pilot-result.json"),
        "gpu_benchmark_sha256":digest(benchmark_path),
        "gpu_small_gate_sha256":digest(gate_path),
        "gpu_manifest_sha256":digest(manifest_path),
        "gpu_config_sha256":digest(config_path),
        "gpu_config":str(config_path),"gpu_job_root":str(benchmark_path.parent)}


def submit(args):
    source = args.source.resolve(strict=True)
    pin = source_pin(source)
    if pin != args.pin:
        raise RuntimeError("Clean exact execution source required")
    gpu_root = args.gpu_job_root.resolve(strict=True)
    gpu, gpu_manifest, sealed = producer(gpu_root,pin)
    if Path(gpu_manifest["source"]).resolve() != source:
        raise RuntimeError("GPU/CPU execution-helper checkouts must be identical")
    recipe = Path(__file__).resolve()
    recipe_pin = source_pin(recipe.parents[3])
    root = args.root
    if not root.is_absolute():
        raise ValueError("Fresh absolute CPU output root required")
    root.mkdir(mode=0o700)
    (root/"logs").mkdir()
    config_path = root/"runtime-config.json"
    save_new(config_path,cpu_configuration(gpu))
    manifest = {
        "kind":"cpu","systems":["peptide"],"source":str(source),
        "helper_pin":pin,"science_pin":pin,"gpu_helper_pin":pin,
        "config":str(config_path),"config_sha256":digest(config_path),
        "inputs":gpu_manifest["inputs"],"inputs_sha256":gpu_manifest["inputs_sha256"],
        "cpu_water_result":gpu_manifest["cpu_water_result"],
        "cpu_water_sha256":gpu_manifest["cpu_water_sha256"],
        "allocation_wall_seconds":7200,"account":"gts-cs207-chemx","qos":"embers",
        "automatic_retry":False,"submission_recipe":str(recipe),
        "submission_recipe_pin":recipe_pin,"submission_recipe_sha256":digest(recipe),
        **sealed}
    save_new(root/"manifest.json",manifest)
    env = {k:v for k,v in os.environ.items() if not k.startswith(("SBATCH_","SLURM_"))}
    env.update(DISP0_SOURCE=str(source),DISP0_PIN=pin,DISP0_RUN_ROOT=str(root),
               DISP0_GATE_ENV=gpu["dependency_environment"],DISP0_SPLIT_THREADS="24",
               DISP0_SPLIT_KIND="cpu",DISP0_SPLIT_DRIVER="benchmark_suite.py")
    argv = command(source,root)
    for trial, name in ((True,"test-only.json"),(False,"submission.json")):
        if not trial:save_new(root/"SUBMISSION_ATTEMPTED.json",{"command":argv})
        response = subprocess.run([argv[0],"--test-only",*argv[1:]] if trial else argv,
                                  env=env,capture_output=True,text=True,timeout=60)
        save_new(root/name,{"returncode":response.returncode,
                           "stdout":response.stdout,"stderr":response.stderr})
        print(response.stdout+response.stderr,flush=True)
        if response.returncode:
            raise RuntimeError("Scheduler failure/ambiguity; no automatic retry")
    job = response.stdout.strip().split(";")[0]
    if not job.isdecimal():raise RuntimeError("Ambiguous scheduler job ID")
    with (root/"job-id.txt").open("x") as stream:stream.write(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(root),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--pin",required=True)
    parser.add_argument("--gpu-job-root",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True)
    submit(parser.parse_args())


if __name__ == "__main__":
    main()
