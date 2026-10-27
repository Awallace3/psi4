"""Explicit current-stage benchmark: one GPU producer, independent CPU jobs.

This is a new experimental batch, not promotion of an old failed allocation or
a fabricated small-H200 release. Whole-stage CPU proof, installed-code hashes,
native GPU admission, raw component and same-gauge oracle checks stay mandatory.
No CPU reference or oracle runs while holding the H200 allocation.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from fragment_gate import require_oracle_config
from full_inputs import load_point
from gpu_runtime import digest, read_config
from owned_process import termination_signals
from repeat_gauge import compare_repeat
from run_full import compare_energy, execute_owned, save
from run_gate import compare, environment
from split_jobs import validate_allocation
from split_scratch import OwnedScratch
from submit_split import save_new, source_pin
from validate_cpu_allocation import fields

SYSTEMS = ("peptide", "nanotube", "protein83")
WALL = {"gpu": 28800, "peptide": 7200, "nanotube": 86400, "protein83": 86400}
CACHE_BYTES = 64 * 1024**2


def command(source, root, label, account, parent=None):
    gpu = label == "gpu"
    if gpu and parent is not None:
        raise ValueError("GPU allocation cannot wait for CPU references")
    if not gpu and (not parent or not parent.isdecimal()):
        raise ValueError("CPU job requires the exact accepted GPU producer ID")
    wall = WALL[label]
    argv = ["sbatch", "--parsable", "--job-name=disp0-bench-" + label,
            "--account=" + account, "--qos=inferno",
            "--partition=" + ("gpu-h200" if gpu else "cpu-small"),
            "--nodes=1", "--ntasks=1", "--cpus-per-task=" + ("8" if gpu else "24"),
            "--mem=160G", f"--time={wall//3600:02d}:00:00",
            "--no-requeue", "--signal=TERM@120", "--export=ALL",
            "--chdir=" + str(root), "--output=" + str(root/"logs/%x-%j.out"),
            "--error=" + str(root/"logs/%x-%j.err")]
    if gpu:
        argv.append("--gres=gpu:h200:1")
    else:
        argv.extend(("--exclusive", "--constraint=core24&gold6226&localNVMe",
                     "--dependency=afterany:" + parent, "--kill-on-invalid-dep=yes"))
    return argv + [str(source/"devtools/disp0/phoenix_validation/split_job.sbatch")]


def worker_command(helper, manifest, config, output, *, system=None, oracle=None):
    gpu = manifest["kind"] == "gpu"
    if gpu and oracle is not None:
        raise ValueError("Never schedule an independent CPU oracle on a GPU node")
    argv = [str(Path(config["stage"])/"bin/psi4"),
            config["dependency_environment"]+"/bin/python",
            str(helper/"fragment_gate.py"), "--config", manifest["config"],
            "--config-sha256", manifest["config_sha256"], "--arm", "gpu" if gpu else "cpu",
            "--scaled", "--benchmark", "--output", str(output),
            "--block-size", "24" if system else "5"]
    if system:
        argv.extend(("--inputs", manifest["inputs"], "--inputs-sha256",
                     manifest["inputs_sha256"], "--system", system))
    if gpu:
        argv.extend(("--factor-cache-bytes", str(CACHE_BYTES)))
    if oracle:
        argv.extend(("--oracle-from", str(oracle), "--oracle-config",
                     manifest["gpu_config"], "--oracle-config-sha256",
                     manifest["gpu_config_sha256"]))
    return ["/bin/bash", "-c",
            'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B "$3" "${@:4}"',
            "fsapt-benchmark", *argv]


def summary(values):
    if len(values) != 3 or any(not 0 < float(x) < float("inf") for x in values):
        raise ValueError("Three finite positive fresh-process timings required")
    return {"n": 3, "median_seconds": statistics.median(values),
            "minimum_seconds": min(values), "maximum_seconds": max(values)}


def execute(root):
    # The payload has already created this directory. Own it before any
    # configuration/scheduler/native preflight can fail.
    owner = OwnedScratch(Path(os.environ["TMPDIR"]).resolve(strict=True),
                         os.environ["DISP0_SPLIT_SCRATCH_CREATED"], root)
    groups = set()
    path = root/"benchmark-result.json"
    try:
        _execute(root, groups)
    except BaseException as error:
        document = json.loads(path.read_text()) if path.exists() else {}
        if document.get("status") != "FAILED_SOME_CASES":
            document.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        save(path, document)
        raise
    finally:
        document = json.loads(path.read_text()) if path.exists() else {"status": "FAILED"}
        try:
            document["scratch_cleanup"] = owner.cleanup(groups)
        except BaseException as error:
            document.update(status="FAILED", accepted_speedup_count=0,
                            scratch_cleanup={"complete": False,
                                             "error": f"{type(error).__name__}: {error}"})
            for case in document.get("cases", {}).values():
                case.update(accepted_speedup=False, qualification_invalidated_by_cleanup=True)
                if case.get("status", "").startswith("PASS_"):
                    case["status"] = "UNQUALIFIED_CLEANUP_FAILED"
            raise
        finally:
            save(path, document)


def _execute(root, groups):
    manifest = json.loads((root.parent/"manifest.json").read_text())
    kind = manifest["kind"]
    config, _ = read_config(Path(manifest["config"]), manifest["config_sha256"])
    if config["source_pin"] != manifest["science_pin"]:
        raise RuntimeError("Benchmark whole-stage science pin changed")
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
                                  text=True, timeout=30)
    validate_allocation(fields(raw), manifest)
    if len(os.sched_getaffinity(0)) < config["threads"]:
        raise RuntimeError("Insufficient allocated CPU affinity")
    scratch = Path(os.environ["TMPDIR"]).resolve(strict=True)
    if kind == "cpu":
        mount = fields(subprocess.check_output(
            ["findmnt", "-T", str(scratch), "-n", "-P", "-o", "SOURCE,FSTYPE,TARGET"],
            text=True, timeout=30))
        if mount.get("FSTYPE") not in ("xfs", "ext4", "ext3", "btrfs"):
            raise RuntimeError("CPU references require actual node-local scratch")
        disk = os.statvfs(scratch)
        if disk.f_bavail*disk.f_frsize < 256*1024**3:
            raise RuntimeError("CPU node-local scratch headroom below 256 GiB")
    helper = Path(__file__).resolve().parent
    deadline = time.monotonic()+manifest["allocation_wall_seconds"]-180
    result = {"status": "STARTED", "kind": kind, "source_pin": config["source_pin"],
              "helper_pin": manifest["helper_pin"], "scheduler_job": raw,
              "cases": {}, "processes": [], "accepted_speedup_count": 0,
              "resource_policy": {"cpu_reference_threads": 24, "gpu_host_threads": 8,
                                  "gpu_count": 1, "gpu_factor_cache_bytes": CACHE_BYTES},
              "scope": "INSTRUMENTED_WHOLE_ENERGY_BENCHMARK_NOT_UNINSTRUMENTED_PRODUCTION"}
    checkpoint = lambda: save(root/"benchmark-result.json", result)

    def child(label, system=None, oracle=None):
        output = root/label
        work = scratch/label
        work.mkdir(mode=0o700)
        argv = worker_command(helper, manifest, config, output, system=system, oracle=oracle)
        env = environment(config, gpu=kind == "gpu")
        env.update(TMPDIR=str(work), SCRATCH=str(work), PSI_SCRATCH=str(work))
        row = {"label": label, "argv": argv}
        result["processes"].append(row)
        checkpoint()
        remaining = min(28800, deadline-time.monotonic())
        if remaining <= 180:
            raise TimeoutError("Bounded benchmark allocation exhausted")
        with (root/(label+".stdout")).open("x") as out, (root/(label+".stderr")).open("x") as err:
            def launched(pid):
                groups.add(pid)
                row["pid"] = pid
                checkpoint()
            def reaped(pid):
                groups.discard(pid)
                row["owned_group_reaped"] = True
            code = execute_owned(argv, env, out, err, remaining, launched, reaped=reaped)
        row["returncode"] = code
        checkpoint()
        if code:
            raise RuntimeError("Scientific benchmark child failed: "+label)
        doc = json.loads((output/"result.json").read_text())
        if (doc["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                or doc["source_pin"] != manifest["science_pin"]
                or doc["runtime_config_sha256"] != manifest["config_sha256"]
                or doc["arm"] != kind or not doc["benchmark"]
                or doc.get("system") != system):
            raise RuntimeError("Incomplete/mismatched benchmark worker")
        for filename, checksum in doc["artifact_sha256"].items():
            if digest(output/filename) != checksum:
                raise RuntimeError("Benchmark snapshot seal changed")
        row["result_sha256"] = digest(output/"result.json")
        checkpoint()
        return output, doc

    checkpoint()
    try:
        if kind == "gpu":
            control_path, control = child("water-control")
            baseline = json.loads(Path(manifest["cpu_water_result"]).read_text())
            if digest(Path(manifest["cpu_water_result"])) != manifest["cpu_water_sha256"]:
                raise RuntimeError("Completed native CPU water control changed")
            if baseline["source_pin"] != config["source_pin"]:
                raise RuntimeError("CPU water control has a different scientific pin")
            result["water_energy_check"] = compare_energy(baseline, control)
            result["water_component_checks"] = compare(baseline, control)
            result["water_control_sha256"] = digest(control_path/"result.json")
            checkpoint()
        for system in manifest["systems"]:
            case = {"status": "STARTED", "samples": [], "accepted_speedup": False}
            result["cases"][system] = case
            checkpoint()
            try:
                _, dimensions = load_point(Path(manifest["inputs"]), manifest["inputs_sha256"], system)
                for repeat in range(3):
                    path, doc = child(f"{system}-r{repeat}", system)
                    case["samples"].append({"path": str(path), "result_sha256": digest(path/"result.json"),
                                            "seconds": doc["benchmark_whole_energy_seconds"]})
                    checkpoint()
                    if repeat:
                        compare_repeat(Path(case["samples"][0]["path"]), path,
                                       dimensions=dimensions, arm=kind)
                case["timing"] = summary([row["seconds"] for row in case["samples"]])
                case["status"] = "PASS_REPEATS_AWAITING_OTHER_RESOURCE_CLASS"
                checkpoint()
                if kind == "cpu":
                    producer = Path(manifest["gpu_job_root"])
                    if not (producer/"benchmark-result.json").is_file():
                        case["status"] = "PASS_CPU_REFERENCES_GPU_PRODUCER_UNAVAILABLE"
                        checkpoint()
                        continue
                    proof = json.loads((producer/"benchmark-result.json").read_text())
                    if manifest.get("gpu_manifest_sha256"):
                        if digest(producer.parent/"manifest.json") != manifest["gpu_manifest_sha256"]:
                            raise RuntimeError("Pinned GPU submission manifest changed")
                    peer, _ = read_config(Path(manifest["gpu_config"]), manifest["gpu_config_sha256"])
                    require_oracle_config(config, peer)
                    if (proof["source_pin"] != manifest["science_pin"] or
                            proof["helper_pin"] != manifest.get("gpu_helper_pin", manifest["helper_pin"])):
                        raise RuntimeError("GPU producer provenance mismatch")
                    gpu_case = proof["cases"].get(system, {})
                    if (proof.get("scratch_cleanup", {}).get("complete") is not True or
                            gpu_case.get("qualification_invalidated_by_cleanup") or
                            gpu_case.get("status") != "PASS_REPEATS_AWAITING_OTHER_RESOURCE_CLASS"):
                        case["status"] = "PASS_CPU_REFERENCES_GPU_CASE_UNQUALIFIED"
                        checkpoint()
                        continue
                    if len(gpu_case["samples"]) != 3:
                        raise RuntimeError("Three sealed GPU repeats required")
                    checks = []
                    for repeat, gpu_row in enumerate(gpu_case["samples"]):
                        gpu_path = Path(gpu_row["path"])
                        if digest(gpu_path/"result.json") != gpu_row["result_sha256"]:
                            raise RuntimeError("Completed GPU sample seal changed")
                        gpu = json.loads((gpu_path/"result.json").read_text())
                        if (gpu["source_pin"] != manifest["science_pin"]
                                or gpu["runtime_config_sha256"] != manifest["gpu_config_sha256"]
                                or gpu["arm"] != "gpu" or gpu["system"] != system
                                or not gpu["benchmark"]):
                            raise RuntimeError("GPU benchmark sample provenance mismatch")
                        for filename, checksum in gpu["artifact_sha256"].items():
                            if digest(gpu_path/filename) != checksum:
                                raise RuntimeError("GPU oracle input seal changed")
                        cpu = json.loads((Path(case["samples"][repeat]["path"])/"result.json").read_text())
                        for key in ("retained_orbital_rank", "virtual_orbitals"):
                            if cpu[key] != gpu[key]:
                                raise RuntimeError("CPU/GPU retained orbital spaces differ")
                        cpu_options, gpu_options = dict(cpu["options"]), dict(gpu["options"])
                        if cpu_options.pop("use_cuest") or not gpu_options.pop("use_cuest"):
                            raise RuntimeError("CPU/GPU native backend policy differs")
                        for options in (cpu_options, gpu_options):
                            options.pop("fisapt_fsapt_filepath")
                        if cpu_options != gpu_options:
                            raise RuntimeError("CPU/GPU scientific options differ")
                        _, oracle = child(f"{system}-oracle-r{repeat}", system, gpu_path)
                        checks.append({"energy": compare_energy(cpu, gpu), "components": compare(cpu, gpu),
                                       "same_gauge_errors": oracle["same_gauge_errors"]})
                    case.update(status="PASS_MATCHED_INSTRUMENTED_BENCHMARK",
                                independent_checks=checks, gpu_timing=gpu_case["timing"],
                                cpu_over_gpu_ratio=case["timing"]["median_seconds"] /
                                gpu_case["timing"]["median_seconds"], accepted_speedup=True)
                    result["accepted_speedup_count"] += 1
            except (InterruptedError, KeyboardInterrupt):
                raise
            except Exception as error:
                case.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                # A scientific failure is not a retry. Preserve it and use only
                # the already-approved distinct cases in this same allocation.
                if groups:
                    raise RuntimeError("Unproved owned-child cleanup; no further cases") from error
            checkpoint()
        result["status"] = ("FAILED_SOME_CASES" if any(
            c["status"] == "FAILED" for c in result["cases"].values())
            else "COMPLETED_REQUIRES_RECORDED_CASE_QUALIFICATION")
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        checkpoint()
    if result["status"] == "FAILED_SOME_CASES":
        raise RuntimeError("Some benchmark cases failed; original outputs retained")


def prepare(args):
    source = args.source.resolve(strict=True)
    pin = source_pin(source)
    config, _ = read_config(args.config, args.config_sha256)
    for system in SYSTEMS:
        load_point(args.inputs, args.inputs_sha256, system)
    cpu_water = args.config.resolve().parents[1]/"cpu-fragment-scaled/result.json"
    water = json.loads(cpu_water.read_text())
    if water["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON" or water["arm"] != "cpu":
        raise RuntimeError("Completed exact-source native CPU water preflight required")
    if water["source_pin"] != config["source_pin"]:
        raise RuntimeError("CPU control is from a different coherent build")
    producer_root = getattr(args, "gpu_producer_root", None)
    producer = None
    if producer_root:
        producer_root = producer_root.resolve(strict=True)
        producer = json.loads((producer_root/"manifest.json").read_text())
        parent = (producer_root/"job-id.txt").read_text().strip()
        if (not parent.isdecimal() or producer["kind"] != "gpu"
                or producer["science_pin"] != config["source_pin"]
                or producer["inputs_sha256"] != args.inputs_sha256
                or producer["systems"] != list(SYSTEMS)):
            raise RuntimeError("Explicit existing one-GPU producer is not compatible")
        # Consumer-only guard fixes may use a new helper pin, never a new timed
        # worker, scientific implementation, or runtime closure.
        for name in ("fragment_gate.py", "gpu_runtime.py", "full_inputs.py",
                     "repeat_gauge.py", "owned_process.py", "run_full.py", "run_gate.py"):
            if digest(source/"devtools/disp0/phoenix_validation"/name) != digest(
                    Path(producer["source"])/"devtools/disp0/phoenix_validation"/name):
                raise RuntimeError("CPU/GPU timed engine or shared qualification helper differs: "+name)
    args.root.mkdir(mode=0o700)
    labels = list(SYSTEMS) if producer else ["gpu", *SYSTEMS]
    for label in labels:
        root = args.root/label
        root.mkdir()
        (root/"logs").mkdir()
        gpu = label == "gpu"
        selected = {**config, "threads": 8 if gpu else 24, "host_memory": "96 GiB",
                    "device_memory_limit_bytes": 120*1024**3, "native_persistent_bytes": 12*1024**3}
        save_new(root/"runtime-config.json", selected)
        draft = {"kind": "gpu" if gpu else "cpu", "systems": list(SYSTEMS) if gpu else [label],
                 "source": str(source), "helper_pin": pin, "science_pin": config["source_pin"],
                 "config": str(root/"runtime-config.json"),
                 "config_sha256": digest(root/"runtime-config.json"),
                 "inputs": str(args.inputs.resolve()), "inputs_sha256": args.inputs_sha256,
                 "allocation_wall_seconds": WALL[label], "cpu_water_result": str(cpu_water),
                 "cpu_water_sha256": digest(cpu_water), "automatic_retry": False,
                 "new_experimental_benchmark_not_previous_H200_release": True}
        if producer:
            draft.update(gpu_helper_pin=producer["helper_pin"],
                         gpu_manifest_sha256=digest(producer_root/"manifest.json"))
        save_new(root/"draft-manifest.json", draft)
    save_new(args.root/"plan.json", {"status": "PREPARED_NOT_SUBMITTED", "source": str(source),
                                    "helper_pin": pin, "labels": labels,
                                    "gpu_root": str(producer_root or args.root/"gpu")})
    print("PREPARED="+str(args.root), flush=True)


def submit(args):
    plan = json.loads((args.root/"plan.json").read_text())
    source = Path(plan["source"])
    if source_pin(source) != plan["helper_pin"]:
        raise RuntimeError("Immutable benchmark helper checkout changed")
    if args.submit_one not in plan["labels"]:
        raise RuntimeError("Role not included in this prepared plan; no GPU duplication")
    root = args.root/args.submit_one
    draft = json.loads((root/"draft-manifest.json").read_text())
    config, _ = read_config(Path(draft["config"]), draft["config_sha256"])
    parent = None
    if args.submit_one != "gpu":
        producer_root = Path(plan.get("gpu_root", str(args.root/"gpu")))
        parent = (producer_root/"job-id.txt").read_text().strip()
        draft.update(gpu_job_root=str(producer_root/("job-"+parent)),
                     gpu_config=str(producer_root/"runtime-config.json"),
                     gpu_config_sha256=digest(producer_root/"runtime-config.json"))
    manifest = {**draft, "account": args.account, "qos": "inferno"}
    save_new(root/"manifest.json", manifest)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SBATCH_", "SLURM_"))}
    env.update(DISP0_SOURCE=str(source), DISP0_PIN=plan["helper_pin"], DISP0_RUN_ROOT=str(root),
               DISP0_GATE_ENV=config["dependency_environment"], DISP0_SPLIT_THREADS=str(config["threads"]),
               DISP0_SPLIT_KIND=draft["kind"], DISP0_SPLIT_DRIVER="benchmark_suite.py")
    argv = command(source, root, args.submit_one, args.account, parent)
    trial = subprocess.run([argv[0], "--test-only", *argv[1:]], env=env, capture_output=True,
                           text=True, timeout=60)
    save_new(root/"test-only.json", {"returncode": trial.returncode, "stdout": trial.stdout,
                                    "stderr": trial.stderr})
    print(trial.stdout+trial.stderr, flush=True)
    if trial.returncode:
        raise RuntimeError("Exact benchmark route rejected; not submitted")
    save_new(root/"SUBMISSION_ATTEMPTED.json", {"command": argv})
    response = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=60)
    save_new(root/"submission.json", {"returncode": response.returncode, "stdout": response.stdout,
                                     "stderr": response.stderr})
    print(response.stdout+response.stderr, flush=True)
    if response.returncode:
        raise RuntimeError("Failed/ambiguous scheduler response; inspect before any retry")
    job = response.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError("Ambiguous accepted job ID")
    with (root/"job-id.txt").open("x") as stream:
        stream.write(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(root), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--submit-one", choices=("gpu", *SYSTEMS))
    parser.add_argument("--account", choices=("gts-cs207", "gts-cs207-chemx"))
    parser.add_argument("--gpu-producer-root", type=Path,
                        help="Prepare CPU-only consumers for an already-submitted immutable GPU producer")
    for name in ("source", "config", "inputs"):
        parser.add_argument("--"+name, type=Path)
    for name in ("config", "inputs"):
        parser.add_argument("--"+name+"-sha256")
    args = parser.parse_args()
    if args.output:
        execute(args.output.resolve(strict=True))
    elif args.submit_one:
        if not args.root or not args.account:
            parser.error("Explicit prepared root/current account required for inferno submission")
        submit(args)
    else:
        if not all((args.root, args.source, args.config, args.config_sha256,
                    args.inputs, args.inputs_sha256)):
            parser.error("Preparation requires frozen source/config/input paths and hashes")
        if not args.root.is_absolute():
            parser.error("Fresh absolute output root required")
        prepare(args)


if __name__ == "__main__":
    with termination_signals():
        main()
