"""Two fresh protein83 baselines per arm; CPU48 references never reserve GPUs.

No Python source tracing. Preserve timer.dat and native execution counters.
Final ratios require both independent CPU jobs, GPU repeats, and a CPU-only
same-gauge oracle. The oracle is a validation calculation, not a timing sample.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from phoenix_timing import clean, digest, save

ROLES = ("cpu0", "cpu1", "gpu", "compare")


def resource_policy(role):
    if role not in ROLES:
        raise ValueError("Explicit split role required")
    gpu = role == "gpu"
    return {"threads": 8 if gpu else 48, "memory_GiB": 160 if gpu else 360,
            "wall_seconds": 14400 if gpu else 28800,
            "partition": "gpu-h200" if gpu else "cpu-gnr",
            "constraint": "H200" if gpu else "graniterapids"}


def two_sample_summary(values):
    if len(values) != 2 or any(not 0 < value < float("inf") for value in values):
        raise ValueError("Exactly two positive finite samples required")
    return {"samples_seconds": values, "median_seconds": statistics.median(values),
            "min_seconds": min(values), "max_seconds": max(values)}


def verified_sample_seconds(doc, row, *, pin, job, node, arm, config_sha, input_sha, dimensions):
    if (doc["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
            or doc["source_pin"] != pin or doc["job_id"] != job or doc["hostname"] != node
            or doc["arm"] != arm or doc["runtime_config_sha256"] != config_sha
            or doc["input_manifest_sha256"] != input_sha
            or doc["system"] != "protein83" or not doc["scaled"]
            or doc["basis_profile"] != "jun-cc-pvdz"
            or doc["factor_cache_bytes"] != 0 or doc["dispersion_block_size"] != 24
            or doc["orbital_dimensions"] != dimensions or not doc["benchmark"]
            or row["seconds"] != doc["benchmark_whole_energy_seconds"]):
        raise RuntimeError("Producer scientific input/timing identity differs")
    expected_options = {"basis": "jun-cc-pvdz", "e_convergence": 10,
                        "d_convergence": 12, "puream": True, "use_cuest": arm == "gpu"}
    if any(doc["options"].get(key) != value for key, value in expected_options.items()):
        raise RuntimeError("Producer basis/SCF numerical policy differs")
    return doc["benchmark_whole_energy_seconds"]


def allocation(job, role):
    from validate_cpu_allocation import memory_bytes, seconds
    policy = resource_policy(role)
    if (job.get("Account") != "gts-cs207-chemx" or job.get("QOS") != "embers"
            or job.get("Partition") != policy["partition"] or job.get("NumNodes") != "1"
            or job.get("NumTasks") != "1" or job.get("NumCPUs") != str(policy["threads"])
            or job.get("CPUs/Task") != str(policy["threads"])
            or seconds(job["TimeLimit"]) != policy["wall_seconds"]
            or memory_bytes(job["MinMemoryNode"]) != policy["memory_GiB"] * 1024**3):
        raise RuntimeError("Allocation differs from requested split resource policy")
    gpu = [v for v in job.get("AllocTRES", "").split(",") if v.startswith("gres/gpu")]
    if role == "gpu":
        if "gres/gpu:h200=1" not in gpu:
            raise RuntimeError("One H200 required")
    elif gpu or os.environ.get("SLURM_JOB_GPUS"):
        raise RuntimeError("CPU references and oracle must reserve no GPU")


def execute(manifest_path, root):
    manifest = json.loads(manifest_path.read_text())
    if digest(manifest_path) != os.environ["FSAPT_MANIFEST_SHA256"]:
        raise RuntimeError("Immutable role manifest changed")
    role = manifest["role"]
    policy = resource_policy(role)
    source = Path(manifest["source"])
    pin = manifest["pin"]
    clean(source, pin)
    helper = source / "devtools/disp0/phoenix_validation"
    sys.path[:0] = [str(helper), str(source / "devtools/disp0/cachy")]
    from gpu_runtime import read_config, retirement_channel
    from stage_scaling import require_cpu_runtime_admission
    from run_gate import compare, environment
    from run_full import compare_energy, execute_owned
    from repeat_gauge import compare_repeat
    from full_inputs import load_point
    from prepare_inputs import input_basis_profile
    from fragment_gate import require_oracle_config
    from split_scratch import OwnedScratch
    from validate_cpu_allocation import fields
    from profile_sapt_steps import parse_timers

    os.environ.update(DISP0_SOURCE=str(source), DISP0_PIN=pin)
    report = {"status": "STARTED", "role": role, "pin": pin, "samples": [],
              "processes": [], "accepted_speedup_count": 0,
              "manifest_sha256": digest(manifest_path),
              "sanitizers": "OUTSTANDING_LOADER_INCOMPATIBILITY",
              "scope": "UNTRACED_HF_FSAPT_CPU48_VS_H200_HOST8_NOT_EQUAL_CPU_RESOURCES"}
    checkpoint = lambda: save(root / "split-result.json", report)
    groups, owner = set(), None
    checkpoint()
    try:
        raw = subprocess.check_output(
            ["/opt/slurm/current/bin/scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
            text=True, timeout=30)
        allocation(fields(raw), role)
        report.update(scheduler_job=raw, node=os.uname().nodename,
                      cpu_affinity=sorted(os.sched_getaffinity(0)),
                      cpu_hardware=subprocess.check_output(["lscpu"], text=True, timeout=30))
        if len(report["cpu_affinity"]) < policy["threads"]:
            raise RuntimeError("Insufficient allocated CPU affinity")
        if role == "gpu":
            report["gpu_hardware"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
                 "--format=csv,noheader"], text=True, timeout=30)
            if "H200" not in report["gpu_hardware"]:
                raise RuntimeError("Actual H200 required")
        build_root = Path(manifest["build_root"])
        if digest(build_root / "manifest.json") != manifest["build_manifest_sha256"]:
            raise RuntimeError("Prerequisite build manifest changed")
        build = build_root / ("job-" + manifest["build_job"])
        admitted_path = build / "cpu-sealed-admission/runtime-config.json"
        admitted, proof = read_config(admitted_path, digest(admitted_path))
        if proof["pin"] != pin:
            raise RuntimeError("Exact native build required")
        report["cpu_admission"] = require_cpu_runtime_admission(admitted)
        config_path = root / "runtime-config.json"
        save(config_path, {**admitted, "threads": policy["threads"],
                          "sealed_alias_parent": manifest["alias_parent"]})
        config_sha = digest(config_path)
        config, _ = read_config(config_path, config_sha)
        require_cpu_runtime_admission(config)
        report["runtime_config_sha256"] = config_sha
        inputs = Path(manifest["inputs"])
        _, dimensions = load_point(inputs, manifest["inputs_sha256"], "protein83")
        if input_basis_profile(json.loads(inputs.read_text())) != "jun-cc-pvdz":
            raise RuntimeError("Matched jun-cc-pVDZ profile required")
        if role == "gpu":
            parent = Path("/storage/scratch1/3/awallace43")
        else:
            selected = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
            parent = Path(selected) if selected and Path(selected).is_dir() else Path("/scratch")
        parent = parent.resolve(strict=True)
        scratch = parent / ("fsapt-protein83-" + os.environ["SLURM_JOB_ID"])
        scratch.mkdir(mode=0o700)
        owner = OwnedScratch(scratch, str(scratch), root)
        os.environ.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch))
        mount = fields(subprocess.check_output(
            ["findmnt", "-T", str(scratch), "-n", "-P", "-o", "SOURCE,FSTYPE,TARGET"],
            text=True, timeout=30))
        report["scratch_mount"] = mount
        if role != "gpu" and mount.get("FSTYPE") not in ("xfs", "ext4", "ext3", "btrfs"):
            raise RuntimeError("CPU baseline requires actual node-local scratch")
        disk = os.statvfs(scratch)
        report["scratch_free_bytes"] = disk.f_bavail * disk.f_frsize
        if report["scratch_free_bytes"] < 256 * 1024**3:
            raise RuntimeError("Large-case scratch headroom below 256 GiB")
        deadline = time.monotonic() + policy["wall_seconds"] - 240

        def process(label, argv, env, cap=None):
            row = {"label": label, "argv": argv}
            report["processes"].append(row)
            checkpoint()
            remaining = deadline - time.monotonic()
            if remaining < 60:
                raise TimeoutError("Bounded embers allocation exhausted")
            print("START", label, flush=True)
            with (root / (label + ".stdout")).open("x") as out, (
                    root / (label + ".stderr")).open("x") as err, retirement_channel(env) as channel:
                code = execute_owned(argv, env, out, err, min(remaining, cap or remaining),
                                     groups.add, reaped=groups.discard, pass_fds=channel.pass_fds)
                row.update(returncode=code, retirement_reason=channel.reason())
            checkpoint()
            if code or row["retirement_reason"] is not None:
                raise RuntimeError("Scientific child failed: " + label)
            print("DONE", label, flush=True)
            return row

        def worker(label, oracle=None, peer_config=None, *, water=False, scaled=True):
            arm = "gpu" if role == "gpu" else "cpu"
            output, work = root / label, scratch / label
            work.mkdir()
            aliases = Path(config["sealed_alias_parent"]) / (role + "-" + root.name + "-" + label)
            aliases.mkdir(mode=0o700)
            env = environment(config, gpu=arm == "gpu", sealed_alias_directory=aliases)
            env.update(TMPDIR=str(work), SCRATCH=str(work), PSI_SCRATCH=str(work))
            args = [config["dependency_environment"] + "/bin/python", "-B",
                    str(helper / "sealed_bootstrap.py"), "--config", str(config_path),
                    "--config-sha256", config_sha, "--arm", arm, "--benchmark",
                    "--output", str(output), "--scf-convergence", "10",
                    "--scf-density-convergence", "10" if water else "12",
                    "--block-size", "5" if water else "24", "--factor-cache-bytes", "0"]
            if scaled:
                args.append("--scaled")
            if not water:
                args += ["--inputs", str(inputs), "--inputs-sha256",
                         manifest["inputs_sha256"], "--system", "protein83"]
            if oracle is not None:
                args += ["--oracle-from", str(oracle), "--oracle-config", str(peer_config),
                         "--oracle-config-sha256", digest(peer_config)]
            command = ["/bin/bash", "-c",
                       'set -euo pipefail; export PYTHONPATH="${PYTHONPATH-}"; '
                       'eval "$("$1" --psiapi)"; shift; exec "$@"',
                       "protein83", config["stage"] + "/bin/psi4", *args]
            row = process(label, command, env)
            doc = json.loads((output / "result.json").read_text())
            if (doc["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                    or doc["source_pin"] != pin or doc["arm"] != arm
                    or doc["runtime_config_sha256"] != config_sha
                    or doc.get("system") != (None if water else "protein83")
                    or doc["scaled"] != scaled or not doc["benchmark"]
                    or (not water and doc["orbital_dimensions"] != dimensions)):
                raise RuntimeError("Worker provenance/input mismatch")
            for name, checksum in doc["artifact_sha256"].items():
                if digest(output / name) != checksum:
                    raise RuntimeError("Changed scientific artifact: " + name)
            row.update(path=str(output), result_sha256=digest(output / "result.json"),
                       timer_sha256=digest(output / "timer.dat"),
                       seconds=doc["benchmark_whole_energy_seconds"],
                       timers=parse_timers((output / "timer.dat").read_text()))
            checkpoint()
            return row, doc

        if role == "gpu":
            # CPU controls were executed on the CPU build node. Never run a
            # CPU reference or oracle while reserving this H200.
            report["small_gpu_controls"] = {}
            for mode in ("unscaled", "scaled"):
                reference_path = build / ("cpu-fragment-" + mode) / "result.json"
                reference = json.loads(reference_path.read_text())
                if (reference["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                        or reference["source_pin"] != pin or reference["arm"] != "cpu"):
                    raise RuntimeError("Completed native CPU water control required")
                _, actual = worker("water-" + mode, water=True, scaled=mode == "scaled")
                report["small_gpu_controls"][mode] = {
                    "cpu_result_sha256": digest(reference_path),
                    "energy": compare_energy(reference, actual), "terms": compare(reference, actual)}
                checkpoint()
        if role != "compare":
            for index in range(2 if role == "gpu" else 1):
                row, _ = worker("sample-" + str(index))
                report["samples"].append(row)
                checkpoint()
            if role == "gpu":
                report["repeat"] = compare_repeat(root / "sample-0", root / "sample-1",
                                                  dimensions=dimensions)
            report["status"] = "PASS_SAMPLES_AWAITING_CROSS_ARM"
        else:
            peers = {}
            for peer in ("cpu0", "cpu1", "gpu"):
                directory = Path(manifest["campaign"]) / peer
                job = (directory / "job-id.txt").read_text().strip()
                if (not job.isdecimal() or job != manifest["producer_jobs"][peer]
                        or digest(directory / "manifest.json") !=
                        manifest["producer_manifest_sha256"][peer]):
                    raise RuntimeError("Pinned producer job/manifest changed")
                peer_manifest = json.loads((directory / "manifest.json").read_text())
                for key in ("pin", "source", "campaign", "build_root", "build_job",
                            "build_manifest_sha256", "inputs", "inputs_sha256", "alias_parent",
                            "system", "basis", "scf_E", "scf_D", "cache_bytes", "block_size"):
                    if peer_manifest[key] != manifest[key]:
                        raise RuntimeError("Producer policy differs: " + key)
                folder = directory / ("job-" + job)
                if (folder / "job-status.env").read_text().splitlines() != [
                        "exit_code=0", "term_received=0"]:
                    raise RuntimeError("Uninterrupted producer required: " + peer)
                data = json.loads((folder / "split-result.json").read_text())
                if (data["status"] != "PASS_SAMPLES_AWAITING_CROSS_ARM"
                        or data["pin"] != pin or data["role"] != peer
                        or data["manifest_sha256"] != digest(directory / "manifest.json")
                        or not data["scratch_cleanup"]["complete"]
                        or len(data["samples"]) != (2 if peer == "gpu" else 1)):
                    raise RuntimeError("Matching complete producer required: " + peer)
                allocation(fields(data["scheduler_job"]), peer)
                peer_config_path = folder / "runtime-config.json"
                peer_sha = digest(peer_config_path)
                peer_config, _ = read_config(peer_config_path, peer_sha)
                if data["runtime_config_sha256"] != peer_sha:
                    raise RuntimeError("Producer runtime seal changed")
                if peer == "gpu":
                    require_oracle_config(config, peer_config)
                elif peer_config != config or peer_sha != config_sha:
                    raise RuntimeError("CPU repeat runtime differs from this campaign")
                for index, row in enumerate(data["samples"]):
                    output = folder / ("sample-" + str(index))
                    if (Path(row["path"]) != output
                            or digest(output / "result.json") != row["result_sha256"]
                            or digest(output / "timer.dat") != row["timer_sha256"]):
                        raise RuntimeError("Producer seal changed")
                    doc = json.loads((output / "result.json").read_text())
                    seconds = verified_sample_seconds(doc, row, pin=pin, job=job, node=data["node"],
                        arm="gpu" if peer == "gpu" else "cpu", config_sha=peer_sha,
                        input_sha=manifest["inputs_sha256"], dimensions=dimensions)
                    for name, checksum in doc["artifact_sha256"].items():
                        if digest(output / name) != checksum:
                            raise RuntimeError("Producer matrix seal changed")
                    row["seconds"] = seconds
                peers[peer] = (folder, data)
            cpu = [peers[peer][1]["samples"][0] for peer in ("cpu0", "cpu1")]
            gpu = peers["gpu"][1]["samples"]
            report["cpu_repeat"] = compare_repeat(Path(cpu[0]["path"]), Path(cpu[1]["path"]),
                                                  dimensions=dimensions, arm="cpu")
            report["gpu_repeat"] = compare_repeat(Path(gpu[0]["path"]), Path(gpu[1]["path"]),
                                                  dimensions=dimensions)
            report["pair_checks"] = []
            for left, right in zip(cpu, gpu):
                a = json.loads((Path(left["path"]) / "result.json").read_text())
                b = json.loads((Path(right["path"]) / "result.json").read_text())
                report["pair_checks"].append({"energy": compare_energy(a, b), "terms": compare(a, b)})
            _, oracle = worker("same-gauge-oracle", Path(gpu[0]["path"]),
                               peers["gpu"][0] / "runtime-config.json")
            report["same_gauge_fields"] = oracle["same_gauge_compared_fields"]
            report["same_gauge_errors"] = oracle["same_gauge_errors"]
            report["cpu"] = two_sample_summary([row["seconds"] for row in cpu])
            report["gpu"] = two_sample_summary([row["seconds"] for row in gpu])
            report["speedup"] = report["cpu"]["median_seconds"] / report["gpu"]["median_seconds"]
            report["producer_results"] = {peer: {"path": str(folder / "split-result.json"),
                "sha256": digest(folder / "split-result.json")} for peer, (folder, _) in peers.items()}
            report.update(status="PASS_NUMERICAL_TIMING_SANITIZER_OUTSTANDING",
                          accepted_speedup_count=1)
        clean(source, pin)
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
    source = Path(json.loads(args.manifest.read_text())["source"])
    sys.path.insert(0, str(source / "devtools/disp0/phoenix_validation"))
    from owned_process import termination_signals
    with termination_signals():
        execute(args.manifest, args.output)
