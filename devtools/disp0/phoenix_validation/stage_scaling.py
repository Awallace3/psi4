"""Bounded embers stage measurements around an unchanged qualified installation.

GPU: three unprofiled fresh processes, then one separately traced repeat.
CPU: one fresh reference plus its fixed-GPU-gauge oracle per allocation.
No accepted ratio until all three CPU references also pass repeat qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics
import re

SYSTEMS = ("peptide", "nanotube", "protein83")
SCIENCE_PIN = "927283f8ed0a94d42179c35dd2a1a5bce674347e"
NATIVE_GIB = {"peptide": 12, "nanotube": 12, "protein83": 12}
WALL = {"peptide": 3600, "nanotube": 14400, "protein83": 28800}


def pinned_science(pin):
    if not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{40}", pin) is None:
        raise ValueError("Exact scientific commit SHA required")
    return pin


def cpu_wall_seconds(system, profile):
    if system=="protein83" and profile!="def2-svp":
        raise ValueError("Prior 24-core Protein83 reference exceeded 8h; needs a separate approved plan")
    return 3600 if system=="peptide" else 28800


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def configuration(base, aliases, system, kind):
    if (system not in SYSTEMS or kind not in ("gpu", "cpu")
            or not pinned_science(base["source_pin"]) or base["device_family"] != "H200"
            or base.get("identity_contract") != "process-owned-kernel-sealed-snapshots-v1"):
        raise ValueError("Qualified frozen H200 installation and selected system required")
    return {**base, "threads": 8 if kind == "gpu" else 24,
            "sealed_alias_parent": str(aliases), "host_memory": "96 GiB",
            "device_memory_limit_bytes": 120*1024**3,
            "native_persistent_bytes": NATIVE_GIB[system]*1024**3}


def traced_command(command, tracer, trace):
    """Change only the outer entry point; the authenticated bootstrap stays intact."""
    command = list(command)
    bootstrap = Path(command[6])
    if (command[:2] != ["/bin/bash", "-c"]
            or bootstrap.name != "sealed_bootstrap.py"):
        raise ValueError("Expected original sealed worker command")
    command[6:7] = [str(tracer), "--bootstrap", str(bootstrap),
                    "--trace-output", str(trace), "--include-localization"]
    return command


def require_producer(root, system, checksum, *, science_pin=SCIENCE_PIN):
    if (root/"job-status.env").read_text().splitlines() != ["exit_code=0", "term_received=0"]:
        raise RuntimeError("Finalized unpreempted GPU producer required")
    report = json.loads((root/"stage-result.json").read_text())
    if (report["status"] != "PASS_GPU_REPEATS_AND_TRACE_AWAITING_CPU"
            or not report["scratch_cleanup"]["complete"] or report["system"] != system
            or report.get("source_pin") != science_pin or report.get("kind") != "gpu"
            or report["gpu_config_sha256"] != checksum or len(report["samples"]) != 3):
        raise RuntimeError("Matching complete three-repeat GPU stage producer required")
    pids, values = set(), []
    for index, row in enumerate(report["samples"]):
        path = root/f"{system}-r{index}"
        if Path(row["path"]).resolve() != path.resolve() or digest(path/"result.json") != row["result_sha256"]:
            raise RuntimeError("GPU sample identity/seal changed")
        doc = json.loads((path/"result.json").read_text())
        if (doc["source_pin"] != science_pin or doc["arm"] != "gpu" or doc["system"] != system
                or doc["runtime_config_sha256"] != checksum or not doc["benchmark"]
                or doc["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                or doc["benchmark_whole_energy_seconds"] != row["energy_seconds"]
                or doc["pid"] in pids):
            raise RuntimeError("GPU sample provenance/timing changed")
        pids.add(doc["pid"])
        values.append(float(row["energy_seconds"]))
    if (any(not 0 < value < float("inf") for value in values)
            or report["timing"] != {"n": 3, "median_seconds": statistics.median(values),
                                    "minimum_seconds": min(values), "maximum_seconds": max(values)}):
        raise RuntimeError("GPU timing summary changed")
    return report


def cpu_scratch_parent():
    selected = os.environ.get("TMPDIR") or os.environ.get("SLURM_TMPDIR")
    if not selected:
        raise RuntimeError("Scheduler-provided writable CPU scratch required")
    parent = Path(selected).resolve(strict=True)
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise RuntimeError("Scheduler CPU scratch is not writable")
    return parent


def execute(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    science_pin=pinned_science(manifest["science_pin"])
    source = Path(manifest["source"])
    outer = Path(__file__).resolve().parents[3]
    for folder, pin in ((source, science_pin), (outer, manifest["outer_pin"])):
        if (subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"],
                                    text=True).strip() != pin
                or subprocess.check_output(["git", "-C", str(folder), "status", "--porcelain"],
                                           text=True).strip()):
            raise RuntimeError("Frozen science or outer helper changed")
    # Original proof-authenticated runtime/worker helpers, not new copies.
    helper = source/"devtools/disp0/phoenix_validation"
    sys.path.insert(0, str(helper))
    os.environ.update(DISP0_SOURCE=str(source), DISP0_PIN=science_pin)
    from benchmark_suite import worker_command, summary
    from full_inputs import load_point
    from prepare_inputs import input_basis_profile
    from gpu_runtime import read_config, retirement_channel
    from fragment_gate import require_oracle_config
    from repeat_gauge import compare_repeat
    from run_full import execute_owned, compare_energy
    from run_gate import environment, compare
    from split_jobs import validate_allocation
    from split_scratch import OwnedScratch
    from validate_cpu_allocation import fields
    sys.path.insert(0, str(outer/"devtools/disp0/cachy"))
    from profile_sapt_steps import parse_timers
    from run_fragment_term_profile import decomposition

    root = Path(os.environ["DISP0_JOB_RUN"]).resolve(strict=True)
    report_path = root/"stage-result.json"
    result = {"status": "STARTED", "source_pin": science_pin,
              "outer_pin": manifest["outer_pin"], "system": manifest["system"],
              "kind": manifest["kind"], "accepted_speedup_count": 0,
              "processes": [], "samples": [], "manifest_sha256": digest(manifest_path),
              "scope": "UNPROFILED_BASELINES_PLUS_SEPARATE_SYNCHRONIZED_STAGE_DIAGNOSTIC"}
    checkpoint = lambda: write(report_path, result)
    groups, owner = set(), None
    checkpoint()
    try:
        if digest(manifest_path) != os.environ["DISP0_MANIFEST_SHA256"]:
            raise RuntimeError("Immutable submission manifest changed")
        if manifest.get("prerequisite"):
            prerequisite = Path(manifest["prerequisite"])
            peer = json.loads((prerequisite.parent/"manifest.json").read_text())
            require_producer(prerequisite, "peptide", peer["config_sha256"],
                             science_pin=science_pin)
            result["prerequisite_result_sha256"] = digest(prerequisite/"stage-result.json")
        raw = subprocess.check_output(
            ["/opt/slurm/current/bin/scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]],
            text=True, timeout=30)
        validate_allocation(fields(raw), manifest)
        result["scheduler_job"] = raw
        config, _ = read_config(Path(manifest["config"]), manifest["config_sha256"])
        if config["source_pin"]!=science_pin:
            raise RuntimeError("Runtime and submission scientific pins differ")
        if config["threads"] != (8 if manifest["kind"] == "gpu" else 24):
            raise RuntimeError("Wrong worker thread policy")
        if len(os.sched_getaffinity(0)) < config["threads"]:
            raise RuntimeError("Insufficient allocated CPU affinity")
        system = manifest["system"]
        _, dimensions = load_point(Path(manifest["inputs"]), manifest["inputs_sha256"], system)
        profile=input_basis_profile(json.loads(Path(manifest["inputs"]).read_text()))
        if manifest.get("basis_profile", "aug-cc-pvdz")!=profile:
            raise RuntimeError("Submission basis differs from frozen inputs")
        result["basis_profile"]=profile
        # Bind fresh scratch after srun has rewritten step environment variables.
        parent = (Path("/storage/scratch1/3/awallace43") if manifest["kind"] == "gpu"
                  else cpu_scratch_parent())
        parent = parent.resolve(strict=True)
        scratch = parent/("disp0-stage-"+os.environ["SLURM_JOB_ID"])
        scratch.mkdir(mode=0o700)
        owner = OwnedScratch(scratch, str(scratch), root)
        os.environ.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
                          DISP0_SPLIT_SCRATCH_CREATED=str(scratch))
        if manifest["kind"] == "cpu":
            mount = fields(subprocess.check_output(
                ["findmnt", "-T", str(scratch), "-n", "-P", "-o", "SOURCE,FSTYPE,TARGET"],
                text=True, timeout=30))
            if mount.get("FSTYPE") not in ("xfs", "ext4", "ext3", "btrfs"):
                raise RuntimeError("CPU references need actual node-local scratch")
            stat = os.statvfs(scratch)
            if stat.f_bavail*stat.f_frsize < 256*1024**3:
                raise RuntimeError("CPU scratch headroom below 256 GiB")
            result["scratch_mount"] = mount
        deadline = time.monotonic()+manifest["allocation_wall_seconds"]-180

        def run(label, *, trace=False, oracle=None):
            work = scratch/label
            work.mkdir(mode=0o700)
            output = root/label
            aliases = Path(config["sealed_alias_parent"])/(root.name+"-"+label)
            aliases.mkdir(mode=0o700)
            env = environment(config, gpu=manifest["kind"] == "gpu",
                              sealed_alias_directory=aliases)
            env.update(TMPDIR=str(work), SCRATCH=str(work), PSI_SCRATCH=str(work))
            argv = worker_command(helper, manifest, config, output, system=system, oracle=oracle)
            trace_path = root/(label+"-term-windows.json")
            if trace:
                argv = traced_command(argv, outer/"devtools/disp0/cachy/fragment_term_trace.py",
                                      trace_path)
            row = {"label": label, "argv": argv}
            result["processes"].append(row)
            checkpoint()
            remaining = deadline-time.monotonic()
            if remaining < 60:
                raise TimeoutError("Bounded stage allocation exhausted")
            with (root/(label+".stdout")).open("x") as out, (
                    root/(label+".stderr")).open("x") as err, retirement_channel(env) as channel:
                def launched(pid):
                    groups.add(pid)
                    row["pid"] = pid
                    checkpoint()
                def reaped(pid):
                    groups.discard(pid)
                    row["owned_group_reaped"] = True
                started = time.perf_counter()
                code = execute_owned(argv, env, out, err, remaining, launched, reaped=reaped,
                                     pass_fds=channel.pass_fds)
                reason = channel.reason()
            row.update(returncode=code, inclusive_seconds=time.perf_counter()-started,
                       retirement_reason=reason)
            checkpoint()
            if code or reason is not None:
                raise RuntimeError("Scientific child failed: "+label)
            doc = json.loads((output/"result.json").read_text())
            if (doc["source_pin"] != science_pin or doc["arm"] != manifest["kind"]
                    or doc.get("system") != system or not doc["benchmark"]
                    or doc["runtime_config_sha256"] != manifest["config_sha256"]
                    or doc["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                    or doc["orbital_dimensions"] != dimensions):
                raise RuntimeError("Worker provenance/geometry changed")
            for name, checksum in doc["artifact_sha256"].items():
                if digest(output/name) != checksum:
                    raise RuntimeError("Worker artifact seal changed")
            row.update(path=str(output), result_sha256=digest(output/"result.json"),
                       timer_sha256=digest(output/"timer.dat"),
                       psi4_out_sha256=digest(output/"psi4.out"),
                       timers=parse_timers((output/"timer.dat").read_text()),
                       energy_seconds=doc["benchmark_whole_energy_seconds"])
            if trace:
                observation = json.loads(trace_path.read_text())
                if observation["status"] != "PASS_OBSERVED_TERM_BOUNDARIES":
                    raise RuntimeError("Incomplete GPU term trace")
                full = row["timers"]["driver_steps"]["SAPT(DFT): GPU F-SAPT"]["wall_seconds"]
                row.update(term_trace=observation["trace"], trace_sha256=digest(trace_path),
                           decomposition=decomposition(observation["trace"], full))
            checkpoint()
            return output, doc, row

        if manifest["kind"] == "gpu":
            # Every selected larger case has its own short full correctness gate.
            gate = root/"small-gate"
            gate.mkdir()
            env = dict(os.environ)
            with (root/"small-gate.stdout").open("x") as out, (
                    root/"small-gate.stderr").open("x") as err:
                code = execute_owned(
                    [sys.executable, "-B", str(helper/"run_gate.py"), "--config", manifest["config"],
                     "--config-sha256", manifest["config_sha256"], "--output", str(gate)],
                    env, out, err, min(3500, deadline-time.monotonic()), groups.add,
                    reaped=groups.discard)
            proof = json.loads((gate/"gate-result.json").read_text()) if not code else {}
            if (code or proof.get("status") != "PASS_SMALL_H200_ONLY"
                    or not proof.get("gpu_small_case_correctness_verified")):
                raise RuntimeError("Small H200 gate failed; no larger benchmark")
            result["small_gate_sha256"] = digest(gate/"gate-result.json")
            baseline = None
            for i in range(3):
                output, doc, row = run(f"{system}-r{i}")
                if baseline is not None:
                    row["repeat_check"] = compare_repeat(baseline, output,
                                                        dimensions=dimensions, arm="gpu")
                else:
                    baseline = output
                result["samples"].append(row.copy())
                checkpoint()
            result["timing"] = summary([s["energy_seconds"] for s in result["samples"]])
            output, doc, row = run(system+"-traced", trace=True)
            row["repeat_check"] = compare_repeat(baseline, output, dimensions=dimensions, arm="gpu")
            result["trace"] = row
            result["gpu_config_sha256"] = manifest["config_sha256"]
            result["status"] = "PASS_GPU_REPEATS_AND_TRACE_AWAITING_CPU"
        else:
            gpu_root = Path(manifest["gpu_job_root"]).resolve(strict=True)
            if (digest(gpu_root/"stage-result.json") != manifest["gpu_producer_sha256"]
                    or digest(gpu_root.parent/"manifest.json") != manifest["gpu_manifest_sha256"]):
                raise RuntimeError("Finalized GPU producer/manifest changed after CPU submission")
            peer = require_producer(gpu_root, system, manifest["gpu_config_sha256"],
                                    science_pin=science_pin)
            gpu_config, _ = read_config(Path(manifest["gpu_config"]),
                                       manifest["gpu_config_sha256"])
            require_oracle_config(config, gpu_config)
            original = peer["samples"][manifest["repeat"]]
            gpu_path = Path(original["path"])
            if digest(gpu_path/"result.json") != original["result_sha256"]:
                raise RuntimeError("Finalized GPU sample changed")
            gpu = json.loads((gpu_path/"result.json").read_text())
            for name, checksum in gpu["artifact_sha256"].items():
                if digest(gpu_path/name) != checksum:
                    raise RuntimeError("GPU oracle input changed")
            output, cpu, reference = run(system+"-cpu")
            options = []
            for doc in (cpu, gpu):
                selected = dict(doc["options"])
                backend = selected.pop("use_cuest")
                if backend != (doc["arm"] == "gpu"):
                    raise RuntimeError("Wrong native CPU/GPU engine")
                selected.pop("fisapt_fsapt_filepath")
                options.append(selected)
            if options[0] != options[1] or any(cpu[k] != gpu[k] for k in
                    ("retained_orbital_rank", "virtual_orbitals")):
                raise RuntimeError("CPU/GPU scientific options/orbital spaces differ")
            _, oracle, _ = run(system+"-oracle", oracle=gpu_path)
            result.update(reference=reference, gpu_sample=original,
                          energy_check=compare_energy(cpu, gpu), signed_checks=compare(cpu, gpu),
                          same_gauge_errors=oracle["same_gauge_errors"],
                          status="PASS_CPU_REFERENCE_AND_ORACLE_NEEDS_THREE_REPEAT_AGGREGATION")
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if owner is not None:
                result["scratch_cleanup"] = owner.cleanup(groups)
            else:
                result["scratch_cleanup"] = {"complete": not groups, "created": False}
        except BaseException as error:
            result.update(status="FAILED", scratch_cleanup={"complete": False, "error": str(error)})
            raise
        finally:
            checkpoint()


def submit(args):
    if args.system not in SYSTEMS:
        raise ValueError("Protein157 and unrequested systems are excluded")
    base_path = args.base_config.resolve(strict=True)
    base = json.loads(base_path.read_text())
    science_pin=pinned_science(base["source_pin"])
    source = Path(base["source"]).resolve(strict=True)
    outer = Path(__file__).resolve().parents[3]
    pin = subprocess.check_output(["git", "-C", str(outer), "rev-parse", "HEAD"], text=True).strip()
    for folder, expected in ((outer, pin), (source, science_pin)):
        if (subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"],
                                    text=True).strip() != expected
                or subprocess.check_output(["git", "-C", str(folder), "status", "--porcelain"],
                                           text=True).strip()):
            raise RuntimeError("Clean frozen helper/source required")
    sys.path.insert(0, str(source/"devtools/disp0/phoenix_validation"))
    from full_inputs import load_point
    from prepare_inputs import input_basis_profile
    # Login-node preflight reads small immutable proof documents only.
    # Full prefix/CUDA/ELF authentication is repeated inside the allocation.
    proof_path = Path(base["build_proof"])
    proof = json.loads(proof_path.read_text())
    if (proof.get("status") != "PASS" or proof.get("pin") != science_pin
            or digest(proof_path) != base["build_proof_sha256"]):
        raise RuntimeError("Completed exact whole-stage CPU build proof required")
    load_point(args.inputs, args.inputs_sha256, args.system)
    profile=input_basis_profile(json.loads(args.inputs.read_text()))
    wall = WALL[args.system] if args.kind=="gpu" else cpu_wall_seconds(args.system,profile)
    root = args.root
    if not root.is_absolute():
        raise ValueError("Fresh absolute output root required")
    root.mkdir(mode=0o700)
    (root/"logs").mkdir()
    aliases = root/"sealed-workers"
    aliases.mkdir(mode=0o700)
    config = configuration(base, aliases.resolve(strict=True), args.system, args.kind)
    gpu_root = None
    if args.kind == "cpu":
        if args.gpu_job_root is None:
            raise ValueError("Finalized GPU producer required before CPU submission")
        gpu_root = args.gpu_job_root.resolve(strict=True)
        peer_manifest = json.loads((gpu_root.parent/"manifest.json").read_text())
        if peer_manifest["inputs_sha256"] != args.inputs_sha256:
            raise RuntimeError("CPU/GPU frozen input manifests differ")
        config_path = Path(peer_manifest["config"])
        peer = json.loads(config_path.read_text())
        require_producer(gpu_root, args.system, digest(config_path), science_pin=science_pin)
        config = {**peer, "threads": 24}
    config_path = root/"runtime-config.json"
    write(config_path, config)
    manifest = {"source": str(source), "science_pin": science_pin, "outer_pin": pin,
                "basis_profile": profile,
                "kind": args.kind, "system": args.system, "repeat": args.repeat,
                "config": str(config_path), "config_sha256": digest(config_path),
                "inputs": str(args.inputs.resolve(strict=True)), "inputs_sha256": args.inputs_sha256,
                "account": "gts-cs207-chemx", "qos": "embers", "allocation_wall_seconds": wall,
                "automatic_retry": False, "base_config": str(base_path),
                "base_config_sha256": digest(base_path)}
    if gpu_root is not None:
        manifest.update(gpu_job_root=str(gpu_root), gpu_config=peer_manifest["config"],
                        gpu_config_sha256=peer_manifest["config_sha256"],
                        gpu_producer_sha256=digest(gpu_root/"stage-result.json"),
                        gpu_manifest_sha256=digest(gpu_root.parent/"manifest.json"))
    parent = None
    if args.prerequisite is not None:
        prior = args.prerequisite.resolve(strict=True)
        prior_manifest = json.loads((prior/"manifest.json").read_text())
        job = (prior/"job-id.txt").read_text().strip()
        if (not job.isdecimal() or args.kind != "gpu" or args.system == "peptide"
                or prior_manifest["system"] != "peptide" or prior_manifest["kind"] != "gpu"
                or prior_manifest["outer_pin"] != pin or prior_manifest["science_pin"] != science_pin):
            raise ValueError("Same-helper peptide GPU preflight required")
        prerequisite = prior/("job-"+job)
        manifest["prerequisite"] = str(prerequisite)
        if (prerequisite/"job-status.env").exists():
            require_producer(prerequisite, "peptide", prior_manifest["config_sha256"],
                             science_pin=science_pin)
        else:
            parent = job
    write(root/"manifest.json", manifest)
    gpu = args.kind == "gpu"
    argv = ["sbatch", "--parsable", "--job-name=fsapt-stage-"+args.system+"-"+args.kind,
            "--account=gts-cs207-chemx", "--qos=embers",
            "--partition="+("gpu-h200" if gpu else "cpu-small"),
            "--nodes=1", "--ntasks=1", "--cpus-per-task="+("8" if gpu else "24"),
            "--mem=160G", "--time="+f"{wall//3600:02d}:00:00", "--no-requeue",
            "--signal=TERM@120", "--export=ALL", "--chdir="+str(root),
            "--output="+str(root/"logs/%x-%j.out"), "--error="+str(root/"logs/%x-%j.err")]
    argv += (["--constraint=H200", "--gres=gpu:h200:1"] if gpu else
             ["--exclusive", "--constraint=core24&gold6226&localNVMe"])
    argv += [str(Path(__file__).with_suffix(".sbatch"))]
    if parent is not None:
        argv[1:1] = ["--dependency=afterany:"+parent, "--kill-on-invalid-dep=yes"]
    env = {k:v for k,v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
    env.update(DISP0_OUTER=str(outer), DISP0_OUTER_PIN=pin, DISP0_RUN_ROOT=str(root),
               DISP0_GATE_ENV=config["dependency_environment"],
               DISP0_STAGE_THREADS=str(config["threads"]),
               DISP0_MANIFEST_SHA256=digest(root/"manifest.json"))
    for trial, name in ((True, "test-only.json"), (False, "submission.json")):
        if not trial:
            write(root/"SUBMISSION_ATTEMPTED.json", {"command": argv})
        response = subprocess.run([argv[0], "--test-only", *argv[1:]] if trial else argv,
                                  env=env, capture_output=True, text=True, timeout=60)
        write(root/name, {"returncode": response.returncode,
                         "stdout": response.stdout, "stderr": response.stderr})
        print(response.stdout+response.stderr, flush=True)
        if response.returncode:
            raise RuntimeError("Scheduler failure/ambiguity: no automatic retry")
    job = response.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError("Ambiguous job ID; inspect receipt before further submission")
    (root/"job-id.txt").write_text(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(root), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--kind", choices=("gpu", "cpu"))
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--repeat", type=int, choices=range(3), default=0)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inputs-sha256")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--gpu-job-root", type=Path)
    parser.add_argument("--prerequisite", type=Path)
    args = parser.parse_args()
    if args.execute:
        path = args.execute.resolve(strict=True)
        if digest(path) != os.environ["DISP0_MANIFEST_SHA256"]:
            raise RuntimeError("Submitted manifest changed")
        source = Path(json.loads(path.read_text())["source"])
        sys.path.insert(0, str(source/"devtools/disp0/phoenix_validation"))
        from owned_process import termination_signals
        with termination_signals():
            execute(path)
    else:
        if any(getattr(args, k) is None for k in
               ("kind", "system", "base_config", "inputs", "inputs_sha256", "root")):
            parser.error("Explicit kind/system/base config/inputs/hash/fresh root required")
        submit(args)


if __name__ == "__main__":
    main()
