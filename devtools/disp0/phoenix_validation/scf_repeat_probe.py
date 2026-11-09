"""Native JK controls and fixed-gauge cache diagnostics, never accepted SAPT timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from stage_scaling import (configuration, digest, pinned_science,
                           require_cpu_runtime_admission, write, cpu_scratch_parent)

SCOPE = "FIXED_DENSITY_NATIVE_JK_DIAGNOSTIC_NOT_SAPT_TIMING_OR_QUALIFICATION"
WALL = 1800
REPEATS = 3
CALLS = 4
PLANS = 3


def clean(folder, pin):
    if (subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"],
                                text=True).strip() != pin
            or subprocess.check_output(["git", "-C", str(folder), "status", "--porcelain"],
                                       text=True).strip()):
        raise RuntimeError("Frozen diagnostic/source checkout changed")


def require_reference(manifest):
    path = Path(manifest["reference_result"])
    if digest(path) != manifest["reference_result_sha256"]:
        raise RuntimeError("Frozen reference result changed")
    doc = json.loads(path.read_text())
    if (doc["status"] != "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
            or doc["arm"] != "gpu" or doc["system"] != "protein83"
            or doc["source_pin"] != manifest["science_pin"]
            or doc["basis_profile"] != "def2-svp"
            or doc["runtime_config_sha256"] != manifest["reference_config_sha256"]
            or doc["input_manifest_sha256"] != manifest["inputs_sha256"]
            or doc["options"]["use_cuest"] is not True
            or doc["options"]["sapt_dft_functional"] != "HF"
            or doc["options"]["cuest_mixed_precision"] is not False):
        raise RuntimeError("Completed matching FP64 Protein83/SVP reference required")
    return doc


def native_device_identity():
    """Verify the native C++ context without creating another cuEST context."""
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    rows = [[v.strip() for v in line.split(",")] for line in processes.splitlines()]
    uuids = {row[1] for row in rows if len(row)==3 and row[0]==str(os.getpid())}
    if len(uuids)!=1:
        raise RuntimeError("One actual native CUDA context UUID required")
    uuid = next(iter(uuids))
    if any(len(row)==3 and row[1]==uuid and row[0]!=str(os.getpid()) for row in rows):
        raise RuntimeError("Foreign compute context on the allocated H200")
    hardware = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits"], text=True, timeout=20)
    devices = [[v.strip() for v in line.split(",")] for line in hardware.splitlines()]
    device = next((row for row in devices if len(row)==5 and row[0]==uuid), None)
    if device is None or "H200" not in device[1] or device[2]!="9.0":
        raise RuntimeError("Actual native CUDA device is not H200/sm90")
    return {"uuid": uuid, "name": device[1], "compute_capability": device[2],
            "driver": device[3], "memory_MiB": int(device[4]), "pid": os.getpid()}


def sample_plan(core, primary, auxiliary, coefficients, *, gpu=True):
    """Same input bytes on every compute; destroy each plan before the next."""
    import numpy as np
    jk = core.JK.build(primary, auxiliary, jk_type="DF")
    initialized = False
    samples = []
    try:
        if (gpu and jk.name()!="cuESTJK") or (
                not gpu and jk.name() not in ("MemDFJK","DiskDFJK")):
            raise RuntimeError("Native cuEST JK required; no CPU fallback" if gpu else
                               "Independent native CPU DFJK required; no GPU allowed")
        if not gpu:
            jk.set_memory(max(1, int(core.get_memory())//16))
        jk.set_omega_alpha(1.)
        jk.set_omega_beta(0.)
        jk.set_do_J(True)
        jk.set_do_K(True)
        jk.print_header()
        jk.initialize()
        initialized = True
        for _ in range(CALLS):
            jk.C_clear()
            matrix = core.Matrix.from_array(coefficients.copy())
            jk.C_left_add(matrix)
            started = time.perf_counter()
            jk.compute()
            elapsed = time.perf_counter()-started
            np.testing.assert_array_equal(matrix.np, coefficients)
            j, k = (np.array(values[0].np, copy=True) for values in (jk.J(), jk.K()))
            shape = (coefficients.shape[0],)*2
            if j.shape != shape or k.shape != shape or not all(
                    np.isfinite(v).all() and np.any(v) for v in (j, k)):
                raise RuntimeError("Incomplete/nonfinite native JK output")
            actual_density = np.array(jk.D()[0].np, copy=True)
            np.testing.assert_allclose(actual_density, coefficients@coefficients.T,
                                       atol=1e-12, rtol=1e-12)
            samples.append((j, k, elapsed, actual_density, jk.name()))
    finally:
        # finalize() destroys the C++ cuEST plan and its workspaces.
        if initialized:
            jk.finalize()
    return samples


def matrix_delta(left, right, density):
    import numpy as np
    dj, dk = right[0]-left[0], right[1]-left[1]
    return {"J_max_abs": float(np.max(np.abs(dj))),
            "K_max_abs": float(np.max(np.abs(dk))),
            "fixed_D_closed_shell_JK_energy_delta_Eh":
                float(2*np.einsum("ij,ji->", density, dj)-np.einsum("ij,ji->", density, dk))}


def require_gpu_control(root, reference_sha):
    if (root/"job-status.env").read_text().splitlines()!=["exit_code=0","term_received=0"]:
        raise RuntimeError("Finalized clean GPU diagnostic required")
    doc=json.loads((root/"diagnostic-result.json").read_text())
    if (doc["status"]!="EXECUTED_FIXED_DENSITY_DIAGNOSTIC_ONLY"
            or doc["scope"]!=SCOPE or doc["accepted_speedup_count"]!=0
            or not doc["scratch_cleanup"]["complete"] or len(doc["processes"])!=REPEATS):
        raise RuntimeError("Matching complete fixed-density GPU diagnostic required")
    for i,row in enumerate(doc["processes"]):
        path=root/f"r{i}/probe-result.json"
        worker=json.loads(path.read_text())
        if (row["returncode"] or row["retirement_reason"] is not None
                or not row["owned_groups_reaped"] or row["result_sha256"]!=digest(path)
                or worker["status"]!="EXECUTED_FIXED_DENSITY_DIAGNOSTIC_ONLY"
                or worker["reference_result_sha256"]!=reference_sha
                or worker["scf_executed"] or worker["fisapt_executed"]
                or "H200" not in worker["device"]["name"]):
            raise RuntimeError("GPU diagnostic worker/identity seal changed")
    return digest(root/"diagnostic-result.json")


def cpu_comparison(root, manifest, cpu_doc):
    import numpy as np
    gpu=Path(manifest["gpu_control"])
    if require_gpu_control(gpu,manifest["reference_result_sha256"])!=manifest["gpu_control_sha256"]:
        raise RuntimeError("GPU diagnostic changed after CPU submission")
    results=[]
    with np.load(root/"r0/fixed-jk.npz",allow_pickle=False) as cpu:
        for repeat in range(REPEATS):
            doc=json.loads((gpu/f"r{repeat}/probe-result.json").read_text())
            path=gpu/f"r{repeat}/fixed-jk.npz"
            if (digest(path)!=doc["matrices_sha256"]
                    or doc["science_pin"]!=manifest["science_pin"]
                    or doc["fixed_coefficients_sha256"]!=cpu_doc["fixed_coefficients_sha256"]):
                raise RuntimeError("GPU matrix/input/source seal changed")
            with np.load(path,allow_pickle=False) as target:
                np.testing.assert_array_equal(cpu["density"],target["density"])
                for plan in range(PLANS):
                    checks={}
                    for kind in ("J","K"):
                        left,right=cpu[f"{kind}_0_0"],target[f"{kind}_{plan}_0"]
                        # Original independent AO-JK oracle tolerance, unchanged.
                        checks[kind]=bool(np.allclose(left,right,rtol=2e-7,atol=2e-8))
                    results.append({"gpu_repeat":repeat,"gpu_plan":plan,
                        **matrix_delta((cpu["J_0_0"],cpu["K_0_0"]),
                            (target[f"J_{plan}_0"],target[f"K_{plan}_0"]),cpu["density"]),
                        "original_AO_JK_oracle_checks":checks})
    return {"scope":"FIXED_GAUGE_NATIVE_CPU_AO_JK_CHECK_ONLY_NOT_FULL_SAPT_ORACLE",
            "difference_direction":"GPU_minus_CPU",
            "rtol":2e-7,"atol":2e-8,"comparisons":results,
            "all_original_AO_JK_checks_passed":
                all(all(row["original_AO_JK_oracle_checks"].values()) for row in results)}


def child(manifest, config, build, output):
    import numpy as np
    from sealed_bootstrap import bootstrap
    from gpu_runtime import AuditedRuntime
    output = output.resolve(strict=True)
    os.chdir(output)
    # Keep the sealed owner alive through all native cleanup. No source overlay.
    owner = bootstrap(config, build)
    runtime = AuditedRuntime(Path(manifest["config"]), manifest["config_sha256"])
    core = runtime.psi4.core
    gpu=manifest["kind"]=="gpu"
    if not gpu and (os.environ.get("CUDA_VISIBLE_DEVICES")!="" or os.environ.get("SLURM_JOB_GPUS")):
        raise RuntimeError("CPU control must hide and allocate no GPUs")
    if manifest.get("mode")=="native-tests":
        from native_jk_regression import run
        runtime.psi4.set_memory(config["host_memory"])
        runtime.psi4.set_num_threads(config["threads"])
        run(runtime,manifest,output)
        return
    if manifest.get("mode")=="cache-profile":
        from fixed_gauge_cache_probe import run
        runtime.guarded_call(run,runtime,manifest,output)
        return
    doc = require_reference(manifest)
    source = Path(doc["options"]["fisapt_fsapt_filepath"]).parent
    if source != Path(manifest["reference_result"]).parent:
        raise RuntimeError("Reference result does not belong to its original output")
    raw = source/"raw-hf.npz"
    if digest(raw) != doc["artifact_sha256"]["raw-hf.npz"]:
        raise RuntimeError("Raw fixed-wavefunction snapshot changed")
    with np.load(raw, allow_pickle=False) as data:
        co, s, density = (np.array(data[name], copy=True) for name in ("Cocc_A", "S", "D_A"))
    dims = doc["orbital_dimensions"]
    if co.shape != (dims["ao"], dims["occupied"][0]) or not np.isfinite(co).all():
        raise RuntimeError("Complete frozen monomer-A coefficients required")
    np.testing.assert_allclose(co.T@s@co, np.eye(co.shape[1]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(density, co@co.T, atol=1e-12, rtol=1e-12)
    runtime.psi4.set_memory(config["host_memory"])
    runtime.psi4.set_num_threads(config["threads"])
    core.set_output_file(str(output/"psi4.out"), False)
    options = dict(doc["options"])
    options["use_cuest"]=gpu
    options["fisapt_fsapt_filepath"] = str(output/"unused-fsapt")
    runtime.psi4.set_options(options)
    molecule = runtime.psi4.geometry(doc["geometry"])
    molecule.update_geometry()
    from full_inputs import load_point
    point, _ = load_point(
        Path(manifest["inputs"]), manifest["inputs_sha256"], "protein83")
    if point["geometry_sha256"] != doc["geometry_sha256"]:
        raise RuntimeError("Reference geometry differs from frozen inputs")
    inputs = json.loads(Path(manifest["inputs"]).read_text())
    for item in inputs["basis_files"].values():
        if digest(Path(core.get_datadir())/"basis"/item["filename"]) != item["sha256"]:
            raise RuntimeError("Installed frozen basis changed")
    primary = core.BasisSet.build(molecule, "BASIS", options["basis"], puream=1)
    auxiliary = core.BasisSet.build(molecule, "DF_BASIS_SCF", options["df_basis_scf"],
                                    "JKFIT", options["basis"], puream=1)
    if (primary.nbf(), auxiliary.nbf()) != (dims["ao"], point["spherical_basis_counts"]["jkfit"]):
        raise RuntimeError("Native primary/auxiliary dimensions changed")
    report = {"status": "STARTED", "scope": SCOPE, "accepted_speedup_count": 0,
              "pid": os.getpid(), "science_pin": manifest["science_pin"],
              "reference_result_sha256": manifest["reference_result_sha256"],
              "requested_df_fitting_condition":float(core.get_option("SCF","DF_FITTING_CONDITION")),
              "fixed_coefficients_sha256": hashlib.sha256(co.tobytes()).hexdigest(),
              "scf_executed": False, "fisapt_executed": False,
              "gpu_production_executed": False, "plans": []}
    matrices = {"density": density}
    try:
        runtime.require_device_admission()
        # No SCF or native JK has run in this fresh process before plan0.
        # Later plans inherit the shared handle's math mode left by plan0.
        # A second warm rebuild separates cold-only effects from rebuild noise.
        for plan in range(PLANS if gpu else 1):
            samples = runtime.guarded_call(sample_plan, core, primary, auxiliary, co, gpu=gpu)
            report["plans"].append({
                "label": ("cold" if plan == 0 else f"warm{plan}") if gpu else "CPU_reference",
                "compute_seconds_debug_only": [v[2] for v in samples],
                "actual_density_sha256": [hashlib.sha256(v[3].tobytes()).hexdigest()
                                          for v in samples],
                "density_max_abs_from_snapshot": [float(np.max(np.abs(v[3]-density)))
                                                  for v in samples],
                "native_jk_name":samples[0][4],
                "within_plan_deltas": [matrix_delta(samples[0], v, density) for v in samples[1:]],
                "native_plan_finalize_returned": True})
            for call, (j, k, _, _, _) in enumerate(samples):
                matrices[f"J_{plan}_{call}"], matrices[f"K_{plan}_{call}"] = j, k
            matrices[f"D_{plan}_0"] = samples[0][3]
            write(output/"probe-result.json", report)
        if gpu:
            report["cold_to_warm"] = [matrix_delta(
                (matrices["J_0_0"], matrices["K_0_0"]),
                (matrices[f"J_{p}_0"], matrices[f"K_{p}_0"]), density)
                for p in range(1, PLANS)]
            report["warm_rebuild_delta"] = matrix_delta(
                (matrices["J_1_0"], matrices["K_1_0"]),
                (matrices["J_2_0"], matrices["K_2_0"]), density)
        report["device"] = native_device_identity() if gpu else {"gpu_used":False,"threads":24}
        np.savez(output/"fixed-jk.npz", **matrices)
        report["matrices_sha256"] = digest(output/"fixed-jk.npz")
        runtime.verify_optional_modules()
        report.update(status="EXECUTED_FIXED_DENSITY_DIAGNOSTIC_ONLY",
                      immutable_runtime=owner.report())
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output/"probe-result.json", report)


def execute(path, worker=False, output=None):
    from gpu_runtime import read_config, retirement_channel
    from run_full import execute_owned
    from run_gate import environment
    from split_scratch import OwnedScratch
    from split_jobs import validate_allocation
    from validate_cpu_allocation import fields
    manifest = json.loads(path.read_text())
    gpu=manifest["kind"]=="gpu"
    native_tests=manifest.get("mode")=="native-tests"
    cache_profile=manifest.get("mode")=="cache-profile"
    if manifest["kind"] not in ("gpu","cpu"):
        raise ValueError("Explicit independent GPU/CPU diagnostic role required")
    if (native_tests or cache_profile) and not gpu:
        raise ValueError("Native/cache GPU diagnostics cannot run on CPU allocation")
    if digest(path) != os.environ["DISP0_MANIFEST_SHA256"]:
        raise RuntimeError("Frozen diagnostic manifest changed")
    clean(Path(manifest["source"]), manifest["science_pin"])
    clean(Path(__file__).resolve().parents[3], manifest["outer_pin"])
    os.environ.update(DISP0_SOURCE=manifest["source"], DISP0_PIN=manifest["science_pin"])
    config, build = read_config(Path(manifest["config"]), manifest["config_sha256"])
    if (config["source_pin"] != manifest["science_pin"]
            or config["threads"] != (8 if gpu else 24)
            or require_cpu_runtime_admission(config) != manifest["cpu_runtime_admission"]):
        raise RuntimeError("Exact finalized CPU admission required")
    if worker:
        child(manifest, config, build, output)
        return
    import numpy as np
    root = Path(os.environ["DISP0_JOB_RUN"]).resolve(strict=True)
    raw = subprocess.check_output(["/opt/slurm/current/bin/scontrol", "show", "job", "-o",
                                   os.environ["SLURM_JOB_ID"]], text=True, timeout=30)
    validate_allocation(fields(raw), manifest)
    if len(os.sched_getaffinity(0)) < config["threads"]:
        raise RuntimeError("Insufficient allocated host CPUs")
    parent=Path("/storage/scratch1/3/awallace43") if gpu else cpu_scratch_parent()
    scratch = parent/("disp0-fixed-jk-"+os.environ["SLURM_JOB_ID"])
    scratch.mkdir(mode=0o700)
    scratch = scratch.resolve(strict=True)
    owner = OwnedScratch(scratch, str(scratch), root)
    os.environ.update(TMPDIR=str(scratch), SCRATCH=str(scratch), PSI_SCRATCH=str(scratch),
                      DISP0_SPLIT_SCRATCH_CREATED=str(scratch))
    groups = set()
    report = {"status": "STARTED", "scope": manifest["scope"], "accepted_speedup_count": 0,
              "kind":manifest["kind"],"science_pin":manifest["science_pin"],
              "outer_pin":manifest["outer_pin"],"manifest_sha256":digest(path),
              "processes": [], "scheduler_job": raw}
    try:
        if not gpu:
            mount=fields(subprocess.check_output(
                ["findmnt","-T",str(scratch),"-n","-P","-o","SOURCE,FSTYPE,TARGET"],
                text=True,timeout=30))
            stat=os.statvfs(scratch)
            if mount.get("FSTYPE") not in ("xfs","ext4","ext3","btrfs") or (
                    stat.f_bavail*stat.f_frsize<256*1024**3):
                raise RuntimeError("CPU control needs actual local scratch and 256GiB headroom")
            report["scratch_mount"]=mount
        deadline = time.monotonic()+WALL-180
        for repeat in range(REPEATS if gpu and not (native_tests or cache_profile) else 1):
            out = root/f"r{repeat}"
            out.mkdir()
            work = scratch/f"r{repeat}"
            work.mkdir()
            aliases = Path(config["sealed_alias_parent"])/f"r{repeat}"
            aliases.mkdir(mode=0o700)
            env = environment(config, gpu=gpu, sealed_alias_directory=aliases)
            env.update(TMPDIR=str(work), SCRATCH=str(work), PSI_SCRATCH=str(work),
                       DISP0_MANIFEST_SHA256=digest(path))
            argv = ["/bin/bash", "-c",
                    'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B "$3" "${@:4}"',
                    "fixed-native-jk", str(Path(config["stage"])/"bin/psi4"),
                    config["dependency_environment"]+"/bin/python", str(Path(__file__).resolve()),
                    "--worker", str(path), "--output", str(out)]
            row = {"repeat": repeat, "argv": argv}
            report["processes"].append(row)
            with (out/"stdout").open("x") as stdout, (out/"stderr").open("x") as stderr, \
                    retirement_channel(env) as channel:
                code = execute_owned(argv, env, stdout, stderr,
                    min(900 if native_tests else 1500 if cache_profile else 500 if gpu else 1500,
                        deadline-time.monotonic()),
                    groups.add, reaped=groups.discard,
                    pass_fds=channel.pass_fds)
                row.update(returncode=code, retirement_reason=channel.reason(),
                           owned_groups_reaped=not groups)
            if code or row["retirement_reason"] is not None or groups:
                raise RuntimeError("Fixed-density diagnostic worker failed")
            doc = json.loads((out/"probe-result.json").read_text())
            if cache_profile:
                from fixed_gauge_cache_probe import verify
                doc=verify(out,manifest)
                row.update(result_sha256=digest(out/"probe-result.json"),pid=doc["pid"])
                report["cache_diagnostic"]=doc
                write(root/"diagnostic-result.json",report)
                continue
            if native_tests:
                from native_jk_regression import junit_summary
                if (doc["status"]!="PASS_NATIVE_JK_GPU_REGRESSION_ONLY"
                        or doc["science_pin"]!=manifest["science_pin"]
                        or doc["native_test_sha256"]!=manifest["native_test_sha256"]
                        or doc["junit_sha256"]!=digest(out/"native-jk.xml")
                        or doc["junit"]!=junit_summary(out/"native-jk.xml")):
                    raise RuntimeError("Incomplete/changed native GPU regression")
                row.update(result_sha256=digest(out/"probe-result.json"),pid=doc["pid"])
                report["native_jk_regression"]=doc
                write(root/"diagnostic-result.json",report)
                continue
            native_log = (out/"psi4.out").read_text()
            if (doc["status"] != "EXECUTED_FIXED_DENSITY_DIAGNOSTIC_ONLY"
                    or doc["science_pin"] != manifest["science_pin"]
                    or doc["reference_result_sha256"] != manifest["reference_result_sha256"]
                    or doc["matrices_sha256"] != digest(out/"fixed-jk.npz")
                    or (gpu and ("cuESTJK: GPU-Accelerated Density-Fitted" not in native_log
                         or native_log.count("cuESTJK compute_JK:") != PLANS*CALLS))
                    or (not gpu and ("cuESTJK:" in native_log or doc["device"]["gpu_used"]))):
                raise RuntimeError("Incomplete/changed fixed-density diagnostic")
            if repeat:
                first_doc = json.loads((root/"r0/probe-result.json").read_text())
                if doc["fixed_coefficients_sha256"] != first_doc["fixed_coefficients_sha256"]:
                    raise RuntimeError("Fixed coefficient bytes changed between processes")
            row.update(result_sha256=digest(out/"probe-result.json"))
            if any(p.get("pid") == doc["pid"] for p in report["processes"][:-1]):
                raise RuntimeError("Distinct fresh process identities required")
            row["pid"] = doc["pid"]
            write(root/"diagnostic-result.json", report)
        if cache_profile:
            report["status"]="PASS_FIXED_GAUGE_CACHE_DIAGNOSTIC"
        elif native_tests:
            gate=root/"small-gate"
            gate.mkdir()
            argv=[sys.executable,"-B",str(Path(manifest["source"])/
                "devtools/disp0/phoenix_validation/run_gate.py"),"--config",manifest["config"],
                "--config-sha256",manifest["config_sha256"],"--output",str(gate)]
            with (root/"small-gate.stdout").open("x") as stdout, (
                    root/"small-gate.stderr").open("x") as stderr:
                code=execute_owned(argv,dict(os.environ),stdout,stderr,
                    deadline-time.monotonic(),groups.add,reaped=groups.discard)
            if code or groups:
                raise RuntimeError("Full small GPU gate failed after native JK tests")
            proof=json.loads((gate/"gate-result.json").read_text())
            if (proof["status"]!="PASS_SMALL_H200_ONLY"
                    or proof["source_pin"]!=manifest["science_pin"]
                    or not proof["gpu_small_case_correctness_verified"]):
                raise RuntimeError("Incomplete/wrong-source small GPU gate")
            report.update(small_gate_sha256=digest(gate/"gate-result.json"),
                          status="PASS_NATIVE_JK_AND_SMALL_GPU_GATES")
        elif gpu:
            report["fresh_process_deltas"] = []
            with np.load(root/"r0/fixed-jk.npz", allow_pickle=False) as first:
                for repeat in range(1, REPEATS):
                    with np.load(root/f"r{repeat}/fixed-jk.npz", allow_pickle=False) as other:
                        np.testing.assert_array_equal(first["density"], other["density"])
                        report["fresh_process_deltas"].append({
                            "repeat": repeat, "plans": [matrix_delta(
                                (first[f"J_{p}_0"], first[f"K_{p}_0"]),
                                (other[f"J_{p}_0"], other[f"K_{p}_0"]), first["density"])
                                for p in range(PLANS)]})
        else:
            report["independent_native_CPU_comparison"]=cpu_comparison(root,manifest,doc)
        if not (native_tests or cache_profile):
            report["status"] = "EXECUTED_FIXED_DENSITY_DIAGNOSTIC_ONLY"
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            report["scratch_cleanup"] = owner.cleanup(groups)
        except BaseException as error:
            report.update(status="FAILED", scratch_cleanup={"complete": False, "error": str(error)})
            raise
        finally:
            write(root/"diagnostic-result.json", report)


def submit(args):
    cache_profile=getattr(args,"mode","fixed-density")=="cache-profile"
    if cache_profile and args.kind!="gpu":
        raise ValueError("Fixed-gauge cache diagnostic requires H200")
    outer = Path(__file__).resolve().parents[3]
    pin = subprocess.check_output(["git", "-C", str(outer), "rev-parse", "HEAD"], text=True).strip()
    clean(outer, pin)
    base_path = args.config.resolve(strict=True)
    base = json.loads(base_path.read_text())
    source = Path(base["source"])
    science_pin = pinned_science(base["source_pin"])
    clean(source, science_pin)
    sys.path.insert(0, str(source/"devtools/disp0/phoenix_validation"))
    admission = require_cpu_runtime_admission(base)
    proof = Path(base["build_proof"])
    if (digest(proof) != base["build_proof_sha256"]
            or json.loads(proof.read_text())["status"] != "PASS"):
        raise RuntimeError("Matching PASS whole build required")
    reference = args.reference.resolve(strict=True)
    if digest(reference) != args.reference_sha256:
        raise RuntimeError("Frozen reference result changed")
    gpu=args.kind=="gpu"
    if not gpu and args.gpu_control is None:
        raise ValueError("Completed fixed-density GPU diagnostic required for CPU comparison")
    root = args.root
    if not root.is_absolute():
        raise ValueError("Fresh absolute diagnostic root required")
    root.mkdir(mode=0o700)
    (root/"logs").mkdir()
    aliases = root/"sealed-workers"
    aliases.mkdir(mode=0o700)
    config = configuration(base, aliases, "protein83", args.kind)
    config_path = root/"runtime-config.json"
    write(config_path, config)
    # Bind the exact original frozen inputs, not a mutable geometry module.
    peer = json.loads((reference.parent.parent.parent/"manifest.json").read_text())
    manifest = {"kind": args.kind, "science_pin": science_pin, "source": str(source),
                "outer_pin": pin, "scope": SCOPE, "automatic_retry": False,
                "accepted_speedup_count": 0, "config": str(config_path),
                "config_sha256": digest(config_path), "cpu_runtime_admission": admission,
                "reference_result": str(reference), "reference_result_sha256": args.reference_sha256,
                "reference_config_sha256": digest(base_path), "inputs": peer["inputs"],
                "inputs_sha256": peer["inputs_sha256"], "account": "gts-cs207-chemx",
                "qos": "embers", "allocation_wall_seconds": WALL}
    if cache_profile:
        from fixed_gauge_cache_probe import SCOPE as CACHE_SCOPE
        manifest.update(mode="cache-profile",scope=CACHE_SCOPE)
    require_reference(manifest)
    from full_inputs import load_point
    _, dimensions = load_point(Path(manifest["inputs"]), manifest["inputs_sha256"], "protein83")
    if json.loads(reference.read_text())["orbital_dimensions"] != dimensions:
        raise RuntimeError("Frozen reference dimensions differ from selected inputs")
    if not gpu:
        target=args.gpu_control.resolve(strict=True)
        manifest.update(gpu_control=str(target),
            gpu_control_sha256=require_gpu_control(target,args.reference_sha256))
    write(root/"manifest.json", manifest)
    schedule(root,manifest,config,outer)


def schedule(root, manifest, config, outer):
    gpu=manifest["kind"]=="gpu"
    pin=manifest["outer_pin"]
    env = {k:v for k,v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
    env.update(DISP0_OUTER=str(outer), DISP0_OUTER_PIN=pin, DISP0_RUN_ROOT=str(root),
               DISP0_GATE_ENV=config["dependency_environment"],
               DISP0_STAGE_THREADS=str(config["threads"]),
               DISP0_MANIFEST_SHA256=digest(root/"manifest.json"),
               DISP0_EXECUTE_DRIVER="scf_repeat_probe.py")
    argv = ["sbatch", "--parsable", "--job-name=fsapt-"+manifest.get("mode","fixed-jk")+"-"+manifest["kind"],
            "--account=gts-cs207-chemx", "--qos=embers",
            "--partition="+("gpu-h200" if gpu else "cpu-small"),
            "--nodes=1", "--ntasks=1",
            "--cpus-per-task="+str(config["threads"]),
            "--mem=160G", "--time=00:30:00", "--no-requeue",
            "--signal=TERM@120", "--export=ALL", "--chdir="+str(root),
            "--output="+str(root/"logs/%x-%j.out"), "--error="+str(root/"logs/%x-%j.err"),
            str(outer/"devtools/disp0/phoenix_validation/stage_scaling.sbatch")]
    argv[1:1]=(["--constraint=H200","--gres=gpu:h200:1"] if gpu else
               ["--exclusive","--constraint=core24&gold6226&localNVMe"])
    for trial, name in ((True, "test-only.json"), (False, "submission.json")):
        if not trial:
            write(root/"SUBMISSION_ATTEMPTED.json", {"command": argv})
        result = subprocess.run([argv[0], "--test-only", *argv[1:]] if trial else argv,
                                env=env, capture_output=True, text=True, timeout=60)
        write(root/name, {"returncode": result.returncode, "stdout": result.stdout,
                         "stderr": result.stderr})
        print(result.stdout+result.stderr, flush=True)
        if result.returncode:
            raise RuntimeError("Scheduler failure/ambiguity; no automatic retry")
    job = result.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError("Ambiguous scheduler receipt")
    (root/"job-id.txt").write_text(job+"\n")
    print("SUBMITTED_JOB="+job+"\nOUTPUT_ROOT="+str(root), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--kind",choices=("gpu","cpu"),default="gpu")
    parser.add_argument("--gpu-control",type=Path)
    parser.add_argument("--mode",choices=("fixed-density","native-tests","cache-profile"),
                        default="fixed-density")
    args = parser.parse_args()
    if args.execute or args.worker:
        path = (args.execute or args.worker).resolve(strict=True)
        manifest = json.loads(path.read_text())
        sys.path.insert(0, str(Path(manifest["source"])/"devtools/disp0/phoenix_validation"))
        from owned_process import termination_signals
        with termination_signals():
            execute(path, worker=bool(args.worker), output=args.output)
    else:
        required=("config","root") if args.mode=="native-tests" else (
            "config","reference","reference_sha256","root")
        if any(getattr(args,key) is None for key in required):
            parser.error("Explicit config/reference/hash/fresh root required")
        if args.mode=="native-tests":
            from native_jk_regression import submit as submit_native
            submit_native(args,schedule)
        else:
            submit(args)


if __name__ == "__main__":
    main()
