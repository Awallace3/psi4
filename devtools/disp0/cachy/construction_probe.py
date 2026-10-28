"""Bounded fresh-process cuEST construction comparison, NOT an energy benchmark.

Requires an already qualified whole local stage. No SCF, DFMO, Python overlay,
cache-budget change or context-sharing optimization is performed.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"phoenix_validation"))
from gpu_runtime import AuditedRuntime, digest, read_config, require_local_lease
from coherent_stage11 import clean, git, lease, policy, run_command, save
from response_reuse_probe import resource_scope


def measure_pair(runtime, primary, jkfit, rifit, threads, report, checkpoint=lambda:None):
    """Construct two independent owned contexts on one session, retaining LIFO.

    Keep this schedule identical in both arms. Sampled free memory is coexistence
    evidence only, not a native peak bound or permission to change the reserve.
    """
    timings = runtime.plans.PhaseTimings()
    report["phase_timings"] = timings.report()
    report["memory_samples"] = []
    owners = []
    session = None

    def phase(name):
        report["last_started_phase"] = name
        checkpoint()
        return timings.measure(name)

    def sample(label):
        runtime.require_device_admission()
        session.synchronize_device()
        free,total = session.rt(session.runtime.cudaMemGetInfo())
        report["memory_samples"].append({
            "label":label,"cuda_free_bytes":int(free),"cuda_total_bytes":int(total),
            "tracked_live_bytes":session.live_bytes,
            "declared_opaque_reserve_bytes":session.native_bytes,
            "sampled_not_peak":True,
        })
        checkpoint()

    def owner(auxiliary, label):
        with phase(label+".api_admission"):
            api = runtime.abi.admit_factory_api(
                session=session,sdk_root=Path(runtime.config["cuest_root"]),
                archive=Path(runtime.config["cuest_archive"]),cuda_manifest=runtime.manifest,
                cuda_manifest_sha256=runtime.config["cuda_manifest_sha256"])
        with phase(label+".owner_admission"):
            result = runtime.plans.DiagnosticPlanOwner(
                api,primary,auxiliary,coefficient_candidate="UNVALIDATED_PSI4_COEF",
                host_workspace_limit=256*1024**2,jit_compile_threads=threads)
        owners.append((label,result))
        return result

    try:
        runtime.require_device_admission()
        with phase("session_creation"):
            session = runtime.cuda.CUDASession(
                cublas_library=runtime.blas["path"],cublas_sha256=runtime.blas["sha256"],
                memory_limit_bytes=runtime.config["device_memory_limit_bytes"],
                native_persistent_bytes=runtime.config["native_persistent_bytes"])
        with session:
            report["device"] = runtime.device_identity(session)
            sample("session_before_native")
            jk = owner(jkfit,"JKFIT")
            with phase("JKFIT.build_and_lifetime"), jk:
                sample("JKFIT_live")
                ri = owner(rifit,"RIFIT")
                with phase("RIFIT.build_and_lifetime"), ri:
                    sample("JKFIT_RIFIT_coexist")
                    for _,item in owners:
                        if item.context_parameters["jit_compile_threads"] != threads:
                            raise RuntimeError("Requested native JIT thread readback differs")
                sample("after_RIFIT_cleanup")
            sample("after_JKFIT_cleanup")
    finally:
        report["phase_timings"] = timings.report()
        report["owners"] = {label:item.report() for label,item in owners}
        if session is not None:
            report["session_cleanup"] = {
                "complete":session.cleanup_complete,"live_bytes":session.live_bytes,
                "native_owners":len(session.native_owners),
                "peak_explicit_bytes":session.peak_live_bytes,
            }
        checkpoint()
    if (not session.cleanup_complete or session.live_bytes or session.native_owners
            or any(not item.cleanup_complete for _,item in owners)):
        raise RuntimeError("Construction diagnostic did not completely unwind")


def build_bases(psi4, geometry, basis, jkfit, rifit):
    """Only CPU geometry/basis metadata: no wavefunction/SCF/integral oracle."""
    molecule = psi4.geometry(geometry)
    molecule.update_geometry()
    primary = psi4.core.BasisSet.build(molecule,"BASIS",basis,puream=1)
    jk = psi4.core.BasisSet.build(molecule,"DF_BASIS_SCF",jkfit,"JKFIT",basis,puream=1)
    ri = psi4.core.BasisSet.build(molecule,"DF_BASIS_MP2",rifit,"RIFIT",basis,puream=1)
    return primary,jk,ri


def basis_provenance(psi4, names):
    directory = (Path(psi4.core.get_datadir())/"basis").resolve(strict=True)
    records = {}
    for role,name in zip(("primary","JKFIT","RIFIT"),names):
        if Path(name).name != name or name in (".",".."):
            raise ValueError("Explicit installed basis names, not paths, required")
        path = (directory/(name.lower()+".gbs")).resolve(strict=True)
        if path.parent != directory:
            raise RuntimeError("Installed basis escaped the selected data directory")
        records[role] = {"path":str(path),"sha256":digest(path)}
    return records


def helper_identity(config, build, root):
    """Launch helpers and installed science have separate, explicit Git pins."""
    filename = Path(__file__).resolve()
    if build["provenance_sha256"].get(str(filename)) != digest(filename):
        raise RuntimeError("Construction helper differs from the qualified byte inventory")
    helper_pin = git(HERE.parents[2],"rev-parse","HEAD")
    clean(HERE.parents[2],helper_pin)
    clean(root/"source",config["source_pin"])
    return helper_pin


def launch_command(config, python, script, arguments):
    # Match the shared gate's authenticated launcher contract. A sanitized
    # child environment alone cannot locate a private whole Psi4 installation.
    return ["/bin/bash","-c",
        'set -euo pipefail; export PYTHONPATH="${PYTHONPATH-}"; '
        'eval "$("$1" --psiapi)"; exec "$2" -B "$3" "${@:4}"',
        "psi4-construction",str(Path(config["stage"])/"bin/psi4"),
        str(python),str(script),*map(str,arguments)]


def child(args):
    # Same authenticated import order as the whole scientific gate; no module
    # is loaded from the source checkout over an older installed core.
    child_wall,child_cpu = time.perf_counter(),time.process_time()
    from bootstrap_gate import bootstrap
    config,build = read_config(args.config,args.config_sha256)
    if config["schema"] != 2 or config["device_family"] != "RTX3090":
        raise RuntimeError("Construction probe is local RTX3090 only")
    helper_pin = helper_identity(config,build,args.root)
    if digest(args.geometry) != args.geometry_sha256:
        raise RuntimeError("Frozen geometry changed")
    timing = None
    report = {
        "status":"RUNNING","scope":"RTX3090_CONSTRUCTION_DEBUG_ONLY_NOT_SPEEDUP_OR_NUMERICAL_GATE",
        "source_pin":config["source_pin"],"runtime_config_sha256":args.config_sha256,
        "identity_contract":config.get("identity_contract","fresh-bytes-v1"),
        "helper_pin":helper_pin,"helper_sha256":digest(__file__),
        "geometry_sha256":args.geometry_sha256,"basis":args.basis,"jkfit":args.jkfit,
        "rifit":args.rifit,"jit_compile_threads":args.threads,"pid":os.getpid(),
        "resource_scope":resource_scope(),
        "cache_policy":"Fresh per-arm CUDA_CACHE_PATH; vendor/library-wide cache state unknown",
        "upstream_scf_context_present":False,
    }
    def checkpoint():
        save(args.output/"result.json",report)
    checkpoint()
    try:
        bootstrap_wall,bootstrap_cpu = time.perf_counter(),time.process_time()
        preloads,events = bootstrap(config,build)
        report["bootstrap_seconds"] = {
            "wall":time.perf_counter()-bootstrap_wall,"process_cpu":time.process_time()-bootstrap_cpu}
        # Import from the admitted installation only, after bootstrap.
        timing = importlib.import_module("psi4.driver.procrouting.sapt.sapt_cuest_timing").PhaseTimings()
        with timing.measure("runtime_admission"):
            runtime = AuditedRuntime(args.config,args.config_sha256)
        runtime.psi4.set_num_threads(4)
        runtime.psi4.set_memory("2 GiB")
        runtime.psi4.core.set_output_file(str(args.output/"psi4.out"),False)
        names = (args.basis,args.jkfit,args.rifit)
        report["basis_files"] = basis_provenance(runtime.psi4,names)
        with timing.measure("geometry_and_basis_metadata"):
            bases = build_bases(runtime.psi4,args.geometry.read_text(),args.basis,args.jkfit,args.rifit)
        report["basis_dimensions"] = dict(zip(("primary","JKFIT","RIFIT"),(int(b.nbf()) for b in bases)))
        with timing.measure("construction_and_cleanup_total"):
            runtime.guarded_call(measure_pair,runtime,*bases,args.threads,report,checkpoint)
        if basis_provenance(runtime.psi4,names) != report["basis_files"]:
            raise RuntimeError("Installed basis files changed during construction")
        runtime.verify_optional_modules()
        if runtime.identity_owner is not None:
            report["immutable_runtime"]=runtime.identity_owner.report()
        report.update(status="PASS_CONSTRUCTION_ONLY",bootstrap_events=events)
    except BaseException as error:
        report.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if timing is not None:
            report["whole_host_phase_timings"] = timing.report()
        report["child_inclusive_seconds"] = {
            "wall":time.perf_counter()-child_wall,"process_cpu":time.process_time()-child_cpu}
        checkpoint()


def compare(records):
    """Reject incomparable paired runs rather than printing a speculative gain."""
    if {row["jit_compile_threads"] for row in records} != {1,4}:
        raise RuntimeError("Both explicit JIT thread settings must complete")
    first = records[0]
    for row in records:
        for name in ("source_pin","helper_pin","helper_sha256","identity_contract",
                     "runtime_config_sha256","geometry_sha256","basis",
                     "jkfit","rifit","basis_dimensions","basis_files","cache_policy"):
            if row[name] != first[name]:
                raise RuntimeError("Construction arm changed "+name)
        if row["status"] != "PASS_CONSTRUCTION_ONLY":
            raise RuntimeError("Incomplete construction arm")
        for role in ("JKFIT","RIFIT"):
            owner,original = row["owners"][role],first["owners"][role]
            for name in ("primary_fingerprint","auxiliary_fingerprint",
                         "plan_parameter_readback","pair_thresholdPQ","coefficient_candidate",
                         "native_version","native_math_mode"):
                if owner[name] != original[name]:
                    raise RuntimeError("Construction arm changed "+role+" "+name)
            if owner["context_parameter_readback"]["jit_compile_threads"] != row["jit_compile_threads"]:
                raise RuntimeError("Construction arm has wrong native thread readback")
            if owner["native_math_mode"] != 1:
                raise RuntimeError("Construction arm is not native FP64")
            if not owner["cleanup_complete"]:
                raise RuntimeError("Construction arm leaked a native owner")
        cleanup = row["session_cleanup"]
        if not cleanup["complete"] or cleanup["live_bytes"] or cleanup["native_owners"]:
            raise RuntimeError("Construction arm leaked a CUDA session")
    return {
        "status":"PASS_PAIRED_CONSTRUCTION_ONLY",
        "scope":"RTX 3090, debug only; no H200 speedup or energy/numerical-parity claim",
        "runs":records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True,help="Private writable run root, not a shared root")
    parser.add_argument("--attempt",required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--geometry",type=Path,required=True)
    parser.add_argument("--geometry-sha256",required=True)
    parser.add_argument("--basis",default="aug-cc-pvdz")
    parser.add_argument("--jkfit",default="aug-cc-pvdz-jkfit")
    parser.add_argument("--rifit",default="aug-cc-pvdz-ri")
    parser.add_argument("--seconds",type=int,default=600,help="Deadline PER fresh process")
    parser.add_argument("--order",nargs="+",type=int,choices=(1,4),default=[1,4])
    parser.add_argument("--child",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--threads",type=int,choices=(1,4),help=argparse.SUPPRESS)
    parser.add_argument("--output",type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root = args.root.resolve(strict=True)
    args.config = args.config.resolve(strict=True)
    args.geometry = args.geometry.resolve(strict=True)
    if (not args.attempt.replace("-","").isalnum() or not 30 <= args.seconds <= 900
            or set(args.order) != {1,4} or not 2 <= len(args.order) <= 4
            or args.root.parent != Path.home()/"gits"
            or args.root.name in ("psi4-disp0-cachy","psi4-saptdft-cuest-cachy","psi4-xdm-cuest-cachy")):
        raise RuntimeError("Private local root, fresh attempt and bounded 1/4-thread comparison required")
    resource_scope()
    if args.child:
        if (args.threads is None or args.output is None
                or args.output.resolve().parent != args.root/"runs"/args.attempt):
            raise RuntimeError("Explicit owned child output required")
        child(args)
        return
    with lease(args.root) as fd:
        config,build = read_config(args.config,args.config_sha256)
        helper_identity(config,build,args.root)
        if config["schema"] != 2 or config["device_family"] != "RTX3090" or config["threads"] != 4:
            raise RuntimeError("Qualified local four-host-thread configuration required")
        if digest(args.geometry) != args.geometry_sha256:
            raise RuntimeError("Frozen geometry changed")
        output = args.root/"runs"/args.attempt
        output.mkdir(mode=0o700)
        records = []
        for index,threads in enumerate(args.order):
            require_local_lease(policy(args.root))
            folder = output/f"arm-{index}-threads-{threads}"
            folder.mkdir()
            (folder/"scratch").mkdir()
            env = {k:os.environ[k] for k in ("HOME","USER","DISP0_LOCAL_LEASE_FD") if k in os.environ}
            env.update(PATH=str(Path(sys.executable).parent)+":/usr/bin:/bin",
                LD_LIBRARY_PATH=config["startup_library_path"],PYTHONNOUSERSITE="1",
                PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",CUDA_VISIBLE_DEVICES=config["local_lease"]["expected_uuid"],
                CUDA_CACHE_PATH=str(folder/"cuda-cache"),PSI_SCRATCH=str(folder/"scratch"),
                TMPDIR=str(folder/"scratch"))
            arguments = ["--root",str(args.root),"--attempt",args.attempt,
                "--config",str(args.config),"--config-sha256",args.config_sha256,
                "--geometry",str(args.geometry),"--geometry-sha256",args.geometry_sha256,
                "--basis",args.basis,"--jkfit",args.jkfit,"--rifit",args.rifit,
                "--seconds",str(args.seconds),"--child","--threads",str(threads),"--output",str(folder)]
            argv = launch_command(config,sys.executable,Path(__file__),arguments)
            save(folder/"launch.json",{"argv":argv,"deadline_seconds":args.seconds,
                "scope":"CONSTRUCTION_ONLY","native_context_sharing":False,
                "config_sha256":args.config_sha256,"geometry_sha256":args.geometry_sha256})
            run_command(argv,args.root,folder,"construction",env,fd,args.seconds)
            records.append(json.loads((folder/"result.json").read_text()))
        save(output/"comparison.json",compare(records))
        print("PASS_PAIRED_CONSTRUCTION_ONLY",output,flush=True)


if __name__ == "__main__":
    main()
