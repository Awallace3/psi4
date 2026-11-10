"""Bounded Cachy diagnosis using an untouched whole installed stage.

This records phase/stack/profile evidence, not accepted timings or numerical
qualification. It neither bypasses admission nor overlays scientific modules.
Run only under the common GPU flock, no HOLD, and the declared resource scope.
"""
import argparse
import cProfile
from contextlib import ExitStack, contextmanager
import faulthandler
import functools
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pstats
import signal
import sys
import threading
import time
from unittest.mock import patch

PIN = "9cc6784336e5c0ab6d1e99180b66e031e1d95a75"
HOLD = Path("/home/amwalla3/.cache/psi4-cachy/gpu0.HOLD")
WATER = """0 1
O -1.551007 -0.114520 0.
H -1.934259 0.762503 0.
H -0.599677 0.040712 0.
--
0 1
O 1.350625 0.111469 0.
H 1.680398 -0.373741 -0.758561
H 1.680398 -0.373741 0.758561
units angstrom
symmetry c1
no_reorient
no_com
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--block", type=int, choices=(5, 24), default=5)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--audit-only", action="store_true",
                        help="Compare old/new mapped-byte audits, without an energy call")
    parser.add_argument("--candidate-source", type=Path)
    args = parser.parse_args()
    if (not args.attempt.replace("-", "").isalnum() or not 30 <= args.seconds <= 180
            or os.environ.get("DISP0_SERIALIZED_GPU_LEASE") != "bounded-trace-v1"
            or HOLD.exists()):
        raise RuntimeError("Explicit bounded serialized lease, absent HOLD and fresh attempt required")
    root = args.root.resolve(strict=True)
    run = root / "runs" / args.attempt
    run.mkdir()
    (run / "scratch").mkdir()
    sys.path.insert(0, str(root / "metadata"))
    result = {"status": "STARTED", "scope": "DIAGNOSTIC_ONLY_NO_H200_OR_SPEEDUP_CLAIM",
              "scientific_pin": PIN, "block_size": args.block, "pid": os.getpid(),
              "deadline_seconds": args.seconds, "installed_science_modified": False,
              "basis": "aug-cc-pvdz", "threads": 4, "phase_statistics": {}}
    profile = cProfile.Profile()
    started = time.monotonic()
    stop = threading.Event()
    log_lock = threading.RLock()

    def expired(signum, frame):
        signal.alarm(0)  # Never interrupt teardown with a second alarm.
        raise TimeoutError("Bounded diagnostic deadline reached")

    def watch_hold():
        while not stop.wait(1):
            if HOLD.exists():
                result["hold_requested_stop"] = True
                os.kill(os.getpid(), signal.SIGINT)
                return

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.seconds)
    threading.Thread(target=watch_hold, daemon=True).start()
    events = (run / "phases.jsonl").open("x", buffering=1)
    stacks = (run / "stacks.txt").open("x", buffering=1)
    faulthandler.enable(file=stacks, all_threads=True)
    faulthandler.dump_traceback_later(15, repeat=True, file=stacks)

    def event(kind, name, **values):
        row = {"event": kind, "phase": name, "seconds": time.monotonic() - started,
               "thread": threading.current_thread().name, **values}
        with log_lock:
            events.write(json.dumps(row, sort_keys=True) + "\n")
        if kind in ("enter", "error") and name not in ("mapped_file_digest",):
            print("[DISP0-DIAG]", json.dumps(row), flush=True)

    def observe(target, name, label=None):
        original = getattr(target, name)
        label = label or name

        @functools.wraps(original)
        def wrapped(*pos, **kw):
            if HOLD.exists():
                raise InterruptedError("Shared GPU HOLD requested")
            t = time.monotonic()
            details = {}
            if label == "mapped_file_digest":
                path = Path(pos[0])
                details = {"file": str(path), "bytes": path.stat().st_size}
            event("enter", label, **details)
            try:
                return original(*pos, **kw)
            except BaseException as error:
                event("error", label, error=repr(error))
                raise
            finally:
                elapsed = time.monotonic() - t
                with log_lock:
                    row = result["phase_statistics"].setdefault(label, {"calls": 0, "inclusive_seconds": 0.})
                    row["calls"] += 1
                    row["inclusive_seconds"] += elapsed
                event("exit", label, elapsed_seconds=elapsed, **details)
        return patch.object(target, name, wrapped)

    env = None
    try:
        profile.enable()
        event("enter", "whole_stage_admission")
        from stage05_native import NativeDiagnostic
        from physical_hf_raw_gpu import MANIFEST_SHA
        env = NativeDiagnostic(root, run, stage_number="cuest-10", source_pin=PIN)
        event("exit", "whole_stage_admission")
        psi4 = env.psi4
        if args.audit_only:
            if args.candidate_source is None:
                raise ValueError("Explicit Git-staged candidate source required for audit-only comparison")
            source = args.candidate_source.resolve(strict=True)
            candidate_path = source / "psi4/driver/procrouting/sapt/sapt_cuest250_identity.py"
            spec = importlib.util.spec_from_file_location("_disp0_standalone_audit_candidate", candidate_path)
            candidate = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(candidate)
            old = importlib.import_module("psi4.driver.procrouting.sapt.sapt_cuest250_identity")
            native_path = str((Path(env.plan["sdk_root"]) / "lib/13/libcuest.so.0.2.50").resolve(strict=True))
            native = next(owner for owner in env.owners
                          if str(Path(owner._name).resolve(strict=True)) == native_path)
            proof = {"native_path": native_path}
            records = []
            # Main-thread-only cProfile would treat serial hashing and worker
            # hashing differently. Disable it for BOTH timed implementations.
            profile.disable()
            try:
                for repeat in range(3):
                    for label, implementation in (
                        (("serial", old), ("concurrent", candidate)) if repeat % 2 == 0 else
                        (("concurrent", candidate), ("serial", old))
                    ):
                        t = time.monotonic()
                        receipt = implementation.audit_maps(native, proof, env.manifest)
                        records.append({"repeat": repeat, "mode": label,
                                        "seconds": time.monotonic()-t, "receipt": receipt})
            finally:
                profile.enable()
            if any(record["receipt"] != records[0]["receipt"] for record in records):
                raise RuntimeError("Paired fresh byte-audit receipts differ")
            result.update(status="PASS_STANDALONE_FRESH_BYTE_AUDIT_COMPARISON_ONLY",
                          audit_samples=records, candidate_source=str(source),
                          audit_profile_policy="cProfile disabled for BOTH timed byte-audit implementations",
                          candidate_module_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                          audit_scope="Separate pure-Python byte audit; no Psi4 module overlay. "
                                      "No integral/energy/kernel workload requested; ELF/runtime "
                                      "initialization remains startup, not numerical execution proof.")
            return
        import einsums
        einsums.rc.gpu_disable = True
        einsums.rc.threads = 4
        einsums.rc.profile_disable = True
        einsums.rc.profile_report = False
        einsums.rc.max_memory = "1GB"
        einsums.rc.buffer_size = "32MB"
        einsums.rc.scratch_dir = str(run / "scratch")
        module = lambda name: importlib.import_module("psi4.driver.procrouting.sapt." + name)
        identity = module("sapt_cuest250_identity")
        dfmo = module("sapt_cuest_dfmo_cuda")
        terms = module("sapt_jk_terms_ein")
        psi4.set_memory("2 GiB")
        psi4.set_num_threads(4)
        psi4.core.IOManager.shared_object().set_default_path(str(run / "scratch"))
        psi4.core.set_output_file(str(run / "psi4.out"), False)
        psi4.set_options({
            "basis": "aug-cc-pvdz", "puream": True,
            "df_basis_scf": "aug-cc-pvdz-jkfit", "df_basis_sapt": "aug-cc-pvdz-jkfit",
            "df_basis_mp2": "aug-cc-pvdz-ri", "reference": "rhf", "scf_type": "df",
            "use_cuest": True, "cuest_mixed_precision": False, "cuest_gemm_min_dim": 0,
            "e_convergence": 10, "d_convergence": 10, "freeze_core": True,
            "sapt_dft_functional": "HF", "sapt_dft_do_fsapt": "SAPTDFT",
            "sapt_dft_use_einsums": True, "sapt_dft_mp2_disp_alg": "SAPT",
            "sapt_dft_do_disp": True, "sapt_dft_do_dhf": False,
            "sapt_dft_induction_type": "CPKS", "fisapt_link_assignment": "C",
            "fisapt_fsapt_ind_response": True, "fisapt_fsapt_exch_scale": False,
            "fisapt_fsapt_ind_scale": False, "ssapt0_scale": False,
            "fisapt_exch_parperp": False, "fisapt_fsapt_filepath": "none"})

        @contextmanager
        def factory(primary, auxiliary):
            with env.open_plan(primary, auxiliary) as owner:
                with dfmo.CuESTDFMOProvider(
                    session=owner.session, sdk_root=Path(env.plan["sdk_root"]),
                    archive=root / "dependencies/cuest-artifact/cuest_v0.2.50.tar.gz",
                    cuda_manifest=env.manifest, cuda_manifest_sha256=MANIFEST_SHA,
                    handle=owner.handle, plan=owner.plan, handle_pid=owner.pid,
                    primary_basis=primary, auxiliary_basis=auxiliary, keepalive=(owner,),
                    plan_policy=dict(dfmo.POLICY), output_order="C", variable_bytes=16*1024**2
                ) as provider:
                    yield provider

        def forbidden(*pos, **kw):
            raise AssertionError("CPU fragment fallback is forbidden in this GPU diagnostic")

        with ExitStack() as patches:
            for name in ("audit_maps", "mapped_file_digest"):
                patches.enter_context(observe(identity, name))
            patches.enter_context(observe(dfmo.CuESTDFMOProvider, "compute_device"))
            patches.enter_context(observe(terms, "fisapt_cuda"))
            for mod, names in (
                # Patch names at their actual call sites: workflow imports
                # these functions by value, not through their origin modules.
                ("sapt_fisapt_workflow_cuda", (
                    "fragment_terms", "prepare_iao_metrics", "build_iao_device",
                    "localize_iao_device", "felst_device", "fexch_device",
                    "coupled_find_device", "prepare_hf", "blocked_device_dispersion")),
                ("sapt_fisapt_localization_cuda", ("jacobi_localize_device",)),
            ):
                target = module(mod)
                for name in names:
                    patches.enter_context(observe(target, name))
            for name in ("DFHelper", "FISAPT", "IBOLocalizer2"):
                patches.enter_context(patch.object(psi4.core, name, side_effect=forbidden))
            event("enter", "public_energy")
            energy, wave = psi4.energy(
                "sapt(dft)", molecule=psi4.geometry(WATER), return_wfn=True,
                cuda_provider_factory=factory, cuda_dispersion_block_size=args.block)
            result["energy_Eh"] = float(energy)
            result["execution"] = dict(wave._sapt_cuda_fisapt_execution)
            result["status"] = "EXECUTED_DIAGNOSTIC_REQUIRES_INDEPENDENT_COMPARISON"
            event("exit", "public_energy")
    except BaseException as error:
        result.update(status="STOPPED_OR_FAILED_DIAGNOSTIC", error=repr(error))
        raise
    finally:
        signal.alarm(0)
        stop.set()
        faulthandler.cancel_dump_traceback_later()
        profile.disable()
        profile.dump_stats(str(run / "profile.pstats"))
        stats = pstats.Stats(profile)
        records = [{"file": f, "line": line, "function": fun, "calls": nc,
                    "exclusive_seconds": tt, "inclusive_seconds": ct}
                   for (f, line, fun), (_, nc, tt, ct, _) in stats.stats.items()]
        result.update(elapsed_seconds=time.monotonic()-started,
                      native_environment=None if env is None else env.report,
                      profile_scope="cProfile main thread only; JSONL hash timings include worker threads",
                      top_exclusive=sorted(records, key=lambda r:r["exclusive_seconds"], reverse=True)[:20],
                      top_cumulative=sorted(records, key=lambda r:r["inclusive_seconds"], reverse=True)[:25])
        (run / "diagnostic-result.json").write_text(json.dumps(result, indent=2) + "\n")
        events.close()
        # Leave the faulthandler descriptor valid through interpreter shutdown.
        print("[DISP0-DIAG]", result["status"], str(run), flush=True)


if __name__ == "__main__":
    main()
