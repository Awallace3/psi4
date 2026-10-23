"""Fresh-process public CPU/GPU HF fragment gate, not an accepted timing run.

Diagnostic snapshots and primitive probes deliberately remain outside benchmark
claims. A completed GPU process alone is NOT independent numerical qualification.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

import numpy as np
from gpu_runtime import AuditedRuntime, read_config


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
FIELDS = ("Qocc0A", "Qocc0B", "Uocc0A", "Uocc0B", "Locc_A", "Locc_B",
          "Elst_AB", "Exch_AB", "IndAB_AB", "IndBA_AB", "Disp_AB",
          "Ind20u_AB", "Ind20u_BA", "ExchInd20u_AB", "ExchInd20u_BA",
          "Ind20r_AB", "Ind20r_BA", "ExchInd20r_AB", "ExchInd20r_BA",
          "Disp20 occupied pairs", "Exch-Disp20 occupied pairs")


def options(gpu, scaled):
    # SAPTDFT selects the independent exact Python CPU fragment continuation.
    # FISAPT would silently select the historically different native CPU route.
    return {"basis": "aug-cc-pvdz", "puream": True,
        "df_basis_scf": "aug-cc-pvdz-jkfit", "df_basis_sapt": "aug-cc-pvdz-jkfit",
        "df_basis_mp2": "aug-cc-pvdz-ri", "reference": "rhf", "scf_type": "df",
        "use_cuest": gpu, "cuest_mixed_precision": False, "cuest_gemm_min_dim": 0,
        "e_convergence": 10, "d_convergence": 10, "freeze_core": True,
        "sapt_dft_functional": "HF", "sapt_dft_do_fsapt": "SAPTDFT",
        "sapt_dft_use_einsums": True, "sapt_dft_mp2_disp_alg": "SAPT",
        "sapt_dft_do_disp": True, "sapt_dft_do_dhf": scaled,
        "sapt_dft_induction_type": "CPKS", "fisapt_link_assignment": "C",
        "fisapt_fsapt_ind_response": True, "fisapt_fsapt_exch_scale": scaled,
        "fisapt_fsapt_ind_scale": scaled, "ssapt0_scale": False,
        "fisapt_exch_parperp": False, "fisapt_do_fsapt": True}


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def batched_probe(runtime):
    """Real cuBLAS transpose/broadcast controls and high-K CUDA reductions."""
    rng = np.random.default_rng(581)
    config = runtime.config
    session = runtime.cuda.CUDASession(
        cublas_library=runtime.blas["path"], cublas_sha256=runtime.blas["sha256"],
        memory_limit_bytes=config["device_memory_limit_bytes"],
        native_persistent_bytes=config["native_persistent_bytes"])
    record = {}
    try:
        with session:
            record["device"] = runtime.device_identity(session)
            for lt in (False, True):
                for rt in (False, True):
                    for broadcast in ("none", "left", "right"):
                        l = rng.normal(size=(3, 2, 4) if not lt else (3, 4, 2))
                        r = rng.normal(size=(3, 4, 5) if not rt else (3, 5, 4))
                        if broadcast == "left": l = l[0]
                        if broadcast == "right": r = r[0]
                        with session.scope():
                            actual = session.gemm_batched(session.upload(l), session.upload(r),
                                transpose_left=lt, transpose_right=rt)
                            expected = np.matmul(np.swapaxes(l, -1, -2) if lt else l,
                                                 np.swapaxes(r, -1, -2) if rt else r)
                            np.testing.assert_allclose(session.download(actual), expected,
                                                       rtol=1e-12, atol=1e-12)
            # Independent NumPy trajectories are implemented in the CPU fixture;
            # this physical control tests high indices, normalization and ordering.
            module = importlib.import_module(
                "psi4.driver.procrouting.sapt.sapt_fisapt_localization_cuda")
            b = np.zeros((2, 513))
            theta = .27
            b[:, 400:402] = [[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]]
            with session.scope():
                localized = module.jacobi_localize_device(session, session.upload(b),
                    session.upload(np.array([-1.3, -.7])), np.arange(513, dtype=np.int64))
                q = session.download(localized.q)
                np.testing.assert_allclose(q.sum(axis=0), np.ones(2), atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(q[:128], 0., atol=1e-14, rtol=0)
                np.testing.assert_allclose(np.sort(q[400:402],axis=0), [[0.,0.],[1.,1.]],
                                           atol=1e-10,rtol=1e-10)
            record["successful_blas_calls"] = dict(session.blas_calls)
            record["successful_kernel_launches"] = dict(session.kernel_launches)
            if session.blas_calls.get("cublasDgemmStridedBatched", 0) < 12:
                raise RuntimeError("Physical batched BLAS controls did not execute")
    finally:
        record["cleanup"] = {"complete": session.cleanup_complete,
            "live_bytes": session.live_bytes, "native_owners": len(session.native_owners)}
        runtime.report["primitive_probe"] = record
    if not session.cleanup_complete or session.live_bytes or session.native_owners:
        raise RuntimeError("Physical primitive control did not clean up")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--arm", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--distance-change", type=float, default=0.)
    parser.add_argument("--scaled", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir()  # never overwrite a previous diagnostic attempt
    os.chdir(args.output)
    result = {"status": "STARTED", "scope": "DIAGNOSTIC_NOT_ACCEPTED_TIMING",
              "arm": args.arm, "scaled": args.scaled, "distance_change": args.distance_change,
              "pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID")}
    runtime = None
    phases, captured = [], {}
    try:
        config, build = read_config(args.config, args.config_sha256)
        import psi4
        from psi4 import core
        if Path(core.__file__).resolve() != Path(build["module_files"]["psi4.core"]):
            raise RuntimeError("Gate imported a different whole-stage core")
        import einsums
        einsums.rc.gpu_disable = True
        if not einsums.rc.gpu_disable:
            raise RuntimeError("Einsums upstream CPU policy was not applied")
        terms = importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms_ein")
        core.set_output_file(str(args.output / "psi4.out"), False)
        psi4.set_memory(config["host_memory"])
        psi4.set_num_threads(config["threads"])
        selected = options(args.arm == "gpu", args.scaled)
        selected["fisapt_fsapt_filepath"] = str(args.output / "fsapt")
        psi4.set_options(selected)
        text = WATER
        if not np.isfinite(args.distance_change) or abs(args.distance_change) > .2:
            raise ValueError("Bounded finite small-case displacement required")
        # Shift every B atom, not only its oxygen.
        lines, side = [], 0
        for line in text.splitlines():
            if line == "--": side = 1
            fields = line.split()
            if side and fields and fields[0] in ("O", "H"):
                fields[1] = str(float(fields[1]) + args.distance_change)
                line = " ".join(fields)
            lines.append(line)
        text = "\n".join(lines) + "\n"
        molecule = psi4.geometry(text)
        result.update(source_pin=config["source_pin"],
            runtime_config_sha256=args.config_sha256, options=selected, geometry=text,
            upstream_einsums_gpu_disabled=bool(einsums.rc.gpu_disable),
            geometry_sha256=hashlib.sha256(text.encode()).hexdigest())

        def capture(cache):
            for name in FIELDS:
                if name in cache:
                    item = cache[name]
                    captured[name] = np.array(getattr(item, "np", item), dtype=np.float64, copy=True)
            for name in ("Disp20,u", "Exch-Disp20,u"):
                if name in cache: result[name] = float(cache[name])

        def forbidden(*args, **kwargs):
            raise RuntimeError("Forbidden CPU fragment integral/localizer fallback")

        with ExitStack() as stack:
            if args.arm == "gpu":
                runtime = AuditedRuntime(args.config, args.config_sha256)
                batched_probe(runtime)
                original = terms.fisapt_cuda
                def fragment(cache, *positional, **kwargs):
                    from psi4.driver.procrouting.sapt.sapt_fisapt_workflow_cuda import HF_FIELDS
                    np.savez(args.output / "raw-hf.npz", **{
                        name: np.array(cache[name].np,copy=True) for name in (
                            *HF_FIELDS, "J_O", "eps_occ_A", "eps_occ_B", "eps_vir_A", "eps_vir_B")})
                    with ExitStack() as guard:
                        for name in ("DFHelper", "FISAPT", "MintsHelper", "IBOLocalizer2", "ExternalPotential"):
                            guard.enter_context(patch.object(core,name,forbidden))
                        for name in ("localization","flocalization","felst","fexch","find","fdisp0"):
                            guard.enter_context(patch.object(terms,name,forbidden))
                        start = time.perf_counter()
                        out, data = original(cache,*positional,**kwargs)
                        phases.append({"name":"GPU full fragment continuation",
                                       "seconds":time.perf_counter()-start})
                    capture(out)
                    result["fragment_execution"] = data["FISAPT execution"]
                    return out,data
                stack.enter_context(patch.object(terms,"fisapt_cuda",fragment))
            else:
                for name in ("localization","partition","flocalization","felst","fexch","find","fdisp0"):
                    original = getattr(terms,name)
                    def timed(*positional,_name=name,_original=original,**kwargs):
                        start=time.perf_counter()
                        out=_original(*positional,**kwargs)
                        phases.append({"name":_name,"seconds":time.perf_counter()-start})
                        if _name=="fdisp0":capture(out)
                        return out
                    stack.enter_context(patch.object(terms,name,timed))
            start = time.perf_counter()
            energy, wfn = psi4.energy("sapt(dft)", molecule=molecule, return_wfn=True,
                **({"cuda_provider_factory":runtime.provider,"cuda_dispersion_block_size":5}
                   if runtime else {}))
            result["diagnostic_whole_energy_seconds"] = time.perf_counter()-start
        for name in ("Elst_AB","Exch_AB","IndAB_AB","IndBA_AB","Disp_AB"):
            if name not in captured or not np.isfinite(captured[name]).all():
                raise RuntimeError("Incomplete/nonfinite full fragment output: " + name)
            np.testing.assert_allclose(captured[name][-1,:],0.,atol=1e-12,rtol=0)
            np.testing.assert_allclose(captured[name][:,-1],0.,atol=1e-12,rtol=0)
        np.savez(args.output / "fragment-matrices.npz", **captured)
        result["signed_matrix_sums"] = {name:float(np.sum(value)) for name,value in captured.items()
                                        if name.endswith(("_AB","_BA"))}
        result["variables"] = {name:float(value) for name,value in core.scalar_variables().items()}
        result["energy_Eh"] = float(energy)
        if runtime:
            text = (args.output / "psi4.out").read_text()
            if "cuESTJK: GPU-Accelerated Density-Fitted" not in text:
                raise RuntimeError("Native cuEST SCF banner missing")
            if "MemDFJK" in text or "DiskDFJK" in text:
                raise RuntimeError("CPU DFJK fallback banner found")
            if not runtime.report["plans"]:
                raise RuntimeError("No audited GPU fragment plan executed")
        result["status"] = "EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
    except BaseException as error:
        result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["phases"] = phases
        result["max_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if runtime: result["runtime"] = runtime.report
        save(args.output / "result.json",result)


if __name__ == "__main__":
    main()
