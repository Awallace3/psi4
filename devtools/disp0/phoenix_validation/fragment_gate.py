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
          "ZA", "ZB",
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


def replay_hf(core, cache, wfn, raw):
    """CPU-only diagnostic: rebuild AO JK independently in the completed GPU gauge."""
    if core.get_global_option("USE_CUEST"):
        raise RuntimeError("Same-gauge oracle must not use cuEST")
    for name in ("S","V_A","V_B"):
        np.testing.assert_allclose(raw[name],cache[name].np,rtol=1e-10,atol=1e-10,
                                   err_msg="Independent native CPU one-electron "+name)
    s=raw["S"]
    for side in ("A","B"):
        co,cv=raw["Cocc_"+side],raw["Cvir_"+side]
        eo,ev=raw["eps_occ_"+side],raw["eps_vir_"+side]
        if (co.ndim!=2 or cv.ndim!=2 or co.shape[0]!=s.shape[0]
                or cv.shape[0]!=s.shape[0] or co.shape[1]+cv.shape[1]!=s.shape[0]
                or eo.shape!=(co.shape[1],) or ev.shape!=(cv.shape[1],)
                or any(not np.isfinite(value).all() for value in (co,cv,eo,ev))
                or np.min(ev)-np.max(eo)<=0):
            raise ValueError("Incomplete finite full HF orbital/energy spaces in diagnostic")
        for coefficients in (co,cv):
            np.testing.assert_allclose(coefficients.T@s@coefficients,
                np.eye(coefficients.shape[1]),rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(co.T@s@cv,0.,atol=1e-8,rtol=0)
        for name,coefficients in (("D_",co),("P_",cv)):
            np.testing.assert_allclose(raw[name+side],coefficients@coefficients.T,
                rtol=1e-8,atol=1e-8,err_msg="Independent HF projector consistency "+name+side)
    for name in raw.files:
        value=np.array(raw[name],copy=True)
        cache[name]=(core.Vector.from_array(value) if value.ndim==1
                     else core.Matrix.from_array(value))
    ca,cb,da,db,s=(cache[n].np for n in ("Cocc_A","Cocc_B","D_A","D_B","S"))
    jk=core.JK.build(wfn.basisset(),wfn.get_basisset("DF_BASIS_SCF"),jk_type="DF")
    jk.set_memory(max(1,int(core.get_memory())//16))
    jk.initialize()
    try:
        for left,right in ((ca,ca),(cb,cb),(db@s@ca,ca),
                           (db@s@da@s@cb,cb),(da@s@db@s@ca,ca)):
            jk.C_left_add(core.Matrix.from_array(left))
            jk.C_right_add(core.Matrix.from_array(right))
        jk.compute()
        js,ks=jk.J(),jk.K()
        rebuilt={"J_A":js[0].np,"K_A":ks[0].np,"J_B":js[1].np,"K_B":ks[1].np,
                 "J_O":js[2].np,"K_O":ks[2].np.T,"J_P_A":js[4].np,"J_P_B":js[3].np}
        for name,value in rebuilt.items():
            if name in raw.files:
                np.testing.assert_allclose(value,raw[name],rtol=2e-7,atol=2e-8,
                                           err_msg="Independent CPU AO JK "+name)
            cache[name]=core.Matrix.from_array(np.array(value,copy=True))
    finally:
        jk.finalize()


def replay_localization(core, cache, wfn, gpu, iao_results):
    """Use GPU L only as oracle comparison gauge, never as a production input.

    Native CPU IAOs independently reconstruct charges for that gauge; native
    CPU JK/DFHelper then calculate every fragment term in the fixed gauge.
    """
    if len(iao_results)!=3:
        raise RuntimeError("Expected one dimer and two native CPU monomer IAO results")
    s=cache["S"].np
    minimal=wfn.get_basisset("MINAO")
    atoms=np.array([minimal.function_to_center(i) for i in range(minimal.nbf())])
    reports={}
    for side,iao in zip(("A","B"),iao_results[-2:]):
        co=cache["Cocc_"+side].np
        locc=np.asarray(gpu["Locc_"+side])
        rotation=co.T@s@locc
        np.testing.assert_allclose(co@rotation,locc,atol=1e-8,rtol=1e-8)
        np.testing.assert_allclose(rotation.T@rotation,np.eye(rotation.shape[0]),atol=1e-8,rtol=1e-8)
        b=locc.T@s@iao
        q=np.zeros_like(gpu["Qocc0"+side])
        for m,atom in enumerate(atoms):q[atom]+=b[:,m]**2
        np.testing.assert_allclose(q,gpu["Qocc0"+side],atol=2e-7,rtol=2e-7,
                                   err_msg="Independent native CPU IAO charges "+side)
        nf=cache["Lfocc0"+side].shape[1]
        # Frozen/active mixing is forbidden, including in diagnostic gauges.
        np.testing.assert_allclose(rotation[:nf,nf:],0.,atol=1e-8,rtol=0)
        np.testing.assert_allclose(rotation[nf:,:nf],0.,atol=1e-8,rtol=0)
        gradient_squared=0.
        for first,last in ((0,nf),(nf,co.shape[1])):
            for i in range(first,last):
                for j in range(first,i):
                    z=np.bincount(atoms,weights=b[i]*b[j],minlength=q.shape[0])
                    pair_gradient=4*np.sum(z*(q[:,i]**3-q[:,j]**3))
                    gradient_squared+=pair_gradient**2
        gradient=float(np.sqrt(gradient_squared))
        if not np.isfinite(gradient) or gradient>1e-8:
            raise RuntimeError("Independent final-gauge CPU IBO stationarity failed")
        reports[side]={"independent_gradient":gradient,
                       "independent_objective":float(np.sum(q**4)**.25)}
        fields={"Locc_"+side:locc,"Uocc_"+side:rotation,"Qocc0"+side:q,
                "Lfocc0"+side:locc[:,:nf],"Laocc0"+side:locc[:,nf:],
                "Ufocc0"+side:rotation[:nf,:nf],"Uaocc0"+side:rotation[nf:,nf:],
                "Caocc0"+side:co[:,nf:]}
        for name,value in fields.items():cache[name]=core.Matrix.from_array(np.array(value,copy=True))
    return reports


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
    parser.add_argument("--oracle-from", type=Path,
                        help="Completed GPU diagnostic directory; CPU comparison only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.oracle_from:
        args.oracle_from=args.oracle_from.resolve(strict=True)
        if args.arm!="cpu":raise ValueError("GPU production cannot consume diagnostic oracle inputs")
    args.output.mkdir()  # never overwrite a previous diagnostic attempt
    os.chdir(args.output)
    result = {"status": "STARTED", "scope": "DIAGNOSTIC_NOT_ACCEPTED_TIMING",
              "arm": args.arm, "scaled": args.scaled, "distance_change": args.distance_change,
              "pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID")}
    runtime = None
    phases, captured = [], {}
    try:
        config, build = read_config(args.config, args.config_sha256)
        # RuntimeTensor lookup during Psi4's SAPT imports initializes Einsums;
        # rc must be configured BEFORE that first lookup, not merely read back.
        import einsums
        einsums.rc.gpu_disable = True
        if not einsums.rc.gpu_disable:
            raise RuntimeError("Einsums upstream CPU policy was not applied")
        import psi4
        from psi4 import core
        if Path(core.__file__).resolve() != Path(build["module_files"]["psi4.core"]):
            raise RuntimeError("Gate imported a different whole-stage core")
        for module in (psi4, einsums, einsums._core):
            if str(Path(module.__file__).resolve())!=build["module_files"][module.__name__]:
                raise RuntimeError("Gate imported a different CPU-qualified package: "+module.__name__)
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
                key = {"Uocc0A":"Uocc_A","Uocc0B":"Uocc_B"}.get(name,name)
                if name in cache or key in cache:
                    item = cache[name] if name in cache else cache[key]
                    captured[name] = np.array(getattr(item, "np", item), dtype=np.float64, copy=True)
            for name in ("Disp20,u", "Exch-Disp20,u"):
                if name in cache: result[name] = float(cache[name])

        def forbidden(*args, **kwargs):
            raise RuntimeError("Forbidden CPU fragment integral/localizer fallback")

        with ExitStack() as stack:
            raw=gpu=None
            iao_results=[]
            if args.oracle_from:
                previous=json.loads((args.oracle_from/"result.json").read_text())
                if (previous["status"]!="EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
                        or previous["arm"]!="gpu" or previous["geometry_sha256"]!=result["geometry_sha256"]
                        or previous["scaled"]!=args.scaled
                        or previous["runtime_config_sha256"]!=args.config_sha256):
                    raise RuntimeError("Oracle inputs differ from requested completed GPU diagnostic")
                for filename in ("raw-hf.npz","fragment-matrices.npz"):
                    if hashlib.sha256((args.oracle_from/filename).read_bytes()).hexdigest()!=previous["artifact_sha256"][filename]:
                        raise RuntimeError("Completed GPU oracle artifact changed: "+filename)
                raw=stack.enter_context(np.load(args.oracle_from/"raw-hf.npz",allow_pickle=False))
                gpu=stack.enter_context(np.load(args.oracle_from/"fragment-matrices.npz",allow_pickle=False))
                result["oracle_input_result_sha256"]=hashlib.sha256(
                    (args.oracle_from/"result.json").read_bytes()).hexdigest()
                original_localizer=core.IBOLocalizer2
                class CaptureIAO:
                    def __init__(self,*args,**kwargs):self.native=original_localizer(*args,**kwargs)
                    def __getattr__(self,name):return getattr(self.native,name)
                    def localize(self,*args,**kwargs):
                        out=self.native.localize(*args,**kwargs)
                        iao_results.append(np.array(out["A"].np,copy=True))
                        return out
                stack.enter_context(patch.object(core,"IBOLocalizer2",CaptureIAO))
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
                    result["Disp20,u"]=float(data["Disp20,u"])
                    result["Exch-Disp20,u"]=float(data["Exch-Disp20,u"])
                    result["fragment_execution"] = data["FISAPT execution"]
                    return out,data
                stack.enter_context(patch.object(terms,"fisapt_cuda",fragment))
            else:
                store_components=terms._store_dispersion0_components
                def components(cache,disp,exch,*positional,**kwargs):
                    out=store_components(cache,disp,exch,*positional,**kwargs)
                    cache["Disp20 occupied pairs"]=np.array(disp,copy=True)
                    cache["Exch-Disp20 occupied pairs"]=np.array(exch,copy=True)
                    return out
                stack.enter_context(patch.object(terms,"_store_dispersion0_components",components))
                for name in ("localization","partition","flocalization","felst","fexch","find","fdisp0"):
                    original = getattr(terms,name)
                    def timed(*positional,_name=name,_original=original,**kwargs):
                        start=time.perf_counter()
                        if raw is not None and _name=="localization":
                            replay_hf(core,positional[0],positional[1],raw)
                        out=_original(*positional,**kwargs)
                        if gpu is not None and _name=="flocalization":
                            result["independent_ibo"]=replay_localization(
                                core,positional[0],positional[1],gpu,iao_results)
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
        missing=set(FIELDS)-captured.keys()
        if missing:raise RuntimeError("Missing required diagnostic fields: "+repr(sorted(missing)))
        for name,key in (("Disp20 occupied pairs","Disp20,u"),
                         ("Exch-Disp20 occupied pairs","Exch-Disp20,u")):
            np.testing.assert_allclose(np.sum(captured[name]),result[key],
                rtol=1e-11,atol=1e-12,err_msg="Signed pair/scalar reduction "+name)
        if args.oracle_from:
            with np.load(args.oracle_from/"fragment-matrices.npz",allow_pickle=False) as target:
                compared=[]
                errors={}
                for name in FIELDS:
                    if name not in target:raise RuntimeError("Missing required GPU oracle field: "+name)
                    np.testing.assert_allclose(captured[name],target[name],rtol=2e-5,atol=2e-7,
                                               err_msg="Same-gauge independent CPU "+name)
                    compared.append(name)
                    difference=captured[name]-target[name]
                    errors[name]={"maximum_absolute_difference":float(np.max(np.abs(difference))),
                                  "signed_sum_difference":float(np.sum(difference))}
                if not all(name in compared for name in (
                        "Elst_AB","Exch_AB","IndAB_AB","IndBA_AB","Disp_AB",
                        "Ind20r_AB","Ind20r_BA","ExchInd20r_AB","ExchInd20r_BA")):
                    raise RuntimeError("Incomplete same-gauge independent CPU fragment comparisons")
                result["same_gauge_compared_fields"]=compared
                result["same_gauge_errors"]=errors
        result["signed_matrix_sums"] = {name:float(np.sum(value)) for name,value in captured.items()
                                        if name.endswith(("_AB","_BA"))}
        result["variables"] = {name:float(value) for name,value in core.scalar_variables().items()}
        result["energy_Eh"] = float(energy)
        file_fields={"ZA":"ZA","ZB":"ZB","QA":"Qocc0A","QB":"Qocc0B",
            "Elst":"Elst_AB","Exch":"Exch_AB","IndAB":"IndAB_AB","IndBA":"IndBA_AB","Disp":"Disp_AB"}
        expected_files={name+".dat" for name in file_fields}|{"geom.xyz","geom.psi4mol"}
        actual_files={p.name for p in (args.output/"fsapt").iterdir() if p.is_file()}
        if actual_files!=expected_files:
            raise RuntimeError("Incomplete/unexpected standard fragment output files")
        for filename,key in file_fields.items():
            value=np.loadtxt(args.output/"fsapt"/(filename+".dat"))
            np.testing.assert_allclose(value,captured[key],rtol=1e-14,atol=1e-15,
                                       err_msg="Fragment file roundtrip "+filename)
        result["output_file_sha256"]={name:hashlib.sha256(
            (args.output/"fsapt"/name).read_bytes()).hexdigest() for name in sorted(expected_files)}
        result["artifact_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in args.output.glob("*.npz")}
        if runtime:
            text = (args.output / "psi4.out").read_text()
            if "cuESTJK: GPU-Accelerated Density-Fitted" not in text:
                raise RuntimeError("Native cuEST SCF banner missing")
            if "MemDFJK" in text or "DiskDFJK" in text:
                raise RuntimeError("CPU DFJK fallback banner found")
            if not runtime.report["plans"]:
                raise RuntimeError("No audited GPU fragment plan executed")
            for plan in runtime.report["plans"]:
                if (plan["producer"]["calls"]<=0 or plan["producer"]["coefficient_h2d_bytes"]!=0
                        or not plan["owner_cleanup"]["cleanup_complete"]):
                    raise RuntimeError("Missing native device-only DFMO work or native cleanup")
                if plan["session_cleanup"]["successful_blas_calls"].get("cublasDgemmStridedBatched",0)<=0:
                    raise RuntimeError("Batched response exchange did not execute on device")
                launches=plan["session_cleanup"]["successful_kernel_launches"]
                if any(launches.get(name,0)<=0 for name in ("response_finish","ibo_pair")):
                    raise RuntimeError("Coupled response/ordered localization device work missing")
            if args.scaled:
                for kind in ("exchange","induction"):
                    factor=result["fragment_execution"]["scaling"][kind]["factor"]
                    if not np.isfinite(factor) or abs(factor-1.)<=1e-8:
                        raise RuntimeError("Scaled gate did not exercise finite nonunit "+kind+" scaling")
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
