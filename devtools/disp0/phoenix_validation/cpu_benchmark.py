"""One fresh production-scaled CPU full F-SAPT call; no GPU or exact-source solve."""
import argparse
from contextlib import ExitStack
import functools
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from unittest.mock import patch

import numpy as np
from fragment_gate import options, WATER
from gpu_runtime import digest, read_config
from prepare_inputs import SYSTEMS

ARRAYS = {"QA":"FSAPT_QA","QB":"FSAPT_QB","Elst":"FSAPT_ELST_AB",
          "Exch":"FSAPT_EXCH_AB","IndAB":"FSAPT_INDAB_AB","IndBA":"FSAPT_INDBA_AB",
          "Disp":"FSAPT_DISP_AB"}
SCALARS=("Elst10,r","Exch10","Ind20,r","Exch-Ind20,r","Delta HF Correction",
         "Disp20,u","Exch-Disp20,u","Ind20,r (A<-B)","Ind20,r (A->B)",
         "Exch-Ind20,r (A<-B)","Exch-Ind20,r (A->B)")


def production_options(arm):
    if arm not in ("einsums","cpp"):raise ValueError("Explicit CPU backend required")
    value=options(False,True)
    # The numerical GPU gate's tightened screening is diagnostic-only. Keep
    # the established production CPU benchmark at its original native default.
    value.pop("ints_tolerance")
    value.update(sapt_dft_do_fsapt="SAPTDFT" if arm=="einsums" else "FISAPT",
        fisapt_fsapt_ind_response=False,fisapt_fdisp_algorithm="BLOCKED",fisapt_fdisp_block=48,
        fisapt_do_fsapt_disp=True,sapt_dft_do_ddft=False,sapt_dft_d3_ie=False,
        sapt_dft_d4_ie=False,sapt_dft_einsums_gpu=False,minao_basis="CC-PVTZ-MINAO")
    return value


def validate_component_ledger(sums,canonical,energy):
    """Check each fitted partition against its own canonical target.

    DFHelper and JK evaluate localized/canonical Coulomb contributions through
    different fitted paths. Their existing per-component accuracy must not be
    applied instead to a much smaller, cancellation-dominated net energy.
    This never alters a matrix, loosens a component tolerance or renormalizes
    a result. The canonical algebra and cross-backend matrix checks stay strict.
    """
    base=canonical["Ind20,r"]+canonical["Exch-Ind20,r"]
    if not np.isfinite(base) or base==0:
        raise ValueError("Finite nonzero canonical induction scaling denominator required")
    scale=(base+canonical["Delta HF Correction"])/base
    targets={"Elst":canonical["Elst10,r"],"Exch":canonical["Exch10"],
             "Disp":canonical["Disp20,u"]+canonical["Exch-Disp20,u"]}
    for side,direction in (("AB","A<-B"),("BA","A->B")):
        targets["Ind"+side]=scale*(canonical[f"Ind20,r ({direction})"]+
                                   canonical[f"Exch-Ind20,r ({direction})"])
    np.testing.assert_allclose(sum(targets.values()),float(energy),atol=2e-8,rtol=1e-8,
                               err_msg="Canonical component algebra vs interaction energy")
    residuals={}
    for name,target in targets.items():
        atol,rtol=(2e-8,1e-8) if name=="Disp" else (2e-7,2e-5)
        np.testing.assert_allclose(sums[name],target,atol=atol,rtol=rtol,
                                   err_msg="DF consistency of fragment "+name)
        residuals[name]=sums[name]-target
    return {"policy":"UNCHANGED_COMPONENT_TOLERANCES_CANONICAL_ALGEBRA_STRICT",
        "raw_fragment_sum_residual_Eh":sum(sums.values())-float(energy),
        "canonical_sum_residual_Eh":sum(targets.values())-float(energy),
        "component_residual_Eh":residuals,
        "note":"Reported DF consistency residuals are not corrections to the raw matrices."}


class PhaseTape:
    """Exclusive totals are additive even when public setup invokes a timed leaf."""
    def __init__(self,clock=time.perf_counter):
        self.clock=clock;self.stack=[];self.records=[];self.matrices={}

    def keep(self,cache):
        for target,names in (("S",("S",)),("Locc_A",("Locc_A","Locc0A")),
                             ("Locc_B",("Locc_B","Locc0B"))):
            for name in names:
                if name in cache:
                    self.matrices[target]=cache[name]
                    break

    def wrap(self,name,function,*,capture=None):
        @functools.wraps(function)
        def timed(*args,**kwargs):
            frame={"start":self.clock(),"children":0.}
            self.stack.append(frame)
            try:
                value=function(*args,**kwargs)
                if capture is not None:capture(value,args)
                return value
            finally:
                elapsed=self.clock()-frame["start"]
                self.stack.pop()
                if self.stack:self.stack[-1]["children"]+=elapsed
                self.records.append({"name":name,"inclusive_seconds":elapsed,
                                     "exclusive_seconds":elapsed-frame["children"]})
        return timed


def main():
    process_start=time.perf_counter()
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--inputs-sha256",required=True)
    parser.add_argument("--system",choices=(*SYSTEMS,"water-preflight"),required=True)
    parser.add_argument("--arm",choices=("einsums","cpp"),required=True)
    parser.add_argument("--threads",type=int,required=True)
    parser.add_argument("--memory",required=True)
    parser.add_argument("--repeat",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir()
    os.chdir(root)
    result={"status":"STARTED","scope":"CPU_PRODUCTION_SCALED_FULL_FSAPT",
        "arm":args.arm,"system":args.system,"repeat":args.repeat,"pid":os.getpid(),
        "job_id":os.environ.get("SLURM_JOB_ID"),"threads":args.threads,
        "psi4_memory":args.memory,"input_manifest_sha256":args.inputs_sha256,
        "gpu_speedup_eligible":False}
    tape=PhaseTape()
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or os.environ.get("SLURM_JOB_GPUS"):
            raise RuntimeError("CPU benchmark requires hidden GPUs and a CPU-only allocation")
        config,build=read_config(args.config,args.config_sha256)
        if digest(args.inputs)!=args.inputs_sha256:raise RuntimeError("Frozen inputs changed")
        inputs=json.loads(args.inputs.read_text())
        if set(inputs["systems"])!=set(SYSTEMS):raise RuntimeError("Only the three approved systems admitted")
        import einsums as ein
        ein.rc.gpu_disable=True
        ein.rc.profile_disable=True
        ein.rc.profile_report=False
        import psi4
        from psi4 import core
        for module in (psi4,core,ein,ein._core):
            if str(Path(module.__file__).resolve())!=build["module_files"][module.__name__]:
                raise RuntimeError("Foreign CPU runtime module: "+module.__name__)
        import einsums.sealed as sealed
        world=sealed.world()
        if (world["git_commit"]!="06404fa69ce9346443d7d4f19a250f6db9f0f62c"
                or len(sealed.mapped_einsums_libraries())!=1
                or str(Path(world["library_path"]).resolve()) not in build["binary_sha256"]):
            raise RuntimeError("Wrong or mixed optimized Einsums native world")
        terms=importlib.import_module("psi4.driver.procrouting.sapt.sapt_jk_terms_ein")
        sapt_proc=importlib.import_module("psi4.driver.procrouting.sapt.sapt_proc")
        setup=importlib.import_module("psi4.driver.procrouting.sapt.saptdft_fisapt")
        response=importlib.import_module("psi4.driver.procrouting.sapt.sapt_fisapt_cpu_response")
        def forbidden(*args,**kwargs):
            raise RuntimeError("Exact per-source fragment solve is forbidden in the production-scaled benchmark")
        core.set_output_file(str(root/"psi4.out"),False)
        psi4.set_num_threads(args.threads);psi4.set_memory(args.memory)
        if core.get_num_threads()!=args.threads:raise RuntimeError("CPU thread setting did not apply")
        selected=production_options(args.arm)
        selected["fisapt_fsapt_filepath"]="none"
        psi4.set_options(selected)
        if core.get_option("FISAPT","FISAPT_FSAPT_IND_RESPONSE"):
            raise RuntimeError("Production scalar-rescaled fragment mode was not selected")
        for role,item in inputs["basis_files"].items():
            path=Path(core.get_datadir())/"basis"/item["filename"]
            if digest(path)!=item["sha256"]:
                raise RuntimeError("Installed basis file differs from frozen input: "+role)
        point=None if args.system=="water-preflight" else inputs["systems"][args.system]
        geometry=WATER if point is None else point["geometry"]
        if point is not None and digest_text(geometry)!=point["geometry_sha256"]:
            raise RuntimeError("Frozen geometry changed")
        molecule=psi4.geometry(geometry)
        result.update(source_pin=config["source_pin"],dependency_proof_sha256=args.config_sha256,
            options=selected,geometry_sha256=digest_text(geometry),
            basis_files=inputs["basis_files"],stage=config["stage"],
            einsums_profile_disabled=bool(ein.rc.profile_disable),
            einsums_gpu_disabled=bool(ein.rc.gpu_disable),
            einsums_world=world,node=os.uname().nodename,
            localizer_maxiter_python=50,localizer_maxiter_cpp=int(core.get_option("FISAPT","LOCAL_MAXITER")),
            native_memory_safety_factor=float(core.get_option("FISAPT","FISAPT_MEM_SAFETY_FACTOR")),
            minao_basis_sha256=digest(Path(core.get_datadir())/"basis/cc-pvtz-minao.gbs"),
            timed_output_policy="FSAPT disk output disabled; final arrays saved after energy timing",
            instrumentation="Exclusive call timers and retained S/Locc matrix references only; no frame profiler/oracle",
            v2_dispersion_block=terms.FDISP_BLOCK,cpp_dispersion_block=48,
            hardware=json.loads(subprocess.check_output(["lscpu","--json"],text=True,timeout=10)),
            cpu_affinity=sorted(os.sched_getaffinity(0)))
        with ExitStack() as stack:
            stack.enter_context(patch.object(response,"partition_response",forbidden))
            stack.enter_context(patch.object(sapt_proc,"scf_helper",
                tape.wrap("common.scf",sapt_proc.scf_helper)))
            for name in ("build_sapt_jk_cache","electrostatics","exchange","induction"):
                stack.enter_context(patch.object(terms,name,tape.wrap("common."+name,getattr(terms,name))))
            for name in ("localization","partition","flocalization","felst","fexch","find","fdisp0"):
                function=getattr(terms,name)
                capture=(lambda out,args:tape.keep(out)) if name=="fdisp0" else None
                stack.enter_context(patch.object(terms,name,tape.wrap("einsums."+name,function,capture=capture)))
            for name in ("flocalize","felst","fexch","find","fdisp"):
                function=getattr(core.FISAPT,name)
                capture=(lambda out,args:tape.keep(args[0].matrices())) if name=="fdisp" else None
                stack.enter_context(patch.object(core.FISAPT,name,tape.wrap("cpp."+name,function,capture=capture)))
            stack.enter_context(patch.object(setup,"setup_fisapt_object",
                tape.wrap("cpp.setup_excluding_localization",setup.setup_fisapt_object)))
            result["bootstrap_seconds_before_energy"]=time.perf_counter()-process_start
            started=time.perf_counter()
            try:
                energy,wfn=psi4.energy("sapt(dft)",molecule=molecule,return_wfn=True)
            finally:
                result["whole_energy_seconds"]=time.perf_counter()-started
        expected="einsums.fdisp0" if args.arm=="einsums" else "cpp.fdisp"
        if sum(v["name"]==expected for v in tape.records)!=1:
            raise RuntimeError("Requested full fragment backend was not executed exactly once")
        other="cpp.fdisp" if args.arm=="einsums" else "einsums.fdisp0"
        if any(v["name"]==other for v in tape.records):
            raise RuntimeError("Both fragment backends executed in one timing call")
        core.close_outfile()
        cpp_blocks=[]
        import re
        try:
            with (root/"psi4.out").open() as stream:
                for line in stream:
                    if "IBO Localizer 2 failed." in line:
                        raise RuntimeError("Native/Python IBO localization did not converge; timing is unqualified")
                    match=re.search(r"compute blocks of (\d+) x (\d+)",line)
                    if match:cpp_blocks.append([int(v) for v in match.groups()])
        finally:
            core.reopen_outfile()
        result["observed_cpp_dispersion_blocks"]=cpp_blocks
        if args.arm=="cpp" and len(cpp_blocks)!=1:
            raise RuntimeError("Native BLOCKED dispersion execution banner missing/ambiguous")
        arrays={}
        for name,key in ARRAYS.items():
            item=core.array_variable(key)
            arrays[name]=np.array(getattr(item,"np",item),copy=True)
        arrays.update({name:np.array(getattr(item,"np",item),copy=True)
                       for name,item in tape.matrices.items()})
        if set(arrays)!=set(ARRAYS)|{"S","Locc_A","Locc_B"}:
            raise RuntimeError("Missing fragment gauge/output records")
        if any(not np.isfinite(v).all() for v in arrays.values()) or not np.isfinite(energy):
            raise FloatingPointError("Nonfinite full F-SAPT result")
        # Preserve the measured calculation even when a later qualification
        # guard rejects it. Failed records remain FAILED, never timed samples.
        np.savez(root/"matrices.npz",**arrays)
        result.update(energy_Eh=float(energy),matrices_sha256=digest(root/"matrices.npz"),
            signed_matrix_sums={name:float(np.sum(arrays[name]))
                                for name in ("Elst","Exch","IndAB","IndBA","Disp")})
        n=molecule.natom()
        counts=tuple(arrays["Locc_"+s].shape[1] for s in ("A","B"))
        rank=int(wfn.nmo())
        result.update(retained_mo_rank=rank,
                      retained_virtual_counts=[rank-o for o in counts])
        for name in ("Elst","Exch","IndAB","IndBA","Disp"):
            if arrays[name].shape!=(n+counts[0]+1,n+counts[1]+1):
                raise RuntimeError("Full-system fragment dimensions differ")
            np.testing.assert_allclose(arrays[name][-1,:],0.,atol=1e-12,rtol=0)
            np.testing.assert_allclose(arrays[name][:,-1],0.,atol=1e-12,rtol=0)
        sums={name:float(np.sum(arrays[name])) for name in ("Elst","Exch","IndAB","IndBA","Disp")}
        canonical={name:float(core.variable(name)) for name in SCALARS}
        result["canonical"]=canonical
        result["component_ledger"]=validate_component_ledger(sums,canonical,float(energy))
        actual_frozen=[wfn.basisset().n_frozen_core(core.get_global_option("FREEZE_CORE"),
                         molecule.extract_subsets([i],[])) for i in (1,2)]
        result["frozen_occupied"]=actual_frozen
        for axis,nf in enumerate(actual_frozen):
            block=(slice(None,nf),slice(None)) if axis==0 else (slice(None),slice(None,nf))
            occupied=arrays["Disp"][n:n+counts[0],n:n+counts[1]]
            np.testing.assert_allclose(occupied[block],0.,atol=1e-12,rtol=0)
        if point is not None:
            if wfn.basisset().nbf()!=point["spherical_basis_counts"]["primary"]:
                raise RuntimeError("Installed primary basis dimensions changed")
            if counts!=tuple(f["occupied"] for f in point["fragments"]):
                raise RuntimeError("Occupied/frozen setup changed")
            if actual_frozen!=[f["frozen_occupied"] for f in point["fragments"]]:
                raise RuntimeError("Actual frozen occupied counts changed")
        if core.get_option("FISAPT","FISAPT_FSAPT_IND_RESPONSE"):
            raise RuntimeError("Fragment response mode changed during the energy call")
        result.update(status="COMPLETED_CPU_REQUIRES_PAIRED_VALIDATION",
            energy_Eh=float(energy),signed_matrix_sums=sums,
            matrices_sha256=digest(root/"matrices.npz"),
            scalar_variables={k:float(v) for k,v in core.scalar_variables().items()},
            exact_fragment_solver_forbidden=True)
        mapped=set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields=line.split(maxsplit=5)
            if len(fields)==6 and fields[5].startswith("/"):
                path=Path(fields[5])
                if path.name.startswith(("libmkl","libomp","libgomp","libiomp")):
                    mapped.add(path.resolve())
        result["cpu_math_library_sha256"]={str(path):digest(path) for path in sorted(mapped)}
    except BaseException as error:
        result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["phases"]=tape.records
        if "whole_energy_seconds" in result:
            exclusive=sum(v["exclusive_seconds"] for v in tape.records)
            result["unattributed_energy_seconds"]=result["whole_energy_seconds"]-exclusive
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result["usage"]={name:getattr(usage,name) for name in
            ("ru_maxrss","ru_utime","ru_stime","ru_nvcsw","ru_nivcsw","ru_majflt","ru_minflt")}
        result["process_seconds_before_result_write"]=time.perf_counter()-process_start
        with (root/"result.json").open("x") as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)


def digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__=="__main__":main()
