"""Paired CPU-node v2/C++ production comparisons, three fresh repeats per arm."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time
import numpy as np
from gpu_runtime import digest, read_config
from prepare_inputs import SYSTEMS
from run_gate import environment


def orbital_permutation(left,right,s,*,nfrozen=0):
    if left.shape!=right.shape:raise ValueError("Unequal occupied spaces")
    overlap=left.T@s@right
    mapping=np.argmax(np.abs(overlap),axis=0)
    if len(set(mapping.tolist()))!=right.shape[1]:
        raise RuntimeError("IBO gauges do not admit a unique orbital permutation")
    selected=np.abs(overlap[mapping,np.arange(right.shape[1])])
    np.testing.assert_allclose(selected,1.,rtol=1e-7,atol=1e-7,
                               err_msg="Occupied-gauge equivalence")
    if np.any(mapping[:nfrozen]>=nfrozen) or np.any(mapping[nfrozen:]<nfrozen):
        raise RuntimeError("Occupied gauge mixes frozen and active subspaces")
    off=np.abs(overlap).copy();off[mapping,np.arange(right.shape[1])]=0.
    np.testing.assert_allclose(off,0.,atol=1e-7,rtol=0,err_msg="Off-permutation occupied overlap")
    return mapping


def compare(left_path,right_path):
    left=json.loads((left_path/"result.json").read_text())
    right=json.loads((right_path/"result.json").read_text())
    for value in (left,right):
        if value["status"]!="COMPLETED_CPU_REQUIRES_PAIRED_VALIDATION":
            raise RuntimeError("Incomplete CPU result")
    for key in ("source_pin","geometry_sha256","input_manifest_sha256","threads",
                "psi4_memory","stage","node","frozen_occupied"):
        if left[key]!=right[key]:raise RuntimeError("Unmatched CPU comparison: "+key)
    lo,ro=dict(left["options"]),dict(right["options"])
    for value in (lo,ro):
        value.pop("sapt_dft_do_fsapt");value.pop("fisapt_fsapt_filepath")
    if lo!=ro:raise RuntimeError("CPU production science options differ")
    np.testing.assert_allclose(left["energy_Eh"],right["energy_Eh"],rtol=1e-8,atol=2e-8)
    from cpu_benchmark import SCALARS
    for name in SCALARS:
        np.testing.assert_allclose(left["canonical"][name],right["canonical"][name],
                                   atol=2e-8,rtol=1e-8,err_msg="Signed canonical scalar "+name)
    for path,value in ((left_path,left),(right_path,right)):
        if digest(path/"matrices.npz")!=value["matrices_sha256"]:
            raise RuntimeError("Completed CPU matrix artifact changed")
    with np.load(left_path/"matrices.npz",allow_pickle=False) as l:
        with np.load(right_path/"matrices.npz",allow_pickle=False) as r:
            np.testing.assert_allclose(l["S"],r["S"],atol=1e-11,rtol=1e-11)
            mappings=[orbital_permutation(l["Locc_"+side],r["Locc_"+side],l["S"],nfrozen=nf)
                      for side,nf in zip(("A","B"),left["frozen_occupied"])]
            n=l["QA"].shape[0]
            indices=[np.r_[np.arange(n),n+mapping,n+len(mapping)] for mapping in mappings]
            errors={}
            for name in ("Elst","Exch","IndAB","IndBA","Disp"):
                aligned=l[name][np.ix_(*indices)]
                np.testing.assert_allclose(aligned,r[name],atol=2e-7,rtol=2e-5,
                                           err_msg="Full CPU fragment matrix "+name)
                errors[name]=float(np.max(np.abs(aligned-r[name])))
            for name,mapping in zip(("QA","QB"),mappings):
                np.testing.assert_allclose(l[name][:,mapping],r[name],atol=2e-7,rtol=2e-5)
    return {"maximum_absolute_matrix_errors_Eh":errors,
            "left_to_right_occupied_indices":[v.tolist() for v in mappings]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--inputs-sha256",required=True)
    parser.add_argument("--system",choices=SYSTEMS,required=True)
    parser.add_argument("--memory",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config,build=read_config(args.config,args.config_sha256)
    root=args.output.resolve(strict=True)
    scratch=Path(os.environ["TMPDIR"]).resolve(strict=True)
    records=[];checks={}
    result={"status":"STARTED","scope":"PRODUCTION_SCALED_CPU_V2_VS_CPP",
        "system":args.system,"helper_pin":os.environ["DISP0_PIN"],
        "source_pin":config["source_pin"],"samples":[],"accepted_cpu_sample_count":0,
        "accepted_gpu_speedup_count":0,"input_manifest_sha256":args.inputs_sha256,
        "dependency_proof_sha256":args.config_sha256}
    allocation=json.loads((root/"allocation.json").read_text())
    if allocation["status"]!="PASS_CPU_ALLOCATION_ONLY":
        raise RuntimeError("Effective CPU allocation was not admitted")
    result["allocation_sha256"]=digest(root/"allocation.json")
    deadline=time.monotonic()+int(os.environ["DISP0_CPU_WALL_SECONDS"])-180
    def child(label,arm,system,repeat):
        output=root/label
        work=scratch/label;work.mkdir()
        env=environment(config,gpu=False)
        env.update(TMPDIR=str(work),SCRATCH=str(work),PSI_SCRATCH=str(work))
        argv=[str(Path(config["stage"])/"bin/psi4"),config["dependency_environment"]+"/bin/python",
            str(Path(__file__).parent/"cpu_benchmark.py"),"--config",str(args.config),
            "--config-sha256",args.config_sha256,"--inputs",str(args.inputs),
            "--inputs-sha256",args.inputs_sha256,"--system",system,"--arm",arm,
            "--threads","24","--memory",args.memory,"--repeat",str(repeat),"--output",str(output)]
        command=["/bin/bash","-c",
            'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B "$3" "${@:4}"',
            "cpu-fsapt",*argv]
        row={"label":label,"arm":arm,"repeat":repeat,"system":system,"argv":command}
        records.append(row)
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError("Bounded comparison deadline exhausted")
        started=time.perf_counter()
        stop=threading.Event()
        peak=[0]
        def sample_scratch():
            while True:
                total=0
                for path in work.rglob("*"):
                    try:
                        if path.is_file():total+=path.stat().st_size
                    except FileNotFoundError:pass  # native scratch deletion races are expected
                peak[0]=max(peak[0],total)
                if stop.wait(2.):return
        sampler=threading.Thread(target=sample_scratch,daemon=True)
        sampler.start()
        try:
            with (root/(label+".stdout")).open("x") as out,(root/(label+".stderr")).open("x") as err:
                completed=subprocess.run(command,env=env,stdout=out,stderr=err,timeout=remaining)
        finally:
            stop.set();sampler.join(timeout=10)
            row.update(sampled_peak_scratch_logical_bytes=peak[0],scratch_sample_interval_seconds=2.)
        row.update(returncode=completed.returncode,subprocess_wall_seconds=time.perf_counter()-started)
        if completed.returncode:raise RuntimeError("Fresh CPU process failed: "+label)
        document=json.loads((output/"result.json").read_text())
        if document["status"]!="COMPLETED_CPU_REQUIRES_PAIRED_VALIDATION":
            raise RuntimeError("Missing completed CPU artifact: "+label)
        # Only this fresh child-owned scratch directory is cleaned.
        row["scratch_final_logical_bytes"]=sum(p.stat().st_size for p in work.rglob("*") if p.is_file())
        shutil.rmtree(work)
        return document
    try:
        for arm in ("cpp","einsums"):
            child("preflight-"+arm,arm,"water-preflight",-1)
        checks["water-preflight"]=compare(root/"preflight-cpp",root/"preflight-einsums")
        for repeat in range(3):
            order=("cpp","einsums") if (repeat+SYSTEMS.index(args.system))%2==0 else ("einsums","cpp")
            for arm in order:
                document=child(f"r{repeat}-{arm}",arm,args.system,repeat)
                result["samples"].append({"arm":arm,"repeat":repeat,
                    "whole_energy_seconds":document["whole_energy_seconds"],
                    "max_rss_KiB":document["usage"]["ru_maxrss"]})
            checks["repeat-"+str(repeat)]=compare(root/f"r{repeat}-cpp",root/f"r{repeat}-einsums")
            for arm in ("cpp","einsums"):
                if repeat:compare(root/f"r0-{arm}",root/f"r{repeat}-{arm}")
            # Durable partial proof; incomplete or failed batches never lose prior results.
            with (root/f"paired-repeat-{repeat}.json").open("x") as stream:
                json.dump({"status":"PASS_CPU_PAIR","comparison":checks["repeat-"+str(repeat)]},
                          stream,indent=2,sort_keys=True)
        summary={}
        for arm in ("cpp","einsums"):
            values=[r["whole_energy_seconds"] for r in result["samples"] if r["arm"]==arm]
            summary[arm]={"n":len(values),"median_seconds":statistics.median(values),
                          "minimum_seconds":min(values),"maximum_seconds":max(values)}
        result.update(status="PASS_PRODUCTION_CPU_COMPARISON",summary=summary,
            cpp_over_einsums_whole_call_ratio=summary["cpp"]["median_seconds"]/summary["einsums"]["median_seconds"],
            accepted_cpu_sample_count=6)
    except BaseException as error:
        result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result.update(processes=records,comparisons=checks)
        with (root/"comparison-result.json").open("x") as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)


if __name__=="__main__":main()
