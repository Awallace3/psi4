"""Sequential fresh-process diagnostics and independent same-gauge CPU checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from gpu_runtime import read_config,require_local_lease
from repeat_gauge import compare_repeat


def compare(cpu,gpu):
    """Signed aggregate controls accompany, never replace, the matrix oracle."""
    if cpu["geometry_sha256"]!=gpu["geometry_sha256"] or cpu["scaled"]!=gpu["scaled"]:
        raise RuntimeError("CPU/GPU diagnostic inputs are not matched")
    for name in ("geom.xyz","geom.psi4mol","ZA.dat","ZB.dat"):
        if cpu["output_file_sha256"][name]!=gpu["output_file_sha256"][name]:
            raise RuntimeError("CPU/GPU geometry/charge serialization differs: "+name)
    checks={}
    for key in ("Elst_AB","Exch_AB","IndAB_AB","IndBA_AB","Disp_AB",
                "Ind20u_AB","Ind20u_BA","ExchInd20u_AB","ExchInd20u_BA",
                "Ind20r_AB","Ind20r_BA","ExchInd20r_AB","ExchInd20r_BA"):
        left,right=cpu["signed_matrix_sums"][key],gpu["signed_matrix_sums"][key]
        np.testing.assert_allclose(right,left,atol=2e-7,rtol=2e-5,err_msg="Signed aggregate "+key)
        checks[key]={"cpu_Eh":left,"gpu_Eh":right,"difference_Eh":right-left}
    for key in ("Disp20,u","Exch-Disp20,u"):
        left,right=cpu[key],gpu[key]
        np.testing.assert_allclose(right,left,atol=2e-7,rtol=2e-5,err_msg="Signed "+key)
        checks[key]={"cpu_Eh":left,"gpu_Eh":right,"difference_Eh":right-left}
    return checks


def environment(config,*,gpu):
    env=dict(os.environ)
    for name in ("PYTHONHOME","PYTHONPATH","PYTHONOPTIMIZE","LD_PRELOAD",
                 "CUDA_HOME","CUDA_PATH","CUDA_ROOT","CUDAToolkit_ROOT",
                 "CPATH","C_INCLUDE_PATH","CPLUS_INCLUDE_PATH","LIBRARY_PATH"):
        env.pop(name,None)
    if config.get("schema")==2:
        env.update(LD_LIBRARY_PATH=config["startup_library_path"],
            PYTHONPATH=config["optional_python_path"]+":"+config["einsums_prefix"]+"/lib",
            PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS=str(config["threads"]),MKL_NUM_THREADS=str(config["threads"]),
            OPENBLAS_NUM_THREADS=str(config["threads"]),OMP_DYNAMIC="FALSE",
            EINSUMS_PACKED_GEMM_BATCH_PROMOTION="1")
        env["CUDA_VISIBLE_DEVICES"]=config["local_lease"]["expected_uuid"] if gpu else ""
        return env
    cuda_root=str(Path(config["cuda_manifest"]).parent/"cuda13")
    env.update(CUDA_HOME=cuda_root,CUDA_PATH=cuda_root,CUDA_ROOT=cuda_root)
    env["LD_LIBRARY_PATH"]=":".join((
        cuda_root+"/lib64",
        config["einsums_prefix"]+"/lib",config["cuest_root"]+"/lib/13",
        config["dependency_environment"]+"/lib"))
    env["PYTHONPATH"]=config["optional_python_path"]+":"+config["einsums_prefix"]+"/lib"
    if not gpu:env["CUDA_VISIBLE_DEVICES"]=""
    return env


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.config=args.config.resolve(strict=True)
    config,build=read_config(args.config,args.config_sha256)
    local=config.get("schema")==2
    root=args.output.resolve(strict=True)
    helper=Path(__file__).resolve().parent
    result={"status":"STARTED","scope":("SMALL_CACHY_GATE_NOT_PHOENIX_OR_TIMING_PROOF" if local
                                     else "SMALL_H200_GATE_NOT_LARGE_SPACE_OR_TIMING_PROOF"),
        "source_pin":config["source_pin"],"helper_pin":os.environ["DISP0_PIN"],
        "runtime_config_sha256":args.config_sha256,"cases":[]}
    deadline=time.monotonic()+3300
    def child(label,arm,*,scaled=False,change=0.,oracle=None):
        output=root/label
        argv=[str(Path(config["stage"])/"bin/psi4"),
              config["dependency_environment"]+"/bin/python",
              str(helper.parent/"cachy/bootstrap_gate.py" if local else helper/"fragment_gate.py"),
              "--config",str(args.config),"--config-sha256",args.config_sha256,
              "--arm",arm,"--distance-change",str(change),"--output",str(output)]
        if scaled:argv.append("--scaled")
        if oracle:argv.extend(("--oracle-from",str(root/oracle)))
        # Invoke and evaluate only the SHA-verified whole-stage launcher.
        command=["/bin/bash","-c",
            'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B "$3" "${@:4}"',
            "psi4-gate",*argv]
        record={"label":label,"arm":arm,"argv":command}
        result["cases"].append(record)
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError("Bounded whole-gate deadline exhausted")
        with (root/(label+".stdout")).open("x") as out,(root/(label+".stderr")).open("x") as err:
            env=environment(config,gpu=arm=="gpu")
            scratch=root/(label+"-scratch");scratch.mkdir()
            env["PSI_SCRATCH"]=str(scratch)
            if local:
                from owned_process import execute
                fd=require_local_lease(config["local_lease"])
                code=execute(command,env,out,err,min(600,remaining),fd,config["local_lease"])
            else:
                code=subprocess.run(command,env=env,stdout=out,stderr=err,
                                    timeout=min(600,remaining)).returncode
        record["returncode"]=code
        if code:raise RuntimeError("Fresh diagnostic process failed: "+label)
        document=json.loads((output/"result.json").read_text())
        if document["status"]!="EXECUTED_REQUIRES_INDEPENDENT_COMPARISON":
            raise RuntimeError("Incomplete diagnostic artifact: "+label)
        return document
    try:
        for mode,scaled,change in (("unscaled",False,0.),("scaled",True,0.),
                                   ("changed",True,.08)):
            cpu=child("cpu-"+mode,"cpu",scaled=scaled,change=change)
            gpu=child("gpu-"+mode,"gpu",scaled=scaled,change=change)
            oracle=child("oracle-"+mode,"cpu",scaled=scaled,change=change,oracle="gpu-"+mode)
            result.setdefault("signed_aggregate_checks",{})[mode]=compare(cpu,gpu)
            result.setdefault("same_gauge_fields",{})[mode]=oracle["same_gauge_compared_fields"]
        repeat=child("gpu-repeat","gpu",scaled=True)
        baseline=json.loads((root/"gpu-scaled/result.json").read_text())
        result["repeat_gauge_check"]=compare_repeat(root/"gpu-scaled",root/"gpu-repeat")
        changed=json.loads((root/"gpu-changed/result.json").read_text())
        if abs(changed["energy_Eh"]-baseline["energy_Eh"])<1e-8:
            raise RuntimeError("Changed geometry did not change the energy")
        if local:require_local_lease(config["local_lease"])
        result["status"]="PASS_SMALL_CACHY_ONLY" if local else "PASS_SMALL_H200_ONLY"
        result["gpu_small_case_correctness_verified"]=True
        result["large_system_admitted"]=False
        result["accepted_timing_count"]=0
    except BaseException as error:
        result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with (root/"gate-result.json").open("x") as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)


if __name__=="__main__":main()
