"""Tiny parameter probe, not a relaxed oracle or complete GPU qualification.

Uses the unchanged whole stage and authenticated helpers of a completed gate
attempt. No installed module is overlaid. Launch under the common inherited
lease, external resource limits and an explicit short deadline.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--attempt",type=Path,required=True)
    parser.add_argument("--helper-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--supervise",action="store_true")
    args=parser.parse_args()
    attempt=args.attempt.resolve(strict=True);source=args.helper_source.resolve(strict=True)
    gate=json.loads((attempt/"gate-result.json").read_text())
    if (subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()!=gate["helper_pin"]
            or subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip()):
        raise RuntimeError("Original clean authenticated gate helpers required")
    sys.path.insert(0,str(source/"devtools/disp0/phoenix_validation"))
    from gpu_runtime import digest,read_config,AuditedRuntime,require_local_lease
    config_path=attempt/"runtime-config.json"
    config,build=read_config(config_path,gate["runtime_config_sha256"])
    if config["device_family"]!="RTX3090" or config["threads"]!=4:
        raise RuntimeError("Explicit leased Cachy diagnostic only")
    if args.supervise:
        from owned_process import execute
        from run_gate import environment
        control=args.output.resolve();control.mkdir()
        env=environment(config,gpu=True)
        command=["/bin/bash","-c",
                 'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B "$3" "${@:4}"',
                 "one-electron-probe",str(Path(config["stage"])/"bin/psi4"),
                 config["dependency_environment"]+"/bin/python",str(Path(__file__).resolve()),
                 "--attempt",str(attempt),"--helper-source",str(source),"--output",str(control/"probe")]
        code=None;error=None
        try:
            with (control/"probe.stdout").open("x") as out,(control/"probe.stderr").open("x") as err:
                code=execute(command,env,out,err,90,require_local_lease(config["local_lease"]),
                             config["local_lease"])
            if code:raise RuntimeError("One-electron diagnostic child failed")
        except BaseException as problem:
            error=f"{type(problem).__name__}: {problem}"
            raise
        finally:
            (control/"supervision.json").write_text(json.dumps(
                {"scope":"PARAMETER_DIAGNOSTIC_ONLY","returncode":code,"error":error,
                 "command":command,"script_sha256":digest(__file__)},indent=2)+"\n")
        return
    gpu_dir=attempt/"gpu-unscaled"
    previous=json.loads((gpu_dir/"result.json").read_text())
    raw_path=gpu_dir/"raw-hf.npz"
    if (previous["status"]!="EXECUTED_REQUIRES_INDEPENDENT_COMPARISON"
            or previous["artifact_sha256"]["raw-hf.npz"]!=digest(raw_path)):
        raise RuntimeError("Complete unchanged GPU raw reference required")
    output=args.output.resolve();output.mkdir();(output/"scratch").mkdir()
    os.environ["PSI_SCRATCH"]=str(output/"scratch")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=config["local_lease"]["expected_uuid"]:
        raise RuntimeError("Explicit actual leased CUDA visibility required")
    sys.path.insert(0,str(source/"devtools/disp0/cachy"))
    from bootstrap_gate import bootstrap
    owners,events=bootstrap(config,build)
    import numpy as np
    import psi4
    from psi4 import core
    from fragment_gate import WATER
    psi4.set_num_threads(4);psi4.set_memory("2 GiB")
    core.set_output_file(str(output/"psi4.out"),False)
    psi4.set_options({"basis":"aug-cc-pvdz","puream":True,"use_cuest":False,
                     "cuest_mixed_precision":False,"ints_tolerance":1e-12})
    molecule=psi4.geometry(WATER)
    bases={side:core.BasisSet.build(molecule.extract_subsets(real,ghost),
            "BASIS","aug-cc-pvdz",puream=1)
           for side,real,ghost in (("A",1,2),("B",2,1))}
    result={"status":"STARTED","scope":"ONE_ELECTRON_SCREENING_DIAGNOSTIC_ONLY",
        "source_pin":config["source_pin"],"original_helper_pin":gate["helper_pin"],
        "original_gpu_result_sha256":digest(gpu_dir/"result.json"),
        "script_sha256":digest(__file__),"import_events":events,"cases":[]}
    started=time.monotonic()
    session=None
    try:
        with np.load(raw_path,allow_pickle=False) as raw:
            cpu={}
            for side,basis in bases.items():
                mints=core.MintsHelper(basis)
                np.testing.assert_allclose(mints.ao_overlap().np,raw["S"],rtol=1e-10,atol=1e-10)
                cpu[side]=np.array(mints.ao_potential().np,copy=True)
                del mints
            np.savez(output/"cpu-reference.npz",**{"V_"+k:v for k,v in cpu.items()})
            runtime=AuditedRuntime(config_path,gate["runtime_config_sha256"])
            runtime.require_device_admission()
            with runtime.cuda.CUDASession(
                    cublas_library=runtime.blas["path"],cublas_sha256=runtime.blas["sha256"],
                    memory_limit_bytes=config["device_memory_limit_bytes"],
                    native_persistent_bytes=config["native_persistent_bytes"]) as session:
                result["device"]=runtime.device_identity(session)
                api=runtime.abi.admit_factory_api(session=session,
                    sdk_root=Path(config["cuest_root"]),archive=Path(config["cuest_archive"]),
                    cuda_manifest=runtime.manifest,cuda_manifest_sha256=config["cuda_manifest_sha256"])
                for cutoff in (1e-12,1e-14,1e-16):
                    require_local_lease(config["local_lease"])
                    psi4.set_options({"use_cuest":True,"ints_tolerance":cutoff})
                    arrays={}
                    for side,basis in bases.items():
                        mints=core.MintsHelper(basis)
                        actual=np.array(mints.ao_potential().np,copy=True)
                        del mints;gc.collect()
                        api.assert_identity()
                        difference=actual-cpu[side]
                        violations=np.abs(difference)>1e-10+1e-10*np.abs(cpu[side])
                        rows=np.argwhere(violations)
                        result["cases"].append({"cutoff":cutoff,"side":side,
                            "maximum_absolute_difference":float(np.max(np.abs(difference))),
                            "oracle_violations":int(np.count_nonzero(violations)),
                            "difference_from_saved_gpu":float(np.max(np.abs(actual-raw["V_"+side]))),
                            "first_violations":[{"index":row.tolist(),"cpu":float(cpu[side][tuple(row)]),
                                "gpu":float(actual[tuple(row)]),"difference":float(difference[tuple(row)])}
                                for row in rows[:12]]})
                        arrays["V_"+side]=actual
                    np.savez(output/("gpu-cutoff-"+str(cutoff)+".npz"),**arrays)
                runtime.verify_optional_modules()
        result["status"]="EXECUTED_PARAMETER_DIAGNOSTIC_NOT_QUALIFICATION"
    except BaseException as error:
        result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["seconds"]=time.monotonic()-started
        if session is not None:
            result["cleanup"]={"complete":session.cleanup_complete,"live_bytes":session.live_bytes,
                               "native_owners":len(session.native_owners)}
        (output/"probe-result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(result["status"],output,flush=True)


if __name__=="__main__":main()
