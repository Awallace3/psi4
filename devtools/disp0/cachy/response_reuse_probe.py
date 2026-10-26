"""Leased real-CUDA response microbenchmark, NOT a whole Psi4/cuEST gate.

Fixed synthetic FP64 factors isolate response math/transport and scratch reuse.
No installed scientific module, environment, native plan or foreign process is
changed. The old response implementation comes from the recorded Git baseline.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
import uuid

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASELINE="7ebf33179d4b8d4c4a443c8a46e5fcfeb2d3274f"
ROOT=Path("/home/amwalla3/gits/psi4-disp0-cachy")
sys.path.insert(0,str(HERE.parent/"phoenix_validation"))
from gpu_runtime import digest,require_local_lease
from coherent_stage11 import lease,policy,run_command,save,clean,git


def resource_scope():
    """Read back our own cgroup only; no unrelated job/process inspection."""
    groups=Path("/proc/self/cgroup").read_text().splitlines()
    selected=[line[3:] for line in groups if line.startswith("0::")]
    if len(selected)!=1:raise RuntimeError("Explicit cgroup-v2 scope required")
    directory=Path("/sys/fs/cgroup")/selected[0].lstrip("/")
    values={name:(directory/name).read_text().strip()
            for name in ("cpu.max","memory.max","memory.swap.max","pids.max")}
    quota,period=values["cpu.max"].split()
    if (quota=="max" or int(quota)>4*int(period)
            or values["memory.max"]=="max" or int(values["memory.max"])>12*1024**3
            or values["memory.swap.max"]!="0" or values["pids.max"]=="max"
            or int(values["pids.max"])>128):
        raise RuntimeError("Cachy probe requires <=4CPU/12GiB/no-swap/128-task scope")
    return {"path":str(directory),"limits":values}


def numerical_case(np,O,V,Q):
    rng=np.random.default_rng(943+O+V+Q);N=O+V
    ao=rng.normal(size=(Q,N,N))*.002
    ao=(ao+ao.transpose(0,2,1))*.5
    factors={"ov":np.ascontiguousarray(ao[:,:O,O:]),
             "oo":np.ascontiguousarray(ao[:,:O,:O]),
             "vv":np.ascontiguousarray(ao[:,O:,O:])}
    eo=np.linspace(-1.5,-.7,O);ev=np.linspace(.1,1.,V)
    trial=rng.normal(size=(O,V))
    # Independent NumPy equation: no device implementation is used here.
    coulomb=np.einsum("qia,qjb,jb->ia",factors["ov"],factors["ov"],trial,optimize=True)
    first=np.einsum("qij,qab,jb->ia",factors["oo"],factors["vv"],trial,optimize=True)
    second=np.einsum("qib,qja,jb->ia",factors["ov"],factors["ov"],trial,optimize=True)
    truth=(ev[None,:]-eo[:,None])*trial+4*coulomb-first-second
    return factors,eo,ev,trial,truth


def child(args):
    import numpy as np
    require_local_lease(policy(ROOT))
    clean(REPO,args.pin)
    output=args.output.resolve(strict=True)
    report={"status":"RUNNING","scope":"FIXED_FACTOR_CUDA_RESPONSE_ONLY_NOT_WHOLE_FISAPT",
            "source_pin":args.pin,"baseline_pin":BASELINE,"cases":[],"pid":os.getpid()}
    session=None;primary=None
    try:
        report["resource_scope"]=resource_scope()
        manifest_path=ROOT/"runs/native-tmpfs-stage-01/cuda-manifest.json"
        report["cuda_manifest_sha256"]=digest(manifest_path)
        manifest=json.loads(manifest_path.read_text())
        for component in manifest["components"]:
            if digest(component["path"])!=component["sha256"]:
                raise RuntimeError("Frozen native CUDA closure changed")
        cublas=next(c for c in manifest["components"]
                    if Path(c["path"]).name.startswith("libcublas.so."))
        for name in ("cuda-bindings","cuda-pathfinder","numpy"):
            distribution=importlib.metadata.distribution(name)
            if Path(distribution.locate_file("")).resolve()!=ROOT/"environment-v2/lib/python3.13/site-packages":
                raise RuntimeError("Foreign dependency distribution: "+name)
        report["packages"]={n:importlib.metadata.version(n)
                           for n in ("cuda-bindings","cuda-pathfinder","numpy")}
        if report["packages"]["cuda-bindings"]!="13.3.0":
            raise RuntimeError("Exact existing CUDA bindings required")
        package="_disp0_response_microbenchmark"
        module=types.ModuleType(package)
        module.__path__=[str(REPO/"psi4/driver/procrouting/sapt")]
        sys.modules[package]=module
        runtime=importlib.import_module(package+".sapt_cuda_runtime")
        candidate=importlib.import_module(package+".sapt_fisapt_response_cuda")
        old_name=package+".baseline_response"
        old=types.ModuleType(old_name);old.__package__=package
        sys.modules[old_name]=old
        source_path="psi4/driver/procrouting/sapt/sapt_fisapt_response_cuda.py"
        old_source=subprocess.check_output(["git","-C",str(REPO),"show",BASELINE+":"+source_path])
        exec(compile(old_source,BASELINE+":"+source_path,"exec"),old.__dict__)
        report["baseline_response_sha256"]=hashlib.sha256(old_source).hexdigest()
        report["candidate_response_sha256"]=digest(REPO/source_path)
        report["runtime_sha256"]=digest(Path(runtime.__file__))
        session=runtime.CUDASession(cublas_library=cublas["path"],
            cublas_sha256=cublas["sha256"],memory_limit_bytes=512*1024**2,
            native_persistent_bytes=128*1024**2)
        device=session.rt(session.driver.cuCtxGetDevice())
        actual="GPU-"+str(uuid.UUID(bytes=bytes(
            session.rt(session.driver.cuDeviceGetUuid(device)).bytes)))
        if actual!=policy(ROOT)["expected_uuid"]:
            raise RuntimeError("CUDA context differs from leased RTX3090")
        if session.arch!="compute_86":raise RuntimeError("Expected RTX3090 compute86")
        report.update(uuid=actual,arch=session.arch,ptx_sha256=session.ptx_sha256)
        report["hardware_before"]=hardware(actual)
        for O,V,Q,batch in ((2,3,5,2),(8,32,65,7),(16,96,129,7)):
            require_local_lease(policy(ROOT))
            factors,eo,ev,trial,truth=numerical_case(np,O,V,Q)
            record={"shape":[O,V,Q],"q_batch":batch,"timings":[],"errors":[]}
            with session.scope():
                # Fixed-factor producer deliberately does not claim native DFMO.
                class Producer:
                    nao=O+V;naux=Q;closed=False;failed=False
                    primary_basis=object();auxiliary_basis=object()
                    def validate_basis(self,*_):pass
                    def compute_device(self,pairs):
                        if set(pairs)!={"ov","oo","vv"}:
                            raise RuntimeError("Incomplete fixed-factor request")
                        return {n:session.upload(a) for n,a in factors.items()}
                provider=Producer();provider.session=session
                occupied=session.upload(np.eye(O+V)[:,:O].copy())
                virtual=session.upload(np.eye(O+V)[:,O:].copy())
                energies={"o":session.upload(eo),"v":session.upload(ev)}
                workspace=8*(2*O*V+Q+batch*(3*O*V+O*O))
                operators={name:implementation.HFResponseOperator(
                    provider,occupied,virtual,energies,max_workspace_bytes=workspace)
                    for name,implementation in (("baseline",old),("candidate",candidate))}
                dt=session.upload(trial)
                for name in ("baseline","candidate"):
                    with session.scope():
                        actual_host=session.download(operators[name].apply(dt))
                        np.testing.assert_allclose(actual_host,truth,rtol=1e-11,atol=1e-12)
                for repeat in range(3):
                    # Alternate order to expose warmup/order effects.
                    for name in (("baseline","candidate") if repeat%2==0
                                 else ("candidate","baseline")):
                        require_local_lease(policy(ROOT))
                        with session.scope():
                            session.synchronize_device();started=time.perf_counter()
                            result=operators[name].apply(dt)
                            session.synchronize_device()
                            seconds=time.perf_counter()-started
                            actual_host=session.download(result)
                            np.testing.assert_allclose(actual_host,truth,rtol=1e-11,atol=1e-12)
                            record["timings"].append({"arm":name,"repeat":repeat,"seconds":seconds})
                            record["errors"].append(float(np.max(np.abs(actual_host-truth))))
                # Exercise a changed RHS and PCG independently, not timed.
                if O==2:
                    P=O*V;H=np.empty((P,P))
                    for j in range(P):
                        x=np.eye(P)[j].reshape(O,V)
                        H[:,j]=((ev[None,:]-eo[:,None])*x
                            +4*np.einsum("qia,qjb,jb->ia",factors["ov"],factors["ov"],x)
                            -np.einsum("qij,qab,jb->ia",factors["oo"],factors["vv"],x)
                            -np.einsum("qib,qja,jb->ia",factors["ov"],factors["ov"],x)).ravel()
                    record["pcg"]=[]
                    for name,op in operators.items():
                        for rhs in (trial,trial[::-1].copy()+.125,np.zeros_like(trial)):
                            with session.scope():
                                solved=op.solve(session.upload(rhs),rtol=1e-12,atol=1e-14)
                                actual_host=session.download(solved.amplitudes)
                                expected=-np.linalg.solve(H,rhs.ravel()).reshape(O,V)
                                np.testing.assert_allclose(actual_host,expected,rtol=1e-10,atol=1e-12)
                                record["pcg"].append({"arm":name,"iterations":solved.iterations,
                                    "relative_residual":solved.relative_residual,
                                    "max_absolute_error":float(np.max(np.abs(actual_host-expected)))})
            if session.live_bytes!=0:raise RuntimeError("Case scratch/factors leaked")
            record["median_seconds"]={name:statistics.median(
                row["seconds"] for row in record["timings"] if row["arm"]==name)
                for name in operators}
            record["diagnostic_ratio"]=record["median_seconds"]["baseline"]/record["median_seconds"]["candidate"]
            report["cases"].append(record)
            save(output/"result.json",report)
        report.update(kernel_launches=session.kernel_launches,blas_calls=session.blas_calls,
            peak_tracked_bytes=session.peak_live_bytes,hardware_after=hardware(actual))
        if "psi4" in sys.modules:raise RuntimeError("Standalone probe imported installed Psi4")
    except BaseException as error:
        primary=error;report.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if session is not None:
                session.close()
                report["cleanup"]={"complete":session.cleanup_complete,
                                   "live_bytes":session.live_bytes,
                                   "native_owners":len(session.native_owners)}
            if primary is None:report["status"]="PASS_RESPONSE_CUDA_MICROBENCH_ONLY"
        except BaseException as error:
            report.update(status="FAILED_CLEANUP",cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:save(output/"result.json",report)
    print(json.dumps(report,sort_keys=True),flush=True)


def hardware(device):
    return subprocess.check_output(["nvidia-smi","--id="+device,
        "--query-gpu=uuid,name,memory.total,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits"],text=True,timeout=15).strip()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--attempt",required=True)
    parser.add_argument("--pin",required=True)
    parser.add_argument("--child",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if Path.home()!=ROOT.parents[1] or not args.attempt.replace("-","").isalnum():
        raise RuntimeError("Explicit Cachy owner and fresh simple attempt required")
    clean(REPO,args.pin)
    if args.child:
        if args.output is None or args.output!=ROOT/"runs"/args.attempt:
            raise RuntimeError("Explicit owned fresh output required")
        child(args);return
    with lease(ROOT) as fd:
        output=ROOT/"runs"/args.attempt;output.mkdir(mode=0o700)
        manifest=json.loads((ROOT/"runs/native-tmpfs-stage-01/cuda-manifest.json").read_text())
        cuda=Path(manifest["cuda_root"])
        env={k:os.environ[k] for k in ("HOME","USER","DISP0_LOCAL_LEASE_FD") if k in os.environ}
        env.update(PATH=str(ROOT/"environment-v2/bin")+":/usr/bin:/bin",
            LD_LIBRARY_PATH=str(cuda/"lib")+":"+str(ROOT/"environment-v2/lib"),
            PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",
            CUDA_VISIBLE_DEVICES=policy(ROOT)["expected_uuid"])
        save(output/"launch.json",{"pin":args.pin,"baseline":BASELINE,
            "scope":"STANDALONE_FIXED_FACTOR_CUDA_RESPONSE_NOT_FULL_PSI4",
            "deadline_seconds":600,"explicit_memory_limit_bytes":512*1024**2,
            "opaque_reserved_bytes":128*1024**2,"lease":policy(ROOT)})
        run_command([str(ROOT/"environment-v2/bin/python"),"-B",str(Path(__file__)),
            "--attempt",args.attempt,"--pin",args.pin,"--output",str(output),"--child"],
            ROOT,output,"cuda-response-probe",env,fd,600)
        if json.loads((output/"result.json").read_text())["status"]!="PASS_RESPONSE_CUDA_MICROBENCH_ONLY":
            raise RuntimeError("No standalone CUDA response pass")
        print("PASS_RESPONSE_CUDA_MICROBENCH_ONLY",output,flush=True)


if __name__=="__main__":main()
