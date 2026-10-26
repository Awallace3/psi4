"""Leased native-cuEST/CUDA packed-S cache experiment, not whole F-SAPT.

The Gaussian fixture is explicit, not a claimed Psi4 normalization/SCF proof.
All factors used by either timed arm are produced by actual native DFMO.
Downloaded factors are used only by the independent test oracle.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import types
import uuid

from response_reuse_probe import ROOT,REPO,resource_scope,hardware
from coherent_stage11 import clean,lease,policy,run_command,save
from gpu_runtime import digest,require_local_lease
from response_cuda_closure import prepare,canonical_sha,verify_mapped_cuda

BASELINE="c54d98de7f02ce13c29b15e56bd8a68b408867f4"


class GaussianShell:
    def __init__(self,center,am,auxiliary):
        self.ncenter,self.am=center,am
        self.function_index=0
        self.exponent=(.6 if auxiliary else 1.0)+.2*am+.1*center
    nprimitive=1
    @property
    def nfunction(self):return 2*self.am+1
    @property
    def ncartesian(self):return (self.am+1)*(self.am+2)//2
    def coord(self,k):return (4.*self.ncenter,0.,0.)[k]
    def exp(self,p):
        if p!=0:raise IndexError(p)
        return self.exponent
    def coef(self,p):
        if p!=0:raise IndexError(p)
        return 1.0
    original_coef=erd_coef=coef
    def is_pure(self):return True
    def is_cartesian(self):return False


class GaussianBasis:
    def __init__(self,auxiliary=False):
        self.shells=[GaussianShell(c,l,auxiliary) for c in range(2)
                     for l in range(4 if auxiliary else 3)]
        offset=cart=0;self.cart_offsets=[]
        for shell in self.shells:
            shell.function_index=offset;self.cart_offsets.append(cart)
            offset+=shell.nfunction;cart+=shell.ncartesian
    def has_ECP(self):return False
    def n_ecp_core(self):return 0
    def molecule(self):
        return types.SimpleNamespace(natom=lambda:2,x=lambda c:4.*c,
                                     y=lambda c:0.,z=lambda c:0.)
    def nshell(self):return len(self.shells)
    def nbf(self):return sum(s.nfunction for s in self.shells)
    def nao(self):return sum(s.ncartesian for s in self.shells)
    def nprimitive(self):return len(self.shells)
    def max_am(self):return max(s.am for s in self.shells)
    def shell(self,i):return self.shells[i]
    def shell_to_center(self,i):return self.shells[i].ncenter
    def shell_to_basis_function(self,i):return self.shells[i].function_index
    def shell_to_ao_function(self,i):return self.cart_offsets[i]


def fixture(np,nao,seed):
    rng=np.random.default_rng(seed);sizes={"a":2,"b":3,"r":9,"s":7}
    spaces={name:np.ascontiguousarray(rng.normal(size=(nao,sizes[name[0]]))*.1)
        for name in ("a","r","b","s","r1","s1","a2","b2","r3","s3","a4","b4")}
    corrections={name:np.ascontiguousarray(rng.normal(size=shape)*.01) for name,shape in {
        "Qar":(2,9),"SBar":(2,9),"Qbr":(3,9),"Sbr":(3,9),
        "SAbs":(3,7),"Qbs":(3,7),"Sas":(2,7),"Qas":(2,7)}.items()}
    energies={"a":np.array([-1.2,-.8]),"b":np.array([-1.3,-.9,-.6]),
              "r":np.linspace(.1,.8,9),"s":np.linspace(.2,.9,7)}
    rotations=tuple(np.linalg.qr(rng.normal(size=(n,n)))[0] for n in (2,3))
    return spaces,corrections,energies,rotations


def reference(np,rf,sf,c,e,rotations=None):
    """Full independent tensor equations from native factors, oracle-only."""
    v=np.einsum("Qra,Qsb->rasb",rf["Aar"],sf["Abs"])
    x=(np.einsum("Qra,Qsb->rasb",rf["Aar"],sf["Dbs"]+sf["Ebs"])
       +np.einsum("Qra,Qsb->rasb",rf["Dar"]+rf["Ear"],sf["Abs"])
       +np.einsum("ar,bs->rasb",c["Qar"],c["SAbs"])
       +np.einsum("ar,bs->rasb",c["SBar"],c["Qbs"]))
    w=(np.einsum("Qsa,Qrb->sarb",sf["Bas"],rf["Bbr"])
       +np.einsum("Qsa,Qrb->sarb",sf["Cas"],rf["Cbr"])
       +np.einsum("as,br->sarb",c["Sas"],c["Qbr"])
       +np.einsum("as,br->sarb",c["Qas"],c["Sbr"]))
    den=e["a"][None,:,None,None]+e["b"][None,None,None,:]-e["r"][:,None,None,None]-e["s"][None,None,:,None]
    t=v/den
    if rotations is not None:
        ua,ub=rotations
        t,v,x=(np.einsum("al,rasb,bk->rlsk",ua,z,ub) for z in (t,v,x))
        w=np.einsum("al,sarb,bk->slrk",ua,w,ub)
    return 4*np.einsum("rasb,rasb->ab",t,v),-2*(
        np.einsum("rasb,rasb->ab",t,x)+np.einsum("rasb,sarb->ab",t,w))


def child(args):
    import numpy as np
    import time
    require_local_lease(policy(ROOT));clean(REPO,args.pin)
    output=args.output
    launch=json.loads((output/"launch.json").read_text())
    manifest=json.loads((output/"cuda-manifest.json").read_text())
    if digest(output/"cuda-manifest.json")!=launch["cuda_manifest_sha256"]:
        raise RuntimeError("Private frozen CUDA closure changed")
    report={"status":"RUNNING","scope":"NATIVE_DFMO_DISPERSION_CACHE_ONLY_NOT_WHOLE_FISAPT",
            "source_pin":args.pin,"baseline_pin":BASELINE,"runs":[],"pid":os.getpid(),
            "resource_scope":resource_scope()}
    session=None;primary_error=None
    try:
        package="_disp0_native_dispersion_cache_probe"
        pkg=types.ModuleType(package);pkg.__path__=[str(REPO/"psi4/driver/procrouting/sapt")]
        sys.modules[package]=pkg
        def module(name):return importlib.import_module(package+"."+name)
        runtime=module("sapt_cuda_runtime");identity=module("sapt_cuest250_identity")
        backend=module("sapt_dispersion_cuda");setup=module("sapt_dispersion_cuda_setup")
        factory=module("sapt_cuest_factory_abi");owners=module("sapt_cuest_plan_owner")
        dfmo=module("sapt_cuest_dfmo_cuda")
        preloads=[]
        for stem in ("libcudart.so.","libnvJitLink.so.","libcublasLt.so.","libcublas.so.",
                     "libcusparse.so.","libcusolver.so.","libnvrtc-builtins.so.","libnvrtc.so."):
            component=next(c for c in manifest["components"] if Path(c["path"]).name.startswith(stem))
            preloads.append(runtime.library(component["path"],component["sha256"])[0])
        cublas=next(c for c in manifest["components"] if Path(c["path"]).name.startswith("libcublas.so."))
        old_name=package+".old_dispersion";old=types.ModuleType(old_name)
        old.__package__=package;sys.modules[old_name]=old
        source="psi4/driver/procrouting/sapt/sapt_dispersion_cuda.py"
        old_source=subprocess.check_output(["git","-C",str(REPO),"show",BASELINE+":"+source])
        exec(compile(old_source,BASELINE+":"+source,"exec"),old.__dict__)
        report["baseline_backend_sha256"]=hashlib.sha256(old_source).hexdigest()
        report["candidate_backend_sha256"]=digest(REPO/source)
        session=runtime.CUDASession(cublas_library=cublas["path"],cublas_sha256=cublas["sha256"],
            memory_limit_bytes=1024**3,native_persistent_bytes=256*1024**2)
        actual="GPU-"+str(uuid.UUID(bytes=bytes(session.rt(
            session.driver.cuDeviceGetUuid(session.rt(session.driver.cuCtxGetDevice()))).bytes)))
        if actual!=policy(ROOT)["expected_uuid"] or session.arch!="compute_86":
            raise RuntimeError("Actual CUDA context differs from leased RTX3090")
        report.update(uuid=actual,arch=session.arch,ptx_sha256=session.ptx_sha256,
                      hardware_before=hardware(actual))
        sdk=Path(launch["sdk_root"]);archive=Path(launch["archive"])
        api=factory.admit_factory_api(session=session,sdk_root=sdk,archive=archive,
            cuda_manifest=manifest,cuda_manifest_sha256=canonical_sha(manifest))
        primary,auxiliary=GaussianBasis(),GaussianBasis(True)
        report["basis_scope"]="Explicit spherical primitive-Gaussian fixture; not a Psi4 basis-normalization or SCF proof"
        with owners.DiagnosticPlanOwner(api,primary,auxiliary,
                coefficient_candidate=owners.DiagnosticPlanOwner.COEFFICIENT_CANDIDATE) as owner:
            with dfmo.CuESTDFMOProvider(session=session,sdk_root=sdk,archive=archive,
                    cuda_manifest=manifest,cuda_manifest_sha256=canonical_sha(manifest),
                    handle=owner.handle,plan=owner.plan,handle_pid=os.getpid(),
                    primary_basis=primary,auxiliary_basis=auxiliary,keepalive=(owner,),
                    plan_policy=dict(dfmo.POLICY),output_order="C",variable_bytes=32*1024**2) as provider:
                provider.prepare_device_compute()
                report["native_plan_before"]=owner.report()
                report["mapped_cuda_sha256"]=verify_mapped_cuda(identity,manifest)
                for seed in (521,522):
                    require_local_lease(policy(ROOT))
                    sp,c,e,rot=fixture(np,primary.nbf(),seed)
                    with session.scope():
                        dsp={n:session.upload(v) for n,v in sp.items()}
                        dc={n:session.upload(v) for n,v in c.items()}
                        de={n:session.upload(v) for n,v in e.items()}
                        dr=tuple(session.upload(v) for v in rot)
                        with session.scope():
                            # Native factor downloads are separate oracle data,
                            # never inputs to either production arm.
                            rf={n:session.download(v) for n,v in provider.compute_device(
                                setup.coefficient_pairs(session,dsp,"r",0,9)).items()}
                            sf={n:session.download(v) for n,v in provider.compute_device(
                                setup.coefficient_pairs(session,dsp,"s",0,7)).items()}
                        truths={label:reference(np,rf,sf,c,e,rot if label=="localized" else None)
                                for label in ("canonical","localized")}
                        retained={}
                        def execute(label,arm,budget=None,timed=False):
                            require_local_lease(policy(ROOT))
                            implementation=old if arm=="baseline" else backend
                            options={"block_size":3,"rotations":dr if label=="localized" else None}
                            if arm!="baseline":options["factor_cache_bytes"]=budget
                            before=provider.calls;baseline_bytes=session.live_bytes
                            session.synchronize_device();started=time.perf_counter()
                            # Strict negative: the backend may not upload
                            # coefficients/factors/corrections from the host.
                            upload=session.upload
                            def forbidden(*_,**__):raise RuntimeError("Backend attempted host numerical upload")
                            session.upload=forbidden
                            try:
                                result=implementation.blocked_device_dispersion(provider,dsp,dc,de,**options)
                                session.synchronize_device()
                            finally:session.upload=upload
                            seconds=time.perf_counter()-started
                            if session.live_bytes!=baseline_bytes:
                                raise RuntimeError("Packed cache/temporary bytes survived a call")
                            d,x=truths[label]
                            for field,truth in (("Disp20 occupied pairs",d),
                                                ("Exch-Disp20 occupied pairs",x),
                                                ("Disp0 occupied pairs",d+x)):
                                np.testing.assert_allclose(result[field],truth,rtol=1e-11,atol=1e-12)
                            np.testing.assert_allclose([result["Disp20,u"],result["Exch-Disp20,u"],result["Disp0,u"]],
                                [d.sum(),x.sum(),(d+x).sum()],rtol=1e-11,atol=1e-12)
                            row={"seed":seed,"representation":label,"arm":arm,"budget_bytes":budget,
                                "timed":timed,"seconds":seconds,"native_factor_batches":provider.calls-before,
                                "compute_blocks":result["Disp0 execution"]["compute_blocks"],
                                "cache":result["Disp0 execution"].get("factor_cache"),
                                "max_pair_error":max(float(np.max(np.abs(result[field]-truth)))
                                    for field,truth in (("Disp20 occupied pairs",d),("Exch-Disp20 occupied pairs",x))),
                                "signed_components":[result[n] for n in ("Disp20,u","Exch-Disp20,u","Disp0,u")]}
                            expected=12 if arm=="baseline" or not budget else (6 if budget>=8*(2*provider.naux+2)*7*5 else 10)
                            if row["native_factor_batches"]!=expected or row["compute_blocks"]!=9:
                                raise RuntimeError("Unexpected DFMO call reduction or changed contraction count")
                            report["runs"].append(row)
                            retained[label+"_"+arm]=result
                            save(output/"result.json",report)
                        if seed==521:
                            execute("canonical","baseline")
                            execute("canonical","cached",64*1024**2)
                            for repeat in range(3):
                                for arm in (("baseline","cached") if repeat%2==0 else ("cached","baseline")):
                                    execute("canonical",arm,None if arm=="baseline" else 64*1024**2,True)
                            # Exercise no-cache path and the short-tail-only
                            # partial cache using the same real native plan.
                            execute("canonical","uncached",0)
                            execute("canonical","tail-cache",8*(2*provider.naux+2)*5)
                            execute("localized","baseline")
                            execute("localized","cached",64*1024**2)
                        else:
                            execute("canonical","cached",64*1024**2)
                            execute("localized","cached",64*1024**2)
                        values={"rf_"+n:v for n,v in rf.items()}|{"sf_"+n:v for n,v in sf.items()}
                        values.update({"correction_"+n:v for n,v in c.items()})
                        values.update({"energy_"+n:v for n,v in e.items()})
                        values.update(ua=rot[0],ub=rot[1])
                        for name,result in retained.items():
                            values[name+"_d"]=result["Disp20 occupied pairs"]
                            values[name+"_x"]=result["Exch-Disp20 occupied pairs"]
                        artifact=output/f"native-case-{seed}.npz";np.savez(artifact,**values)
                        report.setdefault("artifacts",[]).append({"path":str(artifact),"sha256":digest(artifact)})
                report["provider"]=provider.report()
            report["provider_cleanup_complete"]=provider.cleanup_complete
        report["native_plan_after"]=owner.report()
        report.update(blas_calls=session.blas_calls,kernel_launches=session.kernel_launches,
                      peak_tracked_bytes=session.peak_live_bytes,hardware_after=hardware(actual))
        for arm in ("baseline","cached"):
            samples=[row["seconds"] for row in report["runs"] if row["timed"] and row["arm"]==arm]
            report.setdefault("median_seconds",{})[arm]=statistics.median(samples)
        report["diagnostic_ratio"]=report["median_seconds"]["baseline"]/report["median_seconds"]["cached"]
        if "psi4" in sys.modules:raise RuntimeError("Standalone experiment imported installed Psi4")
    except BaseException as error:
        primary_error=error;report.update(status="FAILED",error=f"{type(error).__name__}: {error}");raise
    finally:
        try:
            if session is not None:
                session.close()
                report["cleanup"]={"complete":session.cleanup_complete,"live_bytes":session.live_bytes,
                                   "native_owners":len(session.native_owners)}
            if primary_error is None:report["status"]="PASS_NATIVE_DFMO_CACHE_ONLY"
        except BaseException as error:
            report.update(status="FAILED_CLEANUP",cleanup_error=f"{type(error).__name__}: {error}");raise
        finally:save(output/"result.json",report)
    print("PASS_NATIVE_DFMO_CACHE_ONLY",output,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--attempt",required=True)
    parser.add_argument("--pin",required=True);parser.add_argument("--child",action="store_true")
    parser.add_argument("--output",type=Path);args=parser.parse_args()
    if Path.home()!=ROOT.parents[1] or not args.attempt.replace("-","").isalnum():
        raise RuntimeError("Explicit Cachy owner and fresh simple attempt required")
    clean(REPO,args.pin)
    if args.child:
        if args.output!=ROOT/"runs"/args.attempt:raise RuntimeError("Exact owned output required")
        child(args);return
    with lease(ROOT) as fd:
        resource_scope();output=ROOT/"runs"/args.attempt;output.mkdir(mode=0o700)
        path,sha=prepare(ROOT,output,args.attempt,with_cuest=True)
        manifest=json.loads(path.read_text())
        staged=json.loads((output/"cuda-restage.json").read_text())
        env={k:os.environ[k] for k in ("HOME","USER","DISP0_LOCAL_LEASE_FD") if k in os.environ}
        env.update(PATH=str(ROOT/"environment-v2/bin")+":/usr/bin:/bin",
            LD_LIBRARY_PATH=manifest["cuda_root"]+"/lib:"+staged["sdk_root"]+"/lib/13:"+str(ROOT/"environment-v2/lib"),
            PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",
            CUDA_VISIBLE_DEVICES=policy(ROOT)["expected_uuid"],CUDA_CACHE_PATH=str(output/"cuda-cache"))
        save(output/"launch.json",{"pin":args.pin,"baseline":BASELINE,"deadline_seconds":600,
            "cuda_manifest_sha256":sha,"sdk_root":staged["sdk_root"],"archive":staged["archive"],
            "scope":"NATIVE_DFMO_CACHE_MICROBENCH_NOT_FULL_PSI4"})
        run_command([str(ROOT/"environment-v2/bin/python"),"-B",str(Path(__file__)),
            "--attempt",args.attempt,"--pin",args.pin,"--child","--output",str(output)],
            ROOT,output,"native-dispersion-cache",env,fd,600)
        if json.loads((output/"result.json").read_text())["status"]!="PASS_NATIVE_DFMO_CACHE_ONLY":
            raise RuntimeError("No native DFMO cache pass")
        print("PASS_NATIVE_DFMO_CACHE_ONLY",output,flush=True)


if __name__=="__main__":main()
