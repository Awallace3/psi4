"""Preserve Cachy10, normally rebuild/install Cachy11, then validate whole bytes.

Only build orchestration is shared with the existing private metadata helpers.
No installed module is replaced individually. Old installs/receipts stay intact.
The outer launcher must enforce 4 CPU/12 GiB and a bounded runtime.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"phoenix_validation"))
from gpu_runtime import digest, require_local_lease
from owned_process import execute

GPU="GPU-d36fd194-bdd6-3d53-96c9-6a67e3225eac"
OLD_PIN="9cc6784336e5c0ab6d1e99180b66e031e1d95a75"
OLD_HEAD="f735f5c194e3e71fb394d13fbcf2819b6e804c0a"


def save(path,value):
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    temporary.replace(path)


def policy(root):
    cache=root.parents[1]/".cache/psi4-cachy"
    return {"lock_path":str(cache/"gpu0.lock"),"hold_path":str(cache/"gpu0.HOLD"),
            "expected_uuid":GPU}


@contextmanager
def lease(root):
    selected=policy(root)
    if Path(selected["hold_path"]).exists():raise RuntimeError("Common GPU HOLD is active")
    fd=os.open(selected["lock_path"],os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)  # Never wait-spin.
        os.set_inheritable(stream.fileno(),True)
        os.environ["DISP0_LOCAL_LEASE_FD"]=str(stream.fileno())
        require_local_lease(selected)
        try:yield stream.fileno()
        finally:
            os.environ.pop("DISP0_LOCAL_LEASE_FD",None)
            # Close our reference, never LOCK_UN the shared open-file
            # description: an inherited owner must retain its lease until exit.


def git(source,*args):
    return subprocess.check_output(["git","-C",str(source),*args],text=True).strip()


def clean(source,pin):
    if git(source,"rev-parse","HEAD")!=pin or git(source,"status","--porcelain"):
        raise RuntimeError("Exact clean frozen source required")


def run_command(argv,root,folder,label,env,fd,seconds):
    """Monitor only this child/HOLD; retain the lease through owned-tree unwind."""
    started=time.monotonic()
    with (folder/(label+".stdout")).open("x") as out,(folder/(label+".stderr")).open("x") as err:
        code=execute(argv,env,out,err,seconds,fd,policy(root),cwd=root)
    record={"label":label,"argv":list(map(str,argv)),"exit":code,
            "seconds":time.monotonic()-started}
    if code:raise RuntimeError(f"{label} failed with exit{code}; see {folder}")
    return record


def previous(root):
    """Same provenance/byte checks as the old snapshotter, explicitly admitting10."""
    from snapshot_psi4_build import owned_receipt
    cpath,c=owned_receipt(root,root/"runs/psi4-cuest-10-config/receipt.json","configured-not-built")
    bpath,b=owned_receipt(root,root/"runs/psi4-cuest-build-10/receipt.json","installed-not-import-validated")
    vpath,v=owned_receipt(root,root/"runs/psi4-cuest-import-13/receipt.json",
                        "passed-whole-installed-cuest-enabled-baseline")
    build=root/"build/psi4-cuest-02";prefix=root/"install/psi4-cuest-10"
    if (c["pin"]!=OLD_PIN or c["build"]!=str(build) or c["prefix"]!=str(prefix)
            or c["source"]!=str(root/"factory-source") or "native_cuest_sdk" not in c
            or any(b[k]!=c[k] for k in ("pin","source","build","prefix"))
            or v["source_pin"]!=OLD_PIN or b["configuration_sha256"]!=digest(cpath)
            or v["build_receipt_sha256"]!=digest(bpath)):
        raise RuntimeError("Previous whole native10 provenance differs")
    from staging_integrity import require_proof,require_packages
    require_proof(c["source_version_proof"],c["source_version_proof_sha256"])
    require_packages(root/"environment-v2",c["package_record_sha256"])
    native_path=vpath.parent/"native-result.json"
    if digest(native_path)!=v["native_result_sha256"]:raise RuntimeError("Old native proof changed")
    native=json.loads(native_path.read_text())
    package=prefix/"lib/python3.13/site-packages/psi4"
    staged=build/"stage/lib/python3.13/site-packages/psi4"
    if (Path(native["psi4_python"]).resolve().parent!=package.resolve()
            or not Path(native["psi4_core"]).resolve().is_relative_to(package.resolve())
            or not native.get("native_cuest_identity") or not (build/"CMakeCache.txt").is_file()
            or Path(c["source_version_proof"])!=cpath.parent/"source-version/validated.json"):
        raise RuntimeError("Previous installed package/core/proof ownership differs")
    actual={p.relative_to(package).as_posix():digest(p) for p in package.rglob("*.py")}
    if (actual!=native["installed_python_files"]
            or {p.relative_to(staged).as_posix():digest(p) for p in staged.rglob("*.py")}!=actual
            or digest(native["psi4_core"])!=native["core_sha256"]
            or digest(staged/Path(native["psi4_core"]).name)!=native["core_sha256"]
            or native["source_version_proof_sha256"]!=c["source_version_proof_sha256"]):
        raise RuntimeError("Old installed/build-stage bytes or source proof changed")
    from stage_native_sonames import verify
    alias,aliases=verify(root)
    if digest(alias)!=v["native_alias_receipt_sha256"]:raise RuntimeError("Native aliases changed")
    return cpath,c,bpath,b,vpath,v


def native_configuration(root,config):
    if (digest(root/"metadata/native-tmpfs-plan.json")!=config["native_deployment_plan_sha256"]
            or digest(root/"runs/native-tmpfs-stage-01/cuda-manifest.json")!=config["native_cuda_manifest_sha256"]):
        raise RuntimeError("Exact deployed native plan/manifest changed")
    plan=json.loads((root/"metadata/native-tmpfs-plan.json").read_text())
    expected=":".join((str(Path(plan["cuda_root"])/"lib"),str(Path(plan["sdk_root"])/"lib/13"),
                       str(root/"environment-v2/lib")))
    if config["native_runtime_library_path"]!=expected:
        raise RuntimeError("Native startup closure differs")


def validation_child(root):
    """Reuse the existing generic whole-stage validator, not its old CLI whitelist."""
    from validate_psi4_stage import child
    from staging_integrity import require_proof,require_packages
    run=root/"runs/psi4-cuest-import-14"
    bpath=root/"runs/psi4-cuest-build-11/receipt.json"
    b=json.loads(bpath.read_text());cpath=Path(b["configuration"])
    if b["status"]!="installed-not-import-validated" or digest(cpath)!=b["configuration_sha256"]:
        raise RuntimeError("Completed whole build11 required")
    c=json.loads(cpath.read_text());clean(Path(c["source"]),c["pin"])
    if (c["status"]!="configured-not-built" or c["prefix"]!=str(root/"install/psi4-cuest-11")
            or c["build"]!=str(root/"build/psi4-cuest-02")
            or any(c[k]!=b[k] for k in ("pin","prefix","build","source"))
            or b["source_version_proof_sha256"]!=c["source_version_proof_sha256"]):
        raise RuntimeError("Explicit whole11 prefix/cache required")
    native_configuration(root,c)
    require_proof(c["source_version_proof"],c["source_version_proof_sha256"])
    require_packages(root/"environment-v2",c["package_record_sha256"])
    if digest(b["nested_install_manifest"])!=b["nested_install_manifest_sha256"]:
        raise RuntimeError("Fresh whole install manifest changed")
    from staging_integrity import require_regenerated
    require_regenerated(b["nested_manifest_before"],b["nested_manifest_after"],b["phase_start_ns"])
    if (Path(b["nested_install_manifest"]).stat().st_mtime_ns!=b["nested_manifest_after"]["mtime_ns"]
            or Path(c["source_version_proof"])!=cpath.parent/"source-version/validated.json"):
        raise RuntimeError("Fresh manifest/proof binding differs")
    child(root,run,Path(c["prefix"]),c["pin"],Path(c["build"]),
          Path(c["source_version_proof"]),Path(c["source"]),Path(b["nested_install_manifest"]),
          c["package_record_sha256"],c)
    # Bind actual packages and native modules for the shared CPU/GPU gate.
    result=json.loads((run/"native-result.json").read_text())
    import psi4,einsums,einsums._core
    import waggle,qcelemental,qcengine,optking
    modules=(psi4,psi4.core,einsums,einsums._core,waggle,qcelemental,qcengine,optking)
    result["module_files"]={m.__name__:str(Path(m.__file__).resolve()) for m in modules}
    dependencies={}
    for package in (einsums,waggle,qcelemental,qcengine,optking):
        for path in Path(package.__file__).resolve().parent.rglob("*.py"):
            dependencies[str(path)]=digest(path)
    for name,checksum in c["package_record_sha256"].items():
        path=root/"environment-v2/conda-meta"/name
        if digest(path)!=checksum:raise RuntimeError("Package record changed during native smoke")
        dependencies[str(path)]=checksum
    result["dependency_python_sha256"]=dependencies
    save(run/"native-result.json",result)


def build(root,pin,fd):
    from configure_incremental_psi4 import replay_commands,package_records
    from staging_integrity import require_proof,require_packages,manifest_identity,require_regenerated
    source=root/"factory-source";clean(source,OLD_HEAD)
    if (source.is_symlink() or source.resolve()!=source
            or Path(git(source,"rev-parse","--show-toplevel")).resolve()!=source
            or source.resolve()==(root/"source").resolve()
            or (root/"build/psi4-cuest-02").resolve()!=root/"build/psi4-cuest-02"):
        raise RuntimeError("Only literal owned factory source/cache may change")
    if git(HERE.parents[2],"rev-parse","HEAD")!=pin:
        raise RuntimeError("Reviewed helper/source Git pin must match")
    cpath,old,bpath,old_build,vpath,old_validation=previous(root)
    native_configuration(root,old)
    paths={name:root/"runs"/name for name in (
        "psi4-cuest-10-snapshot-01","psi4-cuest-11-config","psi4-cuest-build-11","psi4-cuest-import-14")}
    prefix=root/"install/psi4-cuest-11";snapshot=root/"build-snapshots/psi4-cuest-10-snapshot-01"
    if prefix.exists() or snapshot.exists() or any(p.exists() for p in paths.values()):
        raise RuntimeError("Fresh snapshot/config/build/import/prefix required; never overwrite")
    for path in paths.values():path.mkdir(mode=0o700)
    srun=paths["psi4-cuest-10-snapshot-01"]
    run_command([sys.executable,"-B",str(Path(__file__)),"--root",str(root),"--snapshot-child"],
                root,srun,"preserve",dict(os.environ),fd,900)
    records=json.loads((srun/"inventory.json").read_text())
    origin={"configuration":str(cpath),"build_receipt":str(bpath),"validation":str(vpath),
            "configuration_sha256":digest(cpath),"build_receipt_sha256":digest(bpath),
            "validation_sha256":digest(vpath),"pin":old["pin"],"source":old["source"],
            "prefix":old["prefix"],"core_sha256":json.loads(
                (vpath.parent/"native-result.json").read_text())["core_sha256"]}
    snap={"status":"passed-readonly-whole-build-snapshot","snapshot":str(snapshot),
          "build":old["build"],"manifest":str(srun/"inventory.json"),
          "manifest_sha256":digest(srun/"inventory.json"),"old_source_head":OLD_HEAD,
          "provenance":origin,
          "entries":len(records),"file_bytes":sum(v.get("bytes",0) for v in records.values()),
          "script_sha256":digest(__file__)}
    save(srun/"receipt.json",snap)
    require_local_lease(policy(root))
    clean(source,OLD_HEAD)
    subprocess.run(["git","-C",str(source),"merge-base","--is-ancestor",OLD_HEAD,pin],check=True)
    subprocess.run(["git","-C",str(source),"checkout","--detach",pin],check=True)
    clean(source,pin)
    runtime=root/"environment-v2";work=Path(old["build"]);crun=paths["psi4-cuest-11-config"]
    packages,changes=package_records(runtime,old["package_record_sha256"])
    env={k:os.environ[k] for k in ("HOME","USER","DISP0_LOCAL_LEASE_FD") if k in os.environ}
    env.update(PATH=f"{runtime}/bin:/usr/bin:/bin",CONDA_PREFIX=str(runtime),
        CC=f"{runtime}/bin/x86_64-conda-linux-gnu-cc",CXX=f"{runtime}/bin/x86_64-conda-linux-gnu-c++",
        FC=f"{runtime}/bin/x86_64-conda-linux-gnu-gfortran",MATH_ROOT=str(runtime),
        PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",EINSUMS_PACKED_GEMM_BATCH_PROMOTION="1",
        CMAKE_BUILD_PARALLEL_LEVEL="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    config={k:old[k] for k in ("native_cuda_manifest_sha256","native_cuest_sdk",
        "native_deployment_plan_sha256","native_runtime_library_path")}
    config.update(status="running",pin=pin,source=str(source),build=str(work),prefix=str(prefix),
        configuration=str(crun/"receipt.json"),source_version_proof=str(crun/"source-version/validated.json"),
        snapshot_receipt=str(srun/"receipt.json"),snapshot_receipt_sha256=digest(srun/"receipt.json"),
        work_build_origin=origin,
        package_record_sha256=packages,approved_package_record_changes=changes,
        script_sha256=digest(__file__),commands=[])
    save(crun/"receipt.json",config)
    for label,argv in replay_commands(old,source,work,prefix,crun):
        clean(source,pin);require_packages(runtime,packages)
        config["commands"].append(run_command(argv,root,crun,label,env,fd,600))
        if label=="source-version":config["source_version_proof_sha256"]=digest(config["source_version_proof"])
        save(crun/"receipt.json",config)
    config["status"]="configured-not-built";save(crun/"receipt.json",config)
    brun=paths["psi4-cuest-build-11"];manifest=work/"psi4-core-prefix/src/psi4-core-build/install_manifest.txt"
    built={"status":"running","pin":pin,"source":str(source),"build":str(work),"prefix":str(prefix),
           "configuration":str(crun/"receipt.json"),"configuration_sha256":digest(crun/"receipt.json"),
           "source_version_proof":config["source_version_proof"],
           "source_version_proof_sha256":config["source_version_proof_sha256"],
           "nested_manifest_before":manifest_identity(manifest),"phase_start_ns":time.time_ns(),
           "snapshot_receipt":str(srun/"receipt.json"),"script_sha256":digest(__file__),
           "snapshot_receipt_sha256":digest(srun/"receipt.json"),
           "nested_build_parallel_level":1,"commands":[]}
    save(brun/"receipt.json",built)
    for label,argv in (("build",[str(runtime/"bin/cmake"),"--build",str(work),"--parallel","1"]),
                       ("install",[str(runtime/"bin/cmake"),"--install",str(work)])):
        clean(source,pin);require_proof(config["source_version_proof"],config["source_version_proof_sha256"])
        require_packages(runtime,packages)
        print("CACHY_WHOLE_STAGE",label,flush=True)
        built["commands"].append(run_command(argv,root,brun,label,env,fd,1800))
        save(brun/"receipt.json",built)
    built["nested_manifest_after"]=manifest_identity(manifest)
    require_regenerated(built["nested_manifest_before"],built["nested_manifest_after"],built["phase_start_ns"])
    built.update(status="installed-not-import-validated",nested_install_manifest=str(manifest),
                 nested_install_manifest_sha256=digest(manifest))
    save(brun/"receipt.json",built)
    vrun=paths["psi4-cuest-import-14"];(vrun/"scratch").mkdir()
    env.update(LD_LIBRARY_PATH=old_validation["native_runtime_library_path"],PYTHONPATH="",
               PSI_SCRATCH=str(vrun/"scratch"),CUDA_VISIBLE_DEVICES="")
    validation={"status":"running","source_pin":pin,"build_receipt_sha256":digest(brun/"receipt.json"),
        "native_runtime_library_path":env["LD_LIBRARY_PATH"],
        "native_alias_receipt_sha256":old_validation["native_alias_receipt_sha256"],
        "script_sha256":digest(__file__),"commands":[]}
    save(vrun/"receipt.json",validation)
    argv=["/bin/bash","-c",'set -euo pipefail; eval "$("$1" --psiapi)"; exec "$2" -B "$3" --root "$4" --validation-child',
          "cachy-native11",str(prefix/"bin/psi4"),str(runtime/"bin/python"),str(Path(__file__)),str(root)]
    validation["commands"].append(run_command(argv,root,vrun,"native-smoke",env,fd,300))
    native_configuration(root,config)
    from stage_native_sonames import verify
    alias_path,_=verify(root)
    if digest(alias_path)!=validation["native_alias_receipt_sha256"]:
        raise RuntimeError("Startup aliases changed during whole native smoke")
    require_proof(config["source_version_proof"],config["source_version_proof_sha256"])
    clean(source,pin);require_packages(runtime,packages)
    validation.update(status="passed-whole-installed-cuest-enabled-baseline",
                      native_result_sha256=digest(vrun/"native-result.json"))
    save(vrun/"receipt.json",validation)
    clean(source,pin);require_packages(runtime,packages)
    print("PASS_CACHY11_WHOLE_BUILD_CPU_VALIDATION_ONLY",str(vrun),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--pin")
    parser.add_argument("--validation-child",action="store_true")
    parser.add_argument("--snapshot-child",action="store_true")
    args=parser.parse_args();root=args.root.resolve(strict=True)
    if root!=Path.home()/"gits/psi4-disp0-cachy":raise RuntimeError("Explicit owned Cachy root required")
    sys.path.insert(0,str(root/"metadata"))
    if args.snapshot_child:
        require_local_lease(policy(root))
        from snapshot_psi4_build import preserve
        previous(root);clean(root/"factory-source",OLD_HEAD)
        records=preserve(root/"build/psi4-cuest-02",
                         root/"build-snapshots/psi4-cuest-10-snapshot-01")
        save(root/"runs/psi4-cuest-10-snapshot-01/inventory.json",records)
        return
    if args.validation_child:
        require_local_lease(policy(root));validation_child(root);return
    if args.pin is None or re.fullmatch("[0-9a-f]{40}",args.pin) is None:
        raise RuntimeError("Exact reviewed Git pin required")
    with lease(root) as fd:
        try:build(root,args.pin,fd)
        except BaseException as error:
            for name in ("psi4-cuest-11-config","psi4-cuest-build-11","psi4-cuest-import-14"):
                path=root/"runs"/name/"receipt.json"
                if path.exists():
                    value=json.loads(path.read_text())
                    if value.get("status")=="running":
                        value.update(status="failed",error=f"{type(error).__name__}: {error}")
                        save(path,value)
            raise


if __name__=="__main__":main()
