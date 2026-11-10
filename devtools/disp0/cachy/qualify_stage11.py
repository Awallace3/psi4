"""Freeze completed whole Cachy11 CPU proof, then run the shared small gate.

No copied scientific modules, dependency installs, or Phoenix submission.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"phoenix_validation"))
from gpu_runtime import digest,optional_inventory,read_config
from coherent_stage11 import lease,policy,save,clean,git,run_command


def freeze(root,output):
    source=root/"factory-source"
    prefix=root/"install/psi4-cuest-11"
    vpath=root/"runs/psi4-cuest-import-14/receipt.json"
    bpath=root/"runs/psi4-cuest-build-11/receipt.json"
    cpath=root/"runs/psi4-cuest-11-config/receipt.json"
    npath=vpath.parent/"native-result.json"
    v,b,c,n=(json.loads(p.read_text()) for p in (vpath,bpath,cpath,npath))
    pin=c["pin"];clean(source,pin)
    if (v["status"]!="passed-whole-installed-cuest-enabled-baseline"
            or b["status"]!="installed-not-import-validated"
            or c["status"]!="configured-not-built" or v["source_pin"]!=pin
            or b["pin"]!=pin or b["prefix"]!=str(prefix) or c["prefix"]!=str(prefix)
            or v["build_receipt_sha256"]!=digest(bpath)
            or b["configuration_sha256"]!=digest(cpath)
            or v["native_result_sha256"]!=digest(npath)):
        raise RuntimeError("Complete coherent new whole11 CPU qualification required")
    sys.path.insert(0,str(root/"metadata"))
    from staging_integrity import require_proof,require_packages,require_regenerated
    from stage_native_sonames import verify
    require_proof(c["source_version_proof"],c["source_version_proof_sha256"])
    require_packages(root/"environment-v2",c["package_record_sha256"])
    require_regenerated(b["nested_manifest_before"],b["nested_manifest_after"],b["phase_start_ns"])
    if digest(b["nested_install_manifest"])!=b["nested_install_manifest_sha256"]:
        raise RuntimeError("Fresh nested whole manifest changed")
    alias_path,alias=verify(root)
    if digest(alias_path)!=v["native_alias_receipt_sha256"]:
        raise RuntimeError("Startup aliases changed")
    runtime=root/"environment-v2"
    site=runtime/"lib/python3.13/site-packages"
    versions={};names={"cuda"}
    for name in ("cuda-bindings","cuda-pathfinder"):
        distribution=importlib.metadata.distribution(name)
        if Path(distribution.locate_file("")).resolve()!=site.resolve():
            raise RuntimeError("Optional distribution not in selected dependency environment")
        versions[name]=distribution.version
        names.update(Path(str(p)).parts[0] for p in (distribution.files or ())
                     if Path(str(p)).parts and Path(str(p)).parts[0].endswith(".dist-info"))
    if versions!={"cuda-bindings":"13.3.0","cuda-pathfinder":"1.8.3"}:
        raise RuntimeError("Pinned CUDA13.3 bindings/pathfinder required")
    package=prefix/"lib/python3.13/site-packages/psi4"
    if {p.relative_to(package).as_posix():digest(p) for p in package.rglob("*.py")}!=n["installed_python_files"]:
        raise RuntimeError("Installed scientific files changed since whole smoke")
    proof={"status":"PASS","scope":"WHOLE_BUILD_CPU_SMOKE_ONLY_NOT_GPU_QUALIFICATION",
        "pin":pin,"stage":str(prefix),"launcher_sha256":digest(prefix/"bin/psi4"),
        "python_package":str(package),"scientific_python_sha256":n["installed_python_files"],
        "module_files":n["module_files"],"dependency_python_sha256":n["dependency_python_sha256"],
        "binary_sha256":n["native_library_sha256"],"provenance_sha256":{}}
    if proof["module_files"]["psi4.core"] not in proof["binary_sha256"]:
        raise RuntimeError("Smoke did not bind its actual mapped core")
    provenance=[vpath,bpath,cpath,npath,Path(c["source_version_proof"]),
                Path(b["nested_install_manifest"]),alias_path,root/"metadata/native-tmpfs-plan.json"]
    provenance+=list((root/"metadata").glob("*.py"))
    provenance+=list(HERE.glob("*.py"))+list((HERE.parent/"phoenix_validation").glob("*.py"))
    proof["provenance_sha256"]={str(p):digest(p) for p in provenance}
    save(output/"whole-build-proof.json",proof)
    plan=json.loads((root/"metadata/native-tmpfs-plan.json").read_text())
    manifest_path=root/"runs/native-tmpfs-stage-01/cuda-manifest.json"
    manifest=json.loads(manifest_path.read_text())
    import hashlib
    manifest_sha=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),
                                          allow_nan=False).encode()).hexdigest()
    if manifest_sha!=plan["deployment_cuda_manifest_sha256"]:
        raise RuntimeError("Explicit deployed CUDA closure differs")
    lock={"scope":"EXISTING_CONDA_CUDA_BINDINGS_NO_INSTALL_OR_WHEEL_FALLBACK",
          "versions":versions,"package_record_sha256":c["package_record_sha256"]}
    save(output/"optional-artifact-lock.json",lock)
    alias_dir=Path(alias["directory"])
    aliases={str(p):os.readlink(p) for p in alias_dir.iterdir() if p.is_symlink()}
    if not aliases:raise RuntimeError("No verified startup aliases")
    config={"schema":2,"device_family":"RTX3090","local_lease":policy(root),
        "source_pin":pin,"stage":str(prefix),
        "build_proof":str(output/"whole-build-proof.json"),
        "build_proof_sha256":digest(output/"whole-build-proof.json"),
        "wheel_lock":str(output/"optional-artifact-lock.json"),
        "wheel_lock_sha256":digest(output/"optional-artifact-lock.json"),
        "cuda_manifest":str(manifest_path),"cuda_manifest_sha256":manifest_sha,
        "cuest_root":plan["sdk_root"],
        "cuest_archive":str(root/"dependencies/cuest-artifact/cuest_v0.2.50.tar.gz"),
        "einsums_prefix":str(runtime),"dependency_environment":str(runtime),
        "optional_python_path":str(site),"optional_inventory_roots":sorted(names),
        "optional_package_versions":versions,"threads":4,"host_memory":"2 GiB",
        "device_memory_limit_bytes":1024**3,"native_persistent_bytes":512*1024**2,
        "startup_library_path":v["native_runtime_library_path"],
        "native_alias_directory":str(alias_dir),"native_alias_links":aliases,
        "metadata_path":str(root/"metadata"),
        "native_preload_paths":[str(Path(plan["cuda_root"])/"lib"/name)
            for name in plan["explicit_dependency_load_order_after_matching_core_before_einsums"]]
            +[str(Path(plan["sdk_root"])/"lib/13/libcuest.so.0.2.50")]}
    config["optional_files"]=optional_inventory(config)
    save(output/"runtime-config.json",config)
    read_config(output/"runtime-config.json",digest(output/"runtime-config.json"))
    return config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--attempt",required=True)
    args=parser.parse_args();root=args.root.resolve(strict=True)
    if root!=Path.home()/"gits/psi4-disp0-cachy" or not args.attempt.replace("-","").isalnum():
        raise RuntimeError("Explicit owned root and fresh simple attempt required")
    helper_repo=HERE.parents[2];helper_pin=git(helper_repo,"rev-parse","HEAD");clean(helper_repo,helper_pin)
    with lease(root) as fd:
        output=root/"runs"/args.attempt;output.mkdir()
        config=freeze(root,output)
        (output/"scratch").mkdir()
        env={k:os.environ[k] for k in ("HOME","USER","DISP0_LOCAL_LEASE_FD") if k in os.environ}
        env.update(PATH=config["dependency_environment"]+"/bin:/usr/bin:/bin",
            PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
            DISP0_PIN=helper_pin,PSI_SCRATCH=str(output/"scratch"))
        command=[config["dependency_environment"]+"/bin/python","-B",
                 str(HERE.parent/"phoenix_validation/run_gate.py"),"--config",
                 str(output/"runtime-config.json"),"--config-sha256",
                 digest(output/"runtime-config.json"),"--output",str(output)]
        run_command(command,root,output,"whole-small-gate",env,fd,3350)
        result=json.loads((output/"gate-result.json").read_text())
        if result["status"]!="PASS_SMALL_CACHY_ONLY":
            raise RuntimeError("No full new Cachy small-GPU qualification")
        print("PASS_SMALL_CACHY_ONLY",output,flush=True)


if __name__=="__main__":main()
