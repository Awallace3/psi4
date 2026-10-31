"""Normally build/install a fresh whole private prefix; never replace a baseline.

Run only in the established <=4CPU/12GiB/no-swap/128-task Cachy scope.
Nested Ninja is serial. No GPU test, package installation or module overlay.
"""
import argparse
import os
from pathlib import Path
import pwd
import subprocess
import time

from coherent_stage11 import clean,save
from response_reuse_probe import resource_scope


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--pin",required=True)
    args=parser.parse_args()
    root=args.root.resolve(strict=True)
    if root!=Path(pwd.getpwuid(os.getuid()).pw_dir)/"gits/psi4-disp0-perf-cachy":
        raise RuntimeError("Only the private performance root may be built")
    scope=resource_scope()
    clean(root/"source",args.pin)
    prefix=root/("install-owned-sealed-"+args.pin[:10])
    receipt=root/"runs"/("sealed-build-"+args.pin[:10])
    if prefix.exists() or receipt.exists():
        raise RuntimeError("Fresh whole candidate prefix/build receipt required")
    receipt.mkdir(mode=0o700)
    previous={str(p):p.read_bytes().decode() for p in root.glob("install*/BUILT_FROM")}
    environment=root.parent/"psi4-disp0-cachy/environment-v2"
    env=dict(os.environ,PATH=str(environment/"bin")+":/usr/bin:/bin",
        CONDA_PREFIX=str(environment),MATH_ROOT=str(environment),
        CUDA_HOME=str(environment),CUDA_PATH=str(environment),
        CMAKE_BUILD_PARALLEL_LEVEL="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="",PYTHONNOUSERSITE="1",
        LD_LIBRARY_PATH=str(root/"runtime/lib")+":"+str(environment/"lib"))
    for key in ("CPATH","C_INCLUDE_PATH","CPLUS_INCLUDE_PATH","LIBRARY_PATH",
                "PYTHONPATH","PYTHONHOME","PYTHONOPTIMIZE","LD_PRELOAD"):
        env.pop(key,None)
    inner=root/"build/psi4-core-prefix/src/psi4-core-build"
    commands=[
        ["cmake","-S",str(root/"source"),"-B",str(root/"build"),
         "-DCMAKE_INSTALL_PREFIX="+str(prefix)],
        # Build the actual core target, including its unconditional version
        # generation. External-project stamps alone may skip that step.
        ["cmake","--build",str(inner),"--parallel","1"],
        ["cmake","--install",str(inner)],
        ["cmake","--install",str(root/"build")],
    ]
    report={"status":"STARTED","scope":"WHOLE_BUILD_ONLY_NOT_GPU_QUALIFIED",
            "pin":args.pin,"prefix":str(prefix),"resource_scope":scope,
            "preserved_built_from":previous,"commands":[]}
    try:
        for command in commands:
            print("BUILD_COMMAND",command,flush=True)
            started=time.monotonic()
            subprocess.run(command,cwd=root,env=env,check=True,timeout=2700)
            report["commands"].append({"argv":command,"seconds":time.monotonic()-started})
        clean(root/"source",args.pin)
        for path,value in previous.items():
            if Path(path).read_text()!=value:
                raise RuntimeError("Preserved prefix provenance changed: "+path)
        (prefix/"BUILT_FROM").write_text(args.pin+"\n")
        from qualify_construction_root import installed_inventory
        installed_inventory(root,args.pin,prefix)
        report["status"]="BUILD_OK_REQUIRES_FRESH_CPU_QUALIFICATION"
        print("BUILD_OK",prefix,flush=True)
    finally:
        save(receipt/"result.json",report)


if __name__=="__main__":
    main()
