"""Authenticated whole-stage import order for the shared small scientific gate."""
import argparse
import ctypes
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"phoenix_validation"))
from gpu_runtime import digest,read_config,require_local_lease,require_cuda_manifest,verify_qualified_imports


class ConfigureEinsums(importlib.abc.MetaPathFinder):
    """Use the actual E2 loader, configuring rc before caller attribute lookup."""
    def __init__(self,config):
        self.config=config
        self.configured=False

    def find_spec(self,fullname,path=None,target=None):
        if fullname!="einsums":return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None or spec.loader is None:raise ImportError("Real Einsums package required")
        original=spec.loader
        observer=self
        class Loader(importlib.abc.Loader):
            def __getattr__(self,name):
                return getattr(original,name)
            def create_module(self,spec):
                return original.create_module(spec) if hasattr(original,"create_module") else None
            def exec_module(self,module):
                original.exec_module(module)
                module.rc.gpu_disable=True
                module.rc.profile_disable=True
                module.rc.profile_report=False
                module.rc.threads=observer.config["threads"]
                module.rc.max_memory="1GB"
                module.rc.buffer_size="32MB"
                module.rc.scratch_dir=os.environ["PSI_SCRATCH"]
                if not module.rc.gpu_disable:raise RuntimeError("E2 CPU-only upstream policy failed")
                observer.configured=True
        spec.loader=Loader()
        return spec


def bootstrap(config,build):
    require_local_lease(config["local_lease"])
    if "psi4" in sys.modules or "einsums" in sys.modules:
        raise RuntimeError("Cachy bootstrap requires a fresh interpreter")
    if os.environ.get("LD_LIBRARY_PATH")!=config["startup_library_path"] or os.environ.get("LD_PRELOAD"):
        raise RuntimeError("Exact authenticated startup loader required")
    alias_dir=Path(config["native_alias_directory"])
    if {str(p) for p in alias_dir.iterdir()}!=set(config["native_alias_links"]):
        raise RuntimeError("Native alias directory entries changed")
    for filename,target in config["native_alias_links"].items():
        path=Path(filename)
        if not path.is_symlink() or os.readlink(path)!=target:
            raise RuntimeError("Native startup alias changed")
    manifest=require_cuda_manifest(config)
    allowed={item["path"]:item["sha256"] for item in manifest["components"]}
    allowed.update(build["binary_sha256"])
    owners=[]
    for filename in config["native_preload_paths"]:
        if filename not in allowed or digest(filename)!=allowed[filename]:
            raise RuntimeError("Unauthenticated native preload")
        owners.append(ctypes.CDLL(filename,mode=os.RTLD_NOW|os.RTLD_GLOBAL))
    # read_config authenticated the helper and exact installed package/core.
    sys.path.insert(0,config["metadata_path"])
    try:from import_order import import_psi4_core_first
    finally:sys.path.pop(0)
    observer=ConfigureEinsums(config)
    sys.meta_path.insert(0,observer)
    try:
        psi4,events=import_psi4_core_first(Path(config["stage"]),
            before_einsums=lambda core:None,
            expected_core_path=build["module_files"]["psi4.core"],
            expected_core_sha256=build["binary_sha256"][build["module_files"]["psi4.core"]])
    finally:
        sys.meta_path.remove(observer)
    if not observer.configured:raise RuntimeError("E2 pre-lookup policy was not executed")
    verify_qualified_imports(config,build)
    # Keep every preload alive through scientific execution and teardown.
    return owners,events


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True)
    args,remaining=parser.parse_known_args()
    config,build=read_config(args.config,args.config_sha256)
    if config["schema"]!=2 or config["device_family"]!="RTX3090":
        raise RuntimeError("Local bootstrap cannot select a Phoenix runtime")
    owners,events=bootstrap(config,build)
    script=HERE.parent/"phoenix_validation/fragment_gate.py"
    sys.argv=[str(script),"--config",str(args.config),
              "--config-sha256",args.config_sha256,*remaining]
    runpy.run_path(str(script),run_name="__main__")


if __name__=="__main__":main()
