"""Authenticated whole-stage import order for the shared small scientific gate."""
import argparse
import ast
import ctypes
import importlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"phoenix_validation"))
from gpu_runtime import (digest,read_config,require_local_lease,require_cuda_manifest,
    verify_qualified_imports,require_runtime_support,sealed_startup_path,verify_modules)

SEALED = "process-owned-kernel-sealed-snapshots-v1"
NATIVE_SHA = "65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c"


def native_symbols(package):
    """Read authenticated ABI declarations without importing the science package."""
    folder=Path(package)/"driver/procrouting/sapt"
    tree=ast.parse((folder/"sapt_cuest250_identity.py").read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="configure_abi")
    table=next(n.value for n in function.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=="specs" for t in n.targets))
    names=[ast.literal_eval(key) for key in table.keys]
    tree=ast.parse((folder/"sapt_cuest_factory_abi.py").read_text())
    table=next(n.value for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=="FACTORY_ABI" for t in n.targets))
    names.extend(ast.literal_eval(key) for key in table.keys)
    if len(names)!=25 or len(set(names))!=25:
        raise RuntimeError("Canonical native/bootstrap capability table changed")
    return tuple(names)


def prepare_sealed_owner(config,build,manifest):
    """Normal import of authenticated, coherently installed stdlib support."""
    if "psi4" in sys.modules or "einsums" in sys.modules:
        raise RuntimeError("Sealed preparation must precede native consumers")
    root=require_runtime_support(build)
    if root is None:raise RuntimeError("Whole sealed support installation required")
    directory=Path(os.environ["DISP0_SEALED_ALIAS_DIRECTORY"])
    if os.environ.get("LD_LIBRARY_PATH")!=sealed_startup_path(config,directory):
        raise RuntimeError("Exact sealed startup environment required")
    sys.path.insert(0,str(root.parent))
    try:
        support=importlib.import_module("psi4_runtime_support")
        importlib.import_module("psi4_runtime_support.sealed")
    finally:
        sys.path.pop(0)
    verify_modules(("psi4_runtime_support",),root,build["runtime_support_sha256"])
    native=str((Path(config["cuest_root"])/"lib/13/libcuest.so.0.2.50").resolve(strict=True))
    pins={native:NATIVE_SHA}
    for row in manifest["components"]:
        path=str(Path(row["path"]).resolve(strict=True))
        if path in pins:raise RuntimeError("Duplicate sealed input locator")
        pins[path]=row["sha256"]
    owner=support.prepare_sealed_runtime(pins,alias_directory=directory,dedicated_worker=True)
    owner.preload(config["native_preload_paths"])
    owner.verify_symbols(native,native_symbols(build["python_package"]))
    return owner


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
    sealed=config.get("identity_contract")==SEALED
    startup=(sealed_startup_path(config,os.environ["DISP0_SEALED_ALIAS_DIRECTORY"])
             if sealed else config["startup_library_path"])
    if os.environ.get("LD_LIBRARY_PATH")!=startup or os.environ.get("LD_PRELOAD"):
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
    if sealed:
        owner=prepare_sealed_owner(config,build,manifest)
        owners=[owner]
    else:
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
    except BaseException:
        if sealed:owner.invalidate("Sealed core/bootstrap import failed")
        raise
    finally:
        sys.meta_path.remove(observer)
    if not observer.configured:raise RuntimeError("E2 pre-lookup policy was not executed")
    verify_qualified_imports(config,build)
    if sealed:
        native=str((Path(config["cuest_root"])/"lib/13/libcuest.so.0.2.50").resolve(strict=True))
        owner.verify_symbols(native,native_symbols(build["python_package"]))
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
