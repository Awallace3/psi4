"""CPU-smoke/freeze a private whole install for the construction diagnostic.

No dependency installation, scientific overlays, or GPU calculation. Existing
qualified dependency bytes are reused read-only; CUDA/cuEST copies are private.
"""
import argparse
import ast
import ctypes
import importlib.util
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"phoenix_validation"))
from gpu_runtime import digest, optional_inventory, read_config
from coherent_stage11 import clean, lease, policy, run_command, save
from response_reuse_probe import resource_scope
from response_cuda_closure import canonical_sha, COPY_HELPER_SHA


def relocate_manifest(original, directory):
    """Match every copied CUDA DSO by exact bytes, not guessed SONAME versions."""
    expected = {row["sha256"] for row in original["components"]}
    matches = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and not path.name.startswith("libcuest.so"):
            checksum = digest(path)
            if checksum not in expected or checksum in matches:
                raise RuntimeError("Unexpected/duplicate private CUDA file: "+str(path))
            matches[checksum] = path.resolve(strict=True)
    if set(matches) != expected or len(expected) != len(original["components"]):
        raise RuntimeError("Private CUDA closure is incomplete or ambiguous")
    return {
        "cuda_root":str(directory.parent),"cuda_runtime_version":original["cuda_runtime_version"],
        "nvrtc_version":original["nvrtc_version"],
        "components":[{"path":str(matches[row["sha256"]]),"sha256":row["sha256"]}
                      for row in original["components"]],
    }


def installed_inventory(root, pin):
    clean(root/"source",pin)
    if (root/"install/BUILT_FROM").read_text().strip() != pin:
        raise RuntimeError("Private install BUILT_FROM differs from the frozen source")
    package = root/"install/lib/python3.13/site-packages/psi4"
    staged = root/"build/stage/lib/python3.13/site-packages/psi4"
    inventory = {p.relative_to(package).as_posix():digest(p) for p in package.rglob("*.py")}
    if not inventory or inventory != {
            p.relative_to(staged).as_posix():digest(p) for p in staged.rglob("*.py")}:
        raise RuntimeError("Whole installed/staged Python inventories differ")
    configured = root/"build/psi4-core-prefix/src/psi4-core-build"
    generated = {"__init__.py","extras.py","metadata.py"}
    for name,checksum in inventory.items():
        source = root/"source/psi4"/name
        if name in generated:
            if digest(configured/name) != checksum:
                raise RuntimeError("Installed generated file differs from configured output: "+name)
        elif source.is_file() and digest(source) != checksum:
            raise RuntimeError("Installed scientific Python differs from source: "+name)
    # CMake generates metadata.py; it is deliberately not the checked-in
    # inplace placeholder. Bind its actual generated version to the clean pin.
    metadata = ast.parse((package/"metadata.py").read_text())
    values = {node.targets[0].id:node.value.value for node in metadata.body
              if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
              and isinstance(node.value,ast.Constant)}
    if (values.get("__version_long","").rsplit("+",1)[-1] != pin[:7]
            or values.get("__version_is_clean") != "True"):
        raise RuntimeError("Generated installed version does not identify the clean source pin")
    cores = list(package.glob("core*.so"))
    if len(cores) != 1 or digest(cores[0]) != digest(staged/cores[0].name):
        raise RuntimeError("Whole installed/staged native core differs")
    return package,cores[0],inventory


def mapped_binary_inventory(text, prefixes, extra):
    """Include exact admitted SDK preloads even outside the stage/env prefixes."""
    result = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and ".so" in fields[5]:
            path = Path(fields[5])
            if str(path) in extra or any(path.is_relative_to(prefix) for prefix in prefixes):
                result[str(path)] = digest(path)
    if any(result.get(path) != checksum for path,checksum in extra.items()):
        raise RuntimeError("Admitted SDK preload is missing/changed in actual mapped binaries")
    return result


def executable_copy(identity, manifest, sdk, archive, native, target, metadata, old_proof):
    """Per-experiment executable copies; durable build/dependency paths stay put.

    Btrfs mappings on Cachy fail the existing strict inode/device gate. Use the
    already audited same-descriptor copier, never relax the gate or repoint a
    shared alias. The returned SDK/manifest describe this fresh tmpfs copy only.
    """
    if target.exists() or target.is_symlink():
        raise RuntimeError("Fresh private executable-copy directory required")
    files = list(metadata.glob("*.py"))
    proofs = old_proof["provenance_sha256"]
    for path in files:
        if proofs.get(str(path)) != digest(path):
            raise RuntimeError("Read-only copy-helper dependency changed: "+str(path))
    if digest(metadata/"prepare_tmpfs_native.py") != COPY_HELPER_SHA:
        raise RuntimeError("Previously audited native copier changed")
    sys.path.insert(0,str(metadata))
    try:
        from prepare_tmpfs_native import stable_copy, create_private_root, validate_inventory
    finally:
        sys.path.pop(0)
    deployed = json.loads(json.dumps(manifest))
    deployed["cuda_root"] = str(target/"cuda")
    items = []
    for before,after in zip(manifest["components"],deployed["components"],strict=True):
        source = Path(before["path"])
        destination = target/"cuda/lib"/source.name
        after["path"] = str(destination)
        items.append((source,destination,before["sha256"],source.stat().st_size))
    for archived,row in native["header_files"].items():
        relative = archived.removeprefix("cuest_PID/")
        items.append((sdk/relative,target/"cuest_PID"/relative,row["sha256"],row["bytes"]))
    source = Path(native["native_path"])
    items.append((source,target/"cuest_PID/lib/13/libcuest.so.0.2.50",
                  native["native_sha256"],source.stat().st_size))
    total = sum(row[3] for row in items)
    fs = subprocess.check_output(["stat","-f","-c","%T",str(target.parent)],text=True).strip()
    if (fs != "tmpfs" or total > 2*1024**3
            or shutil.disk_usage(target.parent).free < total+1024**3):
        raise MemoryError("Executable copies need tmpfs, <=2GiB, and >=1GiB remaining free")
    create_private_root(target)
    probe = target/"strict-mapping-probe"
    probe.write_bytes(b"private executable-copy strict mapping proof\n")
    with probe.open("rb") as stream, mmap.mmap(
            stream.fileno(),0,flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ|mmap.PROT_EXEC):
        if identity.mapped_file_digest(str(probe),identity.mapped_rows(
                Path("/proc/self/maps").read_text())) != digest(probe):
            raise RuntimeError("Strict mapped identity failed on executable-copy storage")
    copied = [stable_copy(*row) for row in items]
    validate_inventory(target,{row[1] for row in items}|{probe},{})
    proof = identity.verify_files(target/"cuest_PID",archive)
    return deployed,target/"cuest_PID",proof,{
        "scope":"Temporary executable copy only; builds use durable private dependencies",
        "target":str(target),"copied_bytes":total,"files":copied,
        "strict_mapping_probe_passed":True,
        "copy_helper_provenance":{str(p):digest(p) for p in files},
    }


def prepare(root, pin, reference_path, reference_sha, output):
    if digest(reference_path) != reference_sha:
        raise RuntimeError("Frozen reference configuration changed")
    reference = json.loads(reference_path.read_text())
    if (reference["schema"],reference["device_family"]) != (2,"RTX3090"):
        raise RuntimeError("Existing qualified local dependency reference required")
    old_proof_path = Path(reference["build_proof"])
    if digest(old_proof_path) != reference["build_proof_sha256"]:
        raise RuntimeError("Reference CPU build proof changed")
    old_proof = json.loads(old_proof_path.read_text())
    if old_proof["status"] != "PASS":
        raise RuntimeError("Reference CPU build did not pass")
    environment = Path(reference["dependency_environment"]).resolve(strict=True)
    # Validate reused dependency inputs without importing an old scientific
    # stage or requiring its obsolete tmpfs CUDA deployment to still exist.
    dependencies = {**old_proof["dependency_python_sha256"],**old_proof["binary_sha256"]}
    for filename,checksum in dependencies.items():
        if Path(filename).is_relative_to(environment) and digest(filename) != checksum:
            raise RuntimeError("Read-only dependency changed: "+filename)
    if optional_inventory(reference) != reference["optional_files"]:
        raise RuntimeError("Read-only CUDA Python dependencies changed")
    metadata = Path(reference["metadata_path"])
    import_order = metadata/"import_order.py"
    if old_proof["provenance_sha256"].get(str(import_order)) != digest(import_order):
        raise RuntimeError("Authenticated core-first loader changed")
    package,core,inventory = installed_inventory(root,pin)
    original = json.loads(Path(reference["cuda_manifest"]).read_text())
    if canonical_sha(original) != reference["cuda_manifest_sha256"]:
        raise RuntimeError("Reference CUDA manifest changed")
    manifest = relocate_manifest(original,root/"runtime/lib")
    spec = importlib.util.spec_from_file_location(
        "_construction_identity",root/"source/psi4/driver/procrouting/sapt/sapt_cuest250_identity.py")
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    sdk = root/"runtime/cuest"
    archive = root/"dependencies/cuest-artifact/cuest_v0.2.50.tar.gz"
    native = identity.verify_files(sdk,archive)
    manifest,sdk,native,copy_receipt = executable_copy(
        identity,manifest,sdk,archive,native,
        Path("/tmp")/("psi4-disp0-perf-"+output.name),metadata,old_proof)
    save(output/"executable-copy.json",copy_receipt)
    save(output/"cuda-manifest.json",manifest)
    aliases = output/"native-aliases"
    aliases.mkdir()
    alias_links = {}
    # Copies retain their already-versioned SONAME locators from the surviving
    # runtime. Do not modify/repoint any alias in the shared reference root.
    execution_paths = [Path(row["path"]) for row in manifest["components"]]
    execution_paths.append(Path(native["native_path"]))
    for path in execution_paths:
        name = "libcuest.so.0" if path.name.startswith("libcuest.so.") else path.name
        link = aliases/name
        link.symlink_to(path.resolve(strict=True))
        alias_links[str(link)] = str(path.resolve(strict=True))
    paths = []
    for old in reference["native_preload_paths"]:
        if Path(old).name.startswith("libcuest.so"):
            paths.append(native["native_path"])
        else:
            checksum = next(row["sha256"] for row in original["components"] if row["path"] == old)
            paths.append(next(row["path"] for row in manifest["components"] if row["sha256"] == checksum))
    config = {key:reference[key] for key in (
        "optional_python_path","optional_inventory_roots","optional_package_versions",
        "optional_files","einsums_prefix","dependency_environment")}
    lock = {"scope":"READ_ONLY_EXISTING_DEPENDENCIES_NO_INSTALL",
            "reference_config_sha256":reference_sha,
            "versions":config["optional_package_versions"]}
    save(output/"optional-artifact-lock.json",lock)
    config.update(schema=2,device_family="RTX3090",local_lease=policy(root),
        source_pin=pin,stage=str(root/"install"),threads=4,host_memory="2 GiB",
        build_proof=str(output/"whole-build-proof.json"),
        wheel_lock=str(output/"optional-artifact-lock.json"),
        wheel_lock_sha256=digest(output/"optional-artifact-lock.json"),
        cuda_manifest=str(output/"cuda-manifest.json"),cuda_manifest_sha256=canonical_sha(manifest),
        cuest_root=str(sdk),cuest_archive=str(archive),
        device_memory_limit_bytes=1024**3,native_persistent_bytes=512*1024**2,
        startup_library_path=str(aliases)+":"+str(environment/"lib"),
        native_alias_directory=str(aliases),native_alias_links=alias_links,
        metadata_path=str(metadata),native_preload_paths=paths)
    initial = {"config":config,"core":str(core),"core_sha256":digest(core),
        "package":str(package),"scientific_python_sha256":inventory,
        "sdk_native_path":native["native_path"],"sdk_native_sha256":native["native_sha256"],
        "reference_config":str(reference_path),"reference_config_sha256":reference_sha,
        "copy_helper_provenance":copy_receipt["copy_helper_provenance"],
        "reference_build_proof":str(old_proof_path),
        "reference_build_proof_sha256":digest(old_proof_path)}
    save(output/"preparation.json",initial)
    return initial


def child(args):
    if digest(args.preparation) != args.preparation_sha256:
        raise RuntimeError("Frozen private preparation changed")
    initial = json.loads(args.preparation.read_text())
    config = initial["config"]
    clean(args.root/"source",args.pin)
    package,core,inventory = installed_inventory(args.root,args.pin)
    if str(core) != initial["core"] or digest(core) != initial["core_sha256"]:
        raise RuntimeError("Private core changed before smoke")
    if os.environ.get("LD_LIBRARY_PATH") != config["startup_library_path"] or os.environ.get("LD_PRELOAD"):
        raise RuntimeError("Exact private startup library closure required")
    owners = []
    manifest = json.loads(Path(config["cuda_manifest"]).read_text())
    if canonical_sha(manifest) != config["cuda_manifest_sha256"]:
        raise RuntimeError("Private CUDA manifest changed before smoke")
    allowed = {row["path"]:row["sha256"] for row in manifest["components"]}
    allowed[initial["sdk_native_path"]] = initial["sdk_native_sha256"]
    for filename in config["native_preload_paths"]:
        if filename not in allowed or digest(filename) != allowed[filename]:
            raise RuntimeError("Private preload changed")
        owners.append(ctypes.CDLL(filename,mode=os.RTLD_NOW|os.RTLD_GLOBAL))
    sys.path.insert(0,config["metadata_path"])
    try:
        from import_order import import_psi4_core_first
    finally:
        sys.path.pop(0)
    from bootstrap_gate import ConfigureEinsums
    observer = ConfigureEinsums(config)
    sys.path.insert(0,str(package.parent))
    sys.meta_path.insert(0,observer)
    try:
        psi4,events = import_psi4_core_first(Path(config["stage"]),
            before_einsums=lambda core:None,expected_core_path=str(core),
            expected_core_sha256=initial["core_sha256"])
    finally:
        sys.meta_path.remove(observer)
    if not observer.configured or Path(psi4.__file__).resolve().parent != package:
        raise RuntimeError("Whole private import/core-first policy failed")
    import einsums, einsums._core, einsums.sealed, waggle, qcelemental, qcengine, optking
    if einsums.sealed.world()["git_commit"] != "06404fa69ce9346443d7d4f19a250f6db9f0f62c":
        raise RuntimeError("Reused Einsums native world differs")
    psi4.set_memory("2 GiB")
    psi4.set_num_threads(4)
    psi4.core.set_output_file(str(args.output/"cpu-smoke.out"),False)
    psi4.set_options({"use_cuest":False,"scf_type":"pk","reference":"rhf"})
    molecule = psi4.geometry("0 1\nH 0 0 0\nH 0 0 0.74\nunits angstrom\nsymmetry c1")
    energy = float(psi4.energy("hf/sto-3g",molecule=molecule))
    if not math.isfinite(energy) or abs(energy-(-1.1167593074)) > 1e-6:
        raise RuntimeError("Private native CPU RHF smoke differs")
    modules = (psi4,psi4.core,einsums,einsums._core,waggle,qcelemental,qcengine,optking)
    module_files = {m.__name__:str(Path(m.__file__).resolve()) for m in modules}
    dependencies = {}
    for module in (einsums,waggle,qcelemental,qcengine,optking):
        directory = Path(module.__file__).resolve().parent
        for path in directory.rglob("*.py"):
            dependencies[str(path)] = digest(path)
    prefixes = tuple(Path(config[key]).resolve() for key in ("stage","dependency_environment"))
    binary = mapped_binary_inventory(Path("/proc/self/maps").read_text(),prefixes,
        {initial["sdk_native_path"]:initial["sdk_native_sha256"]})
    if str(core) not in binary:
        raise RuntimeError("Native CPU smoke did not map its exact core")
    provenance = [args.preparation,Path(config["metadata_path"])/"import_order.py",
        args.output/"executable-copy.json",
        Path(config["wheel_lock"]),args.root/"install/BUILT_FROM",
        args.root/"build/CMakeCache.txt",
        args.root/"build/psi4-core-prefix/src/psi4-core-build/install_manifest.txt"]
    provenance.extend(Path(name) for name in initial["copy_helper_provenance"])
    # Bind both this helper checkout and the unchanged scientific checkout:
    # the actual probe runs the original science pin's launch/admission helpers.
    for source in {HERE.parents[2],args.root/"source"}:
        for folder in ("devtools/disp0/cachy","devtools/disp0/phoenix_validation"):
            provenance.extend((source/folder).glob("*.py"))
    proof = {
        "status":"PASS","scope":"WHOLE_BUILD_CPU_SMOKE_ONLY_NOT_GPU_QUALIFICATION",
        "pin":args.pin,"stage":config["stage"],"python_package":str(package),
        "launcher_sha256":digest(Path(config["stage"])/"bin/psi4"),
        "scientific_python_sha256":inventory,"module_files":module_files,
        "dependency_python_sha256":dependencies,"binary_sha256":binary,
        "provenance_sha256":{str(p):digest(p) for p in provenance},
        "CPU_RHF_Eh":energy,"bootstrap_events":events,
    }
    installed_inventory(args.root,args.pin)
    save(Path(config["build_proof"]),proof)
    config["build_proof_sha256"] = digest(config["build_proof"])
    save(args.output/"runtime-config.json",config)
    read_config(args.output/"runtime-config.json",digest(args.output/"runtime-config.json"))
    print("PASS_WHOLE_PRIVATE_CPU_SMOKE_ONLY",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--pin",required=True)
    parser.add_argument("--reference-config",type=Path)
    parser.add_argument("--reference-config-sha256")
    parser.add_argument("--attempt",required=True)
    parser.add_argument("--child",action="store_true")
    parser.add_argument("--preparation",type=Path)
    parser.add_argument("--preparation-sha256")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve(strict=True)
    if (args.root != Path.home()/"gits/psi4-disp0-perf-cachy"
            or not args.attempt.replace("-","").isalnum()):
        raise RuntimeError("Explicit private performance root and fresh attempt required")
    resource_scope()
    if args.child:
        from gpu_runtime import require_local_lease
        require_local_lease(policy(args.root))
        child(args)
        return
    with lease(args.root) as fd:
        output = args.root/"runs"/args.attempt
        output.mkdir(mode=0o700)
        (output/"scratch").mkdir()
        initial = prepare(args.root,args.pin,args.reference_config,args.reference_config_sha256,output)
        config = initial["config"]
        env = {k:os.environ[k] for k in ("HOME","USER","DISP0_LOCAL_LEASE_FD") if k in os.environ}
        env.update(PATH=config["dependency_environment"]+"/bin:/usr/bin:/bin",
            PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
            LD_LIBRARY_PATH=config["startup_library_path"],CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
            TMPDIR=str(output/"scratch"),PSI_SCRATCH=str(output/"scratch"))
        argv = [sys.executable,"-B",str(Path(__file__)),"--root",str(args.root),
            "--pin",args.pin,"--attempt",args.attempt,"--child","--output",str(output),
            "--preparation",str(output/"preparation.json"),
            "--preparation-sha256",digest(output/"preparation.json")]
        run_command(argv,args.root,output,"whole-private-cpu-smoke",env,fd,240)
        print("PASS_WHOLE_PRIVATE_CPU_SMOKE_ONLY",output,flush=True)


if __name__ == "__main__":
    main()
