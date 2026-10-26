"""Re-stage exact existing CUDA bytes to fresh executable tmpfs, never envs."""
import hashlib
import json
import mmap
from pathlib import Path
import re
import subprocess
import sys

SOURCE_MANIFEST_SHA="585820b1dab98790b71340de97a356ae473b260941bf8380df11a3fd4aa4869c"
COPY_HELPER_SHA="624d2df59064ee976a84f134264a72c4adb9c0f7757c0c1b83398c870a275666"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()


def relocate(original,target):
    result=json.loads(json.dumps(original));result["cuda_root"]=str(target/"cuda")
    names=[]
    for component in result["components"]:
        name=Path(component["path"]).name
        if not re.fullmatch(r"lib[A-Za-z0-9_-]+\.so\.[0-9.]+",name) or name in names:
            raise RuntimeError("Unique versioned CUDA filenames required")
        names.append(name);component["path"]=str(target/"cuda/lib"/name)
    return result


def prepare(root,output,attempt):
    # Import only the already audited file-copy/resource helpers, not any
    # installed scientific package. Preserve their original same-FD checks.
    sys.path.insert(0,str(root/"metadata"))
    helper=root/"metadata/prepare_tmpfs_native.py"
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=COPY_HELPER_SHA:
        raise RuntimeError("Previously verified native copy helper changed")
    from prepare_tmpfs_native import stable_copy,create_private_root,validate_inventory
    from native_resources import admit_resources
    from gpu_runtime import digest,require_local_lease
    from coherent_stage11 import policy
    require_local_lease(policy(root))
    target=Path("/tmp")/("psi4-disp0-"+attempt)
    if target.exists() or target.is_symlink():raise RuntimeError("Fresh private native root required")
    original=json.loads((root/"runs/cuda-closure-01/manifest.json").read_text())
    if canonical_sha(original)!=SOURCE_MANIFEST_SHA or len(original["components"])!=16:
        raise RuntimeError("Original canonical CUDA closure changed")
    deployed=relocate(original,target);items=[]
    for before,after in zip(original["components"],deployed["components"],strict=True):
        source=Path(before["path"]).resolve(strict=True)
        if not source.is_relative_to(root/"environment-v2"):
            raise RuntimeError("Frozen CUDA source escaped private environment")
        items.append((source,Path(after["path"]),before["sha256"],source.stat().st_size))
    plan=json.loads((root/"metadata/native-tmpfs-plan.json").read_text())
    plan["deployment_root"]=str(target)
    total=sum(row[3] for row in items)
    fstype=subprocess.check_output(["stat","-f","-c","%T",str(target.parent)],text=True).strip()
    if fstype!="tmpfs" or total>min(plan["maximum_copied_bytes"],2*1024**3):
        raise RuntimeError("Existing executable tmpfs and <=2GiB closure required")
    record={"status":"RUNNING","scope":"CUDA_FILE_BYTES_ONLY_NO_SDK_OR_INSTALL_CHANGES",
            "source_manifest_sha256":SOURCE_MANIFEST_SHA,"files":[],
            "target":str(target),"bytes":total,"copy_helper_sha256":COPY_HELPER_SHA,
            "resources_before":admit_resources(plan,total,"copy")}
    def save():
        (output/"cuda-restage.json").write_text(json.dumps(record,indent=2)+"\n")
    save()
    try:
        create_private_root(target)
        # Retain the original strict mapped-inode/device proof on this storage.
        package="_disp0_response_copy_identity"
        import importlib.util
        source=Path(__file__).resolve().parents[3]/"psi4/driver/procrouting/sapt/sapt_cuest250_identity.py"
        spec=importlib.util.spec_from_file_location(package,source)
        identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
        probe=target/"mapping-proof"
        probe.write_bytes(b"private executable mapping identity probe\n")
        with probe.open("rb") as stream,mmap.mmap(stream.fileno(),0,
                flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ|mmap.PROT_EXEC):
            rows=identity.mapped_rows(Path("/proc/self/maps").read_text())
            if identity.mapped_file_digest(str(probe),rows)!=digest(probe):
                raise RuntimeError("Strict executable map identity failed")
        aliases={};files={probe}
        for source,destination,sha,size in items:
            require_local_lease(policy(root))
            record["files"].append(stable_copy(source,destination,sha,size));files.add(destination)
            dynamic=subprocess.check_output(["readelf","-d",str(destination)],text=True)
            sonames=re.findall(r"\(SONAME\).*?\[([^]]+)\]",dynamic)
            if len(sonames)!=1 or Path(sonames[0]).name!=sonames[0]:
                raise RuntimeError("Unique local native SONAME required")
            alias=destination.parent/sonames[0]
            if alias!=destination:
                alias.symlink_to(destination.name);aliases[alias]=destination.name
        validate_inventory(target,files,aliases)
        require_local_lease(policy(root))
        record["resources_after"]=admit_resources(plan,total,"load")
        path=output/"cuda-manifest.json"
        path.write_text(json.dumps(deployed,sort_keys=True,separators=(",",":"))+"\n")
        record.update(status="PASS_BYTE_IDENTICAL_CUDA_RELOCATION_ONLY",
                      manifest=str(path),manifest_sha256=digest(path))
        save();return path,record["manifest_sha256"]
    except BaseException as error:
        record.update(status="FAILED",error=f"{type(error).__name__}: {error}");save();raise


def verify_mapped_cuda(identity,manifest):
    """Keep the exact mapping/byte proof; reject a second CUDA world."""
    rows=identity.mapped_rows(Path("/proc/self/maps").read_text())
    expected={row["path"]:row["sha256"] for row in manifest["components"]}
    stems={Path(path).name.split(".so")[0] for path in expected}
    prefixes=getattr(identity,"CUDA_PREFIXES",tuple(stems))
    loaded={row[2] for row in rows if Path(row[2]).name.startswith(prefixes)}
    required=("libcublas.so.","libcudart.so.","libnvrtc.so.")
    if not all(any(Path(path).name.startswith(stem) for path in loaded) for stem in required):
        raise RuntimeError("Required actual CUDA mappings missing")
    if loaded-set(expected):raise RuntimeError("Unlisted/duplicate CUDA closure mapping")
    result={}
    for path in sorted(loaded):
        actual=identity.mapped_file_digest(path,rows)
        if actual!=expected[path]:raise RuntimeError("Mapped CUDA bytes changed")
        result[path]=actual
    return result
