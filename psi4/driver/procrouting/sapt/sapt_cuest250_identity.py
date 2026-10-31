"""Independent fail-closed DFMO admission for canonical cuEST0.2.50 only.

No library loading or native/GPU calls at import. This is distinct from the
CUDA/Einsums build proof. Native version APIs below are bound from the actual
0.2.50 context_api.h, not inferred from a directory or SONAME.
"""
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile

SDK_VERSION = (0, 2, 50)
ARCHIVE_BYTES = 206185073
ARCHIVE_SHA256 = "a06e81489dea0b652bfc4451bcfe50d3032881a7eb6decc34cdbfc50a2076c0d"
NATIVE_SHA256 = "65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c"
HEADER_COUNT = 59
HEADER_JSON_SHA256 = "644fb4d1b64862e91d9278153aaeb78273a38717209e9ff91d10f4f9f1cecc85"
HEADER_NUL_SHA256 = "3d26a7ef607d02ee294f66f11f1374f3badbd83430f5be28ed0a05e6c8d73d18"
REQUIRED_HEADERS = {
    "include/cuest.h": "1220c8209af57c914dc976244f00a81b6f1922ee2feb4b31d5c594439c22b4aa",
    "include/cuest/context/context_api.h": "eea9ba0be5a9f5739ed947f8b97d4819d7a4f019d52629a33832028d14c6a863",
    "include/cuest/types/cuest_parameter_types.h": "ce1b4313eab84d3027d0ad5202ca17e49baf41573f9c700c9089c6535051d882",
    "include/cuest/types/workspace_api.h": "a413ae8984034467cc0b8271b2d0b3cefbd3d23cf62706c7ddb0fe531eb8d1e1",
    "include/cuest/parameters/parameter_api.h": "1c32d0ea8d3e020c1a7663fa96f66b29d9de5f8b6c54dd785ec86b33c5533781",
    "include/cuest/density_fitted_integrals/density_fitted_mo_integrals_compute_api.h":
        "c93868fcc871fd1fafea374661ad9bc188a3b40646e54eb6b906b5353f06775b",
    "include/cuest/integral_plans/density_fitted_integral_plan_api.h":
        "ceed60c3c738603eec85b9c1a8ef87110d24424ed78e9a93cb7f7f584f8f88fa",
}
CUDA_PREFIXES = ("libcudart", "libcublas", "libcusolver", "libcusparse", "libnvJitLink",
                 "libnvrtc", "libcufft", "libcurand", "libcutensor", "libcupti",
                 "libnvfatbin", "libnvvm", "libnvptxcompiler", "libcudnn",
                 "libcufile", "libnvjpeg")
CUDA13_SONAME_STEMS = ("libcudart", "libcublas", "libcublasLt", "libnvrtc",
                      "libnvrtc-builtins", "libnvJitLink")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(stream):
    result = hashlib.sha256()
    for data in iter(lambda: stream.read(1024**2), b""):
        result.update(data)
    return result.hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return digest(stream)


def header_digests(records):
    archive_records={f"cuest_PID/{name}":record for name,record in records.items()}
    json_bytes=json.dumps(archive_records,sort_keys=True,separators=(",",":")).encode()
    nul_bytes="".join(name.removeprefix("include/")+"\0"+records[name]["sha256"].lower()+"\n"
                      for name in sorted(records)).encode()
    return hashlib.sha256(json_bytes).hexdigest(),hashlib.sha256(nul_bytes).hexdigest()


def mapped_file_digest(path, mappings):
    """Compare/hash the SAME opened inode, with before/after race checks."""
    with Path(path).open("rb") as stream:
        before=os.fstat(stream.fileno())
        inode_device={(inode,device) for _,_,name,inode,device in mappings if name==path}
        expected=(before.st_ino,(os.major(before.st_dev),os.minor(before.st_dev)))
        require(inode_device == {expected},f"Mapped inode/device differs from the hashed file: {path}")
        sha=digest(stream)
        after=os.fstat(stream.fileno())
        current=Path(path).stat()
        fingerprint=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        require(fingerprint(before)==fingerprint(after)==fingerprint(current),
                f"Mapped file changed/replaced while hashing: {path}")
        return sha


def check_cuda13_name(path):
    name=Path(path).name
    for stem in CUDA13_SONAME_STEMS:
        if name.startswith(stem+".so"):
            require(re.match(re.escape(stem)+r"\.so\.13(?:\.|$)",name) is not None,
                    f"Wrong CUDA13 toolkit major in explicit closure: {path}")


def canonical_headers(archive):
    """Derive the complete header manifest from exact canonical archive bytes."""
    archive = Path(archive).resolve(strict=True)
    require(archive.stat().st_size == ARCHIVE_BYTES and file_digest(archive) == ARCHIVE_SHA256,
            "cuEST SDK archive is not canonical0.2.50 (0.2.2/implicit upgrades rejected)")
    records = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.isfile() and member.name.startswith("cuest_PID/include/"):
                relative = member.name.removeprefix("cuest_PID/")
                require(".." not in Path(relative).parts, "Unsafe SDK archive path")
                records[relative] = {"bytes":member.size, "sha256":digest(stream.extractfile(member))}
    require(all(records.get(path, {}).get("sha256") == sha for path, sha in REQUIRED_HEADERS.items()),
            "Canonical cuEST required header signatures differ")
    require(len(records)==HEADER_COUNT and header_digests(records)==(HEADER_JSON_SHA256,HEADER_NUL_SHA256),
            "Canonical whole59-header tree/serialization identity differs")
    return records


def verify_files(sdk_root, archive, *, owner=None):
    root = Path(sdk_root).resolve(strict=True)
    records = canonical_headers(archive)
    actual = {str(path.relative_to(root)) for path in (root/"include").rglob("*") if path.is_file()}
    require(actual == set(records), "cuEST header-tree inventory differs from canonical0.2.50")
    for name, record in records.items():
        path = (root/name).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == record["bytes"]
                and file_digest(path) == record["sha256"], f"Wrong cuEST0.2.50 header: {name}")
    native = (root/"lib/13/libcuest.so.0.2.50").resolve(strict=True)
    require(native.is_relative_to(root), "cuEST CUDA13 native bytes are not canonical0.2.50")
    native_sha = file_digest(native) if owner is None else owner.require_library(native,NATIVE_SHA256)
    require(native_sha == NATIVE_SHA256,
            "cuEST CUDA13 native bytes are not canonical0.2.50")
    tree_sha,nul_sha = header_digests(records)
    return {"sdk_version":"0.2.50", "sdk_root":str(root), "archive_sha256":ARCHIVE_SHA256,
            "header_tree_manifest_sha256":tree_sha, "header_tree_include_relative_nul_sha256":nul_sha,
            "header_tree_manifest_schema":"full archive-member paths -> {bytes,sha256}; compact sorted JSON",
            "header_files":{f"cuest_PID/{name}":value for name,value in records.items()},
            "native_path":str(native), "native_sha256":NATIVE_SHA256,
            "binding":"checked-in ctypes C-ABI from canonical0.2.50 headers",
            "binding_source_sha256":file_digest(__file__),
            "scope":"File identity only; native handle/version/maps still require admission"}


def configure_abi(native):
    """Canonical context/parameter/DFMO signatures, uint64 batch counts."""
    pointer, u64 = ct.c_void_p, ct.c_uint64
    specs = {
        "cuestGetMajorVersion":[pointer,ct.POINTER(ct.c_uint32)],
        "cuestGetMinorVersion":[pointer,ct.POINTER(ct.c_uint32)],
        "cuestGetPatchVersion":[pointer,ct.POINTER(ct.c_uint32)],
        "cuestGetMathMode":[pointer,ct.POINTER(ct.c_int)],
        "cuestParametersCreate":[ct.c_int,ct.POINTER(pointer)],
        "cuestParametersDestroy":[ct.c_int,pointer],
        "cuestDFMOIntegralsComputeWorkspaceQuery":
            [pointer,pointer,pointer,pointer,pointer,u64,ct.POINTER(u64),ct.POINTER(u64),
             pointer,pointer,pointer],
        "cuestDFMOIntegralsCompute":
            [pointer,pointer,pointer,pointer,pointer,u64,ct.POINTER(u64),ct.POINTER(u64),
             pointer,pointer,pointer],
    }
    try:
        for name, args in specs.items():
            function = getattr(native,name)
            function.argtypes, function.restype = args, ct.c_int
    except AttributeError as error:
        raise RuntimeError("Canonical0.2.50 native version/DFMO capability missing") from error
    return specs


def native_version(native, handle):
    require(isinstance(handle,int) and not isinstance(handle,bool) and 0 < handle < 2**64,
            "A valid nonnull admitted cuEST handle is required")
    version = []
    for name in ("Major","Minor","Patch"):
        value = ct.c_uint32()
        status = getattr(native,f"cuestGet{name}Version")(ct.c_void_p(handle),ct.byref(value))
        require(status == 0, f"Native cuEST {name} version query failed: {status}")
        version.append(value.value)
    require(tuple(version) == SDK_VERSION, f"Native cuEST must be0.2.50, not {version}")
    return tuple(version)


def native_math_mode(native, handle):
    """Read the admitted live handle's mode; FP64 buffers alone do not prove it.

    Canonical0.2.50 context_api.h uses an enum output, not uint32_t. Default
    mode0 may use mixed precision; native FP64 mode1 must be set by the owner.
    Never silently mutate a borrowed handle to make it pass admission.
    """
    require(isinstance(handle,int) and not isinstance(handle,bool) and 0 < handle < 2**64,
            "A valid nonnull admitted cuEST handle is required")
    mode = ct.c_int()
    status = native.cuestGetMathMode(ct.c_void_p(handle),ct.byref(mode))
    require(status == 0, f"Native cuEST math mode query failed: {status}")
    require(mode.value == 1, f"Native cuEST FP64 math mode1 required, not {mode.value}")
    return mode.value


def mapped_rows(text, *, resolve_once=False):
    """Preserve all segments; optionally resolve each name once per snapshot.

    The dictionary exists for this invocation only. A subsequent snapshot
    resolves every path again; no mtime, mapped-file or byte proof is cached.
    The compatibility path retains per-segment resolution.
    """
    result = []
    resolved = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            if fields[5].endswith(" (deleted)"):
                require(".so" not in fields[5], "Deleted mapped binary is not admissible")
                continue
            lo, hi = (int(x,16) for x in fields[0].split("-"))
            device=tuple(int(part,16) for part in fields[3].split(":"))
            name = fields[5]
            if not resolve_once:
                canonical = str(Path(name).resolve())
            else:
                if name not in resolved:
                    resolved[name] = str(Path(name).resolve())
                canonical = resolved[name]
            result.append((lo,hi,canonical,int(fields[4]),device))
    return result


def _require_owner(owner):
    from .sapt_cuest_owned_runtime import _OwnedRuntime, current_owned_runtime
    from .sapt_cuest_sealed_runtime import _SealedRuntime
    if (type(owner) not in (_OwnedRuntime,_SealedRuntime)
            or owner is not current_owned_runtime()):
        raise TypeError("The exact admitted process-owned immutable runtime is required")


def _owned_rows(text, owner):
    if owner is not None and owner.contract == "process-owned-kernel-sealed-snapshots-v1":
        return owner.mapped_rows(text)
    return mapped_rows(text,resolve_once=True) if owner is not None else mapped_rows(text)


def audit_symbol_owners(native, path, names, *, owner=None, text=None):
    """Fresh additional-symbol admission; no parsed rows escape or are reused.

    The preceding caller-specific native/manifest admission remains required.
    For an owned runtime, independently bind the image's actual mapped identity
    as well as every function pointer; any failure retires the dedicated worker.
    Legacy callers retain fresh filesystem-name/symbol ownership checks.
    """
    if owner is not None:
        _require_owner(owner)
        owner.check_live()
    try:
        text=Path("/proc/self/maps").read_text() if text is None else text
        rows=_owned_rows(text,owner)
        expected=str(path)
        if owner is not None:
            owner.mapped_sha256(expected,rows)
        addresses={}
        for name in names:
            address=ct.cast(getattr(native,name),ct.c_void_p).value
            owners={p for lo,hi,p,_,_ in rows if lo<=address<hi}
            require(owners=={expected},"Native symbol outside admitted image: "+name)
            addresses[name]=address
        if owner is not None:owner.check_live()
        return addresses
    except BaseException as error:
        if owner is not None:owner.invalidate("Additional native symbol audit failed: "+str(error))
        raise


def audit_maps(native, proof, cuda_manifest, *, text=None, owner=None, extra_symbols=()):
    """Fresh mapping/ABI audit, optionally under an explicit immutable owner.

    Without an owner every mapped file is hashed afresh, as before. The fast
    adapter accepts only the exact published process owner; any audit failure
    revokes it permanently rather than silently falling back to fresh hashing.
    Owned factory callers may request extra native symbols in the same fresh
    snapshot; no mapping rows or byte-proof cache is exposed to callers.
    """
    if owner is None:
        if extra_symbols:
            raise ValueError("Additional symbols require explicit owned-runtime admission")
        return _audit_maps(native,proof,cuda_manifest,text=text)
    _require_owner(owner)
    try:
        owner.check_live()
        result = _audit_maps(native,proof,cuda_manifest,text=text,owner=owner,
                             extra_symbols=extra_symbols)
        owner.check_live()
        return {**result,"identity_contract":owner.contract}
    except BaseException as error:
        owner.invalidate("Mapped native identity audit failed: "+str(error))
        raise


def _audit_maps(native, proof, cuda_manifest, *, text=None, owner=None, extra_symbols=()):
    """Exact mapped native/function ownership and explicitly rooted CUDA closure.

    cuda_manifest: {cuda_root:absolute, components:[{path:absolute,sha256:...}]}.
    Records use logical paths inside ONE CUDA root; exact resolved targets are
    compared with real maps, allowing only explicitly pinned SDK-locator links.
    Driver libcuda.so is not the CUDA toolkit and is separately hardware-gated.
    """
    require("cuest.bindings" not in sys.modules,
            "Unadmitted Python cuEST bindings loaded; this backend uses the pinned C ABI only")
    require(cuda_manifest.get("cuda_runtime_version") == 13030
            and cuda_manifest.get("nvrtc_version") == [13,3],
            "DFMO CUDA closure must declare the explicitly selected13.3 ABI")
    text = Path("/proc/self/maps").read_text() if text is None else text
    rows = _owned_rows(text,owner)
    paths = {name for _,_,name,_,_ in rows}
    expected = proof["native_path"]
    native_paths = {name for name in paths if Path(name).name.startswith("libcuest.so")}
    prove = (lambda path:owner.mapped_sha256(path,rows)) if owner is not None else (
        lambda path:mapped_file_digest(path,rows))
    require(native_paths == {expected} and prove(expected) == NATIVE_SHA256,
            "Missing/foreign/duplicate actual cuEST native world (including0.2.2)")
    for name in (*configure_abi(native),*extra_symbols):
        address = ct.cast(getattr(native,name),ct.c_void_p).value
        owners = {path for lo,hi,path,_,_ in rows if lo <= address < hi}
        require(owners == {expected}, f"cuEST symbol resolved outside canonical0.2.50: {name}")
    root = Path(cuda_manifest["cuda_root"])
    require(root.is_absolute(), "Explicit same-CUDA root required")
    allowed = {}
    for record in cuda_manifest["components"]:
        path = Path(record["path"])
        require(path.is_absolute() and path.is_relative_to(root),
                "CUDA component must be explicitly inside the selected CUDA root")
        canonical = str(path.resolve(strict=True))
        check_cuda13_name(canonical)
        require(len(record["sha256"]) == 64, "Missing CUDA component hash")
        require(canonical not in allowed, "Duplicate CUDA component identity")
        allowed[canonical] = record["sha256"]
    cuda_paths = {path for path in paths if Path(path).name.startswith(CUDA_PREFIXES)}
    require(all(any(Path(path).name.startswith(prefix) for path in cuda_paths)
                for prefix in ("libcudart.so","libcublas.so","libcublasLt.so")),
            "Actual CUDA runtime/cuBLAS closure is not mapped")
    ordered_cuda_paths = sorted(cuda_paths)
    for path in ordered_cuda_paths:
        check_cuda13_name(path)
        require(path in allowed,
                f"Foreign/changed loaded CUDA component: {path}")
    # Legacy audits hash every mapped file afresh through the original
    # same-FD/inode/device/before-and-after checks. Only independent file reads
    # run concurrently; native admission/symbol binding above stays serial.
    # Scoped shutdown joins workers even on BaseException, before any caller
    # can enter native teardown. No cache or detached work survives this call.
    if owner is not None:
        # Kernel lease checks are main-thread-affine and cheap; no hash workers
        # are necessary or allowed for this explicit process-owned adapter.
        for path in ordered_cuda_paths:
            require(prove(path) == allowed[path],
                    f"Foreign/changed loaded CUDA component: {path}")
    else:
        with ThreadPoolExecutor(max_workers=min(4,len(ordered_cuda_paths))) as workers:
            hashes = workers.map(lambda path: mapped_file_digest(path,rows),ordered_cuda_paths)
            for path, observed in zip(ordered_cuda_paths,hashes):
                require(observed == allowed[path],
                        f"Foreign/changed loaded CUDA component: {path}")
    return {"pid":os.getpid(), "native_mapped_path":expected,
            "cuda_root":str(root), "cuda_mapped_sha256":{p:allowed[p] for p in sorted(cuda_paths)},
            "scope":"Actual loaded bytes/ABI ownership; NOT DFMO numerical/device correctness"}
