"""Future-only strict cuEST admission; never loads an unmapped SDK or creates a handle."""
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

CONTRACT = json.loads(Path(__file__).with_name("cuest-sdk-0.2.50.json").read_text())
REQUIRED_VERSION = (0, 2, 50)
ABI = "cuest-status-handle-u32ptr-v1"
CUDA_LIBRARY = re.compile(
    r"lib(?:cudart|cublas(?:Lt)?|cusolver(?:Mg)?|cusparse|curand|cufft(?:w)?|"
    r"cutensor(?:Mg)?|nccl|nvrtc(?:-builtins)?|nvJitLink|nvfatbin|nvptxcompiler)\.so(?:\.\d+)*$")
CUDA12_ABI_MAJORS = {
    "libcudart": 12, "libcublas": 12, "libcublasLt": 12,
    "libcusolver": 11, "libcusolverMg": 11, "libcusparse": 12,
    "libcurand": 10, "libcufft": 11, "libcufftw": 11,
    "libcutensor": 2, "libcutensorMg": 2, "libnccl": 2,
    "libnvrtc": 12, "libnvrtc-builtins": 12, "libnvJitLink": 12,
    "libnvfatbin": 12, "libnvptxcompiler": 12,
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def header_version(text):
    values = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        matches = re.findall(rf"^\s*#define\s+CUEST_VER_{part}\s+(\d+)\s*$", text, re.M)
        if len(matches) != 1:
            raise RuntimeError("Missing/ambiguous exact cuEST version macros")
        values.append(int(matches[0]))
    if tuple(values) != REQUIRED_VERSION:
        raise RuntimeError(f"cuEST 0.2.50 required, not {tuple(values)}")
    return values


def header_tree_identity(include):
    """FDDS/Disp0-reconciled whole-tree serialization, not the archive JSON hash."""
    include = Path(include).resolve(strict=True)
    rows = []
    # pathlib orders components ("cuest/..." before "cuest.h"), not relative
    # POSIX path strings as required by the independently reconciled contract.
    for path in sorted(include.rglob("*.h"), key=lambda p: p.relative_to(include).as_posix()):
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(include):
            raise RuntimeError("SDK header escaped include root")
        name = path.relative_to(include).as_posix()
        rows.append(f"{name}\0{sha256(resolved)}\n")
    return dict(files=len(rows), sha256=hashlib.sha256("".join(rows).encode("utf-8")).hexdigest(),
                serialization="include-relative-path-nul-sha256-newline-v1")


def verify_sdk(sdk):
    """Canonical current header/library bytes + exported capabilities, no dlopen."""
    sdk = Path(sdk).resolve(strict=True)
    tree = header_tree_identity(sdk / "include")
    if tree != CONTRACT["header_tree"]:
        raise RuntimeError("Noncanonical cuEST 0.2.50 whole header tree")
    headers = {}
    for name, expected in CONTRACT["headers"].items():
        p = (sdk / "include" / name).resolve(strict=True)
        if not p.is_relative_to(sdk) or sha256(p) != expected:
            raise RuntimeError(f"Noncanonical cuEST 0.2.50 header: {name}")
        headers[name] = expected
    version = header_version((sdk / "include/cuest.h").read_text())
    # No environment default and no lib/13 -> lib fallback.
    library = (sdk / "lib/13/libcuest.so").resolve(strict=True)
    if not library.is_relative_to(sdk) or sha256(library) != CONTRACT["library_sha256"]:
        raise RuntimeError("Canonical cuEST 0.2.50 CUDA13 library hash required")
    output = subprocess.check_output(
        ["nm", "-D", "--defined-only", str(library)], text=True, timeout=20)
    symbols = {line.split()[-1].split("@")[0] for line in output.splitlines() if line.split()}
    capabilities = {}
    for kind, required in CONTRACT["symbols"].items():
        if not set(required) <= symbols:
            raise RuntimeError(f"Missing canonical cuEST {kind} capabilities")
        capabilities[kind] = required
    return dict(schema=1, version=version, cuda_major=13, sdk=str(sdk),
                library=str(library), library_sha256=CONTRACT["library_sha256"],
                header_sha256=headers, header_tree=tree, capabilities=capabilities, verified=True,
                native_version_queried=False, native_version_abi=ABI)


def verify_build_manifest(manifest):
    """Never accept an old/self-consistent manifest as an exact-version proof."""
    saved = manifest.get("cuest_sdk_identity")
    if not isinstance(saved, dict) or saved.get("verified") is not True:
        raise RuntimeError("Build has no strict cuEST 0.2.50 header/library identity receipt")
    actual = verify_sdk(manifest["cuest_sdk"])
    for key in ("version", "cuda_major", "header_sha256", "header_tree", "library_sha256", "capabilities"):
        if actual[key] != saved.get(key):
            raise RuntimeError(f"Frozen cuEST build identity mismatch: {key}")
    if Path(manifest["cuest_library"]).resolve() != Path(actual["library"]):
        raise RuntimeError("Build selected a different cuEST library")
    if manifest.get("cuest_library_sha256") != CONTRACT["library_sha256"]:
        raise RuntimeError("Build manifest must pin the independent canonical 0.2.50 library")
    verify_cuda_receipt(manifest.get("cuda_linkage_identity"))
    return actual


def cuda_linkage_identity(linkage):
    """Hash the CUDA user-library closure resolved by ldd on the staged core.

    libcuda is the host driver, not a CUDA-toolkit library. Record/verify it in
    device provenance separately. No new native library is loaded here.
    """
    if "not found" in linkage:
        raise RuntimeError("Unresolved build linkage")
    libraries = {}
    for line in linkage.splitlines():
        fields = line.strip().split()
        if fields and CUDA_LIBRARY.fullmatch(fields[0]):
            if len(fields) < 3 or fields[1] != "=>" or not fields[2].startswith("/"):
                raise RuntimeError("Unresolved CUDA user-library linkage")
            path = Path(fields[2]).resolve(strict=True)
            libraries[fields[0]] = dict(path=str(path), sha256=sha256(path))
    verify_cuda_receipt(libraries)
    return libraries


def verify_cuda_filename(name, path, major):
    """A receipt key cannot relabel a different ABI's actual resolved file."""
    actual = Path(path).resolve(strict=True).name
    if (not CUDA_LIBRARY.fullmatch(name) or not CUDA_LIBRARY.fullmatch(actual) or
            not (actual == name or actual.startswith(name + "."))):
        raise RuntimeError("CUDA receipt label differs from resolved library ABI")
    if major == 12:
        match = re.fullmatch(r"(lib.+)\.so\.(\d+)(?:\.\d+)*", actual)
        if not match or CUDA12_ABI_MAJORS.get(match[1]) != int(match[2]):
            raise RuntimeError("Non-CUDA12 ABI in pinned PySCF closure")


def verify_cuda_receipt(libraries, major=13):
    if (not isinstance(libraries, dict) or
            {name for name in libraries if name.startswith("libcudart.so")} != {f"libcudart.so.{major}"}):
        raise RuntimeError(f"Build needs a pinned CUDA{major} user-library closure")
    for name, saved in libraries.items():
        path = Path(saved["path"])
        verify_cuda_filename(name, path, major)
        if (not path.is_absolute() or not path.is_file() or
                sha256(path) != saved["sha256"]):
            raise RuntimeError(f"CUDA build library changed: {name}")


def mapped_cuda_identity(maps, libraries, major=13):
    """Every actual mapped CUDA user-library must belong to the build closure."""
    for name, saved in libraries.items():
        verify_cuda_filename(name, saved["path"], major)
    allowed = {str(Path(saved["path"]).resolve(strict=True)): saved
               for saved in libraries.values()}
    proof = {}
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[5]
        if not CUDA_LIBRARY.fullmatch(Path(path.removesuffix(" (deleted)")).name):
            continue
        if path.endswith(" (deleted)") or path not in allowed:
            raise RuntimeError(f"Mapped CUDA library outside pinned CUDA{major} closure")
        st = Path(path).stat()
        if st.st_ino != int(fields[4]) or tuple(int(v, 16) for v in fields[3].split(":")) != (
                os.major(st.st_dev), os.minor(st.st_dev)):
            raise RuntimeError("CUDA mapped inode/device differs from pinned file")
        if path not in proof:
            if sha256(path) != allowed[path]["sha256"]:
                raise RuntimeError("Mapped CUDA library hash changed")
            proof[path] = dict(allowed[path], inode=st.st_ino, device=fields[3])
    if not any(Path(path).name.startswith(f"libcudart.so.{major}.") or
               Path(path).name == f"libcudart.so.{major}" for path in proof):
        raise RuntimeError(f"No actual mapped CUDA{major} runtime")
    return proof


def read_cuda12_receipt(expected_prefix=None):
    """Require an explicitly pre-reviewed/pinned CUDA12 inventory, never invent it.

    Capture/review of that inventory is a separate future setup operation.
    Merely discovering whatever happens to be loaded is NOT an independent pin.
    """
    name = os.environ.get("GPU_COMPARE_CUDA12_RECEIPT")
    expected = os.environ.get("GPU_COMPARE_CUDA12_RECEIPT_SHA256", "")
    if not name or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError("Explicit pinned GPU4PySCF CUDA12 receipt required")
    path = Path(name)
    if not path.is_absolute():
        raise RuntimeError("GPU4PySCF CUDA12 receipt must be absolute")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("GPU4PySCF CUDA12 receipt changed")
    receipt = json.loads(data)
    if receipt.get("cuda_major") != 12 or receipt.get("schema") != 1:
        raise RuntimeError("Not an approved CUDA12 inventory schema")
    prefix = Path(expected_prefix or sys.prefix).resolve(strict=True)
    if Path(receipt["python_prefix"]).resolve(strict=True) != prefix:
        raise RuntimeError("CUDA12 receipt belongs to a different Python environment")
    verify_cuda_receipt(receipt.get("libraries"), major=12)
    return receipt


def mapped_sdk(maps):
    rows = []
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not Path(fields[5]).name.startswith("libcuest"):
            continue
        if fields[5].endswith(" (deleted)"):
            raise RuntimeError("Mapped cuEST library was deleted/replaced")
        if not re.fullmatch(r"libcuest\.so(?:\.\d+)*", Path(fields[5]).name):
            raise RuntimeError("Unrecognized cuEST native mapping")
        start, stop = (int(x, 16) for x in fields[0].split("-"))
        rows.append(dict(start=start, stop=stop, permissions=fields[1],
                         device=fields[3], inode=int(fields[4]), path=fields[5]))
    return rows


def validate_version_receipt(value):
    """FDDS bridge checks native statuses; Python never guesses the handle ABI."""
    if (type(value) is not tuple or len(value) != 3 or
            any(type(v) is not int for v in value) or value != REQUIRED_VERSION):
        raise RuntimeError("Native live-handle cuEST version is not exact (0, 2, 50)")
    return list(value)


def checked_provider(core):
    """Presence check only. In particular, do not initialize/query a handle here."""
    provider = getattr(core, CONTRACT["native_version_provider"], None)
    if (not inspect.isbuiltin(provider) or
            provider.__module__ != getattr(core, "__name__", None)):
        raise RuntimeError("No approved native live-handle cuEST version provider")
    return provider


def verify_core(core, package):
    """Bind the builtin provider to the independently pinned staged extension."""
    package = Path(package).resolve(strict=True)
    manifest = manifest_for_package(package)
    extension = Path(core.__file__).resolve(strict=True)
    if (extension.parent != package or
            sha256(extension) != manifest["core_sha256"]):
        raise RuntimeError("Runtime core/provider differs from pinned extension")
    checked_provider(core)
    return manifest


def manifest_for_package(package):
    """Only the documented private build/stage/lib/psi4 layout is admitted."""
    package = Path(package).resolve(strict=True)
    if tuple(package.parts[-4:]) != ("build", "stage", "lib", "psi4"):
        raise RuntimeError("Unexpected staged Psi4 package layout")
    manifest = json.loads((package.parents[3] / "build-manifest.json").read_text())
    verify_build_manifest(manifest)
    if Path(manifest["package"]).resolve(strict=True) != package:
        raise RuntimeError("Runtime package differs from strict build receipt")
    return manifest


def _mapped_identity(maps):
    rows = mapped_sdk(maps)
    identities = {(r["device"], r["inode"], r["path"]) for r in rows}
    if len(identities) != 1:
        raise RuntimeError("Exactly one actual cuEST native mapping required")
    device, inode, name = identities.pop()
    path = Path(name)
    st = path.stat()
    dev = ":".join(f"{v:x}" for v in (os.major(st.st_dev), os.minor(st.st_dev)))
    if inode != st.st_ino or tuple(int(v, 16) for v in device.split(":")) != (
            os.major(st.st_dev), os.minor(st.st_dev)):
        raise RuntimeError("cuEST mapped inode/device differs from hashed file")
    identity = sha256(path)
    after = path.stat()
    if (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError("cuEST library changed while hashing")
    if identity != CONTRACT["library_sha256"]:
        raise RuntimeError("Actual mapped cuEST is not canonical 0.2.50 CUDA13")
    cudart = {int(m.group(1)) for m in re.finditer(r"libcudart\.so\.(\d+)", maps)}
    if cudart != {13}:
        raise RuntimeError("cuEST worker needs an unmixed mapped CUDA13 runtime")
    return rows, dict(path=name, device=dev, inode=inode, sha256=identity, cuda_major=13)


def runtime_identity(engine, core=None, package=None, *, require_version=True):
    """Audit this process; only query the bridge AFTER an actual JK made a handle.

    The early, require_version=False check never calls the provider. A build
    without the FDDS-owned bridge is deliberately inadmissible even if its old
    manifest faithfully pins a library that happens to be canonical.
    """
    maps = Path("/proc/self/maps").read_text()
    if engine == "pyscf":
        if mapped_sdk(maps) or re.search(r"libcudart\.so\.13(?:\D|$)", maps):
            raise RuntimeError("cuEST/CUDA13 leaked into CUDA12 GPU4PySCF process")
        receipt = read_cuda12_receipt()
        closure = mapped_cuda_identity(maps, receipt["libraries"], major=12)
        return dict(verified=True, engine=engine, cuest_absent=True,
                    cuda13_absent=True, cuda_linkage_identity=closure,
                    cuda12_receipt_sha256=os.environ["GPU_COMPARE_CUDA12_RECEIPT_SHA256"])
    if engine != "psi4":
        raise ValueError("Unknown worker engine")
    manifest = verify_core(core, package)
    rows, proof = _mapped_identity(maps)
    closure = mapped_cuda_identity(maps, manifest["cuda_linkage_identity"])
    if not hasattr(os, "RTLD_NOLOAD"):
        raise RuntimeError("Cannot verify already-loaded cuEST without RTLD_NOLOAD")
    library = ctypes.CDLL(proof["path"], mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    for names in CONTRACT["symbols"].values():
        for name in names:
            address = ctypes.cast(getattr(library, name), ctypes.c_void_p).value
            if not any(r["start"] <= address < r["stop"] and "x" in r["permissions"] for r in rows):
                raise RuntimeError(f"cuEST symbol resolved outside canonical mapping: {name}")
    # Never call vendor getters via ctypes: each needs the live C++ handle.
    # The provider owns SUCCESS checks and throws if no handle is already live.
    version = validate_version_receipt(checked_provider(core)()) if require_version else None
    final_maps = Path("/proc/self/maps").read_text()
    final, _ = _mapped_identity(final_maps)
    if final != rows:
        raise RuntimeError("cuEST native mappings changed during admission")
    final_closure = mapped_cuda_identity(final_maps, manifest["cuda_linkage_identity"])
    if final_closure != closure:
        raise RuntimeError("CUDA user-library mappings changed during admission")
    return dict(proof, verified=True, admitted=require_version, engine=engine, version=version,
                native_version_queried=require_version, native_version_abi=ABI,
                native_version_provider=CONTRACT["native_version_provider"],
                core_sha256=manifest["core_sha256"],
                cuda_linkage_identity=final_closure,
                capabilities=CONTRACT["symbols"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    text = json.dumps(verify_sdk(args.sdk), sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
