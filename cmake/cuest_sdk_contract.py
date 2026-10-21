#!/usr/bin/env python3
"""Fail-closed cuEST 0.2.50 SDK/ELF admission. No CUDA calls or SDK installation.

The hashes identify the vendor archive, not its unversioned SONAME. Runtime
admission compares the actual process maps to a build-time closure; recording
whatever happened to load is not an admission check. This module is also used
by FindcuEST in both halves of the superbuild.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

VERSION = "0.2.50"
ARCHIVE_SHA256 = "a06e81489dea0b652bfc4451bcfe50d3032881a7eb6decc34cdbfc50a2076c0d"
# Sorted relative include paths, NUL, lowercase file SHA256, newline (59 files).
HEADERS_SHA256 = "3d26a7ef607d02ee294f66f11f1374f3badbd83430f5be28ed0a05e6c8d73d18"
LIBRARY_SHA256 = {
    12: "c097840157a5260a6399978c3793c7e64bacfd04a2a259af2100c8e89f57b108",
    13: "65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c",
}
# CUDA component SONAME majors are NOT all the toolkit major.
CUDA_SONAMES = {
    12: {"libcudart": 12, "libcublas": 12, "libcublasLt": 12,
         "libcusolver": 11, "libcusparse": 12, "libnvrtc": 12, "libnvJitLink": 12,
         "libnvrtc-builtins": 12, "libnvfatbin": 12},
    13: {"libcudart": 13, "libcublas": 13, "libcublasLt": 13,
         "libcusolver": 12, "libcusparse": 12, "libnvrtc": 13, "libnvJitLink": 13,
         "libnvrtc-builtins": 13, "libnvfatbin": 13},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def header_digest(include):
    digest = hashlib.sha256()
    paths = sorted(p for p in Path(include).rglob("*") if p.is_file())
    require(paths, "cuEST headers absent")
    for path in paths:
        # Do not accept a symlink escape into another SDK/header tree.
        require(path.resolve().is_relative_to(Path(include).resolve()), "cuEST header escapes SDK")
        digest.update(path.relative_to(include).as_posix().encode() + b"\0"
                      + sha256(path).encode() + b"\n")
    return digest.hexdigest()


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=45, check=False)
    require(result.returncode == 0, f"{argv[0]} failed: {result.stderr}")
    return result.stdout


def elf_dynamic(library):
    text = command(["readelf", "-d", str(library)])
    needed = re.findall(r"\(NEEDED\).*?\[([^]]+)\]", text)
    soname = re.findall(r"\(SONAME\).*?\[([^]]+)\]", text)
    return needed, soname


def parse_ldd(text):
    require("not found" not in text, "unresolved runtime library")
    found = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(/\S+)\s+\(", line)
        if match:
            name, path = match.groups()
            require(name not in found, f"duplicate ldd entry: {name}")
            found[name] = Path(path).resolve(strict=True)
    return found


def gpu_library(name):
    # libcuda is the site driver, not a toolkit component.
    return name.startswith((
        "libcuest.", "libcudart.", "libcublas.", "libcublasLt.", "libcusolver.",
        "libcusolverMg.", "libcusparse.", "libcurand.", "libcufft.", "libcufftw.",
        "libcufile.", "libcufile_rdma.", "libcudss.", "libcupti.",
        "libnvrtc.", "libnvrtc-builtins.", "libnvJitLink.", "libnvfatbin.",
        "libnvToolsExt.", "libnvblas."))


def check_cuda_names(names, major):
    for name in names:
        match = re.fullmatch(r"(lib[^.]+)\.so\.(\d+)(?:\..*)?", name)
        if match and match[1] in CUDA_SONAMES[major]:
            require(int(match[2]) == CUDA_SONAMES[major][match[1]],
                    f"cross-CUDA closure: {name} is not CUDA {major}")


def cuda_closure(libraries, root, major):
    check_cuda_names(libraries, major)
    closure = {}
    for name, path in libraries.items():
        if gpu_library(name) and not name.startswith("libcuest."):
            require(path.is_relative_to(root), f"CUDA library outside selected runtime root: {path}")
            closure[name] = {"path": str(path), "sha256": sha256(path)}
    return closure


def validate_sdk(root, major, cuda_root, cudart):
    require(major in LIBRARY_SHA256, "only canonical CUDA 12/13 SDK members are supported")
    root, cuda_root = Path(root).resolve(strict=True), Path(cuda_root).resolve(strict=True)
    include = root / "include"
    require(include.resolve(strict=True).is_relative_to(root), "cuEST include directory escapes SDK")
    library = (root / "lib" / str(major) / "libcuest.so").resolve(strict=True)
    require(library.is_relative_to(root / "lib" / str(major)), "cuEST library escapes selected CUDA directory")
    require(sha256(library) == LIBRARY_SHA256[major], "cuEST native library is not canonical 0.2.50")
    require(header_digest(include) == HEADERS_SHA256, "cuEST headers are not canonical 0.2.50")
    needed, soname = elf_dynamic(library)
    require(soname == ["libcuest.so.0"], "unexpected cuEST SONAME")
    check_cuda_names(needed, major)
    symbols = command(["readelf", "--dyn-syms", "--wide", str(library)])
    for symbol in ("cuestDFMOIntegralsCompute", "cuestDFMOIntegralsComputeWorkspaceQuery",
                   "cuestDFIntPlanCreate", "cuestDFIntPlanCreateWorkspaceQuery", "cuestDFIntPlanDestroy",
                   "cuestGetMajorVersion", "cuestGetMinorVersion", "cuestGetPatchVersion"):
        require(any(line.split()[-1:] == [symbol] and " UND " not in line
                    for line in symbols.splitlines()), f"SDK symbol missing: {symbol}")
    linked = parse_ldd(command(["ldd", str(library)]))
    cudart = Path(cudart).resolve(strict=True)
    require(cudart.is_relative_to(cuda_root), "CMake CUDA::cudart escapes selected runtime root")
    _, cudart_soname = elf_dynamic(cudart)
    require(cudart_soname == [f"libcudart.so.{major}"], "CUDA::cudart toolkit major mismatch")
    require(cudart_soname[0] not in linked or linked[cudart_soname[0]] == cudart,
            "SDK resolves a different CUDA runtime than CUDA::cudart")
    linked[cudart_soname[0]] = cudart
    closure = cuda_closure(linked, cuda_root, major)
    for name in needed:
        if gpu_library(name):
            require(name in closure, f"missing CUDA dependency: {name}")
    return dict(version=VERSION, archive_sha256=ARCHIVE_SHA256, sdk_root=str(root),
                include=str(include), header_sha256=HEADERS_SHA256,
                library=str(library), library_sha256=LIBRARY_SHA256[major],
                cuda_major=major, cuda_root=str(cuda_root), cudart=str(cudart),
                cuda_closure=closure)


def compare_identity(actual, expected):
    require(actual == expected, "cuEST SDK/CUDA identity changed; use a fresh isolated build")


def snapshot_core(core, sdk):
    """Bind a built core and full CUDA closure without importing/initializing it."""
    core = Path(core).resolve(strict=True)
    libraries = parse_ldd(command(["ldd", str(core)]))
    cuest = {name: path for name, path in libraries.items() if name.startswith("libcuest.")}
    require(len(cuest) == 1 and list(cuest.values()) == [Path(sdk["library"])],
            "core resolves a different cuEST library")
    require(sha256(next(iter(cuest.values()))) == sdk["library_sha256"], "core cuEST hash mismatch")
    closure = cuda_closure(libraries, Path(sdk["cuda_root"]), sdk["cuda_major"])
    for name, identity in sdk["cuda_closure"].items():
        require(closure.get(name) == identity, f"core CUDA dependency changed: {name}")
    require(any(n.startswith("libcudart.") for n in closure), "core CUDA runtime absent")
    # Explicit, build-time allowlist for CUDA components that may dlopen lazily.
    # Never admit arbitrary libraries just because they appear after GPU work.
    optional = {}
    root = Path(sdk["cuda_root"])
    directories = [root / "lib", root / "lib64", *root.glob("targets/*/lib")]
    for directory in directories:
        for pattern in ("libnvrtc-builtins.so*", "libnvfatbin.so*"):
            for file in directory.glob(pattern):
                path = file.resolve(strict=True)
                require(path.is_relative_to(root), f"lazy CUDA library escapes selected root: {path}")
                _, names = elf_dynamic(path)
                require(len(names) == 1 and gpu_library(names[0]), "invalid lazy CUDA SONAME")
                check_cuda_names(names, sdk["cuda_major"])
                optional[str(path)] = sha256(path)
    return dict(sdk=sdk, core=str(core), core_sha256=sha256(core),
                cuda_closure=closure, optional_cuda=optional)


def admit_runtime(core, expected):
    """Call *inside the importing process*, before any tests or GPU work."""
    sdk = expected["sdk"]
    compare_identity(validate_sdk(sdk["sdk_root"], sdk["cuda_major"], sdk["cuda_root"], sdk["cudart"]), sdk)
    compare_identity(snapshot_core(core, sdk), expected)
    mapped = set()
    core_path = Path(core).resolve(strict=True)
    core_executable = False
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            raw = fields[5]
            candidate = Path(raw.removesuffix(" (deleted)"))
            is_core = candidate.resolve(strict=False) == core_path
            if gpu_library(candidate.name) or is_core:
                require(not raw.endswith(" (deleted)"), "deleted GPU/core library mapped")
                path = candidate
                path = path.resolve(strict=True)
                stat = path.stat()
                device = f"{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x}"
                mapped_device = ":".join(f"{int(n, 16):x}" for n in fields[3].split(":"))
                require(int(fields[4]) == stat.st_ino and mapped_device == device,
                        f"mapped GPU inode/device changed: {path}")
                mapped.add(path)
                core_executable = core_executable or (is_core and "x" in fields[1])
    require(core_executable, "pinned core has no proved executable mapping")
    wanted = {Path(sdk["library"]), core_path} | {
        Path(item["path"]) for item in expected["cuda_closure"].values()}
    allowed = wanted | {Path(p) for p in expected["optional_cuda"]}
    require(wanted <= mapped <= allowed,
            f"actual mapped GPU libraries differ: required {wanted}, allowed {allowed}, got {mapped}")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--cuda-major", type=int, required=True)
    parser.add_argument("--cuda-root", required=True)
    parser.add_argument("--cudart", required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--core", type=Path)
    parser.add_argument("--expected-core", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = validate_sdk(args.sdk_root, args.cuda_major, args.cuda_root, args.cudart)
    if args.expected:
        compare_identity(identity, json.loads(args.expected.read_text()))
    if args.core:
        identity = snapshot_core(args.core, identity)
    if args.expected_core:
        require(args.core is not None, "--expected-core requires --core")
        compare_identity(identity, json.loads(args.expected_core.read_text()))
    # Never overwrite an earlier build's identity with a newly selected SDK.
    if args.output.exists():
        compare_identity(identity, json.loads(args.output.read_text()))
    else:
        args.output.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
