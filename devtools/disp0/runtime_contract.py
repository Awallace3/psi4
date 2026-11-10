"""Revised build-input/installed-loader guards. Never claims CUDA correctness."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

E2_PIN = "d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62"
SCIENCE_PIN = "587e172a33afaadb00eb7f574a5089533f7fc755"
COMPONENTS = {"cudart": 13, "cublas": 13, "cublasLt": 13, "cusolver": 12,
              "cusparse": 12, "nvJitLink": 13}
HEADERS = ("cuda.h", "cuda_runtime.h", "cuda_runtime_api.h", "cublas_v2.h", "cublasLt.h",
           "cusolverDn.h", "cusparse.h", "nvJitLink.h", "crt/host_config.h",
           "driver_types.h", "vector_types.h", "builtin_types.h", "nv/target")
SUPPORT = ("bin/nvcc", "bin/ptxas", "bin/fatbinary", "bin/nvlink", "bin/cudafe++",
           "nvvm/bin/cicc", "nvvm/libdevice/libdevice.10.bc")
FORBIDDEN = ("cuda-12.", "/einsums-build/", "/psi4-build/", "14182584", "14193865", "14196330")
SYSTEM_LIBRARY_ROOTS = ("/usr/lib", "/usr/lib64", "/lib", "/lib64")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=30, **kwargs)


def elf(path):
    text = command(["readelf", "-d", str(path)])
    result = {"needed": [], "rpath": [], "runpath": [], "soname": None}
    for tag, value in re.findall(r"\((NEEDED|RPATH|RUNPATH|SONAME)\)[^\n]*\[([^\]]*)\]", text):
        if tag == "SONAME":
            result["soname"] = value
        elif tag == "NEEDED":
            result["needed"].append(value)
        else:
            result[tag.lower()] = value.split(":")  # preserve empty entries for rejection
    return result


def clean_location(path, *, allowed_stage=None):
    for token in FORBIDDEN:
        if token == "/psi4-build/" and allowed_stage is not None and Path(path).is_relative_to(allowed_stage):
            continue  # This exact new installed Psi4 stage is intentional, not an old build-tree dependency.
        require(token not in str(path), f"Forbidden historical/site12 input: {path}")


def system_library_location(path):
    return any(Path(path).is_relative_to(root) for root in SYSTEM_LIBRARY_ROOTS)


def identity(path, *, allowed_stage=None):
    path = Path(path).resolve(strict=True)
    clean_location(path, allowed_stage=allowed_stage)
    require(path.is_file(), f"Not an input file: {path}")
    return {"path": str(path), "sha256": digest(path)}


def tree_inventory(root):
    """Include symlinked subdirectories, reject cycles, bind EVERY input file."""
    result = {}
    def visit(directory, relative, ancestors):
        real = directory.resolve(strict=True)
        clean_location(real)
        require(real not in ancestors, f"Cyclic SDK input directory: {directory}")
        for child in sorted(directory.iterdir()):
            name = relative / child.name
            if child.is_dir():
                visit(child, name, {*ancestors, real})
            else:
                result[str(name)] = identity(child)
    visit(Path(root), Path("."), set())
    return result


def inspect_sdk(source):
    """Read-only CPU metadata. No library fallback, package install or GPU call."""
    source = Path(source).resolve(strict=True)
    clean_location(source)
    for name in ("bin", "include", "lib64", "nvvm"):
        require((source / name).is_dir(), f"Incomplete CUDA13 developer tree: {name}")
        clean_location((source / name).resolve())
    headers = {name: identity(source / "include" / name) for name in HEADERS}
    require(re.search(r"#\s*define\s+CUDA_VERSION\s+13030\b",
                      (source / "include/cuda.h").read_text()), "Expected CUDA13.3 headers")
    # This macro can live in an included header; bind the defining file, not a
    # fabricated 'version' label or a numerical runtime call.
    definitions = [(p, p.read_text(errors="replace"))
                   for p in (source / "include").glob("*.h") if p.is_file()]
    defining = [p for p, text in definitions if re.search(r"#\s*define\s+CUDART_VERSION\s+13030\b", text)]
    require(defining, "Missing CUDA13.3 CUDART_VERSION definition")
    for path, text in definitions:
        versions = re.findall(r"#\s*define\s+CUDART_VERSION\s+(\d+)\b", text)
        require(all(v == "13030" for v in versions), f"Conflicting runtime header version: {path}")
    support = {name: identity(source / name) for name in SUPPORT}
    trees = {name: tree_inventory(source / name) for name in ("include", "bin", "nvvm")}
    version = command([str(source / "bin/nvcc"), "--version"])
    require("V13.3.73" in version, "Expected nvcc13.3.73")
    libraries = {}
    for family, major in COMPONENTS.items():
        soname = f"lib{family}.so.{major}"
        # A missing unversioned developer link is repaired only in the NEW
        # private locator. Never search PATH/site directories or change a SONAME.
        candidates = {p.resolve() for p in (source / "lib64").glob(f"lib{family}.so*") if p.is_file()}
        matches = [(p, elf(p)) for p in sorted(candidates)]
        for path, info in matches:
            require(not re.match(r"lib(cudart|cublas|cublasLt|nvJitLink)\.so\.12(?:$|\.)",
                                 info["soname"] or ""),
                    f"Mixed CUDA12/13 component inventory: {path}")
        matches = [(p, info) for p, info in matches if info["soname"] == soname]
        require(len(matches) == 1, f"Missing/ambiguous CUDA13 component: {family}")
        path, info = matches[0]
        clean_location(path)
        for needed in info["needed"]:
            require(not re.match(r"lib(cudart|cublas|cublasLt|nvJitLink)\.so\.12(?:$|\.)", needed),
                    f"CUDA12 dependency in {family}: {needed}")
        libraries[family] = {**identity(path), "soname": soname, "needed": info["needed"]}
    archives = {name: identity(source / "lib64" / name)
                for name in ("libcudart_static.a", "libcudadevrt.a")}
    # Pin all potential imported library targets, not only the six mandatory
    # ones. Configure-time inspection rejects any target outside this inventory.
    allowed = {}
    for path in (source / "lib64").rglob("*"):
        if path.is_file():
            item = identity(path)
            allowed[item["path"]] = item["sha256"]
    return {"schema": 1, "scope": "CUDA13 development identities; not device correctness",
            "source": str(source), "include": str((source / "include").resolve()),
            "library_root": str((source / "lib64").resolve()),
            "headers": headers, "cudart_defining_headers": [identity(p) for p in defining],
            "support": support, "input_trees": trees, "nvcc_version": version, "libraries": libraries,
            "archives": archives, "allowed_files": allowed}


def prepare_sdk(source, destination, expected):
    """Create only a fresh private locator over immutable, hash-bound inputs."""
    require(inspect_sdk(source) == expected, "CUDA SDK identities changed after preflight")
    destination = Path(destination)
    require(destination.is_absolute(), "Private SDK destination must be absolute")
    destination.mkdir(mode=0o700)  # fail rather than reuse any previous prefix
    source = Path(source).resolve()
    for name in ("bin", "include", "nvvm"):
        (destination / name).symlink_to(source / name, target_is_directory=True)
    lib = destination / "lib64"
    lib.mkdir()
    for path in (source / "lib64").iterdir():
        (lib / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    for family, item in expected["libraries"].items():
        name = lib / f"lib{family}.so"
        if name.is_symlink():
            name.unlink()  # private locator only; never touches the source SDK
        require(not name.exists(), f"Unexpected private SDK entry: {name}")
        name.symlink_to(item["path"])
        require(elf(name)["soname"] == item["soname"], "Developer alias changed SONAME")
    targets = destination / "targets/x86_64-linux"
    targets.mkdir(parents=True)
    (targets / "include").symlink_to(destination / "include", target_is_directory=True)
    (targets / "lib").symlink_to(lib, target_is_directory=True)
    record = {**expected, "private_prefix": str(destination)}
    write_new(destination / "sdk-contract.json", record)
    return record


def audit_elf(path, expected_paths, *, require_runpath=True):
    info = elf(path)
    entries = info["runpath"] if require_runpath else info["runpath"] + info["rpath"]
    require(not require_runpath or not info["rpath"], f"Unexpected DT_RPATH: {path}")
    require(entries and "" not in entries, f"Missing/empty installed loader path: {path}")
    require(len(entries) == len(set(entries)) and set(entries) == set(expected_paths),
            f"Installed loader paths differ: {path}: {entries}")
    for entry in entries:
        clean_location(entry)
    for needed in info["needed"]:
        require("/" not in needed, f"Absolute build dependency: {path}: {needed}")
        require(not re.match(r"lib(cudart|cublas|cublasLt|nvJitLink)\.so\.12(?:$|\.)", needed),
                f"CUDA12 installed dependency: {needed}")
    return {"path": str(path), "sha256": digest(path), "elf": info}


def parse_loader(text, *, allowed_stage=None):
    require("not found" not in text, "Unresolved installed dependency")
    result = {}
    for name, path in re.findall(r"^\s*(\S+)\s+=>\s+(/\S+)", text, re.M):
        resolved = Path(path).resolve(strict=True)
        clean_location(resolved, allowed_stage=allowed_stage)
        result[name] = resolved
    require(result, "No dynamic-loader dependency evidence")
    return result


def audit_closure(binary, sdk, prefix, environment, *, require_cuda=True):
    prefix, environment = Path(prefix).resolve(), Path(environment).resolve()
    private = Path(sdk["private_prefix"])
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C",
           "LD_LIBRARY_PATH": f"{prefix}/lib:{private}/lib64:{environment}/lib"}
    stage = prefix if not require_cuda else None
    selected = parse_loader(command(["/usr/bin/ldd", str(binary)], env=env), allowed_stage=stage)
    cuda = {v["soname"]: v for v in sdk["libraries"].values()}
    system = re.compile(r"lib(c|m|dl|rt|pthread|util|resolv)\.so(?:\.|$)|ld-linux|libcuda\.so\.1$")
    evidence = {}
    for name, path in selected.items():
        if name in cuda:
            require(str(path) == cuda[name]["path"] and digest(path) == cuda[name]["sha256"],
                    f"Wrong actual CUDA closure: {name}: {path}")
        elif name.startswith(("libcudart", "libcublas", "libcusolver", "libcusparse", "libnvJitLink")):
            raise RuntimeError(f"Unexpected CUDA component: {name}")
        elif path.is_relative_to(prefix) or path.is_relative_to(environment):
            pass
        else:
            require(system.match(name) and system_library_location(path),
                    f"Unapproved installed dependency: {name}: {path}")
        evidence[name] = identity(path, allowed_stage=stage)
    if require_cuda:
        for name in cuda:
            require(name in selected, f"Missing actual CUDA component: {name}")
    return evidence


def mapped_libraries(text):
    paths = set()
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        if not re.search(r"\.so(?:[.\s]|$)", Path(path).name):
            continue
        require(not path.endswith(" (deleted)"), f"Deleted native mapping: {path}")
        paths.add(str(Path(path).resolve(strict=True)))
    return sorted(paths)


def audit_mapped_libraries(paths, sdk, prefix, stage, environment):
    roots = [Path(p).resolve() for p in (prefix, stage, environment)]
    system = re.compile(r"lib(c|m|dl|rt|pthread|util|resolv|nsl|crypt|anl)\.so(?:\.|$)|ld-linux")
    sdk_files = sdk["allowed_files"]
    expected_cuda = {item["path"]: item for item in sdk["libraries"].values()}
    result = {}
    for name in paths:
        path = Path(name).resolve(strict=True)
        clean_location(path, allowed_stage=roots[1])
        require("/stubs/" not in str(path), "A CUDA driver stub was actually mapped")
        own = any(path.is_relative_to(root) for root in roots)
        driver = path.name.startswith(("libcuda.so", "libnvidia-"))
        if driver:
            require(system_library_location(path) or
                    any(path.is_relative_to(root) for root in
                        ("/run/nvidia/driver/usr/lib", "/run/nvidia/driver/usr/lib64")),
                    "Foreign driver mapping")
        elif str(path) in expected_cuda:
            require(digest(path) == expected_cuda[str(path)]["sha256"], "Changed mapped CUDA component")
        elif re.match(r"lib(cudart|cublas|cusolver|cusparse|nvJitLink|nvrtc|cufft|curand|npp|nvfatbin|cupti)", path.name):
            require(sdk_files.get(str(path)) == digest(path), f"Unattested extra CUDA map: {path}")
            info = elf(path)
            for value in [info["soname"] or "", *info["needed"]]:
                require(not re.match(r"lib(cudart|cublas|cublasLt|nvJitLink)\.so\.12(?:$|\.)", value),
                        "CUDA12 component in actual mapped closure")
        elif own:
            pass
        else:
            require(system.match(path.name) and system_library_location(path),
                    f"Foreign actual mapped library: {path}")
        result[str(path)] = digest(path)
    for path in expected_cuda:
        require(path in result, "Required CUDA component absent from actual process")
    return result


def audit_target_report(path, sdk):
    targets = set()
    count = 0
    for line in Path(path).read_text().splitlines():
        fields = line.split("\t")
        require(len(fields) == 5, "Malformed imported CUDA target record")
        target, prop, locator, real, observed = fields
        require(target.startswith("CUDA::") and prop.startswith("IMPORTED_LOCATION"), "Foreign target record")
        require(Path(locator).is_relative_to(Path(sdk["private_prefix"]) / "lib64")
                and str(Path(locator).resolve(strict=True)) == real, "Nonprivate CUDA target locator")
        clean_location(real)
        require(sdk["allowed_files"].get(real) == observed, "Unattested/stale CUDA target identity")
        targets.add(target)
        count += 1
    require({f"CUDA::{family}" for family in COMPONENTS} <= targets, "Missing mandatory CUDA target records")
    return {"targets": sorted(targets), "records": count, "sha256": digest(path)}


def audit_install_report(path, sdk, environment):
    seen = set()
    for line in Path(path).read_text().splitlines():
        fields = line.split("\t")
        require(len(fields) == 2 and fields[0] not in seen, "Malformed/duplicate install target")
        target, value = fields
        origin = "$ORIGIN/.." if target == "PyEinsums" else "$ORIGIN"
        require(value.split(";") == [origin, str(Path(sdk["private_prefix"]) / "lib64"),
                                     str(Path(environment) / "lib")], "Unsafe intended install loader policy")
        seen.add(target)
    require({"Einsums", "PyEinsums"} <= seen, "Incomplete Python/native install target plan")
    return {"targets": sorted(seen), "sha256": digest(path)}


def write_new(path, document):
    with open(path, "x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect-sdk")
    inspect.add_argument("--source", required=True)
    inspect.add_argument("--out", required=True)
    prepare = sub.add_parser("prepare-sdk")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--destination", required=True)
    args = parser.parse_args()
    if args.action == "inspect-sdk":
        result = inspect_sdk(args.source)
        write_new(args.out, result)
    else:
        result = prepare_sdk(args.source, args.destination, json.loads(Path(args.manifest).read_text()))
    print(json.dumps({"status": "INPUT_IDENTITIES_ONLY", "scope": result["scope"]}))


if __name__ == "__main__":
    main()
