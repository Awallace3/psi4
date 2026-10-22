#!/usr/bin/env python3
"""Static CUDA12 inventory evidence, never an automatically approved receipt.

Run with the GPU4PySCF environment's Python on a CPU compute node. Does not
import CuPy/GPU4PySCF, dlopen a library, query a device or initialize CUDA.
Conflicting providers are retained for independent review, not guessed away.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone

from cuest_sdk import CUDA_LIBRARY, sha256, verify_cuda_filename


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed: {result.stderr[:2000]}")
    return result.stdout


def dynamic(path):
    text = command(["readelf", "-d", str(path)])
    sonames = re.findall(r"\(SONAME\).*?\[([^]]+)\]", text)
    if len(sonames) != 1:
        raise RuntimeError(f"Expected exactly one ELF SONAME: {path}")
    return dict(soname=sonames[0],
                needed=re.findall(r"\(NEEDED\).*?\[([^]]+)\]", text),
                rpath=re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^]]+)\]", text),
                raw_dynamic=text)


def stable_identity(path):
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError(f"File changed while hashing: {path}")
    return dict(sha256=digest, bytes=after.st_size, inode=after.st_ino, device=after.st_dev)


def record_proof(file, actual_hash):
    value = file.hash
    if value is None:
        return dict(algorithm=None, expected=None, matches=None)
    matches = None
    if value.mode == "sha256":
        encoded = base64.urlsafe_b64encode(bytes.fromhex(actual_hash)).decode().rstrip("=")
        matches = encoded == value.value
    return dict(algorithm=value.mode, expected=value.value, matches=matches)


def collect(prefix, extra_dirs):
    prefix = Path(prefix).resolve(strict=True)
    if prefix != Path(sys.prefix).resolve(strict=True):
        raise RuntimeError("Capture must use the selected environment's own Python")
    roots = [prefix, *(Path(p).resolve(strict=True) for p in extra_dirs)]
    aliases = {}
    for root in roots:
        for path in root.rglob("lib*.so*"):
            if not CUDA_LIBRARY.fullmatch(path.name) or not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            aliases.setdefault(resolved, set()).add(str(path))
    evidence = {}
    for path, names in sorted(aliases.items()):
        elf = dynamic(path)
        identity = stable_identity(path)
        try:
            verify_cuda_filename(elf["soname"], path, major=12)
            abi_error = None
        except (RuntimeError, OSError) as error:
            abi_error = str(error)
        evidence[str(path)] = dict(path=str(path), **identity, aliases=sorted(names),
                                  cuda12_abi_error=abi_error, origins=[], **elf)

    packages = []
    site = Path(sysconfig.get_paths()["purelib"]).resolve(strict=True)
    loader_sources = {}
    extensions = []
    for dist in importlib.metadata.distributions(path=[str(site)]):
        name = dist.metadata["Name"]
        packages.append(dict(name=name, version=dist.version))
        for file in dist.files or []:
            path = Path(dist.locate_file(file))
            if not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            if str(resolved) in evidence:
                row = evidence[str(resolved)]
                row["origins"].append(dict(kind="wheel-record", name=name,
                                            version=dist.version, file=str(file),
                                            record=record_proof(file, row["sha256"])))
            # Preserve loader implementation bytes for independent precedence review.
            rel = str(file)
            if rel.endswith(".py") and any(rel.startswith(p) for p in ("cupy/", "gpu4pyscf/lib/")):
                source = resolved.read_text()
                if any(term in source for term in ("CDLL", "load_library", "_preload",
                                                   "CUPY_", "libcusolver", "libcudart")):
                    loader_sources[rel] = dict(path=str(resolved), sha256=sha256(resolved),
                                               source=source)
            if ".so" in path.name and any(rel.startswith(p) for p in
                                         ("cupy/", "cupy_backends/", "gpu4pyscf/", "cutensor/")):
                extensions.append(path)
    conda_packages = []
    for manifest in sorted((prefix / "conda-meta").glob("*.json")):
        data = json.loads(manifest.read_text())
        conda_packages.append({key: data.get(key) for key in ("name", "version", "build")})
        for file in data.get("files", []):
            resolved = (prefix / file).resolve(strict=False)
            if str(resolved) in evidence:
                evidence[str(resolved)]["origins"].append(dict(
                    kind="conda-manifest", name=data["name"], version=data["version"],
                    build=data.get("build"), file=file, manifest_sha256=sha256(manifest)))
        for item in data.get("paths_data", {}).get("paths", []):
            resolved = (prefix / item["_path"]).resolve(strict=False)
            if str(resolved) in evidence and item.get("sha256"):
                evidence[str(resolved)]["origins"].append(dict(
                    kind="conda-file-sha256", name=data["name"], version=data["version"],
                    expected=item["sha256"], matches=item["sha256"] == evidence[str(resolved)]["sha256"]))
    linkage = []
    for extension in sorted(set(extensions)):
        text = command(["ldd", str(extension)])
        linkage.append(dict(path=str(extension.resolve()), sha256=sha256(extension),
                            ldd=text, unresolved=("not found" in text)))
    groups = {}
    for row in evidence.values():
        groups.setdefault(row["soname"], []).append(row["path"])
    startup = []
    for path in sorted(site.glob("*.pth")):
        startup.append(dict(path=str(path), sha256=sha256(path), source=path.read_text()))
    for name in ("sitecustomize.py", "usercustomize.py"):
        for directory in sys.path:
            path = Path(directory) / name
            if path.is_file():
                startup.append(dict(path=str(path.resolve()), sha256=sha256(path),
                                    source=path.read_text()))
    return dict(schema=1, approved=False, cuda_major=12, python_prefix=str(prefix),
                python_version=sys.version, job_id=os.environ.get("SLURM_JOB_ID"),
                host=socket.gethostname(), captured_at=datetime.now(timezone.utc).isoformat(),
                capture_script_sha256=sha256(Path(__file__)),
                interpreter=dict(path=sys.executable, resolved=str(Path(sys.executable).resolve()),
                                 sha256=sha256(Path(sys.executable))),
                sys_path=sys.path, startup_files=startup,
                source_commit=command(["git", "-C", str(Path(__file__).resolve().parents[3]),
                                       "rev-parse", "HEAD"]).strip(),
                environment={key: os.environ.get(key) for key in
                             ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONNOUSERSITE",
                              "CUDA_HOME", "CUDA_PATH", "PATH")},
                search_roots=[str(p) for p in roots], libraries=list(evidence.values()),
                providers_by_soname=groups, packages=sorted(packages, key=lambda p: p["name"]),
                conda_packages=conda_packages, loader_sources=loader_sources,
                extension_linkage=linkage,
                note="Static candidate evidence only. Independent selection/review and receipt hash required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--extra-library-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.no_site or not sys.dont_write_bytecode:
        parser.error("Capture must start with python -S -B (no site hooks or bytecode writes)")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Static hashing/capture belongs on a compute node")
    if os.environ.get("LD_PRELOAD") or os.environ.get("PYTHONPATH"):
        parser.error("Capture requires no LD_PRELOAD and empty PYTHONPATH")
    result = collect(args.prefix, args.extra_library_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("CUDA12_CANDIDATE_ONLY", args.output)
    print("LIBRARIES", len(result["libraries"]), "SONAMES", len(result["providers_by_soname"]))


if __name__ == "__main__":
    main()
