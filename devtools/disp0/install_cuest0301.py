#!/usr/bin/env python3
"""Install a private conda-forge CUDA13 cuEST SDK; never alter an existing prefix.

This is package staging, NOT a Psi4 build or GPU qualification. The receipt
deliberately cannot be used as an old cuEST0.2.50 admission/build proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


VERSION = "0.3.0.1"
BUILD = "h345135a_0"
PACKAGES = {
    "libcuest": "b766bf75603dc25162907b3ed8d29d72ebaebded067ee1a6afd8eb174628d1be",
    "libcuest-dev": "0159f8dcf95b2207946a9f5bfd2892b2b9d8200685032e1ae8c29d1ad2eb4d85",
}


def sha256(path):
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_record(record, name):
    expected = (
        record.get("name") == name
        and record.get("version") == VERSION
        and record.get("build") == BUILD
        and record.get("subdir") == "linux-64"
        and record.get("sha256") == PACKAGES[name]
        and record.get("url") == (
            f"https://conda.anaconda.org/conda-forge/linux-64/"
            f"{name}-{VERSION}-{BUILD}.conda"
        )
    )
    if not expected:
        raise RuntimeError(f"Unexpected conda package identity: {name}")


def inventory(prefix):
    packages = []
    files = {}
    for name in PACKAGES:
        path = prefix / "conda-meta" / f"{name}-{VERSION}-{BUILD}.json"
        record = json.loads(path.read_text())
        validate_record(record, name)
        packages.append(record)
        for relative in record["files"]:
            source = prefix / relative
            resolved = source.resolve(strict=True)
            if not resolved.is_relative_to(prefix):
                raise RuntimeError(f"Package file escapes dedicated prefix: {relative}")
            if not resolved.is_file():
                continue
            files[relative] = {
                "target": str(resolved.relative_to(prefix)),
                "bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
    return packages, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Only the explicitly pinned linux-64 CUDA13 builds are supported")
    prefix = args.prefix.expanduser().absolute()
    output = args.output.expanduser().absolute()
    if prefix.exists() or prefix.is_symlink() or output.exists():
        raise RuntimeError("Refusing an existing prefix/output; use fresh owned paths")
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    # CPU package setup only: no device probe, context, or scientific import.
    command = [
        str(args.conda.expanduser()), "create", "--yes", "--json",
        "--override-channels", "--channel", "conda-forge",
        "--strict-channel-priority", "--prefix", str(prefix),
        f"libcuest={VERSION}={BUILD}", f"libcuest-dev={VERSION}={BUILD}",
        "cuda-version=13.3", "cuda-nvrtc=13.3.33", "libnvjitlink=13.3.33",
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "install.json").open("x") as log:
        subprocess.run(command, env=env, stdout=log, check=True)
    prefix = prefix.resolve(strict=True)
    packages, files = inventory(prefix)
    with (output / "explicit.txt").open("x") as log:
        subprocess.run(
            [str(args.conda.expanduser()), "list", "--prefix", str(prefix), "--explicit", "--sha256"],
            env=env, stdout=log, check=True,
        )
    receipt = {
        "status": "PASS_PACKAGE_INSTALL_ONLY",
        "qualified_for_sapt0": False,
        "gpu_executed": False,
        "host": platform.node(),
        "timestamp": time.time(),
        "prefix": str(prefix),
        "installer_sha256": sha256(Path(__file__)),
        "packages": packages,
        "files": files,
        "next_gate": "Rebuild Psi4 and migrate versioned ABI/runtime proof, then CPU/GPU parity",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "status", "qualified_for_sapt0", "gpu_executed", "host", "prefix"
    )}))


if __name__ == "__main__":
    main()
