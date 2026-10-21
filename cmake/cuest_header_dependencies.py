#!/usr/bin/env python3
"""Verify the *consumed* cuEST headers after compilation (no SDK/GPU calls)."""
import argparse
import json
from pathlib import Path
import subprocess

from cuest_sdk_contract import HEADERS_SHA256, header_digest, require, sha256


def admitted_headers(paths, sdk):
    include = (Path(sdk) / "include").resolve(strict=True)
    require(header_digest(include) == HEADERS_SHA256, "SDK headers changed during compilation")
    found = {}
    for path in paths:
        path = Path(path)
        if path.suffix != ".h" or not (path.name == "cuest.h" or "cuest" in path.parts):
            continue
        path = path.resolve(strict=True)
        require(path.is_relative_to(include), f"compiler consumed an unadmitted cuEST header: {path}")
        found[str(path)] = sha256(path)
    require(any(Path(path).name == "cuest.h" for path in found), "no consumed cuEST umbrella header proved")
    return found


def dependencies(build, generator):
    if generator == "Ninja":
        result = subprocess.run(["ninja", "-C", str(build), "-t", "deps"],
                                capture_output=True, text=True, timeout=120, check=False)
        require(result.returncode == 0, f"ninja dependency audit failed: {result.stderr}")
        return [line.strip() for line in result.stdout.splitlines() if line.startswith("    ")]
    if generator == "Unix Makefiles":
        # GCC dependency files escape whitespace and continue lines with '\'.
        import shlex
        paths = []
        for depfile in Path(build).rglob("*.o.d"):
            paths.extend(shlex.split(depfile.read_text().replace("\\\n", " ")))
        return paths
    raise ValueError(f"cuEST consumed-header audit unsupported generator: {generator}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", required=True, type=Path)
    parser.add_argument("--build-directory", required=True, type=Path)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = dependencies(args.build_directory, args.generator)
    found = admitted_headers(paths, args.sdk_root)
    args.output.write_text(json.dumps(found, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
