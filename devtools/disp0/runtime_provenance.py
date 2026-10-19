"""Strict dedicated stage/native pairing check; CPU preparation, not a GPU gate."""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import psi4
from psi4 import core
import einsums as ein
import einsums.sealed as sealed


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    build = Path(os.environ["DISP0_BUILD_DIR"]).resolve()
    prefix = Path(os.environ["DISP0_EIN_PREFIX"]).resolve()
    world = sealed.world()
    assert world["git_commit"] == "d17bb76e4ed3a07fbbc470e6813a5c134dfe0e62", world
    assert world["version_major"] == 2 and world["compiler_major"] == 14, world
    assert Path(psi4.__file__).resolve().is_relative_to(build / "stage")
    assert Path(core.__file__).resolve().is_relative_to(build / "stage")
    assert Path(ein.__file__).resolve().is_relative_to(prefix)
    assert Path(ein._core.__file__).resolve().is_relative_to(prefix)
    mapped = sealed.mapped_einsums_libraries()
    assert len(mapped) == 1, mapped
    assert Path(mapped[0]).resolve().is_relative_to(prefix), mapped
    maps = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                   if "/" in line and any(s in line for s in
                                          ("libEinsums", "libcudart.so", "libcublas.so", "libstdc++", "libmkl", "libxc.so"))})
    assert any("libcudart.so.13" in line for line in maps), maps
    assert any("libcublas.so.13" in line for line in maps), maps
    binaries = [Path(core.__file__).resolve(), Path(ein._core.__file__).resolve(),
                Path(world["library_path"]).resolve()]
    elf = subprocess.check_output(["readelf", "-d", str(binaries[-1])], text=True)
    print(json.dumps({
        "status": "PASS", "scope": "build/import/CPU preparation only; no CUDA device correctness claim",
        "python": platform.python_version(), "psi4": psi4.__file__, "psi4_version": psi4.__version__,
        "einsums": ein.__file__, "world": world, "mapped_libraries": maps,
        "binary_sha256": {str(path): digest(path) for path in binaries},
        "einsums_needed": [line.strip() for line in elf.splitlines() if "NEEDED" in line],
    }, indent=2))


if __name__ == "__main__":
    main()
