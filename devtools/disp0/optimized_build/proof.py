"""Installed optimized E2/Psi4 identity/closure proof, NOT GPU correctness."""
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_contract import (audit_mapped_libraries, digest, elf, mapped_libraries,
                              require)

E2 = "06404fa69ce9346443d7d4f19a250f6db9f0f62c"


def validate_world(world):
    require(world["git_commit"] == E2 and world["version_major"] == 2
            and world["compiler_major"] == 14, "Wrong optimized E2/GNU14 world")


def validate_paths(info, expected):
    require(not info["rpath"], "Old-style DT_RPATH is not allowed for optimized E2")
    require(info["runpath"] == expected, "Installed RUNPATH differs from controlled policy")
    require(all("/" not in name for name in info["needed"]), "Absolute binary dependency")


def main():
    import psi4
    from psi4 import core  # core first, then Python E2
    import einsums as ein
    import einsums.sealed as sealed
    import waggle
    import waggle._core

    run = Path(os.environ["DISP0_OPT_RUN"]).resolve()
    prefix, stage, environment = run / "einsums-stage", run / "psi4-build/stage", run / "environment"
    sdk = json.loads((run / "cuda13/sdk-contract.json").read_text())
    for module in (psi4, core):
        require(Path(module.__file__).resolve().is_relative_to(stage), "Foreign Psi4 stage")
    for module in (ein, ein._core):
        require(Path(module.__file__).resolve().is_relative_to(prefix), "Foreign Python E2 package")
    for module in (waggle, waggle._core):
        require(Path(module.__file__).resolve().is_relative_to(prefix), "Foreign Waggle companion package")
    git_info = run / "einsums-build/libs/Einsums/Config/generated/GitInfo.hpp"
    git_text = git_info.read_text()
    require(f'"{E2}"' in git_text and
            re.search(r"#define\s+EINSUMS_GIT_DIRTY_FLAG\s+0\b", git_text),
            "Generated E2 build identity is wrong or dirty")
    world = sealed.world()
    validate_world(world)
    native = Path(world["library_path"]).resolve()
    require(native.is_relative_to(prefix / "lib"), "E2 bound to build-tree native library")
    require({Path(p).resolve() for p in sealed.mapped_einsums_libraries()} == {native},
            "More than one mapped E2 world")
    maps = mapped_libraries(Path("/proc/self/maps").read_text())
    hashes = audit_mapped_libraries(maps, sdk, prefix, stage, environment)
    for name in (core.__file__, ein._core.__file__, native):
        require(str(Path(name).resolve()) in hashes, "Required binary missing from actual maps")
    waggle_maps = [name for name in maps if Path(name).name.startswith("libwaggle.so")]
    require(len(waggle_maps) == 1 and Path(waggle_maps[0]).is_relative_to(prefix / "lib"),
            "Missing/duplicate/foreign native Waggle companion")
    records = {}
    for path in sorted(prefix.rglob("*.so*")):
        if not path.is_file() or path.is_symlink():
            continue
        info = elf(path)
        origin = "$ORIGIN/.." if path.parent.name in ("einsums", "waggle") else "$ORIGIN"
        validate_paths(info, [origin, str(run / "cuda13/lib64"),
                              str(environment / "lib")])
        records[str(path)] = {"sha256": digest(path), "elf": info}
    require(records, "No installed E2 binaries")
    print(json.dumps({
        "scope": "CPU build/installed import/maps only; NOT GPU/device or physical correctness",
        "status": "PASS", "run": str(run), "job": os.environ["SLURM_JOB_ID"],
        "recipe_pin": os.environ["DISP0_OPT_PIN"], "einsums_world": world,
        "psi4": psi4.__file__, "psi4_core": core.__file__,
        "mapped_library_sha256": hashes, "installed_elf": records,
        "sdk_sha256": digest(run / "cuda13/sdk-contract.json"),
        "generated_gitinfo_sha256": digest(git_info),
        "runtime_lock_sha256": digest(run / "environment-explicit.txt"),
        "build_tools_lock_sha256": digest(run / "clang23-explicit.txt"),
        "actual_gpu_correctness_verified": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
