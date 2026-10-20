"""Future CPU build probe: installed identity/closure, never GPU execution."""

import json
import os
from pathlib import Path

import psi4
from psi4 import core  # core before Python E2
import einsums as ein
import einsums.sealed as sealed

from runtime_contract import (E2_PIN, audit_closure, audit_elf, audit_mapped_libraries,
                              digest, mapped_libraries, require)


def main():
    run = Path(os.environ["DISP0_RUN"]).resolve()
    prefix, stage, environment = run / "einsums-stage", run / "psi4-build/stage", run / "environment"
    sdk = json.loads((run / "cuda13/sdk-contract.json").read_text())
    require(Path(psi4.__file__).resolve().is_relative_to(stage), "Wrong Psi4 Python stage")
    require(Path(core.__file__).resolve().is_relative_to(stage), "Wrong Psi4 native core")
    require(Path(ein.__file__).resolve().is_relative_to(prefix), "Wrong installed E2 Python")
    require(Path(ein._core.__file__).resolve().is_relative_to(prefix), "Wrong installed E2 extension")
    world = sealed.world()
    require(world["git_commit"] == E2_PIN and world["version_major"] == 2
            and world["compiler_major"] == 14, "Wrong E2d17/GNU14 world")
    native = Path(world["library_path"]).resolve()
    require(native.is_relative_to(prefix / "lib"), "Native E2 still bound to a build tree")
    require({Path(p).resolve() for p in sealed.mapped_einsums_libraries()} == {native},
            "Multiple/mismatched Einsums worlds")
    maps = mapped_libraries(Path("/proc/self/maps").read_text())
    map_hashes = audit_mapped_libraries(maps, sdk, prefix, stage, environment)
    binaries = [Path(core.__file__).resolve(), Path(ein._core.__file__).resolve(), native]
    for path in binaries:
        require(str(path) in maps, "Required binary is not actually mapped")
    for family, item in sdk["libraries"].items():
        selected = [p for p in maps if Path(p).name.startswith(f"lib{family}.so")]
        require(selected == [item["path"]] and digest(selected[0]) == item["sha256"],
                f"Actual CUDA map differs from SDK contract: {family}: {selected}")
    elf_records = {}
    for path in {p.resolve() for p in (prefix / "lib").glob("*.so*") if p.is_file()}:
        elf_records[str(path)] = audit_elf(
            path, ["$ORIGIN", str(run / "cuda13/lib64"), str(environment / "lib")])
    extension = Path(ein._core.__file__).resolve()
    elf_records[str(extension)] = audit_elf(
        extension, ["$ORIGIN/..", str(run / "cuda13/lib64"), str(environment / "lib")])
    # Psi4's existing install policy may use DT_RPATH. Its only allowed entries
    # are its own adjacent stage and this fresh dependency environment.
    elf_records[str(binaries[0])] = audit_elf(
        binaries[0], ["$ORIGIN/../", str(environment / "lib")], require_runpath=False)
    closure = {str(path): audit_closure(path, sdk, prefix, environment)
               for path in (extension, native)}
    closure[str(binaries[0])] = audit_closure(binaries[0], sdk, stage, environment, require_cuda=False)
    print(json.dumps({"status": "PASS", "scope": "Installed identity/CPU preparation only; not CUDA correctness",
                      "pid": os.getpid(), "source_pin": os.environ["DISP0_PIN"],
                      "job": os.environ["SLURM_JOB_ID"], "run": str(run),
                      "psi4": psi4.__file__, "psi4_version": psi4.__version__, "einsums": ein.__file__,
                      "world": world, "binary_sha256": {str(p): digest(p) for p in binaries},
                      "mapped_libraries": maps, "elf_audit": elf_records, "closure": closure,
                      "mapped_library_sha256": map_hashes,
                      "sdk_contract_sha256": digest(run / "cuda13/sdk-contract.json"),
                      "actual_gpu_correctness_verified": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
