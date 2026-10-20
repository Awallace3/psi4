"""Optional pytest plugin: bind physical XML to its ACTUAL installed runtime."""

import json
import os
from pathlib import Path

import pytest

from runtime_contract import audit_mapped_libraries, digest, mapped_libraries, require, write_new


def observed():
    import psi4
    from psi4 import core
    import einsums as ein
    import einsums.sealed as sealed
    run = Path(os.environ["DISP0_RUN"]).resolve()
    metadata = json.loads((run / "runtime-provenance.json").read_text())
    native = Path(sealed.world()["library_path"]).resolve()
    require(Path(core.__file__).resolve().is_relative_to(run / "psi4-build/stage"), "Foreign physical core")
    require(Path(ein.__file__).resolve().is_relative_to(run / "einsums-stage"), "Foreign physical Python E2")
    require({Path(p).resolve() for p in sealed.mapped_einsums_libraries()} == {native},
            "Physical test process has mixed Einsums worlds")
    hashes = {str(Path(p).resolve()): digest(p) for p in (core.__file__, ein._core.__file__, native)}
    require(hashes == metadata["binary_sha256"], "Physical process uses different prepared binaries")
    sdk = json.loads((run / "cuda13/sdk-contract.json").read_text())
    maps = mapped_libraries(Path("/proc/self/maps").read_text())
    map_hashes = audit_mapped_libraries(
        maps, sdk, run / "einsums-stage", run / "psi4-build/stage", run / "environment")
    components = {}
    for family, item in sdk["libraries"].items():
        selected = {p for p in maps if Path(p).name.startswith(f"lib{family}.so")}
        require(selected == {item["path"]} and digest(item["path"]) == item["sha256"],
                f"Actual test-process CUDA closure mismatch: {family}")
        components[family] = item["path"]
    return {"pid": os.getpid(), "job": os.environ["SLURM_JOB_ID"],
            "source_pin": os.environ["DISP0_PIN"], "run": str(run),
            "psi4": psi4.__file__, "einsums": ein.__file__, "binary_sha256": hashes,
            "cuda_component_maps": components,
            "mapped_library_sha256": map_hashes,
            "metadata_sha256": digest(run / "runtime-provenance.json")}


def pytest_sessionstart(session):
    session.disp0_physical_start = observed()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    final = observed()
    start = session.disp0_physical_start
    # Lazy imports may add native libraries. Every added map must pass the
    # same policy, while preexisting mappings/identities must remain unchanged.
    require({k: v for k, v in final.items() if k != "mapped_library_sha256"} ==
            {k: v for k, v in start.items() if k != "mapped_library_sha256"},
            "Physical runtime changed across test session")
    require(all(final["mapped_library_sha256"].get(p) == h
                for p, h in start["mapped_library_sha256"].items()), "Native physical mapping changed")
    xml = Path(session.config.option.xmlpath).resolve()
    require(xml.is_relative_to(Path(os.environ["DISP0_RUN"]).resolve() / "psi4-build"),
            "Physical XML outside the new build")
    document = {**final, "exit_status": int(exitstatus), "xml": str(xml),
                "xml_sha256": digest(xml), "actual_gpu_correctness_verified": False}
    write_new(Path(os.environ["DISP0_RUN"]) / "physical-process-proof.json", document)
