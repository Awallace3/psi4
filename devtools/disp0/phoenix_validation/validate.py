"""Fresh whole-stage CPU proof; never substitutes for a physical GPU gate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

E2_PIN = "06404fa69ce9346443d7d4f19a250f6db9f0f62c"
CUEST_SHA = "65c9e559201ee589cb742b919cc1c5d80c45aa480b64aede17ac96aff9e4963c"
HEADERS_SHA = "3d26a7ef607d02ee294f66f11f1374f3badbd83430f5be28ed0a05e6c8d73d18"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def header_identity(include):
    # Canonical ordering is POSIX filename text, NOT pathlib's component-wise
    # ordering: "cuest.h" precedes "cuest/context.h" in the canonical stream.
    records = sorted((p.relative_to(include).as_posix(), digest(p))
                     for p in include.rglob("*") if p.is_file())
    aggregate = b"".join(name.encode() + b"\0" + sha.encode() + b"\n"
                         for name, sha in records)
    return len(records), hashlib.sha256(aggregate).hexdigest()


def prepare(source, run, dependencies, e2, cuda, cuest):
    sys.path.insert(0, str(source / "devtools/disp0"))
    from runtime_contract import inspect_sdk, prepare_sdk

    require((dependencies / "bin/python").is_file(), "Missing dependency interpreter")
    require((e2 / "lib/einsums").is_dir(), "Missing read-only Python E2 stage")
    native = cuest / "lib/13/libcuest.so.0.2.50"
    require(digest(native) == CUEST_SHA, "Wrong canonical cuEST native input")
    count, headers_sha = header_identity(cuest / "include")
    require(count == 59 and headers_sha == HEADERS_SHA,
            f"cuEST canonical whole-header identity mismatch: count={count}, sha256={headers_sha}")
    manifest = inspect_sdk(cuda)
    save(run / "cuda-inputs.json", manifest)
    prepare_sdk(cuda, run / "cuda13", manifest)
    records = sorted((dependencies / "conda-meta").glob("*.json"))
    save(run / "dependency-inputs.json", {
        "scope": "Read-only dependency reuse, NOT adoption of failed old Psi4",
        "environment": str(dependencies), "einsums_prefix": str(e2),
        "old_psi4_core_is_forbidden": True, "cuest_native_sha256": CUEST_SHA,
        "cuest_headers_sha256": HEADERS_SHA,
        "conda_records": {p.name: digest(p) for p in records},
        "einsums_source_pin_required": E2_PIN,
    })


def elf(path):
    return subprocess.check_output(["readelf", "-d", str(path)], text=True, timeout=30)


def validate_world(world, prefix):
    require(world["git_commit"] == E2_PIN and world["version_major"] == 2
            and world["compiler_major"] == 14, "Wrong optimized E2/GNU14 world")
    native = Path(world["library_path"]).resolve()
    require(native.is_relative_to(prefix / "lib"), "Foreign E2 native world")
    return native


def validate_installed(source, run, dependencies, e2):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU proof must hide GPUs")
    stage = (run / "build/stage").resolve()
    import psi4
    from psi4 import core
    for module in (psi4, core):
        require(Path(module.__file__).resolve().is_relative_to(stage), "Foreign/old Psi4 stage")
    require("undefined" not in psi4.__version__, "Undefined Psi4 build metadata")
    from psi4 import metadata
    require(metadata.__version_long.endswith("+" + os.environ["DISP0_PIN"][:7]),
            "Installed Psi4 metadata does not bind the pinned source")
    package = Path(psi4.__file__).resolve().parent
    scientific_python = {p.relative_to(package).as_posix(): digest(p)
                         for p in package.rglob("*.py")}
    for relative, checksum in scientific_python.items():
        if relative.startswith("driver/"):
            require(digest(source / "psi4" / relative) == checksum,
                    "Installed scientific driver differs from the pinned source: " + relative)
    import einsums as ein
    import einsums.sealed as sealed
    import waggle
    import waggle._core
    import qcelemental
    import qcengine
    import optking
    for module in (ein, ein._core, waggle, waggle._core):
        require(Path(module.__file__).resolve().is_relative_to(e2), "Foreign E2/Waggle package")
    world = sealed.world()
    native = validate_world(world, e2)
    require({Path(p).resolve() for p in sealed.mapped_einsums_libraries()} == {native},
            "Multiple mapped Einsums native worlds")
    # The superbuild cache is always present, even when inner layout changes.
    cache_text = (run / "build/CMakeCache.txt").read_text()
    require("ENABLE_cuEST:BOOL=ON" in cache_text and "ENABLE_Einsums:BOOL=OFF" in cache_text,
            "Wrong native feature configuration")
    maps = Path("/proc/self/maps").read_text()
    mapped = sorted({row.split(maxsplit=5)[5] for row in maps.splitlines()
                     if len(row.split(maxsplit=5)) == 6
                     and row.split(maxsplit=5)[5].startswith("/")})
    require(not any("cuda-12." in p or "14196330/einsums-build" in p
                    or "/einsums-build/" in p for p in mapped), "Historical/site12 runtime mapping")
    require(not any("/psi4-build/" in p for p in mapped), "Old Psi4 binary mapping")
    sdk = json.loads((run / "cuda13/sdk-contract.json").read_text())
    for family, record in sdk["libraries"].items():
        for path in mapped:
            if Path(path).name.startswith("lib" + family + ".so"):
                require(digest(path) == record["sha256"], "Foreign CUDA component: " + path)
    require(any(Path(p).name == "libcuest.so.0.2.50" and digest(p) == CUEST_SHA
                for p in mapped), "Canonical cuEST is not mapped by the new native core")
    binaries = (Path(core.__file__).resolve(), Path(ein._core.__file__).resolve(), native,
                Path(waggle._core.__file__).resolve())
    for binary in binaries[1:]:
        require("(RPATH)" not in elf(binary), "Old-style dependency DT_RPATH")
    core.set_output_file(str(run / "cpu-native-smoke.out"), False)
    psi4.set_num_threads(1)
    psi4.set_memory("1 GiB")
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk", "reference": "rhf",
                      "use_cuest": False, "e_convergence": 10, "d_convergence": 10})
    mol = psi4.geometry("0 1\nH 0 0 0\nH 0 0 1.4\nunits bohr\nsymmetry c1\n")
    energy = float(psi4.energy("scf", molecule=mol))
    require(math.isfinite(energy) and abs(energy - (-1.116714325)) < 1e-7,
            "Native CPU RHF smoke disagrees")
    api = json.loads((run / "api-probe.json").read_text())
    require(api["status"] == "PASS", "Independent CPU E2 API probe did not pass")
    guard_text = (run / "cpu-guards.stderr").read_text()
    count = re.search(r"Ran (\d+) tests", guard_text)
    require(count is not None and int(count[1]) >= 244 and guard_text.rstrip().endswith("OK"),
            "Missing/incomplete expanded CPU regressions")
    support = package.parent / "psi4_runtime_support"
    support_inventory = {p.relative_to(support).as_posix(): digest(p)
                         for p in support.rglob("*.py")}
    require(support_inventory and support_inventory == {
        p.relative_to(source / "psi4_runtime_support").as_posix(): digest(p)
        for p in (source / "psi4_runtime_support").rglob("*.py")},
        "Whole staged runtime support differs from the pinned source")
    save(run / "result.json", {
        "status": "PASS", "scope": "CPU_BUILD_VALIDATION_ONLY",
        "gpu_correctness_verified": False, "protein_fisapt_admitted": False,
        "source": str(source), "pin": os.environ["DISP0_PIN"],
        "job_id": os.environ["SLURM_JOB_ID"], "stage": str(stage),
        "psi4_version": psi4.__version__, "einsums_world": world,
        "scientific_python_sha256": scientific_python,
        "runtime_support_package": str(support),
        "runtime_support_sha256": support_inventory,
        "provenance_sha256": {
            str(p): digest(p)
            for folder in ("phoenix_validation", "cachy")
            for p in (source/"devtools/disp0"/folder).glob("*.py")},
        "python_package": str(package),
        "native_cpu_rhf_Eh": energy, "dependency_environment": str(dependencies),
        "cpu_guard_count": int(count[1]), "api_probe_sha256": digest(run / "api-probe.json"),
        "read_only_einsums_dependency": str(e2),
        "dependency_python_sha256": {
            str(p.resolve()): digest(p)
            for module in (ein, waggle, qcelemental, qcengine, optking)
            for p in Path(module.__file__).resolve().parent.rglob("*.py")},
        "module_files": {module.__name__: str(Path(module.__file__).resolve())
                         for module in (psi4, core, ein, ein._core, sealed, waggle,
                                        waggle._core, qcelemental, qcengine, optking)},
        "binary_sha256": {str(p): digest(p) for p in binaries},
        "mapped_paths": mapped,
        "cpu_guard_log_sha256": digest(run / "cpu-guards.stderr"),
        "source_version_proof_sha256": digest(run / "source-version/validated.json"),
        "launcher_sha256": digest(stage / "bin/psi4"),
    })
    print("CPU_BUILD_VALIDATION_ONLY: PASS; no GPU correctness claim")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    source = Path(os.environ["DISP0_SOURCE"]).resolve()
    run = Path(os.environ["DISP0_JOB_RUN"]).resolve()
    dependencies = Path(os.environ["DISP0_DEPENDENCIES"]).resolve()
    e2 = Path(os.environ["DISP0_E2_PREFIX"]).resolve()
    if args.prepare:
        prepare(source, run, dependencies, e2,
                Path(os.environ["DISP0_CUDA_SOURCE"]).resolve(),
                Path(os.environ["DISP0_CUEST_ROOT"]).resolve())
    else:
        validate_installed(source, run, dependencies, e2)


if __name__ == "__main__":
    main()
