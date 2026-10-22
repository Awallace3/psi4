# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Corrected separately pinned driver, using the accepted immutable native v3 build."""
import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "build_smoke_plan.json").read_text())
SOURCE = Path(PLAN["source"])
STAGE = SOURCE / PLAN["build_directory"] / "stage"
ENV = Path(PLAN["environment"]).resolve()
OLD = Path(PLAN["dependency_source"]).resolve()
BUILD_RUN = Path(PLAN["run"])
DEPS = BUILD_RUN / "deps"
RUN = Path("/storage/project/r-cs207-0/awallace43/runs/"
           "ccasp-phoenix-benzene-pilot-20261021/P-003")
SCRATCH = Path("/storage/scratch1/3/awallace43/"
               "ccasp-phoenix-benzene-pilot-20261021-science003")
BUILD_COMMIT = "15cc9a7a9a2bd60e43ebc2533ec99686bca58374"
BUILD_TREE = "8f0aa8fabd476aafc6bc49b824c60ac935a86753"
CORE_SHA = "b283901ddac01e6bf7210a7461fb71cd5b2333cf9e42b99d60271959b40ac7eb"
BUILD_NODE = "atl1-1-02-007-23-1"
FIXTURE = RUN.parent / "P-002/inputs/benzene_s66x8_24_monA.json"  # immutable private input
FIXTURE_SHA = "dbfd1bfcb832ed28c94b486962c253f1e3fb8295617557d11e5cdf17631bb8dd"
KEY = re.compile(
    r"lib(Einsums|mkl|omp|gomp|iomp|int2|xc[.]|gg[.]|hdf5|cpptrace|stdc[+][+]|gcc_s|spdlog|fmt|blas|lapack|adcc|openblas|blis)"
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def identity(commit, tree):
    launcher = HERE.parent.parent
    for path, wanted_commit, wanted_tree in (
        (launcher, commit, tree), (SOURCE, BUILD_COMMIT, BUILD_TREE)
    ):
        if (git(path, "rev-parse", "HEAD") != wanted_commit
                or git(path, "rev-parse", "HEAD^{tree}") != wanted_tree
                or git(path, "status", "--porcelain")):
            raise ValueError("source identity/clean-state mismatch: " + str(path))
    batch = json.loads((BUILD_RUN / "metadata/batch-terminal.json").read_text())
    payload = json.loads((BUILD_RUN / "metadata/payload-complete.json").read_text())
    if (batch != {"exit_code": 0, "term_received": 0, "complete": True}
            or payload["commit"] != BUILD_COMMIT or payload["tree"] != BUILD_TREE
            or payload["core_sha256"] != CORE_SHA):
        raise ValueError("accepted build receipts differ")


def prepare(commit, tree):
    identity(commit, tree)
    if RUN.exists() or SCRATCH.exists():
        raise ValueError("pilot run/scratch parent must be exclusive and new")
    os.umask(0o027)
    for name in ("logs", "metadata", "inputs"):
        (RUN / name).mkdir(parents=True)
    SCRATCH.mkdir()
    write(RUN / "metadata/preparation.json", dict(
        launcher_commit=commit, launcher_tree=tree, build_commit=BUILD_COMMIT,
        core_sha256=CORE_SHA, fixture_sha256=FIXTURE_SHA, node=BUILD_NODE,
        no_science_run=True))


def validate_libraries(loaded, full=True):
    wanted = (OLD / "lib/libEinsums.so.1.1.3").resolve()
    allowed = {wanted, (OLD / "lib/libEinsumsExperimental.so.1.1.3").resolve()}
    paths = {Path(path).resolve() for path in loaded}
    einsums = {path for path in paths if path.name.startswith("libEinsums")}
    if wanted not in einsums or einsums - allowed:
        raise ValueError("wrong Einsums mapped-library role")
    for path in paths - allowed:
        if not path.is_relative_to(ENV):
            raise ValueError("numerical runtime outside selected environment: " + str(path))
    omp = {path for path in paths if re.match(r"lib(omp|gomp|iomp)", path.name)}
    if omp != {(ENV / "lib/libomp.so").resolve()}:
        raise ValueError("OpenMP runtime differs from accepted build")
    if full:
        if ((ENV / "lib/libmkl_gnu_thread.so.2").resolve() not in paths
                or any(path.name.startswith("libmkl_intel_thread") for path in paths)):
            raise ValueError("MKL threading role differs from accepted build")


def snapshot(name, full=True, *, receipt_run=None):
    receipt_run = RUN if receipt_run is None else Path(receipt_run)
    text = Path("/proc/self/maps").read_text()
    with (receipt_run / "metadata" / (name + ".maps")).open("x") as stream:
        stream.write(text)
    paths = set()
    protected = {
        (STAGE / "lib/psi4/core.cpython-313-x86_64-linux-gnu.so").resolve(),
        (OLD / "lib/einsums/core.cpython-313-x86_64-linux-gnu.so").resolve(),
    }
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            deleted = fields[5].endswith(" (deleted)")
            original = Path(fields[5].removesuffix(" (deleted)"))
            path = original.resolve()
            numeric = bool(KEY.match(original.name) or KEY.match(path.name))
            shared_object = bool(re.search(r"\.so(?:\.\d+)*$", original.name))
            if deleted and (numeric or shared_object or path in protected):
                raise ValueError("deleted tracked runtime mapping: " + str(original))
            if numeric:
                paths.add(path)
    loaded = {str(path): sha(path) for path in sorted(paths)}
    validate_libraries(loaded, full)
    pins = json.loads((BUILD_RUN / "metadata/dependency-hashes.json").read_text())
    pins.update(json.loads((BUILD_RUN / "metadata/after-tests.json").read_text()))
    for path, digest in loaded.items():
        if path in pins and pins[path] != digest:
            raise ValueError("runtime dependency hash changed: " + path)
    write(receipt_run / "metadata" / (name + ".json"), loaded)


def pilot_arguments(job, driver_commit, driver_tree, basis_manifest, basis_manifest_sha256):
    return [
        str(HERE / "pilot.py"),
        "--source", str(SOURCE), "--commit", BUILD_COMMIT,
        "--native-tree", BUILD_TREE,
        "--driver-source", str(HERE.parent.parent), "--driver-commit", driver_commit,
        "--driver-tree", driver_tree,
        "--stage", str(STAGE), "--core-sha256", CORE_SHA,
        "--basis-manifest", str(basis_manifest),
        "--basis-manifest-sha256", basis_manifest_sha256,
        "--fixture", str(FIXTURE), "--out", str(RUN / "results"),
        "--scratch", str(SCRATCH / ("P-" + job)),
    ]


def invoke_pilot(pilot):
    try:
        return int(pilot.main())
    except SystemExit as error:
        raise ValueError("pilot preflight exit: " + repr(error.code)) from error


def check_driver_result(code, result):
    if code == 0 and not result.get("science_pass"):
        raise ValueError("driver exit0 without scientific completion")


def initialize_runtime(commit, tree, *, receipt_run=None, operation="pilot"):
    """Reuse accepted strict runtime checks; no scientific calculation is invoked."""
    receipt_run = RUN if receipt_run is None else Path(receipt_run)
    host = socket.gethostname().split(".")[0]
    if (not os.environ.get("SLURM_JOB_ID") or host != BUILD_NODE
            or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]):
        raise ValueError("same Phoenix build-node allocation required for -march=native")
    if not sys.flags.isolated or not sys.flags.no_site:
        raise ValueError("Python -I -S required")
    identity(commit, tree)
    if Path(os.environ.get("PYTHONPATH", "").split(":")[0]) != STAGE / "lib":
        raise ValueError("accepted stage PsiAPI not initialized")
    core_path = STAGE / "lib/psi4/core.cpython-313-x86_64-linux-gnu.so"
    if sha(core_path) != CORE_SHA:
        raise ValueError("accepted core hash changed")
    library = OLD / "lib/libEinsums.so.1.1.3"
    if sha(library) != PLAN["einsums_library_sha256"]:
        raise ValueError("accepted Einsums library changed")
    handle = ctypes.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    snapshot("preloaded", full=False, receipt_run=receipt_run)
    sys.path[:0] = [str(STAGE / "lib"), str(DEPS / "lib"), str(DEPS / "python"),
                    str(ENV / "lib/python3.13/site-packages")]
    os.environ["PSIDATADIR"] = str(STAGE / "share/psi4")
    import numpy
    import adcc
    import einsums
    import psi4
    if (Path(psi4.__file__).resolve() != (STAGE / "lib/psi4/__init__.py").resolve()
            or Path(psi4.core.__file__).resolve() != core_path.resolve()):
        raise ValueError("wrong Psi4 package/core")
    if Path(psi4.core.get_datadir()).resolve() != (STAGE / "share/psi4").resolve():
        raise ValueError("wrong accepted native Psi4 data root")
    if (Path(einsums.__file__).resolve() != (OLD / "lib/einsums/__init__.py").resolve()
            or Path(einsums.core.__file__).resolve()
            != (OLD / "lib/einsums/core.cpython-313-x86_64-linux-gnu.so").resolve()):
        raise ValueError("wrong Einsums Python package/extension")
    pins = json.loads((BUILD_RUN / "metadata/dependency-hashes.json").read_text())
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError("accepted dependency changed: " + path)
    dependencies = {}
    for package, version in PLAN["python_dependencies"].items():
        module = importlib.import_module(package)
        if (module.__version__ != version or not Path(module.__file__).resolve().is_relative_to(
                (OLD / "lib" / package).resolve())):
            raise ValueError("wrong isolated Python dependency: " + package)
        dependencies[package] = dict(version=version, path=module.__file__)
    adcc.set_n_threads(1)
    psi4.set_num_threads(1)
    if adcc.get_n_threads() != 1 or psi4.core.get_num_threads() != 1:
        raise ValueError("numerical thread setup refused")
    snapshot("before-" + operation, receipt_run=receipt_run)
    write(receipt_run / "metadata/runtime.json", dict(
        launcher_commit=commit, launcher_tree=tree, build_commit=BUILD_COMMIT,
        native_source=str(SOURCE), native_tree=BUILD_TREE, stage=str(STAGE),
        driver_source=str(HERE.parent.parent), operation=operation,
        data=psi4.core.get_datadir(),
        core_sha256=CORE_SHA, node=host, job_id=os.environ["SLURM_JOB_ID"],
        python=sys.executable, numpy=numpy.__version__, adcc=adcc.__version__,
        python_dependencies=dependencies, einsums_handle_retained=bool(handle)))
    return psi4, handle


def run(commit, tree, basis_manifest=None, basis_manifest_sha256=None):
    if not basis_manifest or not basis_manifest_sha256:
        raise ValueError("P003 requires a reviewed private basis-only manifest and explicit SHA256")
    if sha(basis_manifest) != basis_manifest_sha256:
        raise ValueError("basis-admission manifest hash mismatch")
    preparation = json.loads((RUN / "metadata/preparation.json").read_text())
    if preparation["launcher_commit"] != commit or preparation["launcher_tree"] != tree:
        raise ValueError("launcher preparation mismatch")
    if sha(FIXTURE) != FIXTURE_SHA:
        raise ValueError("fixture hash mismatch")
    _, handle = initialize_runtime(commit, tree)
    argv = pilot_arguments(os.environ["SLURM_JOB_ID"], commit, tree,
                           basis_manifest, basis_manifest_sha256)
    write(RUN / "metadata/pilot-argv.json", argv)
    spec = importlib.util.spec_from_file_location("corrected_benzene_pilot", argv[0])
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    sys.argv = argv
    code = invoke_pilot(pilot)
    snapshot("after-pilot")
    result = json.loads((RUN / "results/result.json").read_text())
    check_driver_result(code, result)
    write(RUN / "metadata/payload-terminal.json", dict(
        returncode=code, science_pass=bool(result.get("science_pass")),
        original_driver_error=result.get("error"), core_sha256=CORE_SHA,
        driver_commit=commit, driver_tree=tree, native_commit=BUILD_COMMIT,
        basis_manifest_sha256=basis_manifest_sha256, einsums_handle_retained=bool(handle)))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--basis-manifest")
    parser.add_argument("--basis-manifest-sha256")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.commit, args.tree)
        return 0
    try:
        return run(args.commit, args.tree, args.basis_manifest, args.basis_manifest_sha256)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        if (RUN / "metadata").is_dir():
            write(RUN / "metadata/bootstrap-failure.json", dict(traceback=error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
