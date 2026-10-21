# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""One concrete Phoenix build recipe. prepare is stdlib/data-only; run requires allocation."""
import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "build_smoke_plan.json").read_text())
SOURCE = Path(PLAN["source"])
ENV = Path(PLAN["environment"])
OLD = Path(PLAN["dependency_source"])
RUN = Path(PLAN["run"])
DEPS = RUN / "deps"
BUILD = SOURCE / PLAN["build_directory"]
INNER = BUILD / "psi4-core-prefix/src/psi4-core-build"
STAGE = BUILD / "stage"
CM = str(ENV / "bin/cmake")
PY = str(ENV / "bin/python3.13")


def stage_python_dependencies(source, destination):
    """Expose only three already-installed pure-Python packages and metadata."""
    links = {}
    destination.mkdir(parents=True, exist_ok=False)
    for package, version in PLAN["python_dependencies"].items():
        module = source / "lib" / package
        metadata = source / "lib" / f"{package}-{version}.dist-info"
        record = Parser().parsestr((metadata / "METADATA").read_text())
        if not module.is_dir() or record.get("Version") != version:
            raise ValueError("missing exact Python dependency: " + package)
        for src in (module, metadata):
            dst = destination / src.name
            dst.symlink_to(src, target_is_directory=True)
            links[str(dst)] = str(src.resolve())
    return links


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    with Path(p).open("x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(SOURCE), *args], text=True).strip()


def identity(commit, tree):
    if Path(__file__).resolve().parent.parent.parent != SOURCE:
        raise ValueError("script is not in the fixed private source")
    if git("rev-parse", "HEAD") != commit or git("rev-parse", "HEAD^{tree}") != tree:
        raise ValueError("source commit/tree differs")
    if git("status", "--porcelain=v1"):
        raise ValueError("dirty source")
    git("merge-base", "--is-ancestor", PLAN["accepted_adapter_commit"], commit)
    for name, pin in PLAN["accepted_adapter_hashes"].items():
        if sha(HERE / name) != pin:
            raise ValueError("accepted adapter changed: " + name)


def prepare(commit, tree):
    """Authorized login-node staging: no imports, configure, large hashes or old edits."""
    identity(commit, tree)
    if RUN.exists() or BUILD.exists():
        raise ValueError("run/build already exists; do not reuse or delete")
    os.umask(0o027)
    (RUN / "logs").mkdir(parents=True, exist_ok=False)
    (RUN / "metadata").mkdir()
    Path(PLAN["scratch_parent"]).mkdir(exist_ok=False)
    configs = {}
    for pkg in ("Einsums", "TargetHDF5"):
        src = OLD / "share/cmake" / pkg
        dst = DEPS / "share/cmake" / pkg
        dst.mkdir(parents=True)
        for f in sorted(src.glob("*.cmake")):
            # All config data are copied byte-identically, no relocation needed:
            # import prefixes resolve relative to this new isolated prefix.
            data = f.read_bytes()
            (dst / f.name).write_bytes(data)
            configs[str(dst / f.name)] = dict(source=str(f), sha256=sha(f), relocation=None)
    links = {}
    def link(rel):
        src, dst = OLD / rel, DEPS / rel
        if not src.exists():
            raise ValueError("missing exact dependency input: " + rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=src.is_dir())
        links[str(dst)] = str(src.resolve())
    for name in ("Einsums", "EinsumsExperimental", "EinsumsPy"):
        link("include/" + name)
    for stem in ("libEinsums", "libEinsumsExperimental"):
        for suffix in (".so", ".so.1", ".so.1.1.3"):
            link("lib/" + stem + suffix)
    for name in ("__init__.py", "utils.py", "core.cpython-313-x86_64-linux-gnu.so"):
        link("lib/einsums/" + name)
    links.update(stage_python_dependencies(OLD, DEPS / "python"))
    if (DEPS / "include/psi4").exists() or (DEPS / "lib/psi4").exists():
        raise ValueError("unrelated Psi4 exposed")
    write(RUN / "metadata/preparation.json", dict(commit=commit, tree=tree,
          configs=configs, symlinks=links, numerical_hashes="deferred to compute allocation",
          environment=str(ENV), relocation_count=0))
    print("Prepared exclusive logs and isolated dependency config/symlinks; no build/import.")


def call(label, argv):
    argv = list(map(str, argv))
    write(RUN / "metadata" / (label + "-argv.json"), argv)
    print(label + ": " + " ".join(argv), flush=True)
    with (RUN / "logs" / (label + ".log")).open("x") as f:
        p = subprocess.run(argv, cwd=RUN, stdout=f, stderr=subprocess.STDOUT)
    write(RUN / "metadata" / (label + "-exit.json"), dict(returncode=p.returncode))
    p.check_returncode()


def options(inner=False):
    # Deliberate explicit INNER flags, not the known uncorrected EP cfgcmd.
    opts = dict(CMAKE_BUILD_TYPE="Release",
        CMAKE_C_COMPILER=str(ENV / "bin/x86_64-conda-linux-gnu-cc"),
        CMAKE_CXX_COMPILER=str(ENV / "bin/x86_64-conda-linux-gnu-c++"),
        CMAKE_MAKE_PROGRAM=str(ENV / "bin/ninja"),
        CMAKE_PREFIX_PATH=f"{DEPS};{ENV}", CMAKE_FIND_PACKAGE_PREFER_CONFIG="ON",
        CMAKE_FIND_USE_PACKAGE_REGISTRY="OFF", CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY="OFF",
        CMAKE_SUPPRESS_REGENERATION="ON", CMAKE_EXPORT_COMPILE_COMMANDS="ON",
        FETCHCONTENT_FULLY_DISCONNECTED="ON", FETCHCONTENT_UPDATES_DISCONNECTED="ON",
        BUILD_Libint2_GENERATOR="OFF", ENABLE_Einsums="ON", ENABLE_OPENMP="ON",
        ENABLE_XHOST="ON", BUILD_SHARED_LIBS="ON", BUILD_TESTING="ON",
        BLAS_TYPE="MKL", LAPACK_TYPE="MKL",
        BLAS_LIBRARIES=str(ENV / "lib/libmkl_rt.so"),
        LAPACK_LIBRARIES=str(ENV / "lib/libmkl_rt.so"),
        BLAS_INCLUDE_DIRS=str(ENV / "include"), LAPACK_INCLUDE_DIRS=str(ENV / "include"),
        CMAKE_C_FLAGS="-march=native", CMAKE_CXX_FLAGS="-march=native",
        CMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG", CMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG",
        CMAKE_BUILD_RPATH=f"{DEPS}/lib;{ENV}/lib",
        CMAKE_INSTALL_RPATH=f"{DEPS}/lib;{ENV}/lib",
        OpenMP_C_FLAGS="-fopenmp", OpenMP_CXX_FLAGS="-fopenmp",
        OpenMP_C_LIB_NAMES="omp", OpenMP_CXX_LIB_NAMES="omp",
        OpenMP_omp_LIBRARY=str(ENV / "lib/libomp.so"), OpenMP_LIBRARY_DIRS=str(ENV / "lib"),
        CMAKE_INSTALL_PREFIX=str(STAGE if inner else RUN / "unused-outer-install"),
        CMAKE_INSTALL_LIBDIR="lib", CMAKE_INSTALL_BINDIR="bin", CMAKE_INSTALL_DATADIR="share",
        CMAKE_INSTALL_INCLUDEDIR="include", PYMOD_INSTALL_LIBDIR="/",
        psi4_INSTALL_CMAKEDIR="share/cmake/psi4",
        TargetLAPACK_INSTALL_CMAKEDIR="share/cmake/TargetLAPACK",
        TargetHDF5_INSTALL_CMAKEDIR="share/cmake/TargetHDF5",
        psi4_CXX_STANDARD=20, psi4_CMAKE_UNITY_BUILD="ON", psi4_CMAKE_UNITY_BUILD_BATCH_SIZE=8,
        psi4_ENABLE_PRECOMPILE_HEADERS="OFF", psi4_SHGAUSS_ORDERING="gaussian",
        MAX_AM_ERI=5, FC_SYMBOL=2)
    for package, rel in {
        "Einsums": "share/cmake/Einsums", "TargetHDF5": "share/cmake/TargetHDF5",
        "Libint2": "lib/cmake/libint2", "Libxc": "lib/cmake/Libxc",
        "gau2grid": "share/cmake/gau2grid", "pybind11": "share/cmake/pybind11",
        "fmt": "lib/cmake/fmt", "argparse": "lib/cmake/argparse",
        "spdlog": "lib/cmake/spdlog", "cpptrace": "lib/cmake/cpptrace",
        "MKL": "lib/cmake/mkl", "Eigen3": "share/eigen3/cmake",
    }.items():
        opts[package + "_DIR"] = str((DEPS if package in ("Einsums", "TargetHDF5") else ENV) / rel)
        opts["CMAKE_INSIST_FIND_PACKAGE_" + package] = "ON"
    for package in ("qcelemental", "qcengine", "optking", "ZLIB"):
        opts["CMAKE_INSIST_FIND_PACKAGE_" + package] = "ON"
    for prefix in ("Python", "Python3"):
        opts[prefix + "_EXECUTABLE"] = PY
        opts[prefix + "_ROOT_DIR"] = str(ENV)
        opts[prefix + "_INCLUDE_DIR"] = str(ENV / "include/python3.13")
        opts[prefix + "_LIBRARY"] = str(ENV / "lib/libpython3.13.so")
        opts[prefix + "_NumPy_INCLUDE_DIR"] = str(ENV / "lib/python3.13/site-packages/numpy/_core/include")
    for name in ("ambit", "adcc", "bse", "CheMPS2", "cppe", "ddx", "dkh", "ecpint",
                 "libefp", "simint", "gauxc", "gdma", "IntegratorXX", "OpenOrbitalOptimizer",
                 "PCMSolver", "qcmanybody", "snsmp2", "v2rdm_casscf", "psi4fockci",
                 "cct3", "gpu_dfcc", "mdi", "BrianQC", "ccsort", "transqt2",
                 "CYTHONIZE", "PLUGIN_TESTING", "ASAN", "TSAN", "MSAN", "UBSAN",
                 "CODE_COVERAGE", "BOUNDS_CHECK", "GENERIC"):
        opts["ENABLE_" + name] = "OFF"
    if inner:
        opts.update(PSI4_ROOT=str(SOURCE), Fortran_ENABLED="OFF", CMAKE_CXX_STANDARD=20,
                    CMAKE_CXX_STANDARD_REQUIRED="ON", CMAKE_CXX_EXTENSIONS="OFF",
                    CMAKE_UNITY_BUILD="ON", CMAKE_UNITY_BUILD_BATCH_SIZE=8,
                    psi4_SHGSHELL_ORDERING=2,
                    TargetLAPACK_DIR=str(STAGE / "share/cmake/TargetLAPACK"),
                    TargetHDF5_DIR=str(STAGE / "share/cmake/TargetHDF5"))
    return ["-D" + k + "=" + str(v) for k, v in opts.items()]


def graph(directory, inner):
    # Small concrete gate on actual generated files, before compilation.
    for n in directory.rglob("*.ninja"):
        for l in n.read_text().splitlines():
            if "COMMAND =" in l and re.search(r"https?://|git clone|pip install|curl |wget ", l):
                raise ValueError("unexpected network/dependency command: " + l)
    cfgs = list(directory.rglob("*-cfgcmd.txt"))
    if any("psi4-core-prefix" not in str(f) for f in cfgs) or (inner and cfgs):
        raise ValueError("unexpected dependency ExternalProject configuration")
    cache = (directory / "CMakeCache.txt").read_text()
    for key, val in (("Einsums_DIR", str(DEPS / "share/cmake/Einsums")),
                     ("CMAKE_CXX_COMPILER", str(ENV / "bin/x86_64-conda-linux-gnu-c++")),
                     ("Python_EXECUTABLE", PY)):
        if not re.search(r"^" + key + r":[^=]*=" + re.escape(val) + r"$", cache, re.M):
            raise ValueError("unexpected dependency/compiler cache: " + key)
    proof = dict(cache_sha256=sha(directory / "CMakeCache.txt"),
                 external_project_cfgs=list(map(str, cfgs)))
    if inner:
        entries = json.loads((directory / "compile_commands.json").read_text())
        isapol = [x for x in entries if "/isapol.dir/" in x["command"]]
        if not isapol:
            raise ValueError("missing native isapol compile graph")
        for x in entries:
            command = x["command"]
            if str(OLD) in command or "/p4qca/" in command or "ccasp-water-port" in command:
                raise ValueError("unexpected old source/dependency prefix in compilation")
            if "-O3" not in command or "-march=native" not in command:
                raise ValueError("missing Release/native optimization")
        for x in isapol:
            if ("-ffp-contract=off" not in x["command"] or "-std=c++20" not in x["command"]
                    or "psi4_SHGSHELL_ORDERING=2" not in x["command"]):
                raise ValueError("isapol precision/standard/shell-order contract differs")
        if not re.search(r"^CMAKE_UNITY_BUILD:[^=]*=ON$", cache, re.M) or not re.search(
                r"^CMAKE_UNITY_BUILD_BATCH_SIZE:[^=]*=8$", cache, re.M):
            raise ValueError("inner unity contract differs")
        proof.update(compile_units=len(entries), isapol_units=len(isapol),
                     compile_commands_sha256=sha(directory / "compile_commands.json"))
    write(RUN / "metadata" / ("inner-graph.json" if inner else "outer-graph.json"), proof)


def run(commit, tree):
    host = socket.gethostname().split(".")[0]
    if (not os.environ.get("SLURM_JOB_ID") or host.startswith("login")
            or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]):
        raise ValueError("compute allocation only")
    identity(commit, tree)
    prep = json.loads((RUN / "metadata/preparation.json").read_text())
    if prep["commit"] != commit or prep["tree"] != tree or BUILD.exists():
        raise ValueError("wrong preparation or preexisting private build")
    for dst, d in prep["configs"].items():
        if sha(dst) != d["sha256"] or sha(d["source"]) != d["sha256"]:
            raise ValueError("dependency config drift")
    for dst, src in prep["symlinks"].items():
        if str(Path(dst).resolve()) != src:
            raise ValueError("dependency symlink drift")
    library = DEPS / "lib/libEinsums.so.1.1.3"
    if sha(library) != PLAN["einsums_library_sha256"]:
        raise ValueError("Einsums 1.1.3 differs from existing campaign pin")
    scratch = Path(PLAN["scratch_parent"]) / ("build-" + os.environ["SLURM_JOB_ID"])
    scratch.mkdir()
    for key in ("TMPDIR", "SCRATCH", "PSI_SCRATCH"):
        os.environ[key] = str(scratch)
    free = shutil.disk_usage(scratch).free
    if free < 100 * 1024**3:
        raise ValueError("less than 100 GiB shared scratch free for build/smoke")
    frontends = {}
    for compiler, frontend in (("cc", "cc1"), ("c++", "cc1plus")):
        tool = ENV / ("bin/x86_64-conda-linux-gnu-" + compiler)
        f = subprocess.check_output([str(tool), "-print-prog-name=" + frontend], text=True).strip()
        if not Path(f).is_file() or not os.access(f, os.X_OK):
            raise ValueError("conda frontend not executable: " + f)
        frontends[frontend] = str(Path(f).resolve())
    # Required numerical hashes happen here on compute, not on login.
    pins = {str(library.resolve()): sha(library)}
    for rel in ("lib/libEinsumsExperimental.so.1.1.3",
                "lib/einsums/__init__.py", "lib/einsums/utils.py",
                "lib/einsums/core.cpython-313-x86_64-linux-gnu.so"):
        p = DEPS / rel
        pins[str(p.resolve())] = sha(p)
    write(RUN / "metadata/dependency-hashes.json", pins)
    header_pins = {}
    for name in ("Einsums", "EinsumsExperimental", "EinsumsPy"):
        for f in sorted((DEPS / "include" / name).rglob("*")):
            if f.is_file():
                header_pins[str(f)] = sha(f)
    write(RUN / "metadata/einsums-header-hashes.json", header_pins)
    write(RUN / "metadata/allocation.json", dict(commit=commit, tree=tree, host=host,
        job_id=os.environ["SLURM_JOB_ID"], scratch=str(scratch), free_scratch_bytes=free,
        environment=str(ENV), compiler_frontends=frontends, started_unix=time.time()))
    call("python-dependency-imports", [PY, "-I", "-S", "-B",
                                     HERE / "python_dependency_check.py"])
    call("outer-configure", [CM, "-S", SOURCE, "-B", BUILD, "-G", "Ninja", *options()])
    graph(BUILD, False)
    for package in ("lapack", "hdf5"):
        install = BUILD / "external/common" / package / "cmake_install.cmake"
        text = install.read_text()
        if re.search(r"TYPE (STATIC_LIBRARY|SHARED_LIBRARY|MODULE|EXECUTABLE)", text):
            raise ValueError("wrapper install is not config-data only")
        call(package + "-config-data", [CM, "-DCMAKE_INSTALL_PREFIX=" + str(STAGE), "-P", install])
    initial = BUILD / "psi4-core-prefix/tmp/psi4-core-cache-Release.cmake"
    call("inner-configure", [CM, "-S", SOURCE / "psi4", "-B", INNER, "-G", "Ninja",
                            "-C", initial, *options(True)])
    graph(INNER, True)
    call("core-parallel1", [CM, "--build", INNER, "--target", "core", "--parallel", "1"])
    call("metadata", [CM, "--build", INNER, "--target", "update_version", "--parallel", "1"])
    call("project-stage", [CM, "--install", INNER, "--prefix", STAGE])
    cores = list((STAGE / "lib/psi4").glob("core*.so"))
    if len(cores) != 1:
        raise ValueError("expected one private staged core")
    write(RUN / "metadata/core.json", dict(path=str(cores[0]), sha256=sha(cores[0]),
                                          commit=commit, tree=tree))
    call("core-dynamic", ["/usr/bin/readelf", "-d", cores[0]])
    shell = ('set -euo pipefail; export PYTHONPATH=; '
             'psiapi="$("$1" -I -S "$2/bin/psi4" --psiapi)"; eval "$psiapi"; '
             'exec "$1" -I -S -B "$3"')
    call("native-smoke", ["/bin/bash", "-c", shell, "psiapi-smoke", PY, STAGE, HERE / "native_smoke.py"])
    identity(commit, tree)
    write(RUN / "metadata/payload-complete.json", dict(commit=commit, tree=tree,
        core_sha256=sha(cores[0]), finished_unix=time.time(), no_benzene=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "run"))
    p.add_argument("--commit", required=True)
    p.add_argument("--tree", required=True)
    a = p.parse_args()
    try:
        (prepare if a.mode == "prepare" else run)(a.commit, a.tree)
    except subprocess.CalledProcessError as e:
        print("Natural phase failure: " + str(e), file=sys.stderr)
        return e.returncode if e.returncode > 0 else 128 - e.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
