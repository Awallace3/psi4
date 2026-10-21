# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Only run as the build allocation's final child. Five explicit small native tests."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "build_smoke_plan.json").read_text())
S, E, D, R = (Path(P[k]) for k in ("source", "environment", "dependency_source", "run"))
STAGE = S / P["build_directory"] / "stage"
DEPS = R / "deps"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(name, obj):
    with (R / "metadata" / name).open("x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def closure(name, require_einsums=True):
    text = Path("/proc/self/maps").read_text()
    (R / "metadata" / (name + ".maps")).write_text(text)
    paths = {Path(l.split()[-1]).resolve() for l in text.splitlines()
             if l.split()[-1].startswith("/") and ".so" in l.split()[-1]}
    einsums = {p for p in paths if p.name.startswith("libEinsums")}
    wanted = (D / "lib/libEinsums.so.1.1.3").resolve()
    allowed = {wanted, (D / "lib/libEinsumsExperimental.so.1.1.3").resolve()}
    if (einsums - allowed or (require_einsums and wanted not in einsums)
            or (not require_einsums and einsums)):
        raise ValueError("wrong/early Einsums SONAME role: " + str(einsums))
    key = {p for p in paths if re.match(
        r"lib(Einsums|mkl|omp|gomp|iomp|int2|xc[.]|gg[.]|hdf5|cpptrace|stdc[+][+]|gcc_s|spdlog|fmt|blas|lapack|adcc|openblas|blis)", p.name)}
    for p in key - allowed:
        if not p.is_relative_to(E):
            raise ValueError("key runtime outside selected environment: " + str(p))
    omp = {p for p in key if re.match(r"lib(omp|gomp|iomp)", p.name)}
    if len(omp) > 1:
        raise ValueError("multiple OpenMP runtimes: " + str(omp))
    write(name + ".json", {str(p): sha(p) for p in sorted(key)})
    return key


def main():
    host = socket.gethostname().split(".")[0]
    if (not os.environ.get("SLURM_JOB_ID") or host.startswith("login")
            or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]):
        raise ValueError("compute allocation only")
    if not sys.flags.no_site or not sys.flags.isolated:
        raise ValueError("start Python -I -S; no scientific startup hooks")
    if Path(os.environ["PYTHONPATH"].split(":")[0]) != STAGE / "lib":
        raise ValueError("PsiAPI was not initialized by this private stage")
    pins = json.loads((R / "metadata/dependency-hashes.json").read_text())
    library = D / "lib/libEinsums.so.1.1.3"
    if sha(library) != P["einsums_library_sha256"] or pins[str(library)] != sha(library):
        raise ValueError("pre-import library hash mismatch")
    closure("initial", require_einsums=False)
    # Process-local selection only; existing ELF/RPATH and numerical defaults untouched.
    handle = ctypes.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    closure("preloaded")
    sys.path[:0] = [str(STAGE / "lib"), str(DEPS / "lib"), str(DEPS / "python"),
                    str(E / "lib/python3.13/site-packages")]
    os.environ["PSIDATADIR"] = str(STAGE / "share/psi4")
    import numpy
    import adcc
    import einsums
    import psi4
    from psi4 import core
    import importlib
    python_dependencies = {}
    for name, version in P["python_dependencies"].items():
        module = importlib.import_module(name)
        if (module.__version__ != version or not Path(module.__file__).resolve().is_relative_to(
                (D / "lib" / name).resolve())):
            raise ValueError("wrong isolated Python dependency: " + name)
        python_dependencies[name] = dict(version=module.__version__, path=module.__file__)
    core_pin = json.loads((R / "metadata/core.json").read_text())
    if Path(core.__file__).resolve() != Path(core_pin["path"]) or sha(core.__file__) != core_pin["sha256"]:
        raise ValueError("wrong private core")
    if not Path(psi4.__file__).resolve().is_relative_to(STAGE):
        raise ValueError("wrong Psi4 package")
    if Path(einsums.__file__).resolve() != D / "lib/einsums/__init__.py":
        raise ValueError("wrong Einsums Python package")
    if Path(einsums.core.__file__).resolve() != D / "lib/einsums/core.cpython-313-x86_64-linux-gnu.so":
        raise ValueError("wrong Einsums Python extension")
    for f, h in pins.items():
        if sha(f) != h:
            raise ValueError("dependency identity drift: " + f)
    psi4.set_num_threads(1)
    psi4.set_memory(4 * 1024**3)
    adcc.set_n_threads(1)
    if adcc.get_n_threads() != 1 or core.get_num_threads() != 1:
        raise ValueError("numerical thread setup refused")
    if not Path(core.get_datadir()).resolve().is_relative_to(STAGE / "share/psi4"):
        raise ValueError("wrong Psi4 data")
    before = closure("before-tests")
    write("imports.json", dict(python=sys.executable, numpy=numpy.__version__, adcc=adcc.__version__,
        psi4=psi4.__file__, core=core.__file__, core_sha256=sha(core.__file__),
        data=core.get_datadir(), einsums=einsums.__file__, einsums_extension=einsums.core.__file__,
        python_dependencies=python_dependencies,
        backend_equivalence="not claimed"))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    import pytest
    sys.path.insert(0, str(S / "tests/pytests"))  # tracked test helpers, not a Psi4 package tree
    argv = ["-c", str(S / "pytest.ini"), "--import-mode=importlib", "-p", "no:cacheprovider",
            "-q", "-o", "addopts=", "--basetemp", str(R / "pytest-tmp"),
            "--junitxml", str(R / "metadata/junit.xml")]
    argv += [str(S / "tests/pytests" / selector) for selector in P["selectors"]]
    write("pytest-argv.json", argv)
    rc = int(pytest.main(argv))
    after = closure("after-tests")
    write("smoke-outcome.json", dict(pytest_returncode=rc, no_benzene=True,
          newly_mapped_key_libraries=sorted(map(str, after - before)),
          einsums_handle_retained=bool(handle)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
