"""New staged ABI/path/mapping probe; no energy, density, or GPU prewarming call."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy import (RUNTIME_PATH, check_required_runtime_maps, check_runtime_maps,
                    require, sha, write_json)


def maps():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        value = line.split()[-1]
        if value.startswith("/") and Path(value).is_file():
            paths.add(value)
    return {p: sha(p) for p in sorted(paths)}


def main():
    output = Path(sys.argv[1])
    observed_path = output.with_name(output.stem + "-observed.json")
    observed = dict(validation="UNVALIDATED", pid=os.getpid(), maps={},
                    env={k: os.getenv(k) for k in ("LD_LIBRARY_PATH", "PYTHONPATH", "OMP_PROC_BIND",
                         "OMP_PLACES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")})
    stage = Path(os.environ["MAJOR2_EXPECTED_STAGE"]).resolve()
    psi4 = None
    import_error = None
    try:
        sys.argv.append("--einsums:no-attach-debugger")
        try:
            import psi4
        finally:
            sys.argv.pop()
        observed.update(module=psi4.__file__, core=psi4.core.__file__,
                        core_sha256=sha(psi4.core.__file__),
                        psi4_version=getattr(psi4, "__version__", None),
                        release_freed_memory=hasattr(psi4.core, "release_freed_memory"))
        # Core-first import ordering is intentional: PyEinsums has DT_RPATH
        # into E/lib. Do not construct a cuEST object or query a GPU here.
        import einsums
        observed.update(pyeinsums=einsums.__file__, pyeinsums_core=einsums.core.__file__,
                        pyeinsums_version=getattr(einsums, "__version__", None),
                        pyeinsums_core_sha256=sha(einsums.core.__file__))
        runtime = os.environ["MAJOR2_HELPER_RUNTIME_ENV"]
        observed["helper_runtime_env"] = dict(path=runtime, sha256=sha(runtime),
                                             text=Path(runtime).read_text())
    except BaseException as exc:
        observed["import_error"] = f"{type(exc).__name__}: {exc}"
        import_error = exc
    finally:
        # Exclusive diagnostic survives import AND validation failure. It is
        # explicitly never a validated receipt; no assertions precede it.
        observed["maps_text"] = Path("/proc/self/maps").read_text()
        try:
            observed["maps"] = maps()
        except BaseException as exc:
            observed["mapping_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_json(observed_path, observed)
    if psi4 is None:
        raise import_error
    require(Path(psi4.__file__).resolve().is_relative_to(stage / "lib/psi4"), "wrong arm/package import")
    require(Path(psi4.core.__file__).resolve().is_relative_to(stage / "lib/psi4"), "wrong arm/core import")
    if import_error is not None:
        raise import_error
    require(hasattr(psi4.core, "release_freed_memory"), "merged core binding absent")
    require(psi4.addons("cuest"), "cuEST addon unavailable")
    loaded = observed["maps"]
    require(os.environ["LD_LIBRARY_PATH"] == RUNTIME_PATH, "effective arm runtime ordering")
    check_runtime_maps(loaded)
    proof = check_required_runtime_maps(loaded)
    write_json(output, dict(observed, validation="VALIDATED", runtime_proof=proof,
                           openmp_maps={p: h for p, h in loaded.items()
                                        if Path(p).name.startswith(("libomp", "libgomp", "libiomp"))}))


if __name__ == "__main__":
    main()
